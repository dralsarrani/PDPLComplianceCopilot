"""
retrieve(question, k=5): the top PDPL articles for a question.

Wraps the hybrid pipeline from hybrid_retrieval.py and reranking.py as one
function the Gap Analyser and the app can import:
    BM25 (tokenize_arabic)  +  dense embeddings stored in ChromaDB
        -> Reciprocal Rank Fusion (k=60) -> top 40 (optionally one language only)
        -> reranker (mmarco-mMiniLMv2, small and fast)
        -> gpt-4o-mini picks its 8 best of the top 10 (falls back to the small reranker's
           order when OpenAI is not reachable) -> top k

The embeddings are computed once and saved in chroma_db/ (ignored by git).
Delete that folder to rebuild them.

Run from the project root:
    python -m retrieval.retriever "هل يجوز نقل بيانات العملاء خارج المملكة؟"
"""
import json
import sys
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from retrieval.text_processing import tokenize_arabic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_PATH = PROJECT_ROOT / "data" / "processed" / "legal_articles_v1.0.json"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION = "pdpl_articles"
DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Small multilingual reranker. Measured with python -m retrieval.compare_rerankers
# on the 34 answerable gold questions (laptop CPU, seconds of reranking per question):
#   BAAI/bge-reranker-v2-m3, 10 candidates, 512 tokens   Recall@5 71.6%  MRR 0.868  ~20 s
#   this model, 20 candidates, same language, 256 tokens  Recall@5 81.4%  Recall@8 85.8%  MRR 0.877  ~1.7 s
# Then python -m retrieval.compare_boost (seconds of the whole search per question):
#   40 candidates + gpt-4o-mini reorders the top 15 (500 chars each)       Recall@5 90.7%  Recall@8 94.6%  MRR 0.926  ~7.0 s
#   40 candidates + gpt-4o-mini picks 8 of the top 10 (300 chars each)      Recall@5 90.2%  Recall@8 90.2%  MRR 0.929  ~4.6 s  <- used
#   20 candidates + gpt-4o-mini picks 8 of the top 15 (300 chars each)      Recall@5 84.3%  Recall@8 86.8%  MRR 0.936  ~3.3 s
#   (used setting: Arabic Recall@5 88.2%, English 92.2%; rewriting the question first added nothing)
# These numbers were tuned on the same questions, so they are optimistic.
# answer() passes the top 8 articles to the LLM.
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RRF_K = 60
CANDIDATES = 40
LLM_RERANK_MODEL = "gpt-4o-mini"
LLM_RERANK_TOP = 10
LLM_RERANK_CHARS = 300
LLM_RERANK_KEEP = 8
# The reranker reads at most this many tokens of each (question, article) pair:
# the start of an article states its topic, and 256 was both faster and more
# accurate than 512 in the comparison above.
RERANK_MAX_TOKENS = 256

_cache = {}


def _load_dense_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(DENSE_MODEL)


def _load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_TOKENS)


def _setup():
    """Load articles, BM25 and the ChromaDB collection once per process."""
    if "articles" in _cache:
        return
    articles = json.loads(ARTICLES_PATH.read_text(encoding="utf-8"))
    texts = [a["normalized_text"] for a in articles]
    _cache["articles"] = articles
    _cache["by_id"] = {a["record_id"]: i for i, a in enumerate(articles)}
    _cache["bm25"] = BM25Okapi([tokenize_arabic(t) for t in texts])
    _cache["dense"] = _load_dense_model()

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    if collection.count() != len(articles):
        print("Building ChromaDB embeddings (first run only)...")
        client.delete_collection(COLLECTION)
        collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        embeddings = _cache["dense"].encode(texts).tolist()
        collection.add(
            ids=[a["record_id"] for a in articles],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"document_code": a["document_code"], "article_number": a["article_number"],
                        "language": a["language"]} for a in articles],
        )
    _cache["collection"] = collection


def _rrf(rankings):
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1 / (RRF_K + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _llm_rerank(question, ranked, articles=None):
    """Let gpt-4o-mini pick and order its LLM_RERANK_KEEP best of the first
    LLM_RERANK_TOP of ranked [(index, score)]; the rest keep the small reranker's order.
    On any OpenAI problem the order is kept as it is, so search still works offline.
    articles: the list the indexes point into (default: the PDPL corpus; the NCA
    controls pass their own, see ecc_retriever.py)."""
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    articles = articles or _cache["articles"]
    head, tail = ranked[:LLM_RERANK_TOP], ranked[LLM_RERANK_TOP:]
    listing = "\n\n".join(f"[{articles[i]['record_id']}] {articles[i]['article_title']}\n"
                           f"{articles[i]['official_text'][:LLM_RERANK_CHARS]}" for i, _ in head)
    prompt = (f"Question: {question}\n\nArticles:\n{listing}\n\n"
              f"Rank the articles by how directly they answer the question, most relevant first. "
              f"Return the {LLM_RERANK_KEEP} most relevant ids. "
              f'Return JSON: {{"ranking": ["id", ...]}}')
    try:
        load_dotenv()
        response = OpenAI().chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", LLM_RERANK_MODEL), temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}])
        order = json.loads(response.choices[0].message.content).get("ranking", [])
    except Exception as error:
        print(f"LLM reranking skipped ({type(error).__name__}); using the small reranker's order.")
        return ranked
    by_rid = {articles[i]["record_id"]: (i, s) for i, s in head}
    reordered = [by_rid.pop(rid) for rid in order if rid in by_rid]
    reordered += [item for item in head if articles[item[0]]["record_id"] in by_rid]
    return reordered + tail


def retrieve(question, k=5, rerank=True, candidates=CANDIDATES, language=None, llm_rerank=True):
    """Return the top k articles as dicts:
    {record_id, document_code, article_number, article_title, language, official_text, score}
    score is the small reranker's score (or the RRF score when rerank=False); with
    llm_rerank the order comes from gpt-4o-mini, so the scores need not be sorted.
    language ("ar" or "en") keeps only articles in that language, so the Arabic
    and English copies of one article do not take two of the candidate places."""
    _setup()
    articles = _cache["articles"]

    bm25_scores = _cache["bm25"].get_scores(tokenize_arabic(question))
    bm25_ranking = sorted(range(len(articles)), key=lambda i: bm25_scores[i], reverse=True)

    query_embedding = _cache["dense"].encode([question]).tolist()
    dense = _cache["collection"].query(query_embeddings=query_embedding, n_results=len(articles))
    dense_ranking = [_cache["by_id"][rid] for rid in dense["ids"][0]]

    fused = _rrf([bm25_ranking, dense_ranking])
    if language:
        fused = [i for i in fused if articles[i]["language"] == language]
    candidates = fused[:candidates]

    if rerank:
        if "reranker" not in _cache:
            _cache["reranker"] = _load_reranker()
        pairs = [(question, f"عنوان المادة: {articles[i]['article_title']}\n"
                            f"النص القانوني: {articles[i]['official_text']}") for i in candidates]
        scores = [float(s) for s in _cache["reranker"].predict(pairs)]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        if llm_rerank:
            ranked = _llm_rerank(question, ranked)
    else:
        ranked = [(i, 1 / (RRF_K + rank + 1)) for rank, i in enumerate(candidates)]

    results = []
    for i, score in ranked[:k]:
        a = articles[i]
        results.append({
            "record_id": a["record_id"],
            "document_code": a["document_code"],
            "article_number": a["article_number"],
            "article_title": a["article_title"],
            "language": a["language"],
            "official_text": a["official_text"],
            "score": round(score, 4),
        })
    return results


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or input("Enter your question: ")
    for rank, r in enumerate(retrieve(query), start=1):
        print(f"{rank}. {r['record_id']}  score={r['score']}")
