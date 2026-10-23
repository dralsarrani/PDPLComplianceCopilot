"""
retrieve_ecc(question, k=5): the top NCA Essential Cybersecurity Controls (ECC-2:2024)
for a question. The second corpus next to the PDPL (src/nca/build_ecc.py builds it).

The same pipeline as retriever.py, on its own index so the PDPL search does not change:
    BM25 + dense embeddings (ChromaDB collection "ecc_controls")
        -> Reciprocal Rank Fusion -> top 40
        -> the same small reranker -> gpt-4o-mini picks its 8 best of the top 10 -> top k

Each control is there in Arabic and English (like the PDPL articles); language="ar" or
"en" keeps the controls in the question's language.

Run from the project root:
    python -m retrieval.ecc_retriever "How often must penetration tests be done?"
"""
import json
import re
import sys

import chromadb
from rank_bm25 import BM25Okapi

from retrieval import retriever
from retrieval.retriever import CANDIDATES, CHROMA_PATH, PROJECT_ROOT, _llm_rerank, _rrf
from retrieval.text_processing import tokenize_arabic

ECC_PATH = PROJECT_ROOT / "data" / "processed" / "ecc_controls_v2024.json"
COLLECTION = "ecc_controls"

_cache = {}


def available():
    return ECC_PATH.exists()


def _setup():
    """Load the controls, BM25 and their ChromaDB collection once per process. The
    embedding model and the reranker are shared with the PDPL retriever."""
    if "articles" in _cache:
        return
    retriever._setup()  # loads the shared embedding model
    controls = json.loads(ECC_PATH.read_text(encoding="utf-8"))
    # the subdomain title is part of what a control is about ("2-10 Vulnerabilities Management")
    texts = [f"{c['article_title']}. {c['official_text']}" for c in controls]
    _cache["articles"] = controls
    _cache["by_id"] = {c["record_id"]: i for i, c in enumerate(controls)}
    _cache["bm25"] = BM25Okapi([tokenize_arabic(t) for t in texts])
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    if collection.count() != len(controls):
        print("Building ChromaDB embeddings for the NCA controls (first run only)...")
        client.delete_collection(COLLECTION)
        collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        collection.add(ids=[c["record_id"] for c in controls],
                       embeddings=retriever._cache["dense"].encode(texts).tolist(),
                       documents=texts,
                       metadatas=[{"document_code": "ECC", "article_number": c["article_number"]}
                                  for c in controls])
    _cache["collection"] = collection


def retrieve_ecc(question, k=5, rerank=True, llm_rerank=True, language=None):
    """Return the top k controls as dicts with the same keys as retriever.retrieve()."""
    _setup()
    controls = _cache["articles"]
    bm25_scores = _cache["bm25"].get_scores(tokenize_arabic(question))
    bm25_ranking = sorted(range(len(controls)), key=lambda i: bm25_scores[i], reverse=True)
    query_embedding = retriever._cache["dense"].encode([question]).tolist()
    dense = _cache["collection"].query(query_embeddings=query_embedding, n_results=len(controls))
    dense_ranking = [_cache["by_id"][rid] for rid in dense["ids"][0]]
    fused = _rrf([bm25_ranking, dense_ranking])
    if language:
        fused = [i for i in fused if controls[i]["language"] == language]
    candidates = fused[:CANDIDATES]

    if rerank:
        if "reranker" not in retriever._cache:
            retriever._cache["reranker"] = retriever._load_reranker()
        pairs = [(question, f"{controls[i]['article_title']}\n{controls[i]['official_text']}") for i in candidates]
        scores = [float(s) for s in retriever._cache["reranker"].predict(pairs)]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        if llm_rerank:
            ranked = _llm_rerank(question, ranked, articles=controls)
    else:
        ranked = [(i, 1 / (retriever.RRF_K + rank + 1)) for rank, i in enumerate(candidates)]

    return [{"record_id": controls[i]["record_id"], "document_code": "ECC",
             "article_number": controls[i]["article_number"], "article_title": controls[i]["article_title"],
             "language": controls[i]["language"], "official_text": controls[i]["official_text"], "score": round(score, 4)}
            for i, score in ranked[:k]]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or input("Enter your question: ")
    for rank, r in enumerate(retrieve_ecc(query, language="ar" if re.search("[\u0600-\u06FF]", query) else "en"), start=1):
        print(f"{rank}. {r['record_id']}  {r['article_title']}  score={r['score']}")
