"""
Try ways to raise search accuracy on the gold questions, and measure each one.

Settings (each adds to the one before):
  A  the app today: hybrid search, top 20 candidates, small reranker
  B  top 40 candidates instead of 20 (several gold articles sit at hybrid rank 21-40)
  C  + query rewriting: gpt-4o-mini rewrites the question 2 ways in legal wording,
       and the hybrid search runs on all 3 versions (agentic-style retrieval)
  D  + LLM reranking: gpt-4o-mini reorders the top 15 from the small reranker
  E  B + LLM reranking, without query rewriting (C added little, so is it needed?)
  F  E, but the LLM sees the top 10 with 300 characters each and returns only its top 8 (faster)
  G  20 candidates instead of 40 (less small-reranker work), LLM sees the top 15 with 300 characters, returns its top 8

Prints Recall@5, Recall@8, MRR and seconds per question, for all questions and
for each language (the brief asks for each language separately).
C and D call OpenAI (about 2 small calls per question, a few cents in total);
they are skipped when OPENAI_API_KEY is not set.

Tuning on these questions makes their numbers optimistic: keep the setting that
wins here, then report it on questions the search was not tuned on.

Run from the project root:
    python -m retrieval.compare_boost          all settings
    python -m retrieval.compare_boost E        only the settings named
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from retrieval import retriever
from retrieval.text_processing import tokenize_arabic

GOLD_PATH = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "gold_qa.json"
LLM_MODEL = "gpt-4o-mini"


def llm_json(prompt):
    from openai import OpenAI
    response = OpenAI().chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", LLM_MODEL), temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}])
    return json.loads(response.choices[0].message.content)


def rewrite(question, language):
    lang = "Arabic" if language == "ar" else "English"
    prompt = (f"A user asked this question about the Saudi Personal Data Protection Law (PDPL), "
              f"its Implementing Regulation and its Transfer Regulation:\n{question}\n\n"
              f"Write 2 short search queries in {lang} that use the formal legal terms the law itself "
              f"would use for the topics in this question (for example the name of the right, "
              f"obligation or procedure). Do not answer the question. "
              f'Return JSON: {{"queries": ["...", "..."]}}')
    return [q for q in llm_json(prompt).get("queries", []) if isinstance(q, str)][:2]


def candidates(queries, language, n):
    """Hybrid search on several queries at once: one BM25 and one dense ranking per
    query, all fused with Reciprocal Rank Fusion, then one language, top n."""
    retriever._setup()
    c = retriever._cache
    articles = c["articles"]
    rankings = []
    for q in queries:
        scores = c["bm25"].get_scores(tokenize_arabic(q))
        rankings.append(sorted(range(len(articles)), key=lambda i: scores[i], reverse=True))
        dense = c["collection"].query(query_embeddings=c["dense"].encode([q]).tolist(), n_results=len(articles))
        rankings.append([c["by_id"][rid] for rid in dense["ids"][0]])
    fused = [i for i in retriever._rrf(rankings) if articles[i]["language"] == language]
    return fused[:n]


def rerank_small(question, idxs):
    c = retriever._cache
    if "reranker" not in c:
        c["reranker"] = retriever._load_reranker()
    a = c["articles"]
    pairs = [(question, f"عنوان المادة: {a[i]['article_title']}\nالنص القانوني: {a[i]['official_text']}") for i in idxs]
    scores = [float(s) for s in c["reranker"].predict(pairs)]
    return [i for i, _ in sorted(zip(idxs, scores), key=lambda x: x[1], reverse=True)]


def rerank_llm(question, idxs, top=15, chars=500, keep=None):
    a = retriever._cache["articles"]
    head = idxs[:top]
    listing = "\n\n".join(f"[{a[i]['record_id']}] {a[i]['article_title']}\n{a[i]['official_text'][:chars]}" for i in head)
    prompt = (f"Question: {question}\n\nArticles:\n{listing}\n\n"
              f"Rank the articles by how directly they answer the question, most relevant first. "
              + (f"Return the {keep} most relevant ids. " if keep else "Include every article id exactly once. ") +
              f'Return JSON: {{"ranking": ["id", ...]}}')
    order = llm_json(prompt).get("ranking", [])
    by_rid = {a[i]["record_id"]: i for i in head}
    ranked = [by_rid[r] for r in order if r in by_rid]
    ranked += [i for i in head if i not in ranked]
    return ranked + idxs[top:]


def search(q, setting):
    question, language = q["question"], q["language"]
    queries = [question]
    if setting in ("C", "D"):
        queries += rewrite(question, language)
    idxs = candidates(queries, language, 20 if setting in ("A", "G") else 40)
    idxs = rerank_small(question, idxs)
    if setting in ("D", "E"):
        idxs = rerank_llm(question, idxs)
    elif setting == "F":
        idxs = rerank_llm(question, idxs, top=10, chars=300, keep=8)
    elif setting == "G":
        idxs = rerank_llm(question, idxs, top=15, chars=300, keep=8)
    return [retriever._cache["articles"][i]["record_id"] for i in idxs]


def scores(rows):
    n = len(rows) or 1
    return (f"Recall@5 {100 * sum(r[0] for r in rows) / n:.1f}%  Recall@8 {100 * sum(r[1] for r in rows) / n:.1f}%  "
            f"MRR {sum(r[2] for r in rows) / n:.3f}")


def main():
    load_dotenv()
    questions = [q for q in json.loads(GOLD_PATH.read_text(encoding="utf-8"))["questions"] if q["gold_article_ids"]]
    print(f"Answerable questions: {len(questions)}")
    settings = ["A", "B"] + (["C", "D", "E", "F", "G"] if os.environ.get("OPENAI_API_KEY") else [])
    chosen = [a.upper() for a in sys.argv[1:]]
    if chosen:
        settings = [s for s in settings if s in chosen]
    for setting in settings:
        rows, seconds = {"ar": [], "en": []}, []
        for q in questions:
            start = time.perf_counter()
            ranked = search(q, setting)
            seconds.append(time.perf_counter() - start)
            gold = set(q["gold_article_ids"])
            rows[q["language"]].append((len(set(ranked[:5]) & gold) / len(gold), len(set(ranked[:8]) & gold) / len(gold),
                                        next((1 / r for r, aid in enumerate(ranked, start=1) if aid in gold), 0.0)))
        print(f"\n{setting}: {scores(rows['ar'] + rows['en'])}  seconds per question {sum(seconds) / len(seconds):.1f}")
        print(f"   Arabic : {scores(rows['ar'])}")
        print(f"   English: {scores(rows['en'])}")


if __name__ == "__main__":
    main()
