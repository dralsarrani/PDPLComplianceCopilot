"""
Search accuracy for each language separately, for every search method.

The brief asks for Recall@5 and MRR "for each language". For every answerable
question in data/evaluation/gold_qa.json this compares:
    BM25            keywords only
    Embeddings      meaning only (paraphrase-multilingual-MiniLM)
    Hybrid          both, fused with Reciprocal Rank Fusion
    Hybrid+rerank   hybrid top 40, ordered by the small reranker
    Final (app)     hybrid+rerank, then gpt-4o-mini picks its 8 best of the top 10
Every method keeps only articles in the question's language, like the app.
"Final" calls OpenAI and is skipped when OPENAI_API_KEY is not set.

Prints a table and saves it to results/retrieval_by_language.csv.

Run from the project root:
    python -m retrieval.evaluate_by_language
"""
import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from retrieval import retriever
from retrieval.text_processing import tokenize_arabic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = PROJECT_ROOT / "data" / "evaluation" / "gold_qa.json"
OUT_PATH = PROJECT_ROOT / "results" / "retrieval_by_language.csv"


def rankings(question, language):
    """Return {method: [record_id, ...]} for one question."""
    retriever._setup()
    c = retriever._cache
    articles = c["articles"]
    same = lambda idxs: [i for i in idxs if articles[i]["language"] == language]

    scores = c["bm25"].get_scores(tokenize_arabic(question))
    bm25 = sorted(range(len(articles)), key=lambda i: scores[i], reverse=True)
    dense_ids = c["collection"].query(query_embeddings=c["dense"].encode([question]).tolist(),
                                      n_results=len(articles))["ids"][0]
    dense = [c["by_id"][rid] for rid in dense_ids]
    hybrid = retriever._rrf([bm25, dense])

    ids = lambda idxs: [articles[i]["record_id"] for i in idxs]
    result = {"BM25": ids(same(bm25)), "Embeddings": ids(same(dense)), "Hybrid": ids(same(hybrid))}
    reranked = retriever.retrieve(question, k=retriever.CANDIDATES, language=language, llm_rerank=False)
    result["Hybrid+rerank"] = [a["record_id"] for a in reranked]
    if os.environ.get("OPENAI_API_KEY"):
        final = retriever.retrieve(question, k=retriever.CANDIDATES, language=language)
        result["Final (app)"] = [a["record_id"] for a in final]
    return result


def main():
    load_dotenv()
    questions = [q for q in json.loads(GOLD_PATH.read_text(encoding="utf-8"))["questions"] if q["gold_article_ids"]]
    print(f"Answerable questions: {len(questions)} "
          f"(Arabic {sum(q['language'] == 'ar' for q in questions)}, English {sum(q['language'] == 'en' for q in questions)})")
    scores = {}  # (method, language) -> [(recall5, reciprocal rank)]
    for n, q in enumerate(questions, start=1):
        print(f"  {n}/{len(questions)} {q['id']}", end="\r")
        gold = set(q["gold_article_ids"])
        for method, ranked in rankings(q["question"], q["language"]).items():
            rr = next((1 / r for r, aid in enumerate(ranked, start=1) if aid in gold), 0.0)
            row = (len(set(ranked[:5]) & gold) / len(gold), rr)
            scores.setdefault((method, q["language"]), []).append(row)
            scores.setdefault((method, "all"), []).append(row)

    rows = []
    for method in dict.fromkeys(m for m, _ in scores):
        for language in ("ar", "en", "all"):
            values = scores[method, language]
            rows.append({"method": method, "language": language, "questions": len(values),
                         "recall_at_5": round(100 * sum(v[0] for v in values) / len(values), 1),
                         "mrr": round(sum(v[1] for v in values) / len(values), 3)})

    print(f"\n{'Method':<15}{'Arabic R@5':>12}{'Arabic MRR':>12}{'English R@5':>13}{'English MRR':>13}{'All R@5':>10}{'All MRR':>10}")
    for method in dict.fromkeys(r["method"] for r in rows):
        r = {x["language"]: x for x in rows if x["method"] == method}
        print(f"{method:<15}{r['ar']['recall_at_5']:>11}%{r['ar']['mrr']:>12}{r['en']['recall_at_5']:>12}%"
              f"{r['en']['mrr']:>13}{r['all']['recall_at_5']:>9}%{r['all']['mrr']:>10}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
