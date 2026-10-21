import json
from pathlib import Path

from retrieval import retriever
from retrieval.text_processing import tokenize_arabic

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "data" / "evaluation" / "gold_qa.json"

questions = json.loads(
    GOLD_PATH.read_text(encoding="utf-8")
)["questions"]

questions = [q for q in questions if q["gold_article_ids"]]

retriever._setup()
cache = retriever._cache
articles = cache["articles"]


def evaluate(method):
    recalls = []
    mrrs = []

    for q in questions:
        question = q["question"]
        language = q["language"]
        gold = set(q["gold_article_ids"])

        if method == "BM25":
            scores = cache["bm25"].get_scores(
                tokenize_arabic(question)
            )

            ranked = sorted(
                range(len(articles)),
                key=lambda i: scores[i],
                reverse=True
            )

            ids = [
                articles[i]["record_id"]
                for i in ranked
                if articles[i]["language"] == language
            ]

        else:
            results = retriever.retrieve(
                question,
                k=40,
                rerank=(method == "Hybrid + Reranker"),
                candidates=40,
                language=language,
                llm_rerank=False
            )

            ids = [a["record_id"] for a in results]

        recall = len(set(ids[:5]) & gold) / len(gold)

        mrr = next(
            (1 / rank for rank, aid in enumerate(ids, 1)
             if aid in gold),
            0
        )

        recalls.append(recall)
        mrrs.append(mrr)

    return (
        100 * sum(recalls) / len(recalls),
        sum(mrrs) / len(mrrs)
    )


for method in ["BM25", "Hybrid", "Hybrid + Reranker"]:
    recall, mrr = evaluate(method)

    print(
        f"{method}: Recall@5={recall:.1f}% | MRR={mrr:.3f}"
    )