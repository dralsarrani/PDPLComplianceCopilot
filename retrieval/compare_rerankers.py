"""
Compare reranker models on the gold questions: accuracy and speed.

For every answerable question in data/evaluation/gold_qa.json the hybrid search
(BM25 + embeddings) picks the candidates, then the reranker orders them.
Settings differ in the reranker, the number of candidates and whether only
articles in the question's language are kept. Prints Recall@5, Recall@8, MRR and
the average seconds per question spent reranking.

Run from the project root (downloads the small model the first time):
    python -m retrieval.compare_rerankers            all settings (~15 min)
    python -m retrieval.compare_rerankers --fast     skip the large model (~3 min)
"""
import json
import sys
import time
from pathlib import Path

from sentence_transformers import CrossEncoder

from retrieval import retriever

GOLD_PATH = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "gold_qa.json"
SMALL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
# name: (reranker model, candidates, keep only the question's language, max tokens per pair)
SETTINGS = {
    "small, 10 candidates": (SMALL, 10, False, 512),
    "small, 20 candidates, same language": (SMALL, 20, True, 512),
    "small, 20 candidates, same language, 256 tokens (used by the app)": (SMALL, 20, True, 256),
    "large bge, 10 candidates": ("BAAI/bge-reranker-v2-m3", 10, False, 512),
}


def main():
    questions = [q for q in json.loads(GOLD_PATH.read_text(encoding="utf-8"))["questions"]
                 if q["gold_article_ids"]]
    print(f"Answerable questions: {len(questions)}")
    models = {}
    for name, (model_name, n_candidates, same_language, max_tokens) in SETTINGS.items():
        if "--fast" in sys.argv and "large" in name:
            continue
        if (model_name, max_tokens) not in models:
            models[model_name, max_tokens] = CrossEncoder(model_name, max_length=max_tokens)
        model = models[model_name, max_tokens]
        recall5, recall8, reciprocal_ranks, seconds = [], [], [], []
        for q in questions:
            articles = retriever.retrieve(q["question"], k=n_candidates, rerank=False, candidates=n_candidates,
                                          language=q["language"] if same_language else None)
            start = time.perf_counter()
            pairs = [(q["question"], f"عنوان المادة: {a['article_title']}\nالنص القانوني: {a['official_text']}")
                     for a in articles]
            scores = [float(s) for s in model.predict(pairs)]
            articles = [a for a, _ in sorted(zip(articles, scores), key=lambda x: x[1], reverse=True)]
            seconds.append(time.perf_counter() - start)
            ranked = [a["record_id"] for a in articles]
            gold = set(q["gold_article_ids"])
            recall5.append(len(set(ranked[:5]) & gold) / len(gold))
            recall8.append(len(set(ranked[:8]) & gold) / len(gold))
            reciprocal_ranks.append(next((1 / r for r, aid in enumerate(ranked, start=1) if aid in gold), 0.0))
        n = len(questions)
        print(f"{name}: Recall@5 {100 * sum(recall5) / n:.1f}%  Recall@8 {100 * sum(recall8) / n:.1f}%  "
              f"MRR {sum(reciprocal_ranks) / n:.3f}  seconds per question {sum(seconds) / n:.1f}")


if __name__ == "__main__":
    main()
