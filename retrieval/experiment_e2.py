import json
from pathlib import Path
from rank_bm25 import BM25Okapi
from retrieval.text_processing import tokenize_arabic

ROOT = Path(__file__).resolve().parent.parent

articles = json.loads(
    (ROOT / "data/processed/legal_articles_v1.0.json").read_text(encoding="utf-8")
)

questions = json.loads(
    (ROOT / "data/evaluation/gold_qa.json").read_text(encoding="utf-8")
)["questions"]

questions = [q for q in questions if q["gold_article_ids"]]

# Method 1: Article-based chunking
article_chunks = [
    {
        "id": a["record_id"],
        "text": a["normalized_text"],
        "language": a["language"]
    }
    for a in articles
]

# Method 2: Fixed-length chunking (200 words)
fixed_chunks = []

for a in articles:
    words = a["normalized_text"].split()

    for start in range(0, len(words), 200):
        chunk = " ".join(words[start:start + 200])

        fixed_chunks.append({
            "id": a["record_id"],
            "text": chunk,
            "language": a["language"]
        })


def evaluate(chunks, language):
    selected = [c for c in chunks if c["language"] == language]

    corpus = [tokenize_arabic(c["text"]) for c in selected]
    bm25 = BM25Okapi(corpus)

    recalls = []
    mrrs = []

    for q in questions:
        if q["language"] != language:
            continue

        scores = bm25.get_scores(tokenize_arabic(q["question"]))

        ranked = sorted(
            range(len(selected)),
            key=lambda i: scores[i],
            reverse=True
        )

        # Deduplicate article IDs
        ids = []
        for i in ranked:
            article_id = selected[i]["id"]
            if article_id not in ids:
                ids.append(article_id)

        gold = set(q["gold_article_ids"])

        recall = len(set(ids[:5]) & gold) / len(gold)
        mrr = next(
            (1 / rank for rank, aid in enumerate(ids, 1) if aid in gold),
            0
        )

        recalls.append(recall)
        mrrs.append(mrr)

    return (
        100 * sum(recalls) / len(recalls),
        sum(mrrs) / len(mrrs)
    )


for language in ["ar", "en"]:
    print(f"\nLanguage: {language}")

    for name, chunks in [
        ("Article-based", article_chunks),
        ("Fixed-length", fixed_chunks)
    ]:
        recall, mrr = evaluate(chunks, language)

        print(
            f"{name}: Recall@5 = {recall:.1f}% | MRR = {mrr:.3f}"
        )