
import json
from pathlib import Path
from retrieval.retriever import retrieve

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "evaluation" / "gold_qa.json"

questions = json.loads(GOLD.read_text(encoding="utf-8"))["questions"]

# Threshold values to compare
thresholds = [None, 0.3, 0.5, 0.7]

rows = []
for q in questions:
    articles = retrieve(
        q["question"],
        k=8,
        language=q["language"],
        llm_rerank=False
    )
    best_score = max((a["score"] for a in articles), default=0)
    rows.append((q, best_score))

for threshold in thresholds:
    correct_abstain = 0
    false_abstain = 0
    unanswerable = 0
    answerable = 0

    for q, score in rows:
        abstain = threshold is not None and score < threshold
        is_answerable = bool(q["gold_article_ids"])

        if is_answerable:
            answerable += 1
            false_abstain += abstain
        else:
            unanswerable += 1
            correct_abstain += abstain

    print(f"\nThreshold: {threshold}")
    print(f"Correct abstention: {correct_abstain}/{unanswerable}")
    print(f"False abstention: {false_abstain}/{answerable}")
