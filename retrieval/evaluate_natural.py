"""
Test "not covered" on 16 natural questions that the PDPL texts do not answer.

The 12 unanswerable gold questions ask for facts no law contains (a date, a phone
number, a count of complaints), so they are easy to refuse. These 16 questions
are written the way our users ask, in formal legal wording, on topics close to
data protection that the PDPL, its Implementing Regulation and its Transfer
Regulation do not address (checked by keyword search in Arabic and English).
Five are marked borderline: no specific article, but general articles exist that
the system may use.

Run from the project root:
    python -m retrieval.evaluate_natural
Details per question go to results/natural_unanswerable_eval.json. Uses OpenAI.
"""
import json
import time
from pathlib import Path

from retrieval.answer import answer
from retrieval.retriever import retrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = PROJECT_ROOT / "data" / "evaluation" / "natural_unanswerable.json"
OUT_PATH = PROJECT_ROOT / "results" / "natural_unanswerable_eval.json"


def main():
    questions = json.loads(GOLD_PATH.read_text(encoding="utf-8"))["questions"]
    retrieve("warm up", llm_rerank=False)  # load the models first so their loading time is not counted
    rows = []
    for n, q in enumerate(questions, start=1):
        start = time.perf_counter()
        result = answer(q["question"])
        seconds = time.perf_counter() - start
        rows.append({"id": q["id"], "language": q["language"], "topic": q["topic"],
                     "borderline": q["borderline"], "covered": result["covered"],
                     "cited": result["citations"], "seconds": round(seconds, 1),
                     "answer": result["answer"]})
        verdict = "answered" if result["covered"] else "not covered"
        print(f"{n}/{len(questions)} {q['id']} ({q['topic']}): {verdict} {result['citations']}")

    def said_not_covered(group):
        return f"{sum(not r['covered'] for r in group)}/{len(group)}"

    clear = [r for r in rows if not r["borderline"]]
    border = [r for r in rows if r["borderline"]]
    print(f"\nSaid not covered, all:        {said_not_covered(rows)}")
    print(f"  clear topics:               {said_not_covered(clear)}")
    print(f"  borderline topics:          {said_not_covered(border)}")
    print(f"  Arabic / English:           {said_not_covered([r for r in rows if r['language'] == 'ar'])}"
          f" / {said_not_covered([r for r in rows if r['language'] == 'en'])}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Details: {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
