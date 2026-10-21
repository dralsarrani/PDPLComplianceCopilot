"""
Test the whole Q&A (retrieve -> answer -> citation check) on the gold questions.

For each question in data/evaluation/gold_qa.json, answer() is run live and compared
with the gold articles:
    answerable questions     answered (not "not covered")
                             right source: at least one cited article is a gold article
                             cited articles that are gold articles (precision)
                             sentences whose quote was found word for word in the article
    unanswerable questions   correctly said "not covered"
Also the average seconds per question. Details per question go to
results/answer_eval.json. Uses OpenAI (one or two calls per question).

Run from the project root:
    python -m retrieval.evaluate_answers
"""
import json
import time
from pathlib import Path

from retrieval.answer import answer
from retrieval.retriever import retrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = PROJECT_ROOT / "data" / "evaluation" / "gold_qa.json"
OUT_PATH = PROJECT_ROOT / "results" / "answer_eval.json"


def pct(part, whole):
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def main():
    questions = json.loads(GOLD_PATH.read_text(encoding="utf-8"))["questions"]
    retrieve("warm up", llm_rerank=False)  # load the models first so their loading time is not counted
    rows = []
    for n, q in enumerate(questions, start=1):
        start = time.perf_counter()
        result = answer(q["question"])
        seconds = time.perf_counter() - start
        gold = set(q["gold_article_ids"])
        cited = result["citations"]
        sentences = result.get("sentences", [])
        rows.append({"id": q["id"], "language": q["language"], "answerable": bool(gold),
                     "covered": result["covered"], "gold": sorted(gold), "cited": cited,
                     "right_source": bool(gold & set(cited)),
                     "cited_in_gold": len(gold & set(cited)),
                     "sentences": len(sentences),
                     "quotes_found": sum(1 for s in sentences if s.get("evidence")),
                     "seconds": round(seconds, 1), "answer": result["answer"]})
        print(f"{n}/{len(questions)} {q['id']}: covered={result['covered']} cited={cited} gold={sorted(gold)}")

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    n_cited = sum(len(r["cited"]) for r in answerable)
    n_sentences = sum(r["sentences"] for r in answerable)
    print(f"\nAnswerable questions: {len(answerable)}")
    print(f"  answered:                       {pct(sum(r['covered'] for r in answerable), len(answerable))}")
    print(f"  right source (a gold article):  {pct(sum(r['right_source'] for r in answerable), len(answerable))}")
    print(f"  cited articles that are gold:   {pct(sum(r['cited_in_gold'] for r in answerable), n_cited)}")
    print(f"  quotes found in the article:    {pct(sum(r['quotes_found'] for r in answerable), n_sentences)}")
    print(f"Unanswerable questions: {len(unanswerable)}")
    print(f"  said not covered:               {pct(sum(not r['covered'] for r in unanswerable), len(unanswerable))}")
    seconds = sorted(r["seconds"] for r in rows)
    print(f"Average seconds per question:     {sum(seconds) / len(seconds):.1f}")
    print(f"Median seconds per question:      {seconds[len(seconds) // 2]:.1f}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Details: {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
