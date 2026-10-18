"""
Test check_citations() on fixed cases with known answers.

data/evaluation/citation_cases.json has Arabic and English sentences with:
    correct        citation supports the sentence      expected: supported
    invented       article id does not exist           expected: not_in_corpus
    unrelated      real article about something else   expected: unsupported
    missing        sentence with no citation           expected: uncited
    not_retrieved  real article the search never found expected: not_retrieved

The sentences are written by hand (not generated), so every run tests the
checker on the same known mistakes. The support step calls OpenAI.

Run from the project root:
    python -m retrieval.evaluate_citation_check
Writes results/citation_check_eval.json
"""
import json
from collections import Counter
from pathlib import Path

from retrieval.citation_check import check_citations

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "citation_cases.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "citation_check_eval.json"


def predicted_label(check):
    row = check["sentences"][0]
    if row["uncited"]:
        return "uncited"
    return row["citations"][0]["status"]


def main():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    rows = []
    print(f"{'Case':<5} {'Lang':<5} {'Type':<14} {'Expected':<14} {'Got':<14}")
    for case in cases:
        check = check_citations([{"text": case["sentence"], "citations": case["citations"]}],
                                case["retrieved"])
        got = predicted_label(check)
        ok = got == case["expected"]
        rows.append({**case, "got": got, "correct": ok, "check": check})
        print(f"{case['id']:<5} {case['lang']:<5} {case['type']:<14} {case['expected']:<14} {got:<14} "
              f"{'OK' if ok else 'WRONG'}")

    total = sum(r["correct"] for r in rows)
    print(f"\nCorrect: {total}/{len(rows)} ({total / len(rows):.0%})")
    by_type, by_lang = Counter(), Counter()
    for r in rows:
        by_type[(r["type"], r["correct"])] += 1
        by_lang[(r["lang"], r["correct"])] += 1
    for t in dict.fromkeys(r["type"] for r in rows):
        n = by_type[(t, True)] + by_type[(t, False)]
        print(f"  {t:<14} {by_type[(t, True)]}/{n}")
    for lang in ("ar", "en"):
        n = by_lang[(lang, True)] + by_lang[(lang, False)]
        print(f"  {lang:<14} {by_lang[(lang, True)]}/{n}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
