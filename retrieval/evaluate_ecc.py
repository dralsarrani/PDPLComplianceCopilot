"""
Evaluate the second corpus (NCA Essential Cybersecurity Controls) and the router.

1. Routing on the cybersecurity questions: did route() pick the source the question
   was written for (data/evaluation/ecc_gold.csv, column expected_source)?
2. Routing on the 60 PDPL gold questions (gold_qa.json + gold_qa_test.json): how many
   stay on "pdpl", so that the PDPL answers and their numbers do not change.
3. Second chance: a cybersecurity question the router sent to the PDPL by mistake is
   answered with answer(), which tries the controls when the PDPL does not answer it;
   and the unanswerable PDPL questions that use cybersecurity words must stay "not covered".
4. Search in the controls: Recall@5 and MRR of retrieve_ecc() on the cybersecurity
   questions that name their correct control(s), in the question's language.

ecc_gold.csv is written by the team (not by the model): one row per question,
    id, question, expected_source (ecc | both | pdpl), gold_control_ids ("2-11-3" or "2-11-3; 2-11-4")
The control numbers are in data/processed/ecc_controls_v2024.csv.

Run from the project root (needs OpenAI for the router and the LLM reranking):
    python -m retrieval.evaluate_ecc
Writes results/ecc_eval/questions.csv and results/ecc_eval/summary.json.
"""
import json
import re
import time
from pathlib import Path

import pandas as pd

from retrieval.answer import answer
from retrieval.ecc_retriever import retrieve_ecc
from retrieval.router import keyword_route, route

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_ECC = PROJECT_ROOT / "data" / "evaluation" / "ecc_gold.csv"
GOLD_PDPL = [PROJECT_ROOT / "data" / "evaluation" / "gold_qa.json",
             PROJECT_ROOT / "data" / "evaluation" / "gold_qa_test.json"]
OUT_DIR = PROJECT_ROOT / "results" / "ecc_eval"
K = 5


def control_ids(cell, language):
    """'2-11-3; 2-11-4', 'ar' -> ['ECC-2-11-3-AR', 'ECC-2-11-4-AR']"""
    return [f"ECC-{c.strip().replace('.', '-')}-{language.upper()}" for c in str(cell).replace(",", ";").split(";")
            if c.strip() and c.strip().lower() != "nan"]


def main():
    gold = pd.read_csv(GOLD_ECC, dtype=str, encoding="utf-8-sig").fillna("")
    rows = []
    for q in gold.itertuples():
        start = time.perf_counter()
        routed = route(q.question)
        language = "ar" if re.search("[\u0600-\u06FF]", q.question) else "en"
        row = {"id": q.id, "question": q.question, "expected_source": q.expected_source.strip(),
               "routed": routed["source"], "route_ok": routed["source"] == q.expected_source.strip(),
               "language": language}
        gold_ids = control_ids(q.gold_control_ids, language)
        if gold_ids:
            found = [r["record_id"] for r in retrieve_ecc(q.question, k=K, language=language)]
            hits = [rank for rank, rid in enumerate(found, start=1) if rid in gold_ids]
            row.update(gold=";".join(gold_ids), retrieved=";".join(found),
                       recall_at_5=len(set(found) & set(gold_ids)) / len(gold_ids),
                       rr=1 / hits[0] if hits else 0.0)
        if routed["source"] == "pdpl" and row["expected_source"] != "pdpl":
            # sent to the PDPL by mistake: does answer()'s second chance still reach the controls?
            row["answered_from"] = answer(q.question)["source"]
        row["seconds"] = round(time.perf_counter() - start, 1)
        rows.append(row)
        print(f"{q.id}: routed {routed['source']:<4} (expected {row['expected_source']})"
              + (f"  R@5 {row['recall_at_5']:.2f}  RR {row['rr']:.2f}" if gold_ids else ""))

    pdpl_routes, moved, second_chance = [], [], []
    for path in GOLD_PDPL:
        for q in json.loads(path.read_text(encoding="utf-8"))["questions"]:
            pdpl_routes.append(route(q["question"])["source"])
            if not q["gold_article_ids"] and keyword_route(q["question"]) != "pdpl":
                # an unanswerable PDPL question with cybersecurity words gets the second chance:
                # it must still be "not covered"
                result = answer(q["question"])
                second_chance.append({"id": q["id"], "covered": result["covered"], "source": result["source"]})
                print(f"PDPL {q['id']} (unanswerable, second chance): covered={result['covered']}")
            if pdpl_routes[-1] != "pdpl":  # a PDPL question sent elsewhere: its PDPL answer would change
                moved.append({"id": q["id"], "question": q["question"], "routed": pdpl_routes[-1]})
                print(f"PDPL {q['id']}: routed {pdpl_routes[-1]}  {q['question']}")
    df = pd.DataFrame(rows)
    with_gold = df.dropna(subset=["rr"]) if "rr" in df else df.iloc[0:0]
    summary = {
        "ecc_questions": len(df),
        "routing_accuracy": round(df["route_ok"].mean(), 3),
        "pdpl_gold_questions": len(pdpl_routes),
        "pdpl_kept_on_pdpl": round(pdpl_routes.count("pdpl") / len(pdpl_routes), 3),
        "pdpl_routes": {s: pdpl_routes.count(s) for s in ("pdpl", "ecc", "both")},
        "ecc_search_questions": len(with_gold),
        "ecc_recall_at_5": round(with_gold["recall_at_5"].mean(), 3) if len(with_gold) else None,
        "ecc_mrr": round(with_gold["rr"].mean(), 3) if len(with_gold) else None,
        "routing_accuracy_with_second_chance": round(
            (df["route_ok"] | (df.get("answered_from", pd.Series(index=df.index, dtype=object)) == df["expected_source"])).mean(), 3),
        "pdpl_unanswerable_second_chance": second_chance,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "questions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(moved, columns=["id", "question", "routed"]).to_csv(
        OUT_DIR / "pdpl_questions_moved.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
