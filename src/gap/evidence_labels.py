"""
Connect the 3A clause classifier to the Gap Analyser (stage 5, simple version).

The Gap Analyser (analyse_v0.py) returns, for each requirement, a sentence quoted
from the policy as evidence. Here the OPP-115 classifier (src/classifier/classify.py)
labels that sentence, and the label is compared with the OPP-115 categories the
checklist links to the requirement (column opp115_categories).

    evidence sentence -> English (gpt-4o-mini, only if it is Arabic)
                      -> classify() -> OPP-115 category with the highest score
                      -> match: that category is one of the requirement's categories

This does not change the analyser's verdicts. It shows the classifier's label next
to each piece of evidence in the app, and measures how well the checklist's link
between OPP-115 categories and PDPL requirements holds on real Saudi policies.

label_evidence(report) adds to each row that has evidence:
    evidence_en     the sentence in English (the same sentence if it was English)
    opp115          the classifier's top category
    opp115_match    True if opp115 is one of the requirement's categories

Run from the project root on the 11 saved analyses in results/gap_v0/:
    python -m src.gap.evidence_labels
Writes results/gap_opp115/evidence_labels.csv and summary.txt.
Translation uses OpenAI; the classifier uses BERT (dralsarrani/opp115-bert),
or the TF-IDF baseline if BERT cannot be loaded.
"""
import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.gap.analyse_v0 import load_checklist

RESULTS_DIR = Path("results/gap_v0")
OUTPUT_DIR = Path("results/gap_opp115")
DEFAULT_MODEL = "gpt-4o-mini"
ARABIC = re.compile(r"[؀-ۿ]")

TRANSLATE_PROMPT = """Translate each Arabic sentence from a privacy policy into English.
Keep the meaning exact. Do not add or leave out anything.
Answer in JSON: {"translations": ["...", "..."]} with one item per sentence, in the same order."""


def translate(sentences, model=None):
    """English for every sentence; English sentences are kept as they are."""
    arabic = [s for s in sentences if ARABIC.search(s)]
    if not arabic:
        return list(sentences)
    load_dotenv()
    response = OpenAI().chat.completions.create(
        model=model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": TRANSLATE_PROMPT},
                  {"role": "user", "content": json.dumps(arabic, ensure_ascii=False)}],
    )
    english = json.loads(response.choices[0].message.content).get("translations", [])
    if len(english) != len(arabic):
        raise ValueError(f"asked for {len(arabic)} translations, got {len(english)}")
    done = dict(zip(arabic, english))
    return [done.get(s, s) for s in sentences]


def requirement_categories(checklist):
    """{req_id: [OPP-115 categories]} from the checklist."""
    return {r.req_id: [c.strip() for c in r.opp115_categories.split(";") if c.strip()]
            for r in checklist.itertuples()}


def label_evidence(report, checklist=None):
    """Add evidence_en, opp115 and opp115_match to every row with evidence.
    Rows already labelled are left as they are. Returns the same report."""
    from src.classifier.classify import classify  # loads torch only when needed

    rows = [r for r in report if r.get("evidence") and "opp115" not in r]
    if not rows:
        return report
    categories = requirement_categories(load_checklist() if checklist is None else checklist)
    english = translate([r["evidence"] for r in rows])
    for row, en, result in zip(rows, english, classify(english)):
        top = max(result["scores"], key=result["scores"].get)
        row["evidence_en"] = en
        row["opp115"] = top
        row["opp115_match"] = top in categories.get(row["req_id"], [])
        row["opp115_model"] = result["model"]
    return report


def main():
    checklist = load_checklist()
    categories = requirement_categories(checklist)
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        report = label_evidence(json.loads(path.read_text(encoding="utf-8")), checklist)
        for r in report:
            if "opp115" in r:
                rows.append({"policy": path.stem, "req_id": r["req_id"], "status": r["status"],
                             "arabic": bool(ARABIC.search(r["evidence"])),
                             "expected": "; ".join(categories.get(r["req_id"], [])),
                             "opp115": r["opp115"], "match": r["opp115_match"],
                             "model": r["opp115_model"],
                             "evidence": r["evidence"], "evidence_en": r["evidence_en"]})
        print(f"{path.stem}: {sum('opp115' in r for r in report)} evidence sentences labelled")

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "evidence_labels.csv", index=False, encoding="utf-8-sig")

    def rate(part):
        return f"{part['match'].mean() * 100:.1f}% ({part['match'].sum()}/{len(part)})"

    per_req = df.groupby("req_id").apply(lambda g: pd.Series({
        "expected": g["expected"].iloc[0],
        "match": f"{g['match'].mean() * 100:.0f}%",
        "n": len(g),
        "most_common_label": g["opp115"].mode().iloc[0]}))
    lines = [f"Policies: {df['policy'].nunique()}   evidence sentences: {len(df)}   "
             f"classifier: {', '.join(sorted(df['model'].unique()))}",
             f"Top OPP-115 label is one of the requirement's categories: {rate(df)}",
             f"  Arabic evidence (translated): {rate(df[df['arabic']])}" if df["arabic"].any() else "",
             f"  English evidence:             {rate(df[~df['arabic']])}" if (~df["arabic"]).any() else "",
             "", "Per requirement:", per_req.to_string()]
    summary = "\n".join(line for line in lines if line is not None)
    (OUTPUT_DIR / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    print("\nSaved", OUTPUT_DIR / "evidence_labels.csv", "and", OUTPUT_DIR / "summary.txt")


if __name__ == "__main__":
    main()
