"""
Evaluate the Gap Analyser against the team's own labels (stage 6).

The team labelled 11 Saudi privacy policies by hand: for each of the 17 checklist
requirements, covered / partial / missing (data/labels/team_labels.csv). The labels
were made without seeing the analyser's answers. Team rule: a requirement the policy
never mentions is missing.

This script runs the analyser (src/gap/analyse_v0.py) on the same 11 policies and
compares its answer with the team's label for every (policy, requirement) pair.

Prints and saves (results/gap_eval/):
    summary.txt            overall accuracy, accuracy if partial and covered are merged,
                           Cohen's kappa, confusion matrix, precision/recall/F1 per class
    per_requirement.csv    accuracy for each requirement over the 11 policies
    per_policy.csv         accuracy for each policy over the 17 requirements
    disagreements.csv      every pair where the analyser and the team differ,
                           with the analyser's quote and reason (for error analysis)

Analyser results already in results/gap_v0/ are reused; pass --rerun to run the
analyser again on every policy (about 4 small gpt-4o-mini calls per policy).
Pass --v1 to evaluate analyse_v1 (yes/no questions) instead: results/gap_v1/, results/gap_eval_v1/.

Run from the project root:
    python -m src.gap.evaluate_gap
    python -m src.gap.evaluate_gap --rerun
    python -m src.gap.evaluate_gap --v1 --rerun
"""
import json
import sys
from pathlib import Path

import pandas as pd

from src.gap import analyse_v0, analyse_v1

LABELS = Path("data/labels/team_labels.csv")
POLICIES = Path("data/policies")
TO_STATUS = {"مغطى": "covered", "جزئي": "partial", "ناقص": "missing"}
CLASSES = ["covered", "partial", "missing"]


def analyser_results(policy, rerun, analyser):
    path = analyser.OUTPUT_DIR / f"{policy}.json"
    if rerun or not path.exists():
        print(f"Running the analyser on {policy}...")
        report = analyser.analyse((POLICIES / f"{policy}.txt").read_text(encoding="utf-8"))
        analyser.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {r["req_id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def kappa(team, model):
    """Cohen's kappa: agreement corrected for the agreement expected by chance."""
    n = len(team)
    observed = sum(t == m for t, m in zip(team, model)) / n
    expected = sum((team.count(c) / n) * (model.count(c) / n) for c in CLASSES)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main():
    rerun = "--rerun" in sys.argv
    analyser = analyse_v1 if "--v1" in sys.argv else analyse_v0
    eval_dir = Path("results/gap_eval_v1" if analyser is analyse_v1 else "results/gap_eval")
    labels = pd.read_csv(LABELS).fillna("")
    labels["team"] = labels["label"].map(TO_STATUS)

    rows = []
    for policy in labels["policy"].unique():
        results = analyser_results(policy, rerun, analyser)
        for r in labels[labels["policy"] == policy].itertuples():
            a = results.get(r.req, {})
            rows.append({"policy": policy, "req": r.req, "labeller": r.labeller,
                         "team": r.team, "analyser": a.get("status", "error"),
                         "evidence": a.get("evidence", ""), "reason": a.get("reason", ""),
                         "team_note": r.note})
    df = pd.DataFrame(rows)
    df["agree"] = df["team"] == df["analyser"]
    merge = {"covered": "mentioned", "partial": "mentioned", "missing": "missing"}
    df["agree_2way"] = df["team"].map(merge) == df["analyser"].map(merge)

    team, model = df["team"].tolist(), df["analyser"].tolist()
    lines = [f"Policies: {df['policy'].nunique()}   pairs (policy x requirement): {len(df)}",
             f"Accuracy (3 classes): {df['agree'].mean():.1%}",
             f"Accuracy (mentioned vs missing): {df['agree_2way'].mean():.1%}",
             f"Cohen's kappa (3 classes): {kappa(team, model):.3f}",
             "",
             "Confusion matrix (rows = team, columns = analyser):",
             pd.crosstab(df["team"], df["analyser"]).reindex(index=CLASSES, columns=CLASSES, fill_value=0).to_string(),
             "",
             "Per class        precision  recall  F1     team count"]
    for c in CLASSES:
        tp = ((df["team"] == c) & (df["analyser"] == c)).sum()
        p = tp / max((df["analyser"] == c).sum(), 1)
        r = tp / max((df["team"] == c).sum(), 1)
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        lines.append(f"{c:<16} {p:>9.2f} {r:>7.2f} {f1:>6.2f}   {(df['team'] == c).sum()}")
    summary = "\n".join(lines)
    print(summary)

    per_req = df.groupby("req")["agree"].agg(accuracy="mean", pairs="count").reset_index()
    per_pol = df.groupby("policy")["agree"].agg(accuracy="mean", pairs="count").reset_index()
    print("\nAccuracy per requirement:")
    print(per_req.to_string(index=False, float_format=lambda x: f"{x:.0%}"))
    print("\nAccuracy per policy:")
    print(per_pol.to_string(index=False, float_format=lambda x: f"{x:.0%}"))

    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    per_req.to_csv(eval_dir / "per_requirement.csv", index=False)
    per_pol.to_csv(eval_dir / "per_policy.csv", index=False)
    df[~df["agree"]].drop(columns=["agree", "agree_2way"]).to_csv(
        eval_dir / "disagreements.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved to {eval_dir}/ ({(~df['agree']).sum()} disagreements in disagreements.csv)")


if __name__ == "__main__":
    main()
