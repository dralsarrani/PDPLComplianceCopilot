"""
Gap Analyser v1 (3B): like v0, but the model answers small yes/no questions
instead of choosing covered / partial / missing itself.

Why: on the team's 11 labelled policies (python -m src.gap.evaluate_gap), v0 was too
lenient. In 22 of 32 cases the team said partial, v0 said covered: it saw the topic
mentioned and stopped there. The team labelled with 2-5 yes/no questions per requirement
(data/checklist/pdpl_subquestions.json, the same questions as the labelling page),
so v1 asks the model the same questions and turns the answers into a status the same way:
    all yes -> covered,  some yes -> partial,  all no -> missing

The questions come from the labelling method, not from the disagreements, but the
change was chosen after seeing v0's errors on the same policies, so v1's score on
them is somewhat optimistic. Report v0 and v1 side by side.

Output has the same fields as v0 plus "answers" (the yes/no per question).

Run from the project root:
    python -m src.gap.analyse_v1 data/policies/tamara.txt
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.gap.analyse_v0 import (DEFAULT_MODEL, GROUPS, MAX_POLICY_CHARS, load_checklist,
                                normalise, quote_in_policy)

SUBQUESTIONS = Path("data/checklist/pdpl_subquestions.json")
OUTPUT_DIR = Path("results/gap_v1")

SYSTEM_PROMPT = """You check privacy policies against the Saudi Personal Data Protection Law (PDPL).
For each requirement you get a few yes/no questions about the policy.
Answer each question "yes" only if the policy text clearly says it; otherwise "no".
Read the whole policy before answering no: the answer can be in any section.
Judge only from the policy text. Do not assume anything the policy does not say.
If at least one answer is yes, copy one short sentence from the policy word for word as evidence,
in the policy's own language, without adding "..." or punctuation; otherwise evidence is "".
Answer in JSON: {"results": [{"req_id": "...", "answers": ["yes", "no", ...], "evidence": "...", "reason": "..."}]}
with one item per requirement, in the same order, and one answer per question in the same order.
reason is one short sentence in English."""


def status_from(answers):
    yes = sum(a == "yes" for a in answers)
    if not answers:
        return "error"
    return "covered" if yes == len(answers) else "partial" if yes else "missing"


def build_prompt(policy_text, group, subquestions):
    lines = []
    for r in group.itertuples():
        qs = "\n".join(f"   {n}. {q}" for n, q in enumerate(subquestions[r.req_id], start=1))
        lines.append(f"{r.req_id}: {r.requirement_en}\n{qs}")
    return ("Requirements and questions:\n" + "\n".join(lines)
            + "\n\nPrivacy policy:\n\"\"\"\n" + policy_text[:MAX_POLICY_CHARS] + "\n\"\"\"")


def analyse(policy_text, checklist=None, model=None):
    load_dotenv()
    checklist = load_checklist() if checklist is None else checklist
    subquestions = json.loads(SUBQUESTIONS.read_text(encoding="utf-8"))
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    client = OpenAI()

    def ask(group):
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": build_prompt(policy_text, group, subquestions)}],
        )
        return json.loads(response.choices[0].message.content).get("results", [])

    size = -(-len(checklist) // GROUPS)  # rounded up
    groups = [checklist.iloc[i:i + size] for i in range(0, len(checklist), size)]
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        replies = {a.get("req_id"): a for results in pool.map(ask, groups) for a in results}

    policy_norm = normalise(policy_text)
    report = []
    for r in checklist.itertuples():
        a = replies.get(r.req_id, {})
        n = len(subquestions[r.req_id])
        answers = [str(x).strip().lower() for x in a.get("answers", [])][:n]
        status = status_from(answers) if len(answers) == n else "error"
        evidence = a.get("evidence", "") if status != "missing" else ""
        report.append({
            "req_id": r.req_id,
            "requirement": r.requirement_en,
            "articles": r.articles,
            "applies": r.applies,
            "condition": r.condition,
            "status": status,
            "answers": answers,
            "evidence": evidence,
            "evidence_found": quote_in_policy(evidence, policy_norm),
            "reason": a.get("reason", ""),
        })
    return report


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.gap.analyse_v1 data/policies/<name>.txt")
        sys.exit(1)
    policy_path = Path(sys.argv[1])
    report = analyse(policy_path.read_text(encoding="utf-8"))
    for row in report:
        print(f"{row['req_id']}  {row['status']:<9} {'/'.join(row['answers']):<22} {row['requirement'][:55]}")
    print("\nSummary:", pd.Series([r["status"] for r in report]).value_counts().to_dict())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{policy_path.stem}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", out)


if __name__ == "__main__":
    main()
