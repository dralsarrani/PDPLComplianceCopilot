"""
Gap Analyser v0 (3B): check a privacy policy against the PDPL checklist.

v0 uses OpenAI only (no classifier, no article retrieval yet). The model reads
the whole policy and, for each requirement in the checklist, returns
covered / partial / missing, with a sentence quoted from the policy as evidence.

The model judges only whether the policy covers each requirement. Conditional
requirements (e.g. R16, R17) are not sent with their condition, because the
model then mixed the condition into the requirement. A conditional requirement
that is missing is kept as missing and printed with its condition, so a person
can decide whether the condition applies to the company.

Every quote is checked against the policy text: if the model returns a quote
that is not in the policy, evidence_found is False and the result should not
be trusted.

Reads   data/checklist/pdpl_checklist.csv
        the policy .txt file given on the command line
        OPENAI_API_KEY (and optional OPENAI_MODEL) from .env
Writes  results/gap_v0/<policy name>.json

Run from the project root:
    python -m src.gap.analyse_v0 data/policies/tamara.txt
"""
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

CHECKLIST = Path("data/checklist/pdpl_checklist.csv")
OUTPUT_DIR = Path("results/gap_v0")
DEFAULT_MODEL = "gpt-4o-mini"
MAX_POLICY_CHARS = 60000
STATUSES = ["covered", "partial", "missing"]

SYSTEM_PROMPT = """You check privacy policies against the Saudi Personal Data Protection Law (PDPL).
For each requirement you get, decide what the policy says about it:
- covered: the policy clearly and specifically states it.
- partial: the policy mentions it, but vaguely or only in part.
- missing: the policy says nothing about it.
Read the whole policy before answering missing: a requirement can be in any section.
Judge only from the policy text. Do not assume anything the policy does not say.
For covered or partial, copy one short sentence from the policy word for word as evidence,
in the policy's own language, without adding "..." or punctuation.
For missing, evidence is an empty string.
Answer in JSON: {"results": [{"req_id": "...", "status": "...", "evidence": "...", "reason": "..."}]}
with one item per requirement, in the same order. reason is one short sentence in English."""


def load_checklist(path=CHECKLIST):
    return pd.read_csv(path).fillna("")


def normalise(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_policy(evidence, policy_norm):
    """True if the quote is in the policy. Ignores "..." and punctuation the
    model adds at the ends of the quote (e.g. a final full stop)."""
    parts = [p.strip(" .,;:!?\"'“”…") for p in re.split(r"\.\.\.|…", normalise(evidence))]
    parts = [p for p in parts if p]
    return bool(parts) and all(p in policy_norm for p in parts)


def build_prompt(policy_text, checklist):
    lines = []
    for r in checklist.itertuples():
        lines.append(f"{r.req_id}: {r.requirement_en}")
    return ("Requirements:\n" + "\n".join(lines)
            + "\n\nPrivacy policy:\n\"\"\"\n" + policy_text[:MAX_POLICY_CHARS] + "\n\"\"\"")


def analyse(policy_text, checklist=None, model=None):
    """Return one dict per checklist requirement:
    {req_id, requirement, articles, applies, condition, status, evidence,
    evidence_found, reason}"""
    load_dotenv()
    checklist = load_checklist() if checklist is None else checklist
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": build_prompt(policy_text, checklist)}],
    )
    answers = {a.get("req_id"): a for a in json.loads(response.choices[0].message.content).get("results", [])}

    policy_norm = normalise(policy_text)
    report = []
    for r in checklist.itertuples():
        a = answers.get(r.req_id, {})
        status = a.get("status", "")
        if status not in STATUSES:
            status = "error"
        evidence = a.get("evidence", "")
        report.append({
            "req_id": r.req_id,
            "requirement": r.requirement_en,
            "articles": r.articles,
            "applies": r.applies,
            "condition": r.condition,
            "status": status,
            "evidence": evidence,
            "evidence_found": quote_in_policy(evidence, policy_norm),
            "reason": a.get("reason", ""),
        })
    return report


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.gap.analyse_v0 data/policies/<name>.txt")
        sys.exit(1)
    policy_path = Path(sys.argv[1])
    report = analyse(policy_path.read_text(encoding="utf-8"))

    for row in report:
        flag = "" if row["evidence_found"] or not row["evidence"] else "  (quote not found in policy)"
        if row["applies"] == "conditional" and row["status"] == "missing":
            flag += f"  (required only if: {row['condition']})"
        print(f"{row['req_id']}  {row['status']:<15} {row['requirement'][:60]}{flag}")
    counts = pd.Series([r["status"] for r in report]).value_counts().to_dict()
    print("\nSummary:", counts)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{policy_path.stem}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", out)


if __name__ == "__main__":
    main()
