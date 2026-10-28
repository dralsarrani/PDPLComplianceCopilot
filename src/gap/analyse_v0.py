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

The requirements are sent in 4 groups at the same time (4 parallel OpenAI calls),
instead of one long call that writes all 17 answers.

Every quote is checked against the policy text: if the model returns a quote
that is not in the policy, evidence_found is False and the result should not
be trusted.

Reads   data/checklist/pdpl_checklist.csv
        the policy .txt file given on the command line
        OPENAI_API_KEY (and optional OPENAI_MODEL) from .env
        CLAUDE_MODEL=claude-sonnet-5-5 in .env (with ANTHROPIC_API_KEY) uses Claude
        instead (src/claude_llm.py); without them, gpt-4o-mini
Writes  results/gap_v0/<policy name>.json

Run from the project root:
    python -m src.gap.analyse_v0 data/policies/tamara.txt
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.claude_llm import claude_json, claude_model
from src.gap.evidence import find_spans

CHECKLIST = Path("data/checklist/pdpl_checklist.csv")
OUTPUT_DIR = Path("results/gap_v0")
DEFAULT_MODEL = "gpt-4o-mini"
MAX_POLICY_CHARS = 60000
STATUSES = ["covered", "partial", "missing"]
GROUPS = 4  # parallel OpenAI calls, each judging a quarter of the requirements

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


def quote_in_policy(evidence, policy_text):
    """True if the quote's words are in the policy, in the same order (see evidence.py:
    case, spacing, punctuation, quote marks and Arabic letter forms are ignored, and
    a quote with "..." is checked piece by piece)."""
    return bool(evidence) and bool(find_spans(policy_text, evidence))


def build_prompt(policy_text, checklist):
    lines = []
    for r in checklist.itertuples():
        lines.append(f"{r.req_id}: {r.requirement_en}")
    return ("Requirements:\n" + "\n".join(lines)
            + "\n\nPrivacy policy:\n\"\"\"\n" + policy_text[:MAX_POLICY_CHARS] + "\n\"\"\"")


def gap_model():
    """The model the analyser uses: Claude when CLAUDE_MODEL is set up, else OPENAI_MODEL."""
    load_dotenv()
    return claude_model() or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)


def analyse(policy_text, checklist=None, model=None):
    """Return one dict per checklist requirement:
    {req_id, requirement, articles, applies, condition, status, evidence,
    evidence_found, reason}"""
    load_dotenv()
    checklist = load_checklist() if checklist is None else checklist
    model = model or gap_model()

    def ask(group):
        if model.startswith("claude"):
            return claude_json(build_prompt(policy_text, group), SYSTEM_PROMPT, model).get("results", [])
        response = OpenAI().chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": build_prompt(policy_text, group)}],
        )
        return json.loads(response.choices[0].message.content).get("results", [])

    size = -(-len(checklist) // GROUPS)  # rounded up
    groups = [checklist.iloc[i:i + size] for i in range(0, len(checklist), size)]
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        answers = {a.get("req_id"): a for results in pool.map(ask, groups) for a in results}

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
