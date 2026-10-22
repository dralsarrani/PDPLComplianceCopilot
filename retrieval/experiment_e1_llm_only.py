"""
E1, LLM only: the same model answers the gold questions WITHOUT retrieval.

gpt-4o-mini gets only the question (no articles) and is asked for the same
JSON as answer(): sentences with article ids and a quote, or covered=false.
It is told the id format of our corpus so its citations can be checked.
The scores are compared with the full system (retrieve -> answer -> citation
check) saved in results/answer_eval.json by retrieval.evaluate_answers.

    answerable questions     answered (not "not covered")
                             right source: a cited article is a gold article
                             (language ignored, so PDPL-A15-EN counts for PDPL-A15-AR,
                             which favours the LLM-only run)
                             cited ids that do not exist in the corpus (invented)
                             sentences whose quote is found word for word in the cited article
    unanswerable questions   correctly said "not covered"

Run from the project root (uses OpenAI, one call per question):
    python -m retrieval.experiment_e1_llm_only
Writes results/e1_llm_only.json and results/e1_llm_only.csv.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from retrieval.answer import DEFAULT_MODEL, find_evidence
from retrieval.citation_check import normalise_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = PROJECT_ROOT / "data" / "evaluation" / "gold_qa.json"
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "legal_articles_v1.0.json"
SYSTEM_EVAL_PATH = PROJECT_ROOT / "results" / "answer_eval.json"
OUT_JSON = PROJECT_ROOT / "results" / "e1_llm_only.json"
OUT_CSV = PROJECT_ROOT / "results" / "e1_llm_only.csv"

SYSTEM_PROMPT = """You answer questions about the Saudi Personal Data Protection Law (PDPL),
its Implementing Regulation and the Regulation on Personal Data Transfer outside the Kingdom,
from your own knowledge. No articles are given to you.
- Answer in the same language as the question, in a few clear sentences.
- Give each sentence the ids of the articles it relies on, in this format:
  PDPL-A15-AR = PDPL article 15, IR-A04-AR = Implementing Regulation article 4,
  TR-A07-AR = Transfer Regulation article 7 (use -EN instead of -AR for English questions).
- For each sentence, give in "quote" the words of that article that support it, as exactly as you can.
- If you do not know, or the PDPL texts do not answer the question, set covered to false
  and leave sentences empty.
Reply in JSON: {"covered": true or false,
                "sentences": [{"text": "...", "citations": ["PDPL-A15-AR", ...],
                               "quote": "..."}, ...]}"""


def pct(part, whole):
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def article(cid):
    """'PDPL-A15-AR' -> 'PDPL-A15': the article without its language."""
    return cid.rsplit("-", 1)[0]


def ask(client, model, question):
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"Question: {question}"}],
    )
    return json.loads(response.choices[0].message.content)


def main():
    load_dotenv()
    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    client = OpenAI()
    questions = json.loads(GOLD_PATH.read_text(encoding="utf-8"))["questions"]
    articles_by_id = {a["record_id"]: a for a in json.loads(CORPUS_PATH.read_text(encoding="utf-8"))}

    with ThreadPoolExecutor(max_workers=8) as pool:
        replies = list(pool.map(lambda q: ask(client, model, q["question"]), questions))

    rows = []
    for q, reply in zip(questions, replies):
        gold = set(q["gold_article_ids"])
        sentences = reply.get("sentences", []) if reply.get("covered") else []
        cited = sorted({normalise_id(c) for s in sentences for c in s.get("citations", [])})
        for s in sentences:
            s["citations"] = [normalise_id(c) for c in s.get("citations", [])]
        quotes_found = sum(1 for s in sentences
                           if find_evidence(s.get("quote", ""), s["citations"], articles_by_id))
        rows.append({"id": q["id"], "language": q["language"], "answerable": bool(gold),
                     "covered": bool(sentences), "gold": sorted(gold), "cited": cited,
                     "right_source": bool({article(g) for g in gold} & {article(c) for c in cited}),
                     "invented": [c for c in cited if c not in articles_by_id],
                     "sentences": len(sentences), "quotes_found": quotes_found,
                     "answer": " ".join(s.get("text", "") for s in sentences) or "not covered"})

    def scores(rows, has_invented=True):
        answerable = [r for r in rows if r["answerable"]]
        unanswerable = [r for r in rows if not r["answerable"]]
        n_cited = sum(len(r["cited"]) for r in answerable)
        result = {
            "Answerable questions answered": pct(sum(r["covered"] for r in answerable), len(answerable)),
            "Right source (a gold article cited)": pct(sum(r["right_source"] for r in answerable), len(answerable)),
            "Quotes found word for word": pct(sum(r["quotes_found"] for r in answerable),
                                              sum(r["sentences"] for r in answerable)),
            "Unanswerable questions said not covered": pct(sum(not r["covered"] for r in unanswerable),
                                                           len(unanswerable)),
        }
        if has_invented:
            result["Cited ids that do not exist"] = pct(sum(len(r["invented"]) for r in answerable), n_cited)
        return result

    table = {"LLM only (no retrieval)": scores(rows)}
    if SYSTEM_EVAL_PATH.exists():
        system = json.loads(SYSTEM_EVAL_PATH.read_text(encoding="utf-8"))
        table["Full system (retrieval + rerank + citation check)"] = scores(system, has_invented=False)
    df = pd.DataFrame(table).fillna("0% (checked)")
    print(f"Model: {model}   questions: {len(rows)}\n")
    print(df.to_string())
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(OUT_CSV, encoding="utf-8-sig")
    print(f"\nSaved {OUT_JSON.relative_to(PROJECT_ROOT)} and {OUT_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
