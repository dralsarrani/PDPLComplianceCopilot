"""
E5, third row: the answer model WITHOUT the "not covered" instruction.

Table 10 compares a retrieval-score threshold with the final system, whose
prompt tells the model to say "not covered" when the articles do not answer.
This script adds the missing case: the same retrieved articles and the same
model (gpt-4o-mini, temperature 0), but a prompt with the refusal line removed.

Both prompts are run here on the 40 development questions, so the two rows
use the same retrieval and are directly comparable. A question counts as
refused when the model returns no answer sentences (or says it cannot answer).
Nothing is cached or written to results/answers_cache.json.

Run from the project root (needs OPENAI_API_KEY in .env, about 5 minutes):
    python -m retrieval.experiment_e5_no_refusal
Writes results/e5_no_refusal.json and results/e5_no_refusal.csv
"""
import csv
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from retrieval.answer import ARTICLES_FOR_ANSWER, SYSTEM_PROMPT, build_prompt, question_language
from retrieval.retriever import retrieve

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "evaluation" / "gold_qa.json"
OUT_JSON = ROOT / "results" / "e5_no_refusal.json"
OUT_CSV = ROOT / "results" / "e5_no_refusal.csv"
MODEL = "gpt-4o-mini"

REFUSAL_LINE = "- If the articles do not answer the question, set covered to false and leave sentences empty.\n"
assert REFUSAL_LINE in SYSTEM_PROMPT, "the answer prompt changed; update REFUSAL_LINE"
PROMPTS = {
    "with_instruction": SYSTEM_PROMPT,
    # Same prompt, refusal line removed; the model can still return no sentences.
    "without_instruction": SYSTEM_PROMPT.replace(REFUSAL_LINE, ""),
}
CANT = ("not covered", "do not answer", "does not answer", "غير مشمول", "لا تجيب", "لم أجد")


def refused(reply):
    sentences = reply.get("sentences") or []
    if reply.get("covered") is False or not sentences:
        return True
    text = " ".join(s.get("text", "") for s in sentences).lower()
    return any(word in text for word in CANT)


def main():
    load_dotenv()
    client = OpenAI()
    questions = json.loads(GOLD.read_text(encoding="utf-8"))["questions"]
    rows = []
    for i, q in enumerate(questions, 1):
        articles = retrieve(q["question"], k=ARTICLES_FOR_ANSWER, language=question_language(q["question"]))
        row = {"id": q["id"], "answerable": bool(q["gold_article_ids"])}
        for name, prompt in PROMPTS.items():
            response = client.chat.completions.create(
                model=MODEL, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": build_prompt(q["question"], articles)}])
            row[name] = refused(json.loads(response.choices[0].message.content))
        rows.append(row)
        print(f"{i}/{len(questions)} {q['id']} answerable={row['answerable']} "
              f"refused with={row['with_instruction']} without={row['without_instruction']}")

    summary = {}
    for name in PROMPTS:
        un = [r for r in rows if not r["answerable"]]
        an = [r for r in rows if r["answerable"]]
        summary[name] = {"correctly_refused_unanswerable": f"{sum(r[name] for r in un)}/{len(un)}",
                         "wrongly_refused_answerable": f"{sum(r[name] for r in an)}/{len(an)}"}
    OUT_JSON.write_text(json.dumps({"model": MODEL, "summary": summary, "rows": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "correctly_refused_unanswerable", "wrongly_refused_answerable"])
        for name, s in summary.items():
            writer.writerow([name, s["correctly_refused_unanswerable"], s["wrongly_refused_answerable"]])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
