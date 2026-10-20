"""
check_citations(sentences, retrieved_ids): verify every citation in an answer.

For each sentence of the answer and each article it cites:
    1. exists?      the id is one of the 180 articles in the corpus    else: not_in_corpus
    2. retrieved?   the article was among the ones retrieve() returned else: not_retrieved
    3. supports?    an LLM reads the sentence and the article text     -> supported /
                                                                          partial /
                                                                          unsupported
A sentence with no citation at all is flagged as uncited.
Step 3 is skipped for a citation whose quote answer() already found word for word
in the article (quoted): that one is supported, which saves an LLM call.

Only supported and partial citations are valid. Everything else is a problem,
so an invented or unrelated article is never shown to the user as evidence.

Reads OPENAI_API_KEY (and optional OPENAI_MODEL) from .env
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_PATH = PROJECT_ROOT / "data" / "processed" / "legal_articles_v1.0.json"
DEFAULT_MODEL = "gpt-4o-mini"
MAX_ARTICLE_CHARS = 4000
VALID = ("supported", "partial")
ID_PATTERN = re.compile(r"^(PDPL|IR|TR)-A(\d+)-(AR|EN)$")

JUDGE_PROMPT = """You check citations in answers about the Saudi Personal Data Protection Law (PDPL).
For each numbered pair you get a sentence and the text of the article it cites.
Decide whether the article text supports the sentence:
- supported: the article clearly states what the sentence says.
- partial: the article supports part of the sentence, but not all of it.
- unsupported: the article does not say this, or is about something else.
Judge only from the article text. The sentence and the article may be in different languages.
Reply in JSON: {"verdicts": [{"n": 1, "verdict": "...", "reason": "..."}]}
with one item per pair. reason is one short sentence in English."""

_articles = {}


def load_articles():
    if not _articles:
        for a in json.loads(ARTICLES_PATH.read_text(encoding="utf-8")):
            _articles[a["record_id"]] = a
    return _articles


def normalise_id(citation):
    """'[pdpl-a4-en]' -> 'PDPL-A04-EN'. Unknown formats are returned cleaned but unchanged."""
    cleaned = citation.strip().strip("[]").upper().replace(" ", "")
    match = ID_PATTERN.match(cleaned)
    if match:
        return f"{match[1]}-A{int(match[2]):02d}-{match[3]}"
    return cleaned


def judge(pairs, client=None, model=None):
    """pairs: [(sentence, article_text)] -> [(verdict, reason)] in the same order."""
    if not pairs:
        return []
    load_dotenv()
    client = client or OpenAI()
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    prompt = "\n\n".join(f"{n}. Sentence: {s}\nArticle: {t[:MAX_ARTICLE_CHARS]}"
                         for n, (s, t) in enumerate(pairs, start=1))
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": JUDGE_PROMPT},
                  {"role": "user", "content": prompt}],
    )
    verdicts = {v.get("n"): v for v in json.loads(response.choices[0].message.content).get("verdicts", [])}
    results = []
    for n in range(1, len(pairs) + 1):
        v = verdicts.get(n, {})
        verdict = v.get("verdict", "")
        results.append((verdict if verdict in ("supported", "partial", "unsupported") else "error",
                        v.get("reason", "")))
    return results


def check_citations(sentences, retrieved_ids, client=None, model=None, quoted=None):
    """sentences: [{"text": "...", "citations": ["PDPL-A15-AR", ...]}]
    retrieved_ids: record ids returned by retrieve()
    quoted: {(sentence index, id)} whose quote was found in the article text

    Returns {"sentences": [{text, uncited, citations: [{id, status, reason}]}],
             "valid_citations": [...], "problems": [{sentence, id, status, reason}], "ok": bool}"""
    articles = load_articles()
    retrieved = {normalise_id(r) for r in retrieved_ids}
    report, to_judge = [], []

    quoted = quoted or set()
    for i, s in enumerate(sentences):
        row = {"text": s.get("text", ""), "citations": []}
        for raw in s.get("citations", []):
            cid = normalise_id(raw)
            if cid not in articles:
                c = {"id": cid, "status": "not_in_corpus", "reason": "No article with this id in the corpus."}
            elif cid not in retrieved:
                c = {"id": cid, "status": "not_retrieved",
                     "reason": "Real article, but not one of the retrieved articles the answer was based on."}
            elif (i, cid) in quoted:
                c = {"id": cid, "status": "supported", "reason": "Quote found word for word in the article."}
            else:
                c = {"id": cid, "status": "pending", "reason": ""}
                to_judge.append((c, row["text"], articles[cid]["official_text"]))
            row["citations"].append(c)
        row["uncited"] = not row["citations"]
        report.append(row)

    verdicts = judge([(text, article) for _, text, article in to_judge], client=client, model=model)
    for (c, _, _), (verdict, reason) in zip(to_judge, verdicts):
        c["status"], c["reason"] = verdict, reason

    valid, problems = [], []
    for i, row in enumerate(report):
        if row["uncited"]:
            problems.append({"sentence": i, "id": None, "status": "uncited",
                             "reason": "Sentence has no citation."})
        for c in row["citations"]:
            if c["status"] in VALID:
                if c["id"] not in valid:
                    valid.append(c["id"])
            else:
                problems.append({"sentence": i, **c})
    return {"sentences": report, "valid_citations": valid, "problems": problems, "ok": not problems}
