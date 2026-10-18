"""
answer(question): a cited PDPL answer from the retrieved articles only.

    retrieve(question)  ->  OpenAI answers from those articles only, sentence
    by sentence, each sentence with the ids it relies on, or says "not covered"
    ->  check_citations() (retrieval/citation_check.py)
    ->  saved to results/answers_cache.json

check_citations() keeps only citations that exist in the corpus, were
retrieved, and support their sentence. Invented, unrelated or missing
citations are reported in "problems", and a sentence with no valid citation
is left out of the answer shown to the user.

Every live answer is saved in results/answers_cache.json. If OpenAI cannot be
reached (no network on demo day), answer() returns the saved answer for the
same question instead. This is the demo fallback (cached responses).

Reads OPENAI_API_KEY (and optional OPENAI_MODEL) from .env

Run from the project root:
    python -m retrieval.answer "هل يجوز نقل بيانات العملاء خارج المملكة؟"
"""
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from retrieval.citation_check import check_citations
from retrieval.retriever import retrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "results" / "answers_cache.json"
DEFAULT_MODEL = "gpt-4o-mini"
NOT_COVERED = {"ar": "غير مشمول: لم أجد في نصوص النظام ما يجيب على هذا السؤال.",
               "en": "Not covered: the PDPL texts I searched do not answer this question."}

SYSTEM_PROMPT = """You answer questions about the Saudi Personal Data Protection Law (PDPL)
using ONLY the articles given to you. Do not use any outside knowledge.
- Answer in the same language as the question, in a few clear sentences.
- Give each sentence the ids of the articles it relies on, e.g. ["PDPL-A15-AR"].
  Use only ids from the articles given to you.
- If the articles do not answer the question, set covered to false and leave sentences empty.
Reply in JSON: {"covered": true or false,
                "sentences": [{"text": "...", "citations": ["PDPL-A15-AR", ...]}, ...]}"""


def question_language(text):
    return "ar" if re.search(r"[؀-ۿ]", text) else "en"


def cache_key(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_to_cache(question, result):
    cache = load_cache()
    cache[cache_key(question)] = result
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def build_prompt(question, articles):
    context = "\n\n".join(f"[{a['record_id']}] {a['article_title']}\n{a['official_text']}" for a in articles)
    return f"Articles:\n{context}\n\nQuestion: {question}"


def answer(question, k=5, model=None):
    """Return {question, covered, answer, citations, problems, citation_check,
    retrieved, from_cache}. Falls back to the cached answer if OpenAI fails."""
    load_dotenv()
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    try:
        articles = retrieve(question, k=k)
        response = OpenAI().chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": build_prompt(question, articles)}],
        )
        reply = json.loads(response.choices[0].message.content)
        sentences = reply.get("sentences", []) if reply.get("covered") else []
        check = check_citations(sentences, [a["record_id"] for a in articles], model=model)
    except Exception as error:
        cached = load_cache().get(cache_key(question))
        if cached is None:
            raise RuntimeError(f"OpenAI failed and no cached answer for this question: {error}") from error
        return {**cached, "from_cache": True}

    # Only sentences backed by at least one valid citation are shown to the user.
    backed = [row["text"] for row in check["sentences"]
              if any(c["status"] in ("supported", "partial") for c in row["citations"])]
    covered = bool(backed)
    result = {
        "question": question,
        "covered": covered,
        "answer": " ".join(backed) if covered else NOT_COVERED[question_language(question)],
        "citations": check["valid_citations"],
        "problems": check["problems"],
        "citation_check": check["sentences"],
        "retrieved": [{"record_id": a["record_id"], "score": a["score"]} for a in articles],
        "from_cache": False,
    }
    save_to_cache(question, result)
    return result


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or input("Enter your question: ")
    result = answer(query)
    print(f"\nQuestion: {query}\n")
    print("Retrieved articles (after reranking):")
    for rank, r in enumerate(result["retrieved"], start=1):
        print(f"Rank {rank}: {r['record_id']:<12} | Score: {r['score']}")
    print("\nAnswer:")
    print(result["answer"])
    print("\nCitations:", ", ".join(result["citations"]) or "none")
    print("\nCitation check:")
    for i, row in enumerate(result.get("citation_check", []), start=1):
        checks = ", ".join(f"{c['id']} {c['status']}" for c in row["citations"]) or "uncited"
        print(f"  Sentence {i}: {checks}")
    if result.get("problems"):
        print(f"Problems found: {len(result['problems'])} (not shown to the user)")
    if result["from_cache"]:
        print("(cached answer: OpenAI was not reachable)")
