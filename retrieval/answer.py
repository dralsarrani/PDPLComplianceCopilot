"""
answer(question): a cited PDPL answer from the retrieved articles only.

    retrieve(question)  ->  OpenAI answers from those articles only,
    cites record ids, or says "not covered"  ->  citation check
    ->  saved to results/answers_cache.json

The citation check drops any cited article that was not in the retrieved
articles, so the answer can only point to text the system actually found.

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

from retrieval.retriever import retrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "results" / "answers_cache.json"
DEFAULT_MODEL = "gpt-4o-mini"
NOT_COVERED = {"ar": "غير مشمول: لم أجد في نصوص النظام ما يجيب على هذا السؤال.",
               "en": "Not covered: the PDPL texts I searched do not answer this question."}

SYSTEM_PROMPT = """You answer questions about the Saudi Personal Data Protection Law (PDPL)
using ONLY the articles given to you. Do not use any outside knowledge.
- Answer in the same language as the question, in a few clear sentences.
- Cite every article you rely on by its id in square brackets, e.g. [PDPL-A15-AR].
- If the articles do not answer the question, set covered to false and leave answer empty.
Reply in JSON: {"covered": true or false, "answer": "...", "citations": ["PDPL-A15-AR", ...]}"""


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
    """Return {question, covered, answer, citations, dropped_citations,
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
    except Exception as error:
        cached = load_cache().get(cache_key(question))
        if cached is None:
            raise RuntimeError(f"OpenAI failed and no cached answer for this question: {error}") from error
        return {**cached, "from_cache": True}

    reply = json.loads(response.choices[0].message.content)
    retrieved_ids = [a["record_id"] for a in articles]
    cited = reply.get("citations", [])
    citations = [c for c in cited if c in retrieved_ids]
    covered = bool(reply.get("covered")) and bool(citations)

    result = {
        "question": question,
        "covered": covered,
        "answer": reply.get("answer", "") if covered else NOT_COVERED[question_language(question)],
        "citations": citations if covered else [],
        "dropped_citations": [c for c in cited if c not in retrieved_ids],
        "retrieved": [{"record_id": a["record_id"], "score": a["score"]} for a in articles],
        "from_cache": False,
    }
    save_to_cache(question, result)
    return result


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or input("Enter your question: ")
    result = answer(query)
    print(result["answer"])
    print("Citations:", ", ".join(result["citations"]) or "none")
    if result["dropped_citations"]:
        print("Dropped (not in retrieved articles):", ", ".join(result["dropped_citations"]))
    if result["from_cache"]:
        print("(cached answer: OpenAI was not reachable)")
