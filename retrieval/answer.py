"""
answer(question): a cited PDPL answer from the retrieved articles only.

    retrieve(question)  ->  OpenAI answers from those articles only, sentence
    by sentence, each sentence with the ids it relies on and a short quote
    copied from one of them, or says "not covered"
    ->  the code looks for each quote in the cited article text
    ->  check_citations() (retrieval/citation_check.py); a citation whose quote
        was found in the article needs no second LLM check
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
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from retrieval import ecc_retriever
from retrieval.citation_check import check_citations, normalise_id
from retrieval.retriever import retrieve
from retrieval.router import keyword_route, route

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "results" / "answers_cache.json"
DEFAULT_MODEL = "gpt-4o-mini"
ARTICLES_FOR_ANSWER = 8  # see RERANK_MODEL in retriever.py
NOT_COVERED = {"ar": "غير مشمول: لم أجد في نصوص النظام ما يجيب على هذا السؤال.",
               "en": "Not covered: the PDPL texts I searched do not answer this question."}
NOT_COVERED_ECC = {"ar": "غير مشمول: لم أجد في النصوص التي بحثت فيها ما يجيب على هذا السؤال.",
                   "en": "Not covered: the texts I searched do not answer this question."}
# A question routed to "both" gets the best PDPL articles and the best NCA controls.
BOTH_SPLIT = (5, 4)

SYSTEM_PROMPT = """You answer questions about the Saudi Personal Data Protection Law (PDPL)
using ONLY the articles given to you. Do not use any outside knowledge.
- Answer in the same language as the question, in a few clear sentences.
- Give each sentence the ids of the articles it relies on, e.g. ["PDPL-A15-AR"].
  Copy each id exactly as it appears in brackets before the article. Cite only the article that directly
  states the rule in the sentence, not articles that are only related; usually one.
- For each sentence, copy in "quote" the words from one cited article that support it,
  exactly as written in the article (one short passage, at most 40 words, no changes).
- If the articles do not answer the question, set covered to false and leave sentences empty.
Reply in JSON: {"covered": true or false,
                "sentences": [{"text": "...", "citations": ["PDPL-A15-AR", ...],
                               "quote": "..."}, ...]}"""
# Used when the router sends the question to the NCA controls (alone or with the PDPL).
# The PDPL-only prompt above stays as it was, so PDPL answers and their evaluation do not change.
ECC_SYSTEM_PROMPT = """You answer questions about Saudi regulations using ONLY the texts given to you:
the Personal Data Protection Law (PDPL) and its regulations (ids like PDPL-A15-AR, IR-A04-EN) and/or
the National Cybersecurity Authority's Essential Cybersecurity Controls ECC-2:2024 (ids like ECC-2-10-3-AR).
Do not use any outside knowledge.
- Answer in the same language as the question, in a few clear sentences.
- Give each sentence the ids of the texts it relies on. Copy each id exactly as it appears in brackets.
  Cite only the text that directly states the rule in the sentence; usually one.
- For each sentence, copy in "quote" the words from one cited text that support it, exactly as written
  in that text (one short passage, at most 40 words, no changes).
- If the texts do not answer the question, set covered to false and leave sentences empty.
Reply in JSON: {"covered": true or false,
                "sentences": [{"text": "...", "citations": ["ECC-2-10-3-AR", ...],
                               "quote": "..."}, ...]}"""
ARABIC_MARKS = re.compile(r"[\u064B-\u0652\u0640]")  # short vowels and tatweel
PUNCTUATION = " .,;:!?\"'“”«»…،؛()[]-"
LETTER_FORMS = str.maketrans("أإآةى", "اااهي")
QUOTE_MATCH = 0.9  # share of the quote's words that must be in the article, in order


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


def normalise(text):
    """Lower case, one space between words, no Arabic short vowels or tatweel."""
    return re.sub(r"\s+", " ", ARABIC_MARKS.sub("", text)).strip().lower()


def words(text):
    """Normalised words without punctuation, used to compare a quote with an article.
    Also treats أ إ آ as ا, ة as ه and ى as ي, which the model sometimes mixes up."""
    text = normalise(text).translate(LETTER_FORMS)
    return [w.strip(PUNCTUATION) for w in text.split()]


def locate_quote(quote, article_text):
    """The passage of the article that the quote comes from, in the article's own
    words, or None. Allows small slips by the model (a comma, one changed word):
    at least 90% of the quote's words must be in the article, in the same order and
    close together. A quote with "..." is looked up part by part."""
    article_raw = article_text.split()
    article = [words(w)[0] if words(w) else "" for w in article_raw]
    passages = []
    for part in re.split(r"\.\.\.|…", quote):
        q = [w for w in words(part) if w]
        if not q:
            continue
        best, best_span = 0, None
        window = len(q) + 3
        for start in range(max(1, len(article) - window + 1)):
            blocks = difflib.SequenceMatcher(None, q, article[start:start + window],
                                             autojunk=False).get_matching_blocks()
            matched = sum(b.size for b in blocks)
            if matched > best:
                used = [b for b in blocks if b.size]
                best = matched
                best_span = (start + used[0].b, start + used[-1].b + used[-1].size)
            if best == len(q):
                break
        if best < QUOTE_MATCH * len(q):
            return None
        passages.append(" ".join(article_raw[best_span[0]:best_span[1]]).strip(" .,;:،؛"))
    return " … ".join(passages) or None


def find_evidence(quote, citation_ids, articles_by_id):
    """The first cited article that contains the quote: {"id", "quote"}, where quote is
    the article's own words. None if no cited article contains it."""
    for cid in citation_ids:
        passage = quote and cid in articles_by_id and locate_quote(quote, articles_by_id[cid]["official_text"])
        if passage:
            return {"id": cid, "quote": passage}
    return None


def fix_language(citation, articles_by_id):
    """The model sometimes writes PDPL-A17-AR when it was given PDPL-A17-EN. It is
    the same article in the other language, so use the id it was given."""
    cid = normalise_id(citation)
    other = cid[:-2] + ("EN" if cid.endswith("AR") else "AR")
    return other if cid not in articles_by_id and other in articles_by_id else cid


def build_prompt(question, articles):
    context = "\n\n".join(f"[{a['record_id']}] {a['article_title']}\n{a['official_text']}" for a in articles)
    return f"Articles:\n{context}\n\nQuestion: {question}"


def answer(question, k=ARTICLES_FOR_ANSWER, model=None, on_step=None, source=None):
    """Return {question, covered, answer, sentences, citations, problems,
    citation_check, retrieved, from_cache, timings}. Falls back to the cached
    answer if OpenAI fails.
    sentences: the sentences shown to the user, [{text, citations, evidence}],
    evidence = {"id", "quote"} when the quote was found in the article, else None.
    on_step(name) is called before each step ("retrieve", "answer", "check"),
    so the app can show what is happening.
    source ("pdpl", "ecc" or "both") skips the router; used for the second chance below."""
    on_step = on_step or (lambda name: None)
    load_dotenv()
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    timings = {}  # seconds per step, shown in the app to see what is slow
    try:
        on_step("retrieve")
        start = time.perf_counter()
        routed = ({"source": source, "by": "second chance"} if source
                  else route(question, model=model) if ecc_retriever.available()
                  else {"source": "pdpl", "by": "none"})
        timings["route"] = time.perf_counter() - start
        start = time.perf_counter()
        language = question_language(question)
        if routed["source"] == "pdpl":
            articles = retrieve(question, k=k, language=language)
        elif routed["source"] == "ecc":
            articles = ecc_retriever.retrieve_ecc(question, k=k, language=language)
        else:
            articles = (retrieve(question, k=BOTH_SPLIT[0], language=language)
                        + ecc_retriever.retrieve_ecc(question, k=BOTH_SPLIT[1], language=language))
        timings["retrieve"] = time.perf_counter() - start
        on_step("answer")
        start = time.perf_counter()
        response = OpenAI().chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system",
                       "content": SYSTEM_PROMPT if routed["source"] == "pdpl" else ECC_SYSTEM_PROMPT},
                      {"role": "user", "content": build_prompt(question, articles)}],
        )
        reply = json.loads(response.choices[0].message.content)
        sentences = reply.get("sentences", []) if reply.get("covered") else []
        timings["answer"] = time.perf_counter() - start
        on_step("check")
        start = time.perf_counter()
        articles_by_id = {a["record_id"]: a for a in articles}
        for s in sentences:
            s["citations"] = [fix_language(c, articles_by_id) for c in s.get("citations", [])]
        evidence = [find_evidence(s.get("quote", ""), s["citations"],
                                  articles_by_id) for s in sentences]
        quoted = {(i, e["id"]) for i, e in enumerate(evidence) if e}
        check = check_citations(sentences, list(articles_by_id), model=model, quoted=quoted)
        timings["check"] = time.perf_counter() - start
    except Exception as error:
        cached = load_cache().get(cache_key(question))
        if cached is None:
            raise RuntimeError(f"OpenAI failed and no cached answer for this question: {error}") from error
        return {**cached, "from_cache": True}

    # Only sentences backed by at least one valid citation are shown to the user.
    shown = []
    for found, row in zip(evidence, check["sentences"]):
        valid_ids = [c["id"] for c in row["citations"] if c["status"] in ("supported", "partial")]
        if valid_ids:
            shown.append({"text": row["text"], "citations": valid_ids,
                          "evidence": found if found and found["id"] in valid_ids else None})
    backed = [s["text"] for s in shown]
    covered = bool(backed)
    not_covered = NOT_COVERED if routed["source"] == "pdpl" else NOT_COVERED_ECC
    result = {
        "question": question,
        "source": routed["source"],  # which corpus the router chose: pdpl, ecc or both
        "covered": covered,
        "answer": " ".join(backed) if covered else not_covered[question_language(question)],
        "sentences": shown,
        "citations": check["valid_citations"],
        "problems": check["problems"],
        "citation_check": check["sentences"],
        "retrieved": [{"record_id": a["record_id"], "score": a["score"]} for a in articles],
        "from_cache": False,
        "timings": {step: round(seconds, 1) for step, seconds in timings.items()},
    }
    # Second chance: the router sent a question to the PDPL, the PDPL texts do not answer it,
    # but it uses cybersecurity words (router.keyword_route), so the controls are tried too.
    # A question the PDPL answers never gets here, so PDPL answers do not change.
    if (not covered and routed["source"] == "pdpl" and not source and ecc_retriever.available()
            and keyword_route(question) != "pdpl"):
        second = answer(question, k=k, model=model, on_step=on_step, source="ecc")
        if second["covered"]:
            second["timings"] = {s: round(result["timings"].get(s, 0) + t, 1) for s, t in second["timings"].items()}
            return second
    save_to_cache(question, result)
    return result


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or input("Enter your question: ")
    result = answer(query)
    print(f"\nQuestion: {query}  (source: {result.get('source', 'pdpl')})\n")
    print("Retrieved articles (after reranking):")
    for rank, r in enumerate(result["retrieved"], start=1):
        print(f"Rank {rank}: {r['record_id']:<12} | Score: {r['score']}")
    print("\nAnswer:")
    print(result["answer"])
    for s in result.get("sentences", []):
        if s["evidence"]:
            print(f"  evidence {s['evidence']['id']}: {s['evidence']['quote']}")
    print("\nCitations:", ", ".join(result["citations"]) or "none")
    print("\nCitation check:")
    for i, row in enumerate(result.get("citation_check", []), start=1):
        checks = ", ".join(f"{c['id']} {c['status']}" for c in row["citations"]) or "uncited"
        print(f"  Sentence {i}: {checks}")
    if result.get("problems"):
        print(f"Problems found: {len(result['problems'])} (not shown to the user)")
    if result["from_cache"]:
        print("(cached answer: OpenAI was not reachable)")
