"""
Check the analyser's quotes against the policy text.

The analyser (analyse_v0.py) gives a quote from the policy as evidence for each
covered or partly covered requirement. A quote counts as verified only when its
words are in the policy, in the same order. The check ignores case, spacing,
punctuation and quote marks ("…" “…” «…»), Arabic short vowels and tatweel, and
the letter forms the model mixes up (أ إ آ -> ا, ة -> ه, ى -> ي). A quote with
"..." is checked piece by piece, and the pieces must come in the policy's order.

When the model reworded the sentence instead of copying it, the quote is not
verified. Then the closest real passage of the policy is looked up, so the app
can show the policy's own words next to the model's quote. That passage is
shown as "closest text", never as a verified quote.

The analyser's status (covered / partial / missing) is not changed here.

    spans = find_spans(text, quote)        -> [(start, end), ...] or []
    nearest(text, quote)                   -> (passage, share of words) or ("", 0)
    report = check_report(report, text)    -> rows with evidence_found,
                                              evidence_state and evidence_near
"""
import re
import unicodedata
from bisect import bisect_right
from collections import Counter
from difflib import SequenceMatcher

MARKS = re.compile(r"[ً-ْٰـ]")  # Arabic short vowels, dagger alef, tatweel
FORMS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي",
                       **{chr(0x660 + d): str(d) for d in range(10)}})
WORD = re.compile(r"\w+")
NEAR_SHARE = 0.6  # a passage must share at least 60% of the quote's words, in order


def word_form(word):
    return MARKS.sub("", unicodedata.normalize("NFKC", word)).lower().translate(FORMS)


def tokens(text):
    """The words of the text in comparable form, with where each one is in the text."""
    found = [(word_form(m.group()), m.start(), m.end()) for m in WORD.finditer(text or "")]
    return [(w, s, e) for w, s, e in found if w]


def pieces(quote):
    """The quote's pieces (split at "..."), each as a list of comparable words."""
    parts = [[t[0] for t in tokens(p)] for p in re.split(r"\.\.\.|…", quote or "")]
    return [p for p in parts if p]


def find_spans(text, quote, text_tokens=None):
    """Where each piece of the quote is in the text, as (start, end) pairs, in order.
    [] if any piece is missing or comes before the piece in front of it. The first
    and last word of a piece may be part of a longer word, so a quote that drops
    an attached و or ف ("نتخذ" for "ونتخذ") still counts."""
    text_tokens = text_tokens if text_tokens is not None else tokens(text)
    words = [t[0] for t in text_tokens]
    joined = " ".join(words)
    starts, at = [], 0  # where each word starts in `joined`
    for w in words:
        starts.append(at)
        at += len(w) + 1
    spans, after = [], 0
    for piece in pieces(quote):
        if after >= len(words):
            return []
        target = " ".join(piece)
        hit = joined.find(target, starts[after])
        if hit < 0:
            return []
        first = bisect_right(starts, hit) - 1
        last = bisect_right(starts, hit + len(target) - 1) - 1
        spans.append((text_tokens[first][1], text_tokens[last][2]))
        after = last + 1
    return spans


def nearest(text, quote, text_tokens=None):
    """The passage of the text closest to the quote, in the text's own words, and the
    share of the quote's words found in it in order. ("", 0) if nothing is close."""
    text_tokens = text_tokens if text_tokens is not None else tokens(text)
    quote_words = [w for p in pieces(quote) for w in p]
    if not quote_words or not text_tokens:
        return "", 0.0
    words = [t[0] for t in text_tokens]
    size, wanted = len(quote_words), Counter(quote_words)
    # windows the size of the quote that share the most words with it ...
    window, shared, scores = Counter(words[:size]), 0, []
    shared = sum(min(c, wanted[w]) for w, c in window.items())
    scores.append((shared, 0))
    for i in range(1, max(1, len(words) - size + 1)):
        out, new = words[i - 1], words[i + size - 1]
        if window[out] <= wanted[out]:
            shared -= 1
        window[out] -= 1
        window[new] += 1
        if window[new] <= wanted[new]:
            shared += 1
        scores.append((shared, i))
    best_share, best = 0.0, None
    # ... then the best of those, with some room on both sides, by words in order
    for _, i in sorted(scores, reverse=True)[:5]:
        lo, hi = max(0, i - size // 2), min(len(words), i + size + size // 2)
        blocks = SequenceMatcher(None, quote_words, words[lo:hi], autojunk=False).get_matching_blocks()
        matched = [b for b in blocks if b.size]
        share = sum(b.size for b in matched) / size
        ends = [b for b in matched if b.size > 1] or matched  # a lone "the" does not start the passage
        if matched and share > best_share:
            best_share = share
            best = (lo + ends[0].b, lo + ends[-1].b + ends[-1].size - 1)
    if best is None or best_share < NEAR_SHARE:
        return "", round(best_share, 2)
    start, end = text_tokens[best[0]][1], text_tokens[best[1]][2]
    return re.sub(r"\s+", " ", text[start:end]).strip(), round(best_share, 2)


def check_report(report, text):
    """The report with each quote checked against the policy text:
    evidence_found  True only if the quote is in the text, word for word and in order
    evidence_state  "verified", "near" (reworded, a close passage was found),
                    "unverified" (nothing close) or "none" (no quote)
    evidence_near   the close passage, in the policy's own words, for "near"
    The status the analyser gave is kept as it is."""
    text_tokens = tokens(text)
    checked = []
    for r in report:
        r = dict(r, evidence_near="", evidence_state="none")
        quote = r.get("evidence") or ""
        r["evidence_found"] = bool(quote) and bool(find_spans(text, quote, text_tokens))
        if r["evidence_found"]:
            r["evidence_state"] = "verified"
        elif quote:
            passage, _ = nearest(text, quote, text_tokens)
            r["evidence_state"] = "near" if passage else "unverified"
            r["evidence_near"] = passage
        checked.append(r)
    return checked
