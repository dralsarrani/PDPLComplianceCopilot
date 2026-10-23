"""
Mark the analyser's evidence inside the policy text (an extra from the brief:
"highlight covered and missing clauses directly in the pasted policy").

The analyser (analyse_v0.py) quotes the policy for each covered or partly covered
requirement. Here each quote is found again in the original text, ignoring case
and spacing like quote_in_policy(), and wrapped in <mark> with its requirement id.
A quote with "..." is marked in pieces. Requirements with no quote are the
missing ones: they have no place in the text, so the app lists them above it.

    spans = find_spans(text, quote)          -> [(start, end), ...] or []
    html = highlight_html(text, report)      -> escaped text with <mark> tags
"""
import html
import re

TONES = {"covered": "ok", "partial": "warn"}


def find_spans(text, quote):
    """Where the quote is in the text, as (start, end) pairs; [] if any piece is missing."""
    pieces = [p.strip(" .,;:!?\"'“”…") for p in re.split(r"\.\.\.|…", quote or "")]
    spans = []
    for piece in filter(None, pieces):
        pattern = r"\s+".join(re.escape(word) for word in piece.split())
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return []
        spans.append(match.span())
    return spans


def highlight_html(text, report, titles=None):
    """The policy text, escaped, with each covered/partial quote in <mark class="hl ok|warn">.
    titles: {req_id: requirement name} for the tooltip. Overlapping quotes: the first one
    in the text keeps the mark and the other ids are added to its label."""
    titles = titles or {}
    marks = []
    for r in report:
        tone = TONES.get(r.get("status"))
        if tone and r.get("evidence"):
            marks += [(start, end, tone, r["req_id"]) for start, end in find_spans(text, r["evidence"])]
    marks.sort()
    merged = []
    for start, end, tone, req in marks:
        if merged and start < merged[-1][1]:  # overlaps the mark before it
            last = merged[-1]
            if req not in last[3]:
                last[3].append(req)
            last[1] = max(last[1], end)
            continue
        merged.append([start, end, tone, [req]])
    out, at = [], 0
    for start, end, tone, reqs in merged:
        out.append(html.escape(text[at:start]))
        tip = html.escape(" · ".join(f"{q} {titles.get(q, '')}".strip() for q in reqs), quote=True)
        out.append(f'<mark class="hl {tone}" title="{tip}">{html.escape(text[start:end])}'
                   f'<sup>{" ".join(reqs)}</sup></mark>')
        at = end
    out.append(html.escape(text[at:]))
    return "".join(out)
