"""
Mark the analyser's evidence inside the policy text (an extra from the brief:
"highlight covered and missing clauses directly in the pasted policy").

The analyser (analyse_v0.py) quotes the policy for each covered or partly covered
requirement. Here each quote is found again in the original text with
evidence.find_spans() (the same check as quote_in_policy()), and wrapped in <mark>
with its requirement id. A quote with "..." is marked in pieces. A quote the model
reworded is marked by the closest real passage (evidence_near) when there is one. Requirements with no quote are the
missing ones: they have no place in the text, so the app lists them above it.

    html = highlight_html(text, report)      -> escaped text with <mark> tags
"""
import html

from src.gap.evidence import find_spans

TONES = {"covered": "ok", "partial": "warn"}


def highlight_html(text, report, titles=None):
    """The policy text, escaped, with each covered/partial quote in <mark class="hl ok|warn">.
    titles: {req_id: requirement name} for the tooltip. Overlapping quotes: the first one
    in the text keeps the mark and the other ids are added to its label."""
    titles = titles or {}
    marks = []
    for r in report:
        tone = TONES.get(r.get("status"))
        quote = r.get("evidence_near") or r.get("evidence")
        if tone and quote:
            marks += [(start, end, tone, r["req_id"]) for start, end in find_spans(text, quote)]
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
