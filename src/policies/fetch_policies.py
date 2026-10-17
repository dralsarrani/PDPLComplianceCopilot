"""
Download the Saudi privacy policies used to evaluate the Gap Analyser (3B).

Reads   data/policies/sources.csv   (one row per policy)
Writes  data/policies/<policy_id>.txt   (clean policy text)
        data/policies/policies.csv      (sources + fetch date, status, length)

Policies with method=manual are skipped: their page blocks scripts or loads
the text with JavaScript, so it is copied by hand from the browser into
data/policies/<policy_id>.txt.

start_line / stop_line (optional) trim leftover menus: the text starts at the
first line that begins with start_line and stops before the first line that
begins with stop_line.

Run from the project root:
    python -m src.policies.fetch_policies
"""

import csv
import re
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup, NavigableString

POLICY_DIR = Path("data/policies")
SOURCES = POLICY_DIR / "sources.csv"
OUTPUT = POLICY_DIR / "policies.csv"

# Some sites refuse requests that do not look like a normal browser.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en,ar;q=0.9",
}

# Tags that start a new line. Everything else (span, a, b...) stays inline.
BLOCK_TAGS = ["p", "div", "li", "ul", "ol", "tr", "table", "section", "article",
              "h1", "h2", "h3", "h4", "h5", "h6", "br", "hr", "dt", "dd"]

# Page parts that are never policy text.
NOISE_TAGS = ["script", "style", "noscript", "svg", "nav", "header", "footer",
              "form", "button", "iframe"]

MIN_CHARS = 1500  # shorter than this usually means the page did not load


def html_to_text(html):
    """Keep the main content of the page, one paragraph per line."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    # HTML comments and CDATA are not visible text.
    for s in soup.find_all(string=True):
        if type(s) is not NavigableString:
            s.extract()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    # Line breaks inside the HTML source are not real paragraph breaks.
    for s in root.find_all(string=True):
        s.replace_with(re.sub(r"\s+", " ", s))
    for tag in root.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    lines = [line.strip(" \u200b\u200d\ufeff") for line in root.get_text().splitlines()]
    return "\n".join(line for line in lines if line)


def trim(text, start_line, stop_line):
    """Cut leftover menus before start_line and from stop_line on.
    A line matches when it starts with the given text."""
    lines = text.splitlines()
    if start_line:
        i = find_line(lines, start_line)
        if i is None:
            print(f"  warning: start_line not found: {start_line}")
        else:
            lines = lines[i:]
    if stop_line:
        i = find_line(lines, stop_line)
        if i is None:
            print(f"  warning: stop_line not found: {stop_line}")
        else:
            lines = lines[:i]
    return "\n".join(lines)


def find_line(lines, prefix):
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            return i
    return None


def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        r.encoding = r.apparent_encoding  # fixes garbled Arabic
    return html_to_text(r.text)


def main():
    with open(SOURCES, encoding="utf-8") as f:
        sources = list(csv.DictReader(f))

    rows = []
    for src in sources:
        out_path = POLICY_DIR / f"{src['policy_id']}.txt"
        status, n_chars = "", 0

        if src["method"] == "manual":
            if out_path.exists():
                n_chars = len(out_path.read_text(encoding="utf-8").strip())
            if n_chars:
                status = "manual"
            else:
                status = "manual: missing, paste the text by hand"
        else:
            try:
                text = trim(fetch(src["url"]), src["start_line"], src["stop_line"])
                out_path.write_text(text, encoding="utf-8")
                n_chars = len(text)
                status = "ok" if n_chars >= MIN_CHARS else "too short, check by hand"
            except requests.RequestException as e:
                status = f"failed: {type(e).__name__}"

        print(f"{src['policy_id']:<15} {status:<40} {n_chars:>7} chars")
        rows.append({**src, "fetched_on": date.today().isoformat(),
                     "status": status, "n_chars": n_chars})

    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
