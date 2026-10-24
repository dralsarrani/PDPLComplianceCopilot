"""
Download the second language of the 11 Saudi policies, for the app.

The 11 policies in data/policies/<policy_id>.txt are the Gap Analyser's test set,
each in the one language the team labelled (9 English, 2 Arabic). The app shows
a policy in the language of its interface, so it also needs the other language:

Reads   data/policies/other_language.csv   (policy, language, link, method)
Writes  data/policies/<language>/<policy_id>.txt
        data/policies/other_language_fetched.csv   (+ date, status, length)

The test set itself is not changed. Same rules as fetch_policies.py: the script
says who it is and follows robots.txt; method=manual pages (scripts blocked or
text loaded by JavaScript) are copied by hand from the browser into
data/policies/<language>/<policy_id>.txt. A page whose text is not in the
expected language (the site sent another language) is not saved.

Run from the project root:
    python -m src.policies.fetch_other_language
"""
import csv
from datetime import date

import requests

from src.policies.fetch_policies import MIN_CHARS, POLICY_DIR, Blocked, fetch, trim
from src.policies.language import text_language

SOURCES = POLICY_DIR / "other_language.csv"
OUTPUT = POLICY_DIR / "other_language_fetched.csv"


def main():
    with open(SOURCES, encoding="utf-8") as f:
        sources = list(csv.DictReader(f))
    rows = []
    for src in sources:
        out_path = POLICY_DIR / src["language"] / f"{src['policy_id']}.txt"
        out_path.parent.mkdir(exist_ok=True)
        status, n_chars = "", 0
        if src["method"] == "manual":
            text = out_path.read_text(encoding="utf-8").strip() if out_path.exists() else ""
            n_chars = len(text)
            if not text:
                status = "manual: missing, paste the text by hand"
            elif text_language(text) != src["language"]:
                status = "manual: not in the expected language"
            else:
                status = "manual"
        else:
            try:
                text = trim(fetch(src["url"]), src["start_line"], src["stop_line"])
                n_chars = len(text)
                if text_language(text) != src["language"]:
                    status = "not in the expected language, check the link"
                else:
                    out_path.write_text(text, encoding="utf-8")
                    status = "ok" if n_chars >= MIN_CHARS else "too short, check by hand"
            except Blocked:
                status = "blocked by robots.txt, copy by hand"
            except requests.RequestException as e:
                status = f"failed: {type(e).__name__}"
        print(f"{src['policy_id']:<15} {src['language']}  {status:<45} {n_chars:>7} chars")
        rows.append({**src, "fetched_on": date.today().isoformat(), "status": status, "n_chars": n_chars})
    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
