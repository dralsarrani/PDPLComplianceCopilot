"""
Build the policy library: privacy policies of well-known companies used in Saudi
Arabia, collected once so the app can show their analysis at once.

Reads   data/library/companies.csv        (company_id, name, site, sector)
Writes  data/library/texts/<id>.txt       (policy text; stays on your computer,
                                           not pushed, because the text belongs
                                           to the company)
        data/library/library.csv          (one row per company: where its policy
                                           is and whether we could collect it)

Collection uses find_policy(): an honest User-Agent, each site's robots.txt
respected, nothing that gets round a site that blocks scripts. A company whose
site blocks us is marked "blocked": open its policy in a browser, copy the text
with add_by_hand.py (it opens the page and saves what you copy).

A company whose text file already exists is not fetched again (use --refresh).

Run from the project root:
    python -m src.policies.build_library
    python -m src.policies.build_library --refresh
"""
import argparse
import csv
import time
from datetime import date
from pathlib import Path

from src.policies.find_policy import BLOCKED, JS_PAGE, NO_ANSWER, NOT_FOUND, find_policy, guess_website

LIBRARY_DIR = Path("data/library")
COMPANIES = LIBRARY_DIR / "companies.csv"
TEXTS_DIR = LIBRARY_DIR / "texts"
OUTPUT = LIBRARY_DIR / "library.csv"
PAUSE = 1  # seconds between companies, to be gentle with the sites

STATUS = {BLOCKED: "blocked", NOT_FOUND: "not_found", NO_ANSWER: "no_answer", JS_PAGE: "javascript"}
STATUS_AR = {"collected": "انجمعت تلقائياً",
             "manual": "منسوخة يدوياً",
             "saved": "محفوظة من قبل",
             "blocked": "الموقع يمنع القراءة الآلية: انسخها يدوياً",
             "not_found": "ما لقينا صفحة السياسة: انسخها يدوياً",
             "javascript": "الصفحة تنبني بـ JavaScript: انسخها يدوياً",
             "no_answer": "الموقع ما رد: جرب لاحقاً أو انسخها يدوياً"}


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect(company):
    """(status, policy address or "", text or "")."""
    try:
        _, _, hint = guess_website(company["name"])  # the privacy address, if OpenAI knows it
    except Exception:
        hint = ""  # no OpenAI: still try the site itself
    try:
        url, text = find_policy(company["site"], hint)
    except ValueError as error:
        return STATUS.get(str(error), "not_found"), hint, ""
    return "collected", url, text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--refresh", action="store_true", help="fetch again even if the text is saved")
    args = parser.parse_args()

    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    before = {row["company_id"]: row for row in read_csv(OUTPUT)} if OUTPUT.exists() else {}
    companies = read_csv(COMPANIES)
    rows = []
    for i, company in enumerate(companies, 1):
        cid = company["company_id"]
        path = TEXTS_DIR / f"{cid}.txt"
        old = before.get(cid, {})
        if path.exists() and not args.refresh:
            status = old["status"] if old.get("status") in ("collected", "manual") else "saved"
            url = old.get("policy_url", "")
            text = path.read_text(encoding="utf-8")
            checked_on = old.get("checked_on", "")
        else:
            status, url, text = collect(company)
            if text:
                path.write_text(text, encoding="utf-8")
            checked_on = date.today().isoformat()
            time.sleep(PAUSE)
        print(f"[{i}/{len(companies)}] {company['name']}: {STATUS_AR[status]}", flush=True)
        rows.append({**company, "policy_url": url, "status": status,
                     "chars": len(text), "checked_on": checked_on})

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ready = sum(row["status"] in ("collected", "manual", "saved") for row in rows)
    print(f"\nجاهزة: {ready} من {len(rows)}")
    manual = [row for row in rows if row["status"] not in ("collected", "manual", "saved")]
    if manual:
        print(f"تحتاج نسخ يدوي ({len(manual)}). شغّل:  python -m src.policies.add_by_hand")
        for row in manual:
            print(f"  {row['company_id']}  {row['name']}  {row['policy_url'] or row['site']}")
    print(f"\nالجدول: {OUTPUT}")


if __name__ == "__main__":
    main()
