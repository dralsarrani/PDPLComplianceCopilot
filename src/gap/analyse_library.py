"""
Analyse every policy in the library once and save the result, so the app can
show a library company's analysis at once (and offline, during the demo).

Reads   data/library/library.csv        (from build_library.py / add_by_hand.py)
        data/library/companies.csv      (Arabic names, for search in the app)
        data/library/texts/<id>.txt
Writes  results/library/<id>.json       {company, dates, text_key, report}

A policy already analysed from the same text is skipped (use --refresh).
Texts that cannot give a fair result are skipped and listed: see EXCLUDE, plus
any text copied out of an Arabic PDF (unreadable) or shorter than MIN_CHARS.

Run from the project root:
    python -m src.gap.analyse_library
"""
import argparse
import csv
import hashlib
import json
import re
from datetime import date
from pathlib import Path

from src.gap.analyse_v0 import analyse

LIBRARY_DIR = Path("data/library")
OUTPUT_DIR = Path("results/library")
MIN_CHARS = 3000
READY = ("collected", "manual", "saved")
PDF_ARABIC = re.compile("[ﭐ-﷿ﹰ-﻿]")
# Checked by hand on 2026-10-19: the saved text is not the company's full policy.
EXCLUDE = {"linkedin": "النص المحفوظ هو سياسة X، انتسخ بالغلط",
           "tabby": "منسوخ من PDF عربي، الحروف والكلمات مقلوبة",
           "temu": "انجمعت صفحة قصيرة بس",
           "paypal": "صفحة مركز الخصوصية وليست السياسة",
           "apple": "صفحة تعريفية وليست السياسة"}


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def text_key(text):
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()[:16]


def problem(cid, text):
    """Why this text should not be analysed, or "" if it is fine."""
    if cid in EXCLUDE:
        return EXCLUDE[cid]
    if len(PDF_ARABIC.findall(text)) > 50:
        return "منسوخ من PDF عربي، الحروف والكلمات مقلوبة"
    if len(text) < MIN_CHARS:
        return f"قصير جداً ({len(text)} حرف)"
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--refresh", action="store_true", help="analyse again even if saved")
    args = parser.parse_args()

    names_ar = {row["company_id"]: row.get("name_ar", "") for row in read_csv(LIBRARY_DIR / "companies.csv")}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    done, skipped = [], []
    for row in read_csv(LIBRARY_DIR / "library.csv"):
        cid, path = row["company_id"], LIBRARY_DIR / "texts" / f"{row['company_id']}.txt"
        if row["status"] not in READY or not path.exists():
            skipped.append((row["name"], "ما انجمعت"))
            continue
        text = path.read_text(encoding="utf-8")
        why = problem(cid, text)
        if why:
            skipped.append((row["name"], why))
            continue
        out = OUTPUT_DIR / f"{cid}.json"
        key = text_key(text)
        if out.exists() and not args.refresh and json.loads(out.read_text(encoding="utf-8"))["text_key"] == key:
            saved = json.loads(out.read_text(encoding="utf-8"))
        else:
            print(f"تحليل {row['name']}...", flush=True)
            saved = {"company_id": cid, "name": row["name"], "name_ar": names_ar.get(cid, ""),
                     "site": row["site"], "sector": row["sector"], "url": row["policy_url"],
                     "fetched_on": row["checked_on"], "analysed_on": date.today().isoformat(),
                     "text_key": key, "report": analyse(text)}
            out.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        counts = {s: sum(r["status"] == s for r in saved["report"]) for s in ("covered", "partial", "missing")}
        done.append((row["name"], counts))

    print(f"\nمحللة: {len(done)}")
    for name, c in sorted(done, key=lambda d: -d[1]["covered"]):
        print(f"  {name:32} مغطى {c['covered']:2}  جزئي {c['partial']:2}  غير مغطى {c['missing']:2}")
    if skipped:
        print(f"\nما انحللت: {len(skipped)}")
        for name, why in skipped:
            print(f"  {name}: {why}")
    print(f"\nالنتائج: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
