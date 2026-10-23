"""
Build the second corpus: the National Cybersecurity Authority's Essential
Cybersecurity Controls (ECC-2:2024), one record per control.

The PDF (downloaded by hand from https://nca.gov.sa/en/regulatory-documents/controls-list/ecc/
into data/nca/) lays each subdomain out as a table: the control number in the left
column, its text in the right one, the number centred on its cell. pdfplumber gives
every word and every table border with its position, so:
    text-column lines between two horizontal borders form one cell,
    the cell gets the control number printed beside it,
    a cell with no number at the top of a page continues the control before it,
    the cells between a subdomain number and the "Controls" label are its title and objective.
Sub-controls (2.2.3.1 ...) stay inside their control's text.

The Arabic PDF (data/nca/ECC-2-2024---NCA.pdf) is read by src/nca/arabic_pdf.py, which
works around its missing ligatures; each control then has an English and an Arabic record,
like the PDPL articles, and the numbers of the two are checked against each other.

Records use the same fields as data/processed/legal_articles_v1.0.json so the
retriever and the citation check can read both:
    record_id ECC-2-3-1-EN / ECC-2-3-1-AR, document_code ECC, article_number 2-3-1,
    article_title = the subdomain ("2-3 Information System and Processing Facilities Protection"),
    chapter = the main domain, objective = the subdomain's objective.

Run from the project root:
    python -m src.nca.build_ecc
Writes data/processed/ecc_controls_v2024.json (and .csv): 108 controls x 2 languages.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pdfplumber

from src.nca.arabic_pdf import parse_arabic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = PROJECT_ROOT / "data" / "nca" / "ECC--2024-EN.pdf"
PDF_PATH_AR = PROJECT_ROOT / "data" / "nca" / "ECC-2-2024---NCA.pdf"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "ecc_controls_v2024.json"
SOURCE_URL = {"en": "https://nca.gov.sa/en/regulatory-documents/controls-list/ecc/",
              "ar": "https://nca.gov.sa/ar/regulatory-documents/controls-list/ecc/"}
FIRST_PAGE, LAST_PAGE = 15, 33   # 0-based pages of "Details of the ECC"; appendices follow
RIGHT_COLUMN = 150               # x where the text column starts (left column: numbers, labels)
BORDER_LEFT, BORDER_WIDTH = 155, 300   # a cell border: starts left of the text column, crosses it
CONTROL = re.compile(r"^\d+-\d+-\d+$")
LABEL = re.compile(r"^\d+[-.]\d+([-.]\d+)?$")  # the PDF prints one subdomain as "1.7"
DOMAINS = {"1": "Cybersecurity Governance", "2": "Cybersecurity Defense", "3": "Cybersecurity Resilience",
           "4": "Third-Party and Cloud Computing Cybersecurity"}
PAGE_FURNITURE = re.compile(r"Essential Cybersecurity Controls|Document classification|TLP:|^\d+$")


def page_lines(page):
    """Lines of the page as (top, bottom, left text, right text), header and footer removed."""
    words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
    lines = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(w["top"] - lines[-1]["top"]) < 3:
            lines[-1]["words"].append(w)
            lines[-1]["bottom"] = max(lines[-1]["bottom"], w["bottom"])
        else:
            lines.append({"top": w["top"], "bottom": w["bottom"], "words": [w]})
    out = []
    for line in lines:
        text = " ".join(w["text"] for w in line["words"])
        if not (60 < line["top"] < 700) or PAGE_FURNITURE.search(text.strip()):
            continue
        left = " ".join(w["text"] for w in line["words"] if w["x0"] < RIGHT_COLUMN).strip()
        right = " ".join(w["text"] for w in sorted(line["words"], key=lambda w: w["x0"])
                         if w["x0"] >= RIGHT_COLUMN).strip()
        out.append((line["top"], line["bottom"], left, right))
    return out


def page_events(page):
    """The page as a list of events in reading order:
    ("row", text, label)  a table cell of the text column; label = the control or
                          subdomain number printed next to it, or None
    ("controls",)         the "Controls" label: the controls of a subdomain start here
    Cells are the bands between the table's full-width horizontal borders."""
    lines = page_lines(page)
    borders = sorted({round(e["top"]) for e in page.edges if e["orientation"] == "h"
                      and e["x0"] <= BORDER_LEFT and e["x1"] - e["x0"] >= BORDER_WIDTH})

    def band(y):
        return sum(1 for b in borders if b <= y)

    cells, labels, events = {}, {}, []
    for top, bottom, left, right in lines:
        middle = (top + bottom) / 2
        if right in DOMAINS.values():  # a main domain heading, not part of any cell
            continue
        if left == "Controls" and not right:
            events.append((top, ("controls",)))
            continue
        if left and LABEL.match(left.split()[0]):
            labels.setdefault(band(middle), left.split()[0].replace(".", "-"))
        if right:
            cell = cells.setdefault(band(middle), [top, []])
            cell[1].append(right)
    for key, (top, texts) in cells.items():
        events.append((top, ("row", " ".join(texts), labels.get(key))))
    return [e for _, e in sorted(events, key=lambda e: e[0])]


def split_header(text):
    """'Cybersecurity Strategy To ensure that ...' -> (title, objective)."""
    match = re.search(r"\bTo [a-z]", text)  # every objective starts "To ensure ..." or "To assess ..."
    if not match:
        return text.strip(), ""
    return text[:match.start()].strip(), text[match.start():].strip()


def parse(pdf_path=PDF_PATH):
    controls, subdomain, objective = [], "", ""
    pending, header_number = [], None  # rows seen since the last control
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[FIRST_PAGE:LAST_PAGE + 1]:
            for event in page_events(page):
                if event[0] == "controls":
                    title, objective = split_header(" ".join(pending))
                    subdomain = f"{header_number} {title}".strip() if header_number else title
                    pending, header_number = [], None
                else:
                    _, text, label = event
                    if label and CONTROL.match(label):
                        if pending and controls and header_number is None:
                            controls[-1]["text"] += " " + " ".join(pending)  # continued over a page
                        pending = []
                        controls.append({"id": label, "domain": DOMAINS[label[0]], "subdomain": subdomain,
                                         "objective": objective, "text": text})
                    else:
                        if label:  # a subdomain number: the header of the next subdomain starts
                            if pending and controls and header_number is None:
                                controls[-1]["text"] += " " + " ".join(pending)
                                pending = []
                            header_number = label
                        pending.append(text)
    return controls


def to_records(controls, language="en"):
    records = []
    for c in controls:
        text = re.sub(r"\s+", " ", c["text"]).strip()
        records.append({
            "record_id": f"ECC-{c['id']}-{language.upper()}", "document_code": "ECC", "article_number": c["id"],
            "article_title": c["subdomain"], "language": language, "chapter": c["domain"],
            "objective": re.sub(r"\s+", " ", c["objective"]).strip(),
            "official_text": text, "normalized_text": text.lower(),
            "source_url": SOURCE_URL[language], "verification_url": SOURCE_URL[language]})
    return records


def main():
    english = parse(PDF_PATH)
    arabic = parse_arabic(PDF_PATH_AR)
    mismatched = [(e["id"], a["id"]) for e, a in zip(english, arabic) if e["id"] != a["id"]]
    if len(english) != len(arabic) or mismatched:
        sys.exit(f"English {len(english)} vs Arabic {len(arabic)} controls; different numbers: {mismatched[:5]}")
    for e, a in zip(english, arabic):
        a["domain"] = e["domain"]  # the domain names are kept in English, like the PDPL chapter field
    records = [r for e, a in zip(to_records(english, "en"), to_records(arabic, "ar")) for r in (e, a)]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(OUT_PATH.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(f"{len(english)} controls x 2 languages -> {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
