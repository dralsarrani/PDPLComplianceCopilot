"""
Add the policies the library could not collect by itself, by hand: for each one
the script opens the policy page in your own browser, you copy the whole page
(Ctrl+A then Ctrl+C), come back and press Enter, and the copied text is saved.

This is the lawful way for sites that do not allow scripts: a person reads the
public page in their browser.

Reads and updates  data/library/library.csv   (run build_library.py first)
Writes             data/library/texts/<id>.txt

Run from the project root:
    python -m src.policies.add_by_hand                  every company still missing
    python -m src.policies.add_by_hand temu paypal      these companies (to replace a bad copy)
"""
import argparse
import csv
import re
import tkinter
import webbrowser
from datetime import date
from urllib.parse import quote_plus

from src.policies.build_library import OUTPUT, TEXTS_DIR, read_csv
from src.policies.fetch_policies import MIN_CHARS
from src.policies.find_policy import as_url

READY = ("collected", "manual", "saved")
# Arabic letters in their "presentation" shapes: what you get when copying Arabic
# out of a PDF viewer. The words also come out in reverse order, so the text is unusable.
PDF_ARABIC = re.compile("[\ufb50-\ufdff\ufe70-\ufeff]")


def clipboard():
    """The text you copied, or "" if the clipboard has no text."""
    root = tkinter.Tk()
    root.withdraw()
    try:
        return root.clipboard_get()
    except tkinter.TclError:
        return ""
    finally:
        root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("ids", nargs="*", help="company ids (default: every company still missing)")
    args = parser.parse_args()

    rows = read_csv(OUTPUT)
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    todo = [row for row in rows if (row["company_id"] in args.ids if args.ids else row["status"] not in READY)]
    unknown = set(args.ids) - {row["company_id"] for row in rows}
    if unknown:
        print("ما لقيت هذي في الجدول:", ", ".join(sorted(unknown)))
    print(f"{len(todo)} سياسة. لكل وحدة: تنفتح الصفحة، اضغط Ctrl+A ثم Ctrl+C، وارجع هنا واضغط Enter.")
    print("لو الصفحة ما فتحت أو قديمة: اكتب g ثم Enter نفتح لك بحث عن سياسة الشركة.")
    print("اكتب s ثم Enter للتخطي، و q للخروج.\n")

    added = 0
    saved = {path.read_text(encoding="utf-8").strip(): path.stem for path in TEXTS_DIR.glob("*.txt")}
    for i, row in enumerate(todo, 1):
        url = row["policy_url"] or as_url(row["site"])
        print(f"[{i}/{len(todo)}] {row['name']}: {url}")
        webbrowser.open(url)
        while True:
            choice = input("  انسخ الصفحة ثم اضغط Enter: ").strip().lower()
            if choice in ("s", "q"):
                break
            if choice == "g":  # the address we have is often out of date: search for the policy
                webbrowser.open("https://www.google.com/search?q=" + quote_plus(f"{row['name']} privacy policy"))
                continue
            if choice:
                continue  # text pasted into the terminal by mistake: ignore it line by line
            text = clipboard().strip()
            if saved.get(text, row["company_id"]) != row["company_id"]:
                print(f"  هذا نفس نص {saved[text]} المحفوظ. انسخ صفحة هذي الشركة.")
                continue
            if len(PDF_ARABIC.findall(text)) > 50:
                print("  النص منسوخ من ملف PDF عربي، فيطلع مقلوب. افتح النسخة الإنجليزية من السياسة وانسخها.")
                continue
            if len(text) < MIN_CHARS:
                print(f"  النص المنسوخ قصير ({len(text)} حرف). تأكد إنك ضغطت Ctrl+A ثم Ctrl+C على صفحة السياسة.")
                continue
            (TEXTS_DIR / f"{row['company_id']}.txt").write_text(text, encoding="utf-8")
            row.update(status="manual", chars=len(text), policy_url=url, checked_on=date.today().isoformat())
            added += 1
            saved[text] = row["company_id"]
            print(f"  انحفظت ({len(text)} حرف).")
            break
        if choice == "q":
            break

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    missing = sum(row["status"] not in READY for row in rows)
    print(f"\nأضفنا {added}. جاهزة: {len(rows) - missing} من {len(rows)}.")


if __name__ == "__main__":
    main()
