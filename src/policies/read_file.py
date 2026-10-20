"""
read_policy_file(name, data): the text of a privacy policy the user uploaded,
for people who already have the policy as a file (a company checking its own
policy, or a page saved as PDF).

    .pdf          -> text of every page (pypdf)
    .docx         -> paragraphs of the Word file (read straight from its XML, no extra library)
    .txt / .md    -> the text itself
    .html / .htm  -> the page's main text, like a fetched page

Raises ValueError with a message for the user when the file cannot be read.
"""
import io
import re
import zipfile
from html import unescape

from src.policies.fetch_policies import html_to_text

TYPES = ["pdf", "docx", "txt", "md", "html", "htm"]
PRESENTATION_FORMS = re.compile(r"[ﭐ-﷿ﹰ-﻿]")  # Arabic letters a PDF stored as shapes


def read_pdf(data):
    from pypdf import PdfReader
    try:
        pages = PdfReader(io.BytesIO(data)).pages
    except Exception:
        raise ValueError("ما قدرنا نفتح ملف الـ PDF. تأكد إنه سليم ومو محمي بكلمة سر.")
    text = "\n".join(page.extract_text() or "" for page in pages)
    if len(PRESENTATION_FORMS.findall(text)) > 50:
        # Arabic saved this way comes out with letters in the wrong order
        raise ValueError("النص العربي في هذا الـ PDF ما ينقرأ صح. ارفع السياسة كملف Word، أو الصق نصها.")
    return text


def read_docx(data):
    try:
        xml = zipfile.ZipFile(io.BytesIO(data)).read("word/document.xml").decode("utf-8")
    except (zipfile.BadZipFile, KeyError):
        raise ValueError("ما قدرنا نفتح ملف الـ Word. احفظه بصيغة docx وجرب مرة ثانية.")
    paragraphs = []
    for paragraph in re.findall(r"<w:p[ >].*?</w:p>", xml, re.DOTALL):
        words = re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", paragraph, re.DOTALL)
        paragraphs.append(unescape("".join(words)))
    return "\n".join(p for p in paragraphs if p.strip())


def read_policy_file(name, data):
    kind = name.rsplit(".", 1)[-1].lower()
    if kind == "pdf":
        text = read_pdf(data)
    elif kind == "docx":
        text = read_docx(data)
    elif kind in ("html", "htm"):
        text = html_to_text(data.decode("utf-8", "ignore"))
    elif kind in ("txt", "md"):
        text = data.decode("utf-8-sig", "ignore")
    else:
        raise ValueError("نوع الملف غير مدعوم. ارفع PDF أو Word أو TXT.")
    if len(text.strip()) < 300:
        raise ValueError("الملف ما فيه نص كافٍ. إذا كان PDF مصوّر (صور بدون نص)، انسخ النص والصقه.")
    return text
