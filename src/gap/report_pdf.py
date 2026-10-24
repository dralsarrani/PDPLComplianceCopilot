"""
The gap report as a PDF with an evidence appendix (an extra from the brief:
"export the gap report as a PDF with an evidence appendix").

    pdf_bytes = build_pdf(report, checklist, name, analysed_on, lang="ar")

Page 1: the company, the share of requirements fully covered, the counts, and a
table of the 17 requirements with their status. Appendix: for each requirement,
the quote from the policy, whether it was found in the policy text, the OPP-115
classifier label (if any) and the law articles behind it.

Arabic needs letter joining and right-to-left order, so fpdf2 uses HarfBuzz
(uharfbuzz) with the IBM Plex Sans Arabic font kept in app/fonts/.
"""
import re
from datetime import date
from pathlib import Path

from fpdf import FPDF
from fpdf.fonts import FontFace

FONTS = Path(__file__).resolve().parents[2] / "app" / "fonts"
MONTHS_AR = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر",
             "أكتوبر", "نوفمبر", "ديسمبر"]
GREEN, GREY, AMBER = (15, 81, 50), (107, 114, 128), (138, 90, 18)
FILL = {"covered": (232, 241, 236), "partial": (251, 243, 226), "missing": (251, 234, 234),
        "unsure": (240, 239, 236), "error": (251, 234, 234)}
WORDS = {
    "ar": {"title": "تقرير الامتثال لنظام حماية البيانات الشخصية",
           "policy": "سياسة الخصوصية لدى", "analysed": "تاريخ التحليل", "made": "تاريخ إصدار التقرير",
           "score": "نسبة المتطلبات المستوفاة بالكامل",
           "counts": "من أصل {total} متطلباً: {covered} مستوفى، {partial} مستوفى جزئياً، {missing} غير مذكور",
           "unsure": "، {n} لم يرد في النسخة المتاحة",
           "id": "الرقم", "req": "المتطلب", "status": "الحالة", "articles": "المرجع",
           "appendix": "ملحق الأدلة: الشواهد المقتبسة من نص السياسة",
           "quote": "الشاهد", "found": "التحقق", "yes": "النص موجود حرفياً في السياسة",
           "no": "دليل غير موثق: النص المقتبس غير موجود في السياسة",
           "near": "دليل غير موثق: أعاد المحلل صياغة الاقتباس، وهذا أقرب نص له في السياسة", "none": "لا يوجد شاهد في نص السياسة",
           "opp": "تصنيف المصنّف (OPP-115)", "match": "يطابق المتطلب", "nomatch": "لا يطابق تصنيفات المتطلب",
           "reason": "سبب الحكم",
           "disclaimer": "هذا التقرير آلي لأغراض التوعية، ولا يُعد رأياً قانونياً. "
                         "أُعدّ بنموذج gpt-4o-mini مقابل قائمة من 17 متطلباً مستخلصة من النظام ولائحتيه.",
           "status_words": {"covered": "مستوفى", "partial": "مستوفى جزئياً", "missing": "غير مذكور",
                            "unsure": "لم يرد في النسخة المتاحة", "error": "تعذر التحليل"}},
    "en": {"title": "PDPL compliance report",
           "policy": "Privacy policy of", "analysed": "Analysed on", "made": "Report made on",
           "score": "Requirements fully covered",
           "counts": "Of {total} requirements: {covered} covered, {partial} partly covered, {missing} missing",
           "unsure": ", {n} not in this copy",
           "id": "ID", "req": "Requirement", "status": "Status", "articles": "Source",
           "appendix": "Evidence appendix: quotes from the policy text",
           "quote": "Quote", "found": "Check", "yes": "Found word for word in the policy",
           "no": "Unverified evidence: the quoted text is not in the policy",
           "near": "Unverified evidence: the analyser reworded the quote; this is the closest text in the policy", "none": "No quote in the policy text",
           "opp": "Classifier label (OPP-115)", "match": "matches the requirement",
           "nomatch": "not one of the requirement's categories", "reason": "Reason",
           "disclaimer": "An automatic report for awareness, not legal advice. Made with gpt-4o-mini "
                         "against a checklist of 17 requirements taken from the PDPL and its regulations.",
           "status_words": {"covered": "Covered", "partial": "Partly covered", "missing": "Missing",
                            "unsure": "Not in this copy", "error": "Could not check"}},
}


def nice_date(day, lang):
    """2026-10-20 -> "20 أكتوبر 2026" / "20 Oct 2026": a dash date turns around inside Arabic text."""
    try:
        d = date.fromisoformat(str(day)[:10])
    except ValueError:
        return str(day) or "-"
    return f"{d.day} {MONTHS_AR[d.month - 1]} {d.year}" if lang == "ar" else d.strftime("%d %b %Y")


def is_latin(text):
    return len(re.findall(r"[A-Za-z]", text)) > len(re.findall(r"[\u0600-\u06FF]", text))


def build_pdf(report, checklist, name, analysed_on="", lang="ar", opp115_names=None):
    """report: rows from analyse_v0 (status, evidence, evidence_found, articles, reason,
    optional opp115/opp115_match). checklist: the pdpl_checklist DataFrame.
    opp115_names: {English category: shown name}, for Arabic labels. Returns the PDF bytes."""
    w = WORDS[lang]
    rtl = lang == "ar"
    align = "R" if rtl else "L"
    titles = {r.req_id: (r.requirement_ar if rtl else r.requirement_en) for r in checklist.itertuples()}

    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_font("Plex", "", FONTS / "IBMPlexSansArabic-Regular.ttf")
    pdf.add_font("Plex", "B", FONTS / "IBMPlexSansArabic-Bold.ttf")
    pdf.set_text_shaping(use_shaping_engine=True, direction="rtl" if rtl else "ltr")
    pdf.add_page()

    def line(text, size=10, style="", color=(17, 24, 39), height=6, own_direction=False):
        """own_direction: a quote or reason keeps its own direction (an English quote in an
        Arabic report is written left to right)."""
        ltr = (own_direction and is_latin(text)) or not rtl
        pdf.set_text_shaping(use_shaping_engine=True, direction="ltr" if ltr else "rtl")
        pdf.set_font("Plex", style, size)
        pdf.set_text_color(*color)
        pdf.multi_cell(0, height, text, align="L" if ltr else "R", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_shaping(use_shaping_engine=True, direction="rtl" if rtl else "ltr")

    counts = {s: sum(r["status"] == s for r in report) for s in FILL}
    checked = len(report) - counts["unsure"]
    percent = round(100 * counts["covered"] / checked) if checked else 0

    line(w["title"], 18, "B", GREEN, 10)
    line(f'{w["policy"]} {name}', 13, "B", height=8)
    line(f'{w["analysed"]}: {nice_date(analysed_on, lang)}   ·   {w["made"]}: {nice_date(date.today(), lang)}', 9, color=GREY)
    pdf.ln(3)
    line(f'{w["score"]}: {percent}%', 15, "B", GREEN, 9)
    summary = w["counts"].format(total=len(report), **counts)
    if counts["unsure"]:
        summary += w["unsure"].format(n=counts["unsure"])
    line(summary, 10)
    pdf.ln(3)

    widths = (14, 96, 32, 48)
    heads = (w["id"], w["req"], w["status"], w["articles"])
    order = {"missing": 0, "error": 0, "unsure": 1, "partial": 2, "covered": 3}
    rows = sorted(report, key=lambda r: (order.get(r["status"], 0), r["req_id"]))
    pdf.set_font("Plex", "", 9)
    head_style = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=GREEN)
    with pdf.table(col_widths=widths[::-1] if rtl else widths, text_align=align,
                   headings_style=head_style, line_height=5.5, padding=1.5) as table:
        cells = [heads, *[(r["req_id"], titles.get(r["req_id"], r["requirement"]),
                           w["status_words"].get(r["status"], r["status"]), r["articles"]) for r in rows]]
        for i, values in enumerate(cells):
            style = None if i == 0 else FontFace(fill_color=FILL.get(rows[i - 1]["status"], (255, 255, 255)))
            row = table.row(style=style)
            for value in (values[::-1] if rtl else values):
                row.cell(str(value))

    pdf.ln(4)
    line(w["disclaimer"], 8, color=GREY, height=5)

    pdf.add_page()
    line(w["appendix"], 15, "B", GREEN, 9)
    pdf.ln(2)
    for r in sorted(report, key=lambda r: r["req_id"]):
        if pdf.will_page_break(30):
            pdf.add_page()
        status = w["status_words"].get(r["status"], r["status"])
        pdf.set_fill_color(*FILL.get(r["status"], (255, 255, 255)))
        pdf.set_font("Plex", "B", 10)
        pdf.set_text_color(17, 24, 39)
        pdf.multi_cell(0, 7, f'{r["req_id"]}  {titles.get(r["req_id"], r["requirement"])}  ({status})',
                       align=align, fill=True, new_x="LMARGIN", new_y="NEXT")
        if r.get("evidence"):
            near = r.get("evidence_state") == "near"
            note = w["yes"] if r.get("evidence_found") else w["near"] if near else w["no"]
            line(f'{w["quote"]} ({note}):', 8.5,
                 color=GREEN if r.get("evidence_found") else AMBER if near else (155, 28, 28), height=5)
            line(f'"{r["evidence_near"] if near else r["evidence"]}"', 9.5, height=5.5, own_direction=True)
            if r.get("opp115"):
                label = (opp115_names or {}).get(r["opp115"], r["opp115"])
                line(f'{w["opp"]}: {label} · {w["match"] if r.get("opp115_match") else w["nomatch"]}',
                     8.5, color=GREY, height=5)
        else:
            line(w["none"], 9.5, color=GREY, height=5.5)
        if r.get("reason"):
            line(f'{w["reason"]}:', 8.5, color=GREY, height=5)
            line(r["reason"], 8.5, color=GREY, height=5, own_direction=True)
        line(f'{w["articles"]}: {r["articles"]}', 8.5, color=GREY, height=5)
        pdf.ln(3)
    return bytes(pdf.output())
