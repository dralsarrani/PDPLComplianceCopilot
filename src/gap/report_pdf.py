"""
The gap report as a PDF with an evidence appendix (an extra from the brief:
"export the gap report as a PDF with an evidence appendix").

    pdf_bytes = build_pdf(report, checklist, name, analysed_on, lang="ar", url=..., texts=...)

It looks like the app (مُصان's dark green and gold, IBM Plex Sans Arabic):
page 1 has the verdict (score ring, verdict word, one-line meaning, a square for
each of the 17 requirements) and the three most important gaps as plain
questions; then a table of the 17 requirements with their status and articles;
then the evidence appendix: for each requirement, the quote from the policy,
whether it was found in the policy text, the OPP-115 classifier label (if any)
and the law articles behind it. Every page ends with the page number and the
note that this is guidance, not a legal opinion.

texts: the app's own wording, so the PDF says what the screen says:
{"verdicts": [(limit, tone, word, meaning)], "questions": {req_id: question},
"why": {req_id: what it means for the person}}.

Arabic needs letter joining and right-to-left order, so fpdf2 uses HarfBuzz
(uharfbuzz) with the IBM Plex Sans Arabic font kept in app/fonts/.
"""
import math
import re
from datetime import date
from pathlib import Path

from fpdf import FPDF
from fpdf.fonts import FontFace

APP = Path(__file__).resolve().parents[2] / "app"
FONTS = APP / "fonts"
LOGO = APP / "assets" / "brand" / "musaan-v2" / "02-Logo-Horizontal" / "MUSAAN-horizontal-dark.png"
MONTHS_AR = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر",
             "أكتوبر", "نوفمبر", "ديسمبر"]
DARK, GREEN, GOLD = (4, 36, 26), (5, 70, 44), (217, 180, 90)
INK, GREY, LINE, BAND = (17, 24, 39), (107, 114, 128), (207, 227, 216), (241, 246, 243)
TONE = {"covered": (5, 120, 70), "partial": (180, 110, 10), "missing": (185, 28, 28),
        "unsure": (107, 114, 128), "error": (185, 28, 28)}
FILL = {"covered": (232, 243, 236), "partial": (253, 244, 224), "missing": (252, 235, 235),
        "unsure": (240, 239, 236), "error": (252, 235, 235)}
VERDICT_TONE = {"ok": "covered", "warn": "partial", "bad": "missing"}
DOCS = {"ar": {"PDPL": "نظام حماية البيانات الشخصية", "IR": "اللائحة التنفيذية",
               "TR": "لائحة نقل البيانات خارج المملكة"},
        "en": {"PDPL": "Personal Data Protection Law", "IR": "Implementing Regulation",
               "TR": "Data Transfer Regulation"}}
SHORT_DOCS = {"ar": {"PDPL": "النظام", "IR": "اللائحة", "TR": "لائحة النقل"},
              "en": {"PDPL": "PDPL", "IR": "IR", "TR": "TR"}}
IR4_LETTERS = {("1", "a"): "أ", ("1", "b"): "ب", ("1", "c"): "ج", ("1", "d"): "د", ("1", "e"): "هـ",
               ("1", "f"): "و", ("1", "g"): "ز", ("5", "b"): "ب", ("5", "c"): "ج"}
WORDS = {
    "ar": {"kicker": "تقرير الامتثال لنظام حماية البيانات الشخصية",
           "policy": "سياسة الخصوصية لدى", "analysed": "تاريخ التحليل", "made": "تاريخ إصدار التقرير",
           "source": "المصدر", "meaning": "الخلاصة",
           "squares": "من أصل {total} متطلباً يشترطها النظام",
           "legend": {"covered": "مستوفى", "partial": "مستوفى جزئياً", "missing": "غير مذكور",
                      "unsure": "لم يرد في النسخة المتاحة"},
           "incomplete": "نتيجة غير مكتملة",
           "incomplete_meaning": "تعذّر التحقق إلا من {checked} من أصل {total} متطلباً، والنسبة محسوبة عليها فقط",
           "gaps": "أبرز الجوانب المتعلقة ببياناتك التي لم تستوفها السياسة",
           "gap_missing": "لم تتناوله السياسة", "gap_partial": "تناولته السياسة بصورة جزئية",
           "all_covered": "تستوفي السياسة جميع المتطلبات",
           "table": "المتطلبات السبعة عشر",
           "id": "الرقم", "req": "المتطلب", "status": "الحالة", "articles": "المرجع",
           "appendix": "ملحق الأدلة: الشواهد المقتبسة من نص السياسة",
           "yes": "دليل موثق: النص موجود في السياسة حرفياً",
           "no": "دليل غير موثق: النص المقتبس غير موجود في السياسة",
           "near": "دليل غير موثق: أعاد المحلل صياغة الاقتباس، وهذا أقرب نص له في السياسة",
           "none": "لا يوجد شاهد في نص السياسة",
           "opp": "فئة الدليل (مصنّف OPP-115)", "match": "يطابق المتطلب", "nomatch": "لا يطابق تصنيفات المتطلب",
           "page": "صفحة {n} من {total}",
           "footer": "مُصان · المعلومات المقدَّمة للاسترشاد فقط، ولا تُعد رأياً قانونياً ملزماً",
           "method": "تحليل آلي بنموذج gpt-4o-mini مقابل 17 متطلباً مستخلصة من نظام حماية البيانات الشخصية "
                     "ولائحتيه، مع التحقق من وجود كل اقتباس في نص السياسة.",
           "status_words": {"covered": "مستوفى", "partial": "مستوفى جزئياً", "missing": "غير مذكور",
                            "unsure": "لم يرد في النسخة المتاحة", "error": "تعذر التحليل"}},
    "en": {"kicker": "PDPL compliance report",
           "policy": "Privacy policy of", "analysed": "Analysed on", "made": "Report made on",
           "source": "Source", "meaning": "Meaning",
           "squares": "Of the {total} requirements of the law",
           "legend": {"covered": "covered", "partial": "partly covered", "missing": "missing",
                      "unsure": "not in this copy"},
           "incomplete": "Incomplete result",
           "incomplete_meaning": "we could check only {checked} of the {total} requirements, and the score is on those only",
           "gaps": "Top questions about your data the policy leaves open",
           "gap_missing": "The policy does not answer this", "gap_partial": "The policy answers this only in part",
           "all_covered": "The policy covers every requirement",
           "table": "The 17 requirements",
           "id": "ID", "req": "Requirement", "status": "Status", "articles": "Source",
           "appendix": "Evidence appendix: quotes from the policy text",
           "yes": "Verified evidence: the text is in the policy word for word",
           "no": "Unverified evidence: the quoted text is not in the policy",
           "near": "Unverified evidence: the analyser reworded the quote; this is the closest text in the policy",
           "none": "No quote in the policy text",
           "opp": "Evidence category (OPP-115 classifier)", "match": "matches the requirement",
           "nomatch": "not one of the requirement's categories",
           "page": "Page {n} of {total}",
           "footer": "MUSAAN · For guidance only; not a binding legal opinion",
           "method": "Automatic analysis with gpt-4o-mini against 17 requirements taken from the PDPL and its "
                     "two regulations; every quote is checked against the policy text.",
           "status_words": {"covered": "Covered", "partial": "Partly covered", "missing": "Missing",
                            "unsure": "Not in this copy", "error": "Could not check"}},
}
ORDER = {"missing": 0, "error": 0, "unsure": 1, "partial": 2, "covered": 3}


def nice_date(day, lang):
    """2026-10-20 -> "20 أكتوبر 2026" / "20 Oct 2026": a dash date turns around inside Arabic text."""
    try:
        d = date.fromisoformat(str(day)[:10])
    except ValueError:
        return str(day) or "-"
    return f"{d.day} {MONTHS_AR[d.month - 1]} {d.year}" if lang == "ar" else d.strftime("%d %b %Y")


def is_latin(text):
    return len(re.findall(r"[A-Za-z]", text)) > len(re.findall(r"[؀-ۿ]", text))


def refs(articles, lang, short=False):
    """'PDPL-A13(1); IR-A04(1b)' -> 'النظام، المادة 13 (1)؛ اللائحة، المادة 4 (1/ب)' (short)
    or with the full names of the texts. Unknown references keep their ID."""
    names = (SHORT_DOCS if short else DOCS)[lang]
    out = []
    for ref in (r.strip() for r in str(articles).split(";")):
        m = re.fullmatch(r"(PDPL|IR|TR)-A0*(\d+)(?:\((\d+)([a-z]?)\))?", ref)
        if not ref:
            continue
        if not m:
            out.append(ref)
            continue
        doc, article, paragraph, letter = m.groups()
        if lang == "en":
            out.append(f"{names[doc]} Art. {article}" + (f"({paragraph})" if paragraph else "")
                       + (f"({letter})" if letter else ""))
            continue
        arabic = IR4_LETTERS.get((paragraph, letter)) if doc == "IR" and article == "4" else None
        if letter and not arabic:
            out.append(ref)
            continue
        part = f" ({paragraph}/{arabic})" if arabic else f" ({paragraph})" if paragraph else ""
        out.append(f"{names[doc]}، المادة {article}{part}")
    return ("؛ " if lang == "ar" else "; ").join(out)


class Report(FPDF):
    def __init__(self, words, rtl):
        super().__init__()
        self.words, self.rtl, self.total = words, rtl, 0

    def footer(self):
        self.set_y(-12)
        self.set_draw_color(*LINE)
        self.line(self.l_margin, self.get_y() - 1.5, self.w - self.r_margin, self.get_y() - 1.5)
        self.set_font("Plex", "", 7.5)
        self.set_text_color(*GREY)
        left, right = self.words["page"].format(n=self.page_no(), total=self.total or ""), self.words["footer"]
        if self.rtl:
            left, right = right, left
        self.cell(0, 5, right, align="R")
        self.set_x(self.l_margin)
        self.cell(0, 5, left, align="L")


def build_pdf(report, checklist, name, analysed_on="", lang="ar", opp115_names=None, url="", texts=None, pages=0):
    """report: rows from analyse_v0 (status, evidence, evidence_found, articles, reason,
    optional opp115/opp115_match). checklist: the pdpl_checklist DataFrame.
    opp115_names: {English category: shown name}, for Arabic labels. url: the policy page.
    texts: the app's verdicts, plain questions and reasons (see the top). Returns the PDF bytes.
    pages: the page count for the footer; the PDF is made twice, the first time to count them."""
    w = WORDS[lang]
    texts = texts or {}
    rtl = lang == "ar"
    align = "R" if rtl else "L"
    titles = {r.req_id: (r.requirement_ar if rtl else r.requirement_en) for r in checklist.itertuples()}
    title = lambda r: titles.get(r["req_id"], r.get("requirement", ""))

    pdf = Report(w, rtl)
    pdf.total = pages
    pdf.set_margins(16, 14, 16)
    pdf.set_auto_page_break(True, margin=18)
    pdf.add_font("Plex", "", FONTS / "IBMPlexSansArabic-Regular.ttf")
    pdf.add_font("Plex", "B", FONTS / "IBMPlexSansArabic-Bold.ttf")
    direction = "rtl" if rtl else "ltr"
    pdf.set_text_shaping(use_shaping_engine=True, direction=direction)
    inner = pdf.w - pdf.l_margin - pdf.r_margin

    def text(value, size=10, style="", color=INK, height=6, own_direction=False, x=None, width=0, fill=None):
        """own_direction: a quote keeps its own direction (an English quote in an Arabic
        report is written left to right)."""
        ltr = (own_direction and is_latin(value)) or not rtl
        pdf.set_text_shaping(use_shaping_engine=True, direction="ltr" if ltr else "rtl")
        pdf.set_font("Plex", style, size)
        pdf.set_text_color(*color)
        if x is not None:
            pdf.set_x(x)
        if fill:
            pdf.set_fill_color(*fill)
        pdf.multi_cell(width, height, value, align="L" if ltr else "R", fill=bool(fill),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_shaping(use_shaping_engine=True, direction=direction)

    counts = {s: sum(r["status"] == s for r in report) for s in FILL}
    total = len(report)
    checked = total - counts["unsure"]
    percent = round(100 * counts["covered"] / checked) if checked else 0
    verdicts = texts.get("verdicts") or [(80, "ok", "", ""), (50, "warn", "", ""), (0, "bad", "", "")]
    tone, word, meaning = next((t, wd, m) for limit, t, wd, m in verdicts if percent >= limit)
    tone = VERDICT_TONE.get(tone, "partial")
    if counts["unsure"]:
        tone, word = "partial", w["incomplete"]
        meaning = w["incomplete_meaning"].format(checked=checked, total=total)

    # ---- page 1: header band with the logo, then the verdict ----
    pdf.add_page()
    band = 30
    pdf.set_fill_color(*DARK)
    pdf.rect(0, 0, pdf.w, band, "F")
    pdf.set_fill_color(*GOLD)
    pdf.rect(0, band, pdf.w, 1.2, "F")
    logo_w = 46
    if LOGO.exists():
        pdf.image(str(LOGO), x=pdf.w - pdf.r_margin - logo_w if rtl else pdf.l_margin, y=8, w=logo_w)
    pdf.set_y(11)
    pdf.set_font("Plex", "B", 11)
    pdf.set_text_color(*GOLD)
    pdf.set_x(pdf.l_margin if rtl else pdf.l_margin + logo_w + 6)
    pdf.cell(inner - logo_w - 6, 6, w["kicker"], align="L" if rtl else "R")
    pdf.set_y(17)
    pdf.set_font("Plex", "", 8.5)
    pdf.set_text_color(200, 220, 210)
    pdf.set_x(pdf.l_margin if rtl else pdf.l_margin + logo_w + 6)
    pdf.cell(inner - logo_w - 6, 5, f'{w["made"]}: {nice_date(date.today(), lang)}', align="L" if rtl else "R")

    pdf.set_y(band + 8)
    text(f'{w["policy"]} {name}', 17, "B", DARK, 9)
    meta = f'{w["analysed"]}: {nice_date(analysed_on, lang)}'
    text(meta, 9, color=GREY, height=5)
    if url:
        # the label in the report's direction, the address beside it left to right
        pdf.set_font("Plex", "", 8)
        pdf.set_text_color(*GREY)
        label = f'{w["source"]}: '
        label_w = pdf.get_string_width(label) + 1
        y = pdf.get_y()
        pdf.set_x(pdf.w - pdf.r_margin - label_w if rtl else pdf.l_margin)
        pdf.cell(label_w, 4.5, label, align=align)
        pdf.set_text_shaping(use_shaping_engine=True, direction="ltr")
        pdf.set_xy(pdf.l_margin if rtl else pdf.l_margin + label_w, y)
        pdf.cell(inner - label_w, 4.5, url[:110], align=align, link=url)
        pdf.set_text_shaping(use_shaping_engine=True, direction=direction)
        pdf.ln(4.5)
    pdf.ln(4)

    # verdict card: ring on the start side, the words beside it
    top, card_h = pdf.get_y(), 54
    pdf.set_fill_color(*BAND)
    pdf.set_draw_color(*LINE)
    pdf.rect(pdf.l_margin, top, inner, card_h, "DF", round_corners=True, corner_radius=3)
    pdf.set_fill_color(*TONE[tone])
    pdf.rect(pdf.l_margin, top, inner, 1.6, "F")
    d = 34
    ring_x = pdf.w - pdf.r_margin - d - 8 if rtl else pdf.l_margin + 8
    ring_y = top + 10
    pdf.set_line_width(4)
    pdf.set_draw_color(225, 232, 228)
    pdf.ellipse(ring_x, ring_y, d, d, "D")
    if percent:
        pdf.set_draw_color(*TONE[tone])
        pdf.arc(ring_x, ring_y, d, -90, -90 + 360 * percent / 100 - (0.01 if percent == 100 else 0))
    pdf.set_line_width(0.2)
    pdf.set_xy(ring_x, ring_y + d / 2 - 5)
    pdf.set_font("Plex", "B", 16)
    pdf.set_text_color(*TONE[tone])
    pdf.set_text_shaping(use_shaping_engine=True, direction="ltr")
    pdf.cell(d, 10, f"{percent}%", align="C")
    pdf.set_text_shaping(use_shaping_engine=True, direction=direction)

    words_x = pdf.l_margin + 8 if rtl else ring_x + d + 10
    words_w = inner - d - 26
    pdf.set_y(top + 8)
    if word:
        text(word, 22, "B", TONE[tone], 11, x=words_x, width=words_w)
    if meaning:
        text(f'{w["meaning"]}: {meaning}', 10, "B", INK, 6, x=words_x, width=words_w)
    pdf.ln(2)
    # a square for each requirement, worst first as in the app
    sq, gap = 5.2, 1.3
    statuses = [r["status"] for r in sorted(report, key=lambda r: -ORDER.get(r["status"], 0))]
    y = pdf.get_y()
    for i, status in enumerate(statuses):
        offset = i * (sq + gap)
        x = words_x + words_w - sq - offset if rtl else words_x + offset
        pdf.set_fill_color(*TONE.get(status, GREY))
        pdf.rect(x, y, sq, sq, "F", round_corners=True, corner_radius=0.8)
    pdf.set_y(y + sq + 2.5)
    legend = ("، " if rtl else ", ").join(f'{counts[s]} {w["legend"][s]}' for s in ("covered", "partial", "missing", "unsure")
                                          if counts[s])
    text(f'{w["squares"].format(total=total)}: {legend}', 8.5, color=GREY, height=5, x=words_x, width=words_w)
    pdf.set_y(top + card_h + 8)

    # the three most important gaps, as the questions a person would ask
    order = {("missing", "always"): 0, ("missing", "conditional"): 1, ("partial", "always"): 2,
             ("partial", "conditional"): 3}
    gaps = sorted((r for r in report if r["status"] in ("missing", "partial")),
                  key=lambda r: (order.get((r["status"], r.get("applies", "always")), 4), r["req_id"]))[:3]
    text(w["gaps"] if gaps else w["all_covered"], 12.5, "B", DARK, 8)
    pdf.ln(1)
    for r in gaps:
        y = pdf.get_y()
        color = TONE[r["status"]]
        pdf.set_fill_color(*FILL[r["status"]])
        x0 = pdf.l_margin
        pdf.rect(x0, y, inner, 20, "F", round_corners=True, corner_radius=2)
        pdf.set_fill_color(*color)
        pdf.rect(pdf.w - pdf.r_margin - 1.4 if rtl else x0, y, 1.4, 20, "F")
        pdf.set_y(y + 2)
        pad = pdf.l_margin + 5
        question = texts.get("questions", {}).get(r["req_id"], title(r))
        text(question, 11, "B", INK, 6, x=pad, width=inner - 10)
        text(w["gap_missing"] if r["status"] == "missing" else w["gap_partial"], 8.5, "B", color, 4.5,
             x=pad, width=inner - 10)
        why = texts.get("why", {}).get(r["req_id"])
        if why:
            text(why, 8.5, color=GREY, height=4.5, x=pad, width=inner - 10)
        pdf.set_y(y + 23)
    pdf.ln(2)
    text(w["method"], 7.5, color=GREY, height=4)

    # ---- the 17 requirements ----
    pdf.add_page()
    text(w["table"], 15, "B", DARK, 9)
    pdf.set_fill_color(*GOLD)
    pdf.rect(pdf.w - pdf.r_margin - 24 if rtl else pdf.l_margin, pdf.get_y(), 24, 0.8, "F")
    pdf.ln(5)
    widths = (13, 92, 28, 45)
    heads = (w["id"], w["req"], w["status"], w["articles"])
    rows = sorted(report, key=lambda r: (ORDER.get(r["status"], 0), r["req_id"]))
    pdf.set_font("Plex", "", 9)
    pdf.set_draw_color(*LINE)
    head_style = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=DARK)
    with pdf.table(col_widths=widths[::-1] if rtl else widths, text_align=align, headings_style=head_style,
                   line_height=5.5, padding=(2, 2.2), borders_layout="HORIZONTAL_LINES",
                   cell_fill_color=(255, 255, 255), cell_fill_mode="ALL") as table:
        head = table.row()
        for value in (heads[::-1] if rtl else heads):
            head.cell(value)
        for r in rows:
            row = table.row()
            status = FontFace(emphasis="BOLD", color=TONE.get(r["status"], GREY),
                              fill_color=FILL.get(r["status"], (255, 255, 255)))
            cells = [(r["req_id"], FontFace(emphasis="BOLD", color=GREY)), (title(r), None),
                     (w["status_words"].get(r["status"], r["status"]), status),
                     (refs(r["articles"], lang, short=True), FontFace(color=GREY, size_pt=8))]
            for value, style in (cells[::-1] if rtl else cells):
                row.cell(str(value), style=style)

    # ---- evidence appendix ----
    pdf.add_page()
    text(w["appendix"], 15, "B", DARK, 9)
    pdf.set_fill_color(*GOLD)
    pdf.rect(pdf.w - pdf.r_margin - 24 if rtl else pdf.l_margin, pdf.get_y(), 24, 0.8, "F")
    pdf.ln(5)
    for r in sorted(report, key=lambda r: r["req_id"]):
        if pdf.will_page_break(40):
            pdf.add_page()
        color = TONE.get(r["status"], GREY)
        status = w["status_words"].get(r["status"], r["status"])
        y = pdf.get_y()
        # header strip: id and title, the status on the other side
        pdf.set_fill_color(*FILL.get(r["status"], BAND))
        pdf.rect(pdf.l_margin, y, inner, 8, "F", round_corners=True, corner_radius=1.5)
        pdf.set_fill_color(*color)
        pdf.rect(pdf.w - pdf.r_margin - 1.4 if rtl else pdf.l_margin, y, 1.4, 8, "F")
        pdf.set_font("Plex", "B", 8.5)
        pdf.set_text_color(*color)
        pdf.set_xy(pdf.l_margin + 3, y + 1.5)
        pdf.cell(inner - 6, 5, status, align="L" if rtl else "R")
        pdf.set_xy(pdf.l_margin + 3, y + 1.5)
        pdf.set_font("Plex", "B", 10)
        pdf.set_text_color(*INK)
        head = f'{r["req_id"]}  {title(r)}'
        pdf.cell(inner - 6, 5, head, align=align)
        pdf.set_y(y + 10)
        if r.get("evidence"):
            near = r.get("evidence_state") == "near"
            found = r.get("evidence_found")
            note = w["yes"] if found else w["near"] if near else w["no"]
            quote = r["evidence_near"] if near else r["evidence"]
            text(f'"{quote}"', 9.5, color=INK, height=5.5, own_direction=True, fill=(248, 250, 249))
            text(note, 8, "B", TONE["covered"] if found else TONE["partial"] if near else TONE["missing"], 5)
            if r.get("opp115"):
                label = (opp115_names or {}).get(r["opp115"], r["opp115"])
                text(f'{w["opp"]}: {label} · {w["match"] if r.get("opp115_match") else w["nomatch"]}',
                     8, color=GREY, height=4.5)
        else:
            text(w["none"], 9, color=GREY, height=5.5)
        if r.get("reason") and not rtl:  # the analyser writes its reason in English
            text(r["reason"], 8, color=GREY, height=4.5)
        text(f'{w["articles"]}: {refs(r["articles"], lang)}', 8, color=GREY, height=4.5)
        pdf.ln(4)
    if not pages:
        return build_pdf(report, checklist, name, analysed_on, lang, opp115_names, url, texts, pdf.page_no())
    return bytes(pdf.output())
