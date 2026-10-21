"""
Builds the Excel sheet the team uses to label the Saudi privacy policies by hand
(stage 3). The labels are the "right answers" the Gap Analyser is measured against
(the brief asks for at least 10 policies labelled by the team).

One sheet per policy: the 17 requirements as plain questions, the policy paragraphs
that look closest to each one (to save reading time), a drop-down for the label
(covered / partial / missing, the same three the analyser uses), a column for the
sentence that proves it, and under the table the whole policy text with numbered
paragraphs, so a "missing" can be checked against all of it.

The analyser's own answers are NOT in the sheet, so the labels stay independent.

Closest paragraphs: a paragraph scores one point for each different keyword of
the requirement it contains (keywords in English and Arabic, below); the 3 best are shown.
This only helps the reader; the label must come from the policy itself.

Reads   data/checklist/pdpl_checklist.csv
        data/policies/policies.csv and data/policies/<policy_id>.txt
Writes  data/labels/gap_labels.xlsx

Run from the project root:
    python -m src.gap.make_label_sheet
"""
import re
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

CHECKLIST = Path("data/checklist/pdpl_checklist.csv")
POLICIES = Path("data/policies")
OUT = Path("data/labels/gap_labels.xlsx")

LABELS = ["مغطى", "جزئي", "ناقص"]  # covered / partial / missing, as in src/gap/analyse_v0.py
TOP_PARAGRAPHS = 3
SHOWN_CHARS = 450  # each close paragraph is cut to this length in the table

# The requirement as a question anyone can answer while reading the policy.
QUESTIONS = {
    "R01": "هل تذكر السياسة اسم الجهة وطريقة التواصل معها بخصوص البيانات؟",
    "R02": "هل تذكر بيانات التواصل مع مسؤول حماية البيانات (DPO)؟",
    "R03": "هل تذكر وش البيانات الشخصية اللي تجمعها؟",
    "R04": "هل تذكر كيف تجمع البيانات؟ (منك مباشرة، من جهازك، من جهات ثانية...)",
    "R05": "هل تذكر ليش تجمع البيانات وتعالجها؟ (الغرض)",
    "R06": "هل تذكر الأساس النظامي للجمع؟ (موافقتك، عقد، التزام نظامي، مصلحة مشروعة...)",
    "R07": "هل توضح أي البيانات إلزامية وأيها اختيارية؟",
    "R08": "هل تذكر وش يصير لو ما قدّمت بياناتك؟",
    "R09": "هل تذكر مع مين تشارك البيانات، وصفتهم؟ (مزودي خدمة، جهات حكومية...)",
    "R10": "هل تذكر إذا البيانات تنتقل أو تُعالج خارج المملكة؟",
    "R11": "هل تذكر كم مدة حفظ البيانات، أو كيف تتحدد المدة؟",
    "R12": "هل تذكر كيف تحفظ البيانات وكيف تتلفها؟",
    "R13": "هل تذكر حقوقك الخمسة: العلم، الوصول، الحصول على نسخة، التصحيح، طلب الإتلاف؟",
    "R14": "هل تذكر كيف تستخدم حقوقك؟ (بريد، نموذج، رقم...)",
    "R15": "هل تذكر كيف تسحب موافقتك؟",
    "R16": "هل تذكر الإجراءات اللي تحمي فيها البيانات؟",
    "R17": "هل تذكر إذا فيه قرارات تُتخذ آلياً بالكامل (بدون إنسان)؟",
}

KEYWORDS = {
    "R01": "contact|address|controller|who we are|email|اتصل|تواصل|عنوان|البريد|المتحكم|من نحن",
    "R02": "data protection officer|dpo|privacy officer|مسؤول حماية البيانات|مسؤول الخصوصية",
    "R03": "we collect|information we collect|data we collect|name|phone|email|location|البيانات التي نجمع|نجمع|الاسم|الجوال|الموقع",
    "R04": "directly|automatically|cookies|third part|you provide|from other sources|تلقائي|ملفات تعريف|مباشرة|مصادر أخرى|تقدمها",
    "R05": "purpose|we use|to provide|in order to|الغرض|الأغراض|نستخدم|لتقديم|بهدف",
    "R06": "legal basis|lawful|consent|contract|legal obligation|legitimate interest|الأساس النظامي|موافقت|عقد|التزام نظامي|مصلحة مشروعة",
    "R07": "mandatory|optional|required|voluntary|إلزامي|اختياري|مطلوب",
    "R08": "not provide|fail to provide|unable to|may not be able|refuse|عدم تقديم|لن نتمكن|لن تتمكن|رفض",
    "R09": "share|disclose|third part|service provider|partner|authorit|نشارك|مشاركة|الإفصاح|أطراف ثالثة|مزودي|الجهات",
    "R10": "outside|transfer|international|abroad|cross-border|country|خارج المملكة|نقل|خارج|دولي|دول",
    "R11": "retain|retention|keep|period|as long as|delete|الاحتفاظ|نحتفظ|مدة|فترة",
    "R12": "store|storage|destroy|delet|erase|server|secure|التخزين|نخزن|حفظ|إتلاف|حذف|خوادم",
    "R13": "right to|rights|access to|copy of|correct|rectif|erasure|be informed|الحق في|حقوق|الوصول|نسخة|تصحيح|إتلاف|العلم",
    "R14": "exercise|your rights|request|contact us|submit|form|ممارسة|طلب|التواصل|نموذج|تقديم",
    "R15": "withdraw|revoke|consent|opt out|unsubscribe|سحب|الرجوع عن|موافقت|إلغاء الاشتراك",
    "R16": "security|protect|encrypt|safeguard|measures|access control|أمن|حماية|تشفير|إجراءات|تدابير",
    "R17": "automated|automatic decision|profiling|algorithm|آلي|المعالجة الآلية|التنميط|خوارزم",
}

GREEN, AMBER, RED = "D9EFE2", "FBEBC8", "F8D7D7"
HEAD_FILL = PatternFill("solid", fgColor="14573A")
THIN = Side(style="thin", color="D5DBD8")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")


def paragraphs(text):
    """One paragraph per non-empty line, as the policy text files are saved."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def is_heading(line):
    return bool(re.match(r"\d+[.)]\s", line)) or (len(line) < 60 and not line.endswith((".", "؛", "،")))


def closest(paras, req_id):
    pattern = re.compile(KEYWORDS[req_id], re.IGNORECASE)

    def hits(text):
        return len({m.lower() for m in pattern.findall(text)})

    scored, heading = [], ""
    for i, p in enumerate(paras):
        if is_heading(p):
            heading = p
        # different keywords count; a line under a matching section heading gets one more point
        score = hits(p) + (1 if heading is not p and hits(heading) else 0)
        if hits(p):
            scored.append((score, i))
    best = sorted(scored, key=lambda s: -s[0])[:TOP_PARAGRAPHS]
    parts, shown = [], set()
    for _, i in sorted(best, key=lambda s: s[1]):
        if i in shown:
            continue
        lines, j = [paras[i]], i + 1
        shown.add(i)
        # a heading or a line ending with ":" is shown with the lines under it
        while (is_heading(paras[i]) or paras[i].endswith(":")) and j < len(paras) and len(lines) < 7 \
                and sum(map(len, lines)) < SHOWN_CHARS and not re.match(r"\d+[.)]\s", paras[j]):
            lines.append(paras[j])
            shown.add(j)
            j += 1
        text = " • ".join(lines)
        parts.append(f"[فقرة {i + 1}] " + (text if len(text) <= SHOWN_CHARS else text[:SHOWN_CHARS] + "…"))
    return "\n\n".join(parts) or "ما لقينا فقرة قريبة، دوري في النص كامل تحت."


def header(ws, row, titles, widths=None):
    for col, title in enumerate(titles, 1):
        cell = ws.cell(row=row, column=col, value=title)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BOX


def readme_sheet(wb, policies):
    ws = wb.active
    ws.title = "اقرأني"
    ws.sheet_view.rightToLeft = True
    ws.column_dimensions["A"].width = 110
    rows = [
        ("تعليم سياسات الخصوصية حسب نظام حماية البيانات الشخصية", Font(bold=True, size=16, color="14573A")),
        ("ليش؟ تعليمكم هو «الإجابة الصحيحة» اللي نقيس فيها دقة محلل الفجوات. المطلوب في المشروع 10 سياسات على الأقل معلَّمة من الفريق.", None),
        ("", None),
        ("الطريقة لكل سياسة (كل ورقة = سياسة):", Font(bold=True)),
        ("1. اقري السؤال في العمود B.", None),
        ("2. شوفي «أقرب فقرات» في العمود C. هذي بس مساعدة، ممكن تكون غلط أو ناقصة.", None),
        ("3. إذا ما لقيتي الجواب فيها، دوري في النص كامل تحت الجدول (Ctrl+F يساعد).", None),
        ("4. اختاري تعليمك من القائمة في العمود D.", None),
        ("5. للمغطى والجزئي: انسخي الجملة اللي تثبت كلامك في العمود E.", None),
        ("6. أي شك أو ملاحظة اكتبيها في العمود F.", None),
        ("", None),
        ("معنى التعليمات (نفس اللي يستخدمها المحلل):", Font(bold=True)),
        ("مغطى: السياسة تذكره بوضوح وتحديد.", PatternFill("solid", fgColor=GREEN)),
        ("جزئي: تذكره بس بشكل عام أو ناقص (مثلاً تذكر 3 حقوق من 5).", PatternFill("solid", fgColor=AMBER)),
        ("ناقص: ما تذكره أبداً. قبل ما تختارين ناقص، تأكدي من النص كامل.", PatternFill("solid", fgColor=RED)),
        ("", None),
        ("مهم:", Font(bold=True)),
        ("• علّمي حسب المكتوب في السياسة بس، مو حسب اللي تتوقعين الشركة تسويه.", None),
        ("• R02 وR16 وR17 مشروطة في النظام، بس علّميها عادي: هل السياسة تذكرها ولا لا.", None),
        ("• لا تشوفين نتيجة المحلل قبل ما تخلصين السياسة، عشان يكون تعليمك مستقل.", None),
        ("• ورقة «التقدم» تحسب كم خلصتي في كل سياسة.", None),
        ("", None),
        (f"السياسات ({len(policies)}): " + "، ".join(policies), None),
    ]
    for i, (text, style) in enumerate(rows, 1):
        cell = ws.cell(row=i, column=1, value=text)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        if isinstance(style, Font):
            cell.font = style
        elif isinstance(style, PatternFill):
            cell.fill = style


def policy_sheet(wb, policy, checklist, text):
    ws = wb.create_sheet(policy["sheet"])
    ws.sheet_view.rightToLeft = True
    for col, width in zip("ABCDEF", (7, 38, 80, 12, 50, 28)):
        ws.column_dimensions[col].width = width
    ws["A1"] = f"{policy['name']}: سياسة الخصوصية"
    ws["A1"].font = Font(bold=True, size=14, color="14573A")
    ws["A2"] = policy["url"]
    ws["A2"].hyperlink = policy["url"]
    ws["A2"].font = Font(color="1F5FBF", underline="single")
    header(ws, 4, ["الرقم", "السؤال", "أقرب فقرات من السياسة (مساعدة فقط)", "تعليمك", "الدليل: جملة من السياسة", "ملاحظة"])
    ws.row_dimensions[4].height = 30
    ws.freeze_panes = "A5"

    paras = paragraphs(text)
    choice = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True,
                            showErrorMessage=True, error="اختاري: مغطى أو جزئي أو ناقص")
    ws.add_data_validation(choice)
    first = 5
    for n, req in enumerate(checklist.itertuples(), first):
        question = QUESTIONS[req.req_id]
        if req.applies == "conditional":
            question += "\n(مشروط في النظام، علّميه عادي)"
        values = [req.req_id, question, closest(paras, req.req_id), None, None, None]
        for col, value in enumerate(values, 1):
            cell = ws.cell(row=n, column=col, value=value)
            cell.alignment = WRAP
            cell.border = BOX
        ws.cell(row=n, column=4).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=n, column=4).fill = PatternFill("solid", fgColor="F5F7F6")
        choice.add(f"D{n}")
        ws.row_dimensions[n].height = 150
    last = first + len(checklist) - 1
    for label, color in zip(LABELS, (GREEN, AMBER, RED)):
        ws.conditional_formatting.add(f"D{first}:D{last}", FormulaRule(formula=[f'D{first}="{label}"'],
                                                                     fill=PatternFill("solid", fgColor=color)))

    start = last + 3
    ws.cell(row=start, column=1, value="نص السياسة كامل").font = Font(bold=True, size=13, color="14573A")
    header(ws, start + 1, ["فقرة", "النص"])
    ws.merge_cells(start_row=start + 1, start_column=2, end_row=start + 1, end_column=6)
    for i, p in enumerate(paras, 1):
        row = start + 1 + i
        ws.cell(row=row, column=1, value=i).alignment = Alignment(horizontal="center", vertical="top")
        ws.cell(row=row, column=2, value=p).alignment = WRAP
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=6)
        ws.row_dimensions[row].height = max(15, 15 * (len(p) // 190 + 1))
    return first, last


def progress_sheet(wb, done):
    ws = wb.create_sheet("التقدم", 1)
    ws.sheet_view.rightToLeft = True
    for col, width in zip("ABCDEF", (22, 12, 10, 10, 10, 10)):
        ws.column_dimensions[col].width = width
    header(ws, 1, ["السياسة", "خلصتي", "مغطى", "جزئي", "ناقص", "باقي"])
    for row, (sheet, first, last) in enumerate(done, 2):
        rng = f"'{sheet}'!D{first}:D{last}"
        ws.cell(row=row, column=1, value=sheet)
        ws.cell(row=row, column=2, value=f'=COUNTA({rng})&" من {last - first + 1}"')
        for col, label in enumerate(LABELS, 3):
            ws.cell(row=row, column=col, value=f'=COUNTIF({rng},"{label}")')
        ws.cell(row=row, column=6, value=f"={last - first + 1}-COUNTA({rng})")
        for col in range(1, 7):
            ws.cell(row=row, column=col).border = BOX
            ws.cell(row=row, column=col).alignment = Alignment(horizontal="center")


def main():
    checklist = pd.read_csv(CHECKLIST)
    policies = pd.read_csv(POLICIES / "policies.csv")
    policies["sheet"] = policies["name"].str.slice(0, 31)  # Excel sheet names are at most 31 characters
    wb = Workbook()
    readme_sheet(wb, list(policies["name"]))
    done = []
    for policy in policies.to_dict("records"):
        text = (POLICIES / f"{policy['policy_id']}.txt").read_text(encoding="utf-8")
        first, last = policy_sheet(wb, policy, checklist, text)
        done.append((policy["sheet"], first, last))
    progress_sheet(wb, done)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"Saved {OUT}: {len(done)} policies x {len(checklist)} requirements")


if __name__ == "__main__":
    main()
