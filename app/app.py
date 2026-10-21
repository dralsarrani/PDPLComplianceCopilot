"""PDPL Compliance Copilot: Streamlit app.

Two pages:
    Q&A               question -> answer() (retrieval/answer.py): answer, cited
                      articles with their text, citation-check problems
    Policy gap check  a company name or website (its privacy page is found and read,
                      src/policies/find_policy.py) or pasted text -> analyse()
                      (src/gap/analyse_v0.py). The 11 saved Saudi policies are not
                      shown, but typing one of their names uses the saved copy:
                      one card per PDPL requirement

Fallback when OpenAI cannot be reached (demo day without network):
    Q&A               answer() returns the cached answer for the same question
    Policy gap check  the last saved result in results/gap_v0/<policy>.json

History: every question and analysis is saved in results/history.json on this
computer and listed in the side panel; a click shows the saved result at once.
"مسح السجل" deletes the file.

Speed: the search models load in a background thread when the app starts, so
the page shows at once. A policy text that was analysed before is shown from
results/gap_cache/ straight away.

Look: app/style.css (right-to-left, IBM Plex Sans Arabic) and .streamlit/config.toml.

Run from the project root:
    python -m streamlit run app/app.py
"""
import hashlib
import html
import json
import re
import sys
import threading
from urllib.parse import quote_plus
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.citation_check import load_articles  # noqa: E402
from src.gap.analyse_v0 import analyse  # noqa: E402
from src.gap.evidence_labels import label_evidence  # noqa: E402
from src.policies.archive import from_archive  # noqa: E402
from src.policies.find_policy import find_policy, guess_website, looks_like_site  # noqa: E402
from src.policies.web_search import from_web_search  # noqa: E402
from src.policies.read_file import TYPES as FILE_TYPES, read_policy_file  # noqa: E402
# retrieval.answer and retrieval.retriever load the search models; they are
# imported in warm_up() so the page does not wait for them.

POLICIES_DIR = PROJECT_ROOT / "data" / "policies"
CHECKLIST_PATH = PROJECT_ROOT / "data" / "checklist" / "pdpl_checklist.csv"
GAP_RESULTS_DIR = PROJECT_ROOT / "results" / "gap_v0"
GAP_CACHE_DIR = PROJECT_ROOT / "results" / "gap_cache"
LIBRARY_DIR = PROJECT_ROOT / "results" / "library"  # analysed ahead by src/gap/analyse_library.py
HISTORY_PATH = PROJECT_ROOT / "results" / "history.json"
HISTORY_MAX = 50
STYLE_PATH = Path(__file__).resolve().parent / "style.css"
DARK_PATH = Path(__file__).resolve().parent / "dark.css"  # night look, used while <html> has class pdpl-dark
EN = st.query_params.get("lang") == "en"  # English page; kept in the address, so a refresh keeps it


def tr(ar, en):
    """The text in the page's language."""
    return en if EN else ar


GAP_MODES = {"search": tr("ابحث عن شركة", "Search a company"), "file": tr("ارفع ملف", "Upload a file")}
MIN_PASTED = 300  # a search longer than this is a pasted policy, not a name
NAMES_AR = {"hungerstation": "هنقرستيشن", "careem": "كريم", "alrajhi": "مصرف الراجحي", "alinma": "مصرف الإنماء",
            "tamara": "تمارا", "jarir": "مكتبة جرير", "stc": "إس تي سي", "saudia": "الخطوط السعودية",
            "nahdi": "صيدليات النهدي", "ksu": "جامعة الملك سعود", "kfupm": "جامعة البترول والمعادن"}
SECTORS_AR = {"Banking": "بنوك", "Fintech (BNPL)": "تقنية مالية", "Telecom": "اتصالات",
              "Food delivery": "توصيل طعام", "Ride-hailing": "نقل ركاب", "Retail": "تجزئة",
              "Airline": "طيران", "Health / pharmacy": "صحة وصيدليات", "Education": "جامعات"}

PAGES = {"qa": tr("اسأل عن النظام", "Ask the law"), "gap": tr("تحليل سياسة الخصوصية", "Check a privacy policy")}
DOCUMENTS = {"PDPL": tr("نظام حماية البيانات الشخصية", "Personal Data Protection Law"),
             "IR": tr("اللائحة التنفيذية", "Implementing Regulation"),
             "TR": tr("لائحة نقل البيانات خارج المملكة", "Data Transfer Regulation")}
SHORT_DOCUMENTS = {"PDPL": tr("النظام", "PDPL"), "IR": tr("اللائحة", "IR"), "TR": tr("لائحة النقل", "TR")}
STATUS = {"covered": (tr("مغطى", "Covered"), "ok"),
          "partial": (tr("مغطى جزئياً", "Partly covered"), "warn"),
          "missing": (tr("غير مغطى", "Missing"), "bad"),
          "error": (tr("تعذر التحليل", "Could not check"), "bad")}
FILTERS = {"all": tr("الكل", "All"), "missing": STATUS["missing"][0], "partial": STATUS["partial"][0],
           "covered": STATUS["covered"][0]}
PROBLEM_LABELS = {"not_in_corpus": tr("مادة غير موجودة في النظام", "Article not in the law"),
                  "not_retrieved": tr("مادة لم يسترجعها البحث", "Article not found by the search"),
                  "unsupported": tr("المادة لا تدعم الجملة", "Article does not support the sentence"),
                  "uncited": tr("جملة بدون استشهاد", "Sentence without a citation"),
                  "error": tr("تعذر الفحص", "Could not check")}
# Messages from src/policies/ are written in Arabic; their English for the English page.
MESSAGES_EN = {
    "الموقع لا يسمح بقراءة صفحاته آلياً، ونحترم ذلك.": "The site does not allow its pages to be read by programs, and we respect that.",
    "ما لقينا صفحة سياسة الخصوصية في موقع الشركة.": "We could not find the privacy policy page on the company's site.",
    "موقع الشركة ما رد على طلبنا.": "The company's site did not answer.",
    "صفحة سياسة الخصوصية ما فيها نص كافٍ، غالباً الموقع يعرضها بـ JavaScript.":
        "The privacy page has too little text; the site probably shows it with JavaScript.",
    "ما لقينا نسخة من السياسة في الأرشيف العام.": "No copy of the policy was found in the public archives.",
    "ما لقينا سياسة الخصوصية حتى بالبحث في الإنترنت.": "The privacy policy was not found even with a web search.",
    "ما قدرنا نفتح ملف الـ PDF. تأكد إنه سليم ومو محمي بكلمة سر.":
        "We could not open the PDF. Check that it is not damaged or password-protected.",
    "النص العربي في هذا الـ PDF ما ينقرأ صح. ارفع السياسة كملف Word، أو الصق نصها.":
        "The Arabic text in this PDF cannot be read correctly. Upload the policy as a Word file, or paste its text.",
    "ما قدرنا نفتح ملف الـ Word. احفظه بصيغة docx وجرب مرة ثانية.":
        "We could not open the Word file. Save it as .docx and try again.",
    "نوع الملف غير مدعوم. ارفع PDF أو Word أو TXT.": "This file type is not supported. Upload a PDF, Word or TXT file.",
    "الملف ما فيه نص كافٍ. إذا كان PDF مصوّر (صور بدون نص)، انسخ النص والصقه.":
        "The file has too little text. If it is a scanned PDF (pictures, no text), copy the text and paste it.",
}


def message(text):
    """A message from src/policies/ in the page's language."""
    if not EN:
        return text
    if text.startswith("تعذر البحث في الإنترنت"):
        return "The web search for the policy failed."
    return MESSAGES_EN.get(text, text)
# Small line icons (inline SVG, colour follows the text)
def _svg(path, size=16):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{path}</svg>')


ICONS = {"ok": _svg('<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.5 2.5L16 9.5"/>'),
         "warn": _svg('<circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor"/>'),
         "bad": _svg('<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/>'),
         "shield": _svg('<path d="M12 3l7 3v5c0 4.5-3 8.5-7 10-4-1.5-7-5.5-7-10V6z"/><path d="M9 12l2 2 4-4"/>', 30),
         "question": _svg('<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 0 1 4.8 1c0 1.7-2.3 2-2.3 3.5"/><path d="M12 17h.01"/>', 22),
         "doc": _svg('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/>', 22),
         "check": _svg('<path d="M5 12.5l4.5 4.5L19 7.5"/>', 14),
         "info": _svg('<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>')}


st.set_page_config(page_title="PDPL Compliance Copilot", page_icon="⚖️", layout="wide",
                   initial_sidebar_state="collapsed")
LTR_PATH = Path(__file__).resolve().parent / "ltr.css"  # left-to-right layout for the English page
LOOK = (STYLE_PATH.read_text(encoding="utf-8") + DARK_PATH.read_text(encoding="utf-8")
        + (LTR_PATH.read_text(encoding="utf-8") if EN else ""))
st.markdown(f"<style>{LOOK}</style>", unsafe_allow_html=True)  # one element, so the page does not shift

def go_to(page):
    st.session_state.page_now = page


NAV_ICONS = {"qa": ('<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.8" '
                    'stroke-linecap="round" stroke-linejoin="round"><path d="M4 5h16v11H9l-5 4z"/><path d="M8 9h8M8 12h5"/></svg>'),
             "gap": ('<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.8" '
                     'stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H6v18h12V7z"/><path d="M14 3v4h4"/>'
                     '<path d="M9 14l2 2 4-4"/></svg>')}


def switch_language():
    st.query_params["lang"] = "ar" if EN else "en"


# Page behaviour that needs the browser, run from a tiny hidden frame of the same page:
# - the moon / sun button switches the night look at once, with no rerun of the app
#   (the choice is kept in the browser, so a refresh keeps it);
# - the page switch shows the other page at once (both pages are already built);
# - a click anywhere outside the open history panel closes it, like its arrow.
# The frame is made again on some reruns, and a listener from a removed frame stops
# working, so the newest frame always replaces the old listeners.
PAGE_SCRIPT = """<script>
const doc = window.parent.document;
function setLook(dark) {
  doc.documentElement.classList.toggle("pdpl-dark", dark);
  try { window.parent.localStorage.setItem("pdplLook", dark ? "dark" : "light"); } catch (e) {}
}

let saved = "light";
try { saved = window.parent.localStorage.getItem("pdplLook") || "light"; } catch (e) {}
setLook(saved === "dark");

const old = doc.pdplHandlers || {};
if (old.click) doc.removeEventListener("click", old.click);
if (old.down) doc.removeEventListener("mousedown", old.down);
doc.pdplHandlers = {
  click: (event) => {
    if (event.target.closest(".theme-btn")) setLook(!doc.documentElement.classList.contains("pdpl-dark"));
    const tab = event.target.closest(".page-nav button");
    if (tab && tab.closest(".page-nav").dataset.p !== tab.dataset.page) {
      tab.closest(".page-nav").dataset.p = tab.dataset.page;  // CSS shows that page now
      window.parent.scrollTo(0, 0);
      const go = doc.querySelector(".st-key-go_" + tab.dataset.page + " button");
      if (go) go.click();  // tells Python, quietly, for the next runs
    }
  },
  down: (event) => {
    const panel = doc.querySelector('section[data-testid="stSidebar"]');
    if (!panel || panel.getAttribute("data-collapsed") !== "false" || panel.contains(event.target)) return;
    if (event.target.closest('[data-testid="stExpandSidebarButton"], [data-testid="stToast"]')) return;
    const close = panel.querySelector('[data-testid="stSidebarCollapseButton"] button');
    if (close) close.click();
  },
};
doc.addEventListener("click", doc.pdplHandlers.click);
doc.addEventListener("mousedown", doc.pdplHandlers.down);
</script>"""
MOON = ('<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>')
SUN = ('<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" '
       'stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4'
       'M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>')


def esc(text):
    return html.escape(str(text))


def article_label(record_id):
    """'PDPL-A15-AR' -> 'نظام حماية البيانات الشخصية، المادة 15'"""
    doc, number, _ = record_id.split("-")
    return tr(f"{DOCUMENTS.get(doc, doc)}، المادة {int(number[1:])}", f"{DOCUMENTS.get(doc, doc)}, Article {int(number[1:])}")


def articles_short(articles):
    """'IR-A04(1b); PDPL-A13(3)' -> 'اللائحة م4(1b)، النظام م13(3)'"""
    def one(ref):
        m = re.match(r"(PDPL|IR|TR)-A0*(\d+)(.*)", ref.strip())
        return tr(f"{SHORT_DOCUMENTS[m[1]]} م{m[2]}{m[3]}", f"{SHORT_DOCUMENTS[m[1]]} Art. {m[2]}{m[3]}") if m else ref.strip()
    return tr("، ", ", ").join(one(r) for r in articles.split(";") if r.strip())


def count_articles(n):
    if EN:
        return "1 article" if n == 1 else f"{n} articles"
    return {1: "مادة واحدة", 2: "مادتين"}.get(n, f"{n} مواد" if n <= 10 else f"{n} مادة")


HERO = {"qa": (tr("وش تبي تعرف عن حماية بياناتك؟", "What do you want to know about your data?"),
               tr("إجابات من نصوص النظام واللوائح، مع رقم المادة",
                  "Answers from the text of the law and its regulations, with the article number")),
        "gap": (tr("حلّل سياسة الخصوصية لأي شركة", "Check any company's privacy policy"),
                tr("نقارن السياسة بمتطلبات النظام، ونعرض الدليل من نصها",
                   "We compare the policy with what the law requires, and show the proof from its text"))}


def hero(page):
    """Big green banner: brand on the right, page headline in the middle (nav sits on its left)."""
    title, subtitle = HERO[page]
    st.markdown(
        f'<div class="hero"><div class="brand"><div class="app-logo">{ICONS["shield"]}</div><div>'
        '<div class="app-title">' + tr("مساعد الامتثال لنظام حماية البيانات الشخصية",
                                          "Saudi Personal Data Protection Assistant") + '</div>'
        '<div class="app-subtitle">PDPL Compliance Copilot</div></div></div>'
        f'<div class="hero-center"><h1>{title}</h1><p>{subtitle}</p></div></div>',
        unsafe_allow_html=True)


def show_error(error):
    message, detail = error[:2]
    st.error(message)
    if detail:  # only real technical errors; plain messages are already clear
        with st.expander(tr("التفاصيل التقنية", "Technical details")):
            st.code(detail, language=None)


# ---------- Q&A ----------

@st.cache_resource(show_spinner=False)
def warm_up():
    """Start loading BM25, the embeddings and the reranker in the background when
    the app starts, so the page shows at once and the first question is not slower
    than the others. Returns the thread; ask() waits for it."""
    def load():
        from retrieval.retriever import retrieve
        retrieve("ما هي البيانات الشخصية؟", llm_rerank=False)
    thread = threading.Thread(target=load, daemon=True)
    thread.start()
    return thread


ASK_STEPS = {"retrieve": tr("البحث في 180 مادة نظامية...", "Searching 180 legal articles..."),
             "answer": tr("كتابة الإجابة من نصوص المواد...", "Writing the answer from the articles..."),
             "check": tr("التحقق من كل استشهاد...", "Checking every citation...")}


def ask(question):
    """Shows the current step while answer() runs, then removes it."""
    progress = st.empty()
    with progress.status(ASK_STEPS["retrieve"]) as status:
        warm_up().join()  # the models are still loading only if the app has just started
        from retrieval.answer import answer
        try:
            st.session_state.qa_result = answer(question, on_step=lambda step: status.update(label=ASK_STEPS[step]))
            st.session_state.qa_error = None
            add_history("qa", question.strip(), {"result": st.session_state.qa_result})
        except RuntimeError as error:
            st.session_state.qa_result = None
            st.session_state.qa_error = (tr("تعذر الاتصال بـ OpenAI، ولا توجد إجابة محفوظة لهذا السؤال.",
                                            "Could not reach OpenAI, and there is no saved answer for this question."),
                                         str(error.__cause__ or error))
    progress.empty()


def short_article(record_id):
    """'PDPL-A29-AR' -> 'النظام م29'"""
    doc, number, _ = record_id.split("-")
    return tr(f"{SHORT_DOCUMENTS.get(doc, doc)} م{int(number[1:])}", f"{SHORT_DOCUMENTS.get(doc, doc)} Art. {int(number[1:])}")


def answer_sentences(result):
    """Each sentence with its article tags and, under it, the words quoted from the
    article (shown only when the code found them in the article text)."""
    if not result.get("sentences"):  # answers cached before sentences were saved
        return f'<div class="answer-body" dir="auto">{esc(result["answer"])}</div>'
    parts = []
    for s in result["sentences"]:
        tags = "".join(f'<span class="cite-tag">{esc(short_article(c))}</span>' for c in s["citations"])
        evidence = s.get("evidence")
        quote = (f'<div class="evidence-quote" dir="auto">«{esc(evidence["quote"])}»'
                 f'<div class="evidence-meta">{ICONS["check"]}{tr("نص حرفي من", "Word for word from")} {esc(article_label(evidence["id"]))}</div></div>'
                 if evidence else "")
        parts.append(f'<div class="answer-sentence"><div dir="auto">{esc(s["text"])} {tags}</div>{quote}</div>')
    return '<div class="answer-body">' + "".join(parts) + "</div>"


def page_qa():
    # a form, so pressing Enter sends the question like the button does
    with st.container(key="askbox"), st.form("ask_form", border=False):
        text_col, button_col = st.columns([6, 1], vertical_alignment="center")
        question = text_col.text_input(tr("السؤال", "Question"), key="question", label_visibility="collapsed",
                                       placeholder=tr("اكتب سؤالك بالعربية أو الإنجليزية", "Type your question in Arabic or English"))
        clicked = button_col.form_submit_button(tr("اسأل", "Ask"), type="primary", use_container_width=True)
    if clicked:
        if question.strip():
            ask(question)
        else:
            st.warning(tr("الرجاء كتابة السؤال أولاً.", "Please type a question first."))

    if st.session_state.get("qa_error"):
        show_error(st.session_state.qa_error)
    result = st.session_state.get("qa_result")
    if not result:
        return

    if result["from_cache"]:
        st.markdown(f'<div class="note">{tr("إجابة محفوظة مسبقاً: تعذر الاتصال بـ OpenAI.", "A saved answer: OpenAI could not be reached.")}</div>',
                    unsafe_allow_html=True)
    if not result["covered"]:
        st.markdown(f'<div class="answer-card muted"><div class="answer-head">{tr("الإجابة", "Answer")}</div>'
                    f'<div class="answer-body">{esc(result["answer"])}</div></div>', unsafe_allow_html=True)
        return

    n = len(result["citations"])
    articles = load_articles()
    sources = []
    for cid in result["citations"]:
        doc, number, _ = cid.split("-")
        text = articles[cid]["official_text"]
        sources.append(f'<div class="source"><div class="source-num"><small>{tr("مادة", "Art.")}</small>{int(number[1:])}</div><div>'
                       f'<div class="source-title">{esc(DOCUMENTS.get(doc, doc))}</div>'
                       f'<div class="source-doc" dir="auto">{esc(text[:90])}...</div></div></div>')
    st.markdown('<div class="answer-card split"><div class="answer-main">'
                f'<div class="answer-title">{tr("الإجابة", "Answer")}</div>'
                f'{answer_sentences(result)}'
                f'<span class="verified">{ICONS["check"]}{tr("مستندة إلى", "Based on")} {count_articles(n)}</span></div>'
                f'<div class="answer-side"><div class="side-title">{tr("المصادر", "Sources")}</div>{"".join(sources)}</div></div>',
                unsafe_allow_html=True)
    with st.expander(tr("عرض نص المواد كاملاً", "Show the full text of the articles")):
        for cid in result["citations"]:
            st.markdown(f'<div class="label">{esc(article_label(cid))}</div>'
                        f'<div class="article-text" dir="auto">{esc(articles[cid]["official_text"])}</div>',
                        unsafe_allow_html=True)

    checked = sum(len(row["citations"]) for row in result.get("citation_check", []))
    problems = result.get("problems", [])
    summary = tr(f"تم التحقق من {checked} استشهاد", f"{checked} citations checked")
    if problems:
        summary += tr(f"، واستُبعد {len(problems)} لا تظهر في الإجابة", f"; {len(problems)} removed from the answer")
    st.markdown(f'<div class="check-line">{ICONS["check"]}{summary}</div>', unsafe_allow_html=True)
    if problems:
        with st.expander(tr("تفاصيل الاستشهادات المستبعدة", "Removed citations")):
            for p in problems:
                sentence = result["citation_check"][p["sentence"]]["text"]
                ref = f" ({p['id']})" if p["id"] else ""
                st.markdown(f'<div class="problem"><b>{PROBLEM_LABELS.get(p["status"], p["status"])}{esc(ref)}</b>'
                            f'<div dir="auto">{esc(sentence)}</div></div>', unsafe_allow_html=True)


# ---------- Policy gap check ----------

def list_policies():
    """The saved example policies: {policy_id: row of policies.csv}."""
    policies = {}
    csv_path = POLICIES_DIR / "policies.csv"
    if csv_path.exists():
        for row in pd.read_csv(csv_path).fillna("").itertuples():
            if (POLICIES_DIR / f"{row.policy_id}.txt").exists():
                policies[row.policy_id] = row
    return policies


def list_library():
    """Companies analysed ahead of time: {company_id: saved result}."""
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(LIBRARY_DIR.glob("*.json"))}


def load_checklist_ar():
    checklist = pd.read_csv(CHECKLIST_PATH).fillna("")
    return {r.req_id: r for r in checklist.itertuples()}


GAP_STEPS = {"guess": tr("البحث عن موقع الشركة...", "Finding the company's website..."),
             "find": tr("البحث عن صفحة سياسة الخصوصية في موقع الشركة...", "Finding the privacy page on the company's site..."),
             "archive": tr("البحث عن نسخة السياسة في الأرشيف العام...", "Looking for a copy in the public archives..."),
             "web": tr("البحث عن السياسة في الإنترنت...", "Searching the web for the policy..."),
             "analyse": tr("مقارنة السياسة بـ17 متطلباً من النظام...", "Comparing the policy with the law's 17 requirements..."),
             "check": tr("التحقق من أن كل اقتباس موجود في نص السياسة...", "Checking that every quote is in the policy..."),
             "classify": tr("تصنيف كل دليل بمصنّف OPP-115...", "Labelling each piece of evidence with the OPP-115 classifier...")}

# OPP-115 categories of the 3A classifier, in Arabic
OPP115_AR = {"First Party Collection/Use": "جمع الجهة للبيانات واستخدامها",
             "Third Party Sharing/Collection": "مشاركة البيانات مع طرف ثالث",
             "User Choice/Control": "خيارات المستخدم وتحكمه",
             "User Access, Edit and Deletion": "وصول المستخدم لبياناته وتعديلها وحذفها",
             "Data Retention": "مدة حفظ البيانات",
             "Data Security": "أمن البيانات",
             "Policy Change": "تغيير السياسة",
             "Do Not Track": "عدم التتبع",
             "International and Specific Audiences": "فئات خاصة ودول أخرى",
             "Other": "أخرى"}


def text_key(text):
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()[:16]


class GapError(Exception):
    """(message for the user, technical detail, company the user can open themselves)
    when the analysis cannot run."""


def policy_help(source):
    """Where the user can open the policy in their own browser: the privacy page we
    were told about, or a web search for it."""
    if source.get("hint"):
        return source["name"], source["hint"]
    return source["name"], "https://www.google.com/search?q=" + quote_plus(f"{source['name']} privacy policy")


def paste_help(name, url):
    """When the site cannot be read for us: the user opens the policy in their own
    browser and pastes it right here. Returns a paste source, or None."""
    with st.container(key="pastehelp"):
        steps = [tr("افتح السياسة من الزر تحت", "Open the policy with the button below"),
                 tr("انسخ نصها كله: <kbd>Ctrl+A</kbd> ثم <kbd>Ctrl+C</kbd>", "Copy all its text: <kbd>Ctrl+A</kbd> then <kbd>Ctrl+C</kbd>"),
                 tr("الصقه في الخانة: <kbd>Ctrl+V</kbd>، واضغط حلّل", "Paste it in the box: <kbd>Ctrl+V</kbd>, then press Check")]
        st.markdown('<div class="paste-steps">' + "".join(f'<div><span class="num">{i}</span>{step}</div>'
                                                          for i, step in enumerate(steps, 1)) + "</div>",
                    unsafe_allow_html=True)
        st.link_button(tr(f"افتح سياسة {name}", f"Open {name}'s policy"), url, icon=":material/open_in_new:")
        with st.form("paste_help_form", border=False):
            text = st.text_area(tr("نص السياسة", "Policy text"), height=140, label_visibility="collapsed",
                                placeholder=tr(f"الصق نص سياسة {name} هنا", f"Paste {name}'s policy here"))
            submitted = st.form_submit_button(tr("حلّل", "Check"), type="primary")
    if not submitted:
        return None
    if len(text.strip()) < 300:
        st.warning(tr("النص قصير جداً. الصق نص السياسة كاملاً.", "The text is too short. Paste the whole policy."))
        return None
    return {"kind": "paste", "name": name, "text": text, "date": date.today().isoformat()}


def run_analysis(source):
    """source: {"kind": "site" | "paste" | "example", "name", "text" or "site", "id", "date"}.
    A "site" source with no site (only a company name) first gets its website from guess_website().
    Shows the steps while it works. The same text analysed before is shown from
    results/gap_cache/ at once; an example falls back to results/gap_v0/ when
    OpenAI cannot be reached."""
    st.session_state.gap_report = None
    st.session_state.gap_error = None
    st.session_state.gap_saved_on = None
    progress = st.empty()
    try:
        with progress.status(GAP_STEPS["find" if source["kind"] == "site" else "analyse"]) as status:
            if source["kind"] == "site" and not source.get("site"):
                status.update(label=GAP_STEPS["guess"])
                try:
                    official, source["site"], source["hint"] = guess_website(source["name"])
                except Exception as error:
                    raise GapError(tr("تعذر الاتصال بـ OpenAI للبحث عن موقع الشركة.",
                                      "Could not reach OpenAI to find the company's website."), str(error))
                if not source["site"]:
                    raise GapError(tr(f"ما عرفنا موقع \"{source['name']}\". الصق رابط موقعها (مثل company.com.sa) "
                                      "أو ارفع ملف السياسة.",
                                      f"We do not know the website of \"{source['name']}\". Paste its link "
                                      "(like company.com) or upload the policy file."), "", policy_help(source))
                source["name"] = official or source["name"]
                library = list_library()
                cid = library_by_site(source["site"], library)
                if cid:  # another spelling of a company analysed ahead
                    progress.empty()
                    open_library(cid, library)
                    return
                status.update(label=GAP_STEPS["find"])
            if source["kind"] == "site":
                try:
                    source["url"], source["text"] = find_policy(source["site"], source.get("hint", ""))
                except ValueError as error:
                    # the site cannot be read for us: a public archive's copy instead
                    status.update(label=GAP_STEPS["archive"])
                    try:
                        source["url"], source["text"], copied_on, source["archive"] = from_archive(
                            source["name"], source["site"], source.get("hint", ""))
                        source["date"] = copied_on or source["date"]
                    except ValueError:
                        # last try: OpenAI's web search writes out what the policy says
                        status.update(label=GAP_STEPS["web"])
                        try:
                            source["url"], source["text"], source["date"] = from_web_search(
                                source["name"], source["site"])
                            source["via"] = "web"
                        except ValueError:
                            raise GapError(tr(f"{error} (الموقع: {source['site']}) تقدر تفتحها بنفسك وتلصقها هنا:",
                                              f"{message(str(error))} (site: {source['site']}) You can open it "
                                              "yourself and paste it here:"), "", policy_help(source))
                status.update(label=GAP_STEPS["analyse"])
            saved = GAP_CACHE_DIR / f"{text_key(source['text'])}.json"
            if saved.exists():
                cached = json.loads(saved.read_text(encoding="utf-8"))
                report, st.session_state.gap_saved_on = cached["report"], cached["date"]
                st.session_state.gap_cached = False
            else:
                try:
                    report = analyse(source["text"])
                    status.update(label=GAP_STEPS["check"])
                    st.session_state.gap_cached = False
                    GAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    saved.write_text(json.dumps({"date": date.today().isoformat(), "report": report},
                                                ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as error:
                    fallback = GAP_RESULTS_DIR / f"{source.get('id')}.json"
                    if source["kind"] != "example" or not fallback.exists():
                        raise GapError(tr("تعذر الاتصال بـ OpenAI، ولا يوجد تحليل محفوظ لهذه السياسة.",
                                          "Could not reach OpenAI, and there is no saved analysis of this policy."),
                                       str(error))
                    report = json.loads(fallback.read_text(encoding="utf-8"))
                    st.session_state.gap_cached = True
            if any(r.get("evidence") and "opp115" not in r for r in report):
                status.update(label=GAP_STEPS["classify"])
                try:  # the analysis is still shown if the classifier or OpenAI cannot run
                    label_evidence(report)
                    if saved.exists() and not st.session_state.gap_cached:
                        cached = json.loads(saved.read_text(encoding="utf-8"))
                        saved.write_text(json.dumps({"date": cached["date"], "report": report},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as error:
                    print(f"[app] OPP-115 labels skipped: {error}")
    except GapError as error:  # shown under the search box instead of the steps
        progress.empty()
        st.session_state.gap_error = error.args
        return
    progress.empty()
    st.session_state.gap_report = report
    st.session_state.gap_source = source
    add_history("gap", source["name"], {"source": source, "report": report})


def source_note(source, saved_on=None):
    """Which copy of the policy was analysed, and that this is not a legal ruling."""
    name, day = esc(source["name"]), esc(source["date"])
    if source["kind"] in ("example", "library"):
        copy = tr(f"لنسخة سياسة {name} المحفوظة بتاريخ {day}", f"of the copy of {name}'s policy saved on {day}")
    elif source.get("via") == "web":
        copy = tr(f"لنص سياسة {name} كما جمعه بحث OpenAI من الإنترنت بتاريخ {day}، وقد يختلف قليلاً عن نصها الأصلي",
                  f"of {name}'s policy as gathered by an OpenAI web search on {day}; it may differ a little from the original")
    elif source.get("archive"):
        copy = tr(f"لنسخة السياسة المحفوظة في الأرشيف العام ({esc(source['archive'])}) بتاريخ {day}",
                  f"of the copy kept in a public archive ({esc(source['archive'])}) on {day}")
    elif source["kind"] == "site":
        copy = tr(f"لنص الصفحة كما جُلب بتاريخ {day}", f"of the page as read on {day}")
    elif source["kind"] == "file":
        copy = tr(f"للملف المرفوع ({name}) بتاريخ {day}", f"of the uploaded file ({name}) on {day}")
    else:
        copy = tr(f"للنص الملصوق بتاريخ {day}", f"of the pasted text on {day}")
    when = tr(f" (نتيجة محفوظة من تحليل بتاريخ {esc(saved_on)})", f" (saved result of an analysis on {esc(saved_on)})") if saved_on else ""
    link = (f' <a href="{esc(source["url"])}" target="_blank" dir="ltr">{esc(source["url"])}</a>'
            if source.get("url") else "")
    text = tr(f"تحليل آلي {copy}، وليس حكماً قانونياً{when}.", f"Automatic analysis {copy}. Not a legal ruling{when}.")
    return f'<div class="source-note">{ICONS["info"]}{text}{link}</div>'


def example_source(policy_id, policies):
    row = policies[policy_id]
    return {"kind": "example", "id": policy_id, "name": row.name if EN else NAMES_AR.get(policy_id, row.name), "url": row.url,
            "date": row.fetched_on, "text": (POLICIES_DIR / f"{policy_id}.txt").read_text(encoding="utf-8")}


def squash_name(text):
    """'شي إن' and 'شي ان' -> 'شيان', 'Booking.com' -> 'bookingcom': the same name
    however it is spelled (hamza forms, taa marbuta, spaces, dots)."""
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text.lower())  # harakat and tatweel
    text = re.sub(r"[إأآٱ]", "ا", text).replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"[^\w]|_", "", text)


def match_saved(query, policies, library):
    """Saved policies that match the query, as [(kind, id, name)]: kind "example"
    for the 11 Saudi policies, "library" for the companies analysed ahead.
    A name typed in full wins over names that only contain it ("x" -> X, not Netflix)."""
    q = squash_name(query)
    if not q:
        return []
    entries = [("example", pid, NAMES_AR.get(pid, row.name),
                [pid, row.name, NAMES_AR.get(pid, ""), row.sector, SECTORS_AR.get(row.sector, "")])
               for pid, row in policies.items()]
    entries += [("library", cid, item["name"],
                 [cid, item["name"], item.get("name_ar", ""), *item.get("name_ar", "").split(),
                  *re.split(r"\s*/\s*|\s+\(", item["name"]), item["site"], item["site"].split(".")[0]])
                for cid, item in library.items()]
    exact = [e for e in entries if q in (squash_name(w) for w in e[3] if w)]
    found = exact or [e for e in entries if len(q) >= 3 and any(q in squash_name(w) for w in e[3] if w)]
    return [(kind, key, name) for kind, key, name, _ in found]


def library_by_site(site, library):
    """The company analysed ahead whose website this is, or None ('www.shein.com' -> shein)."""
    bare = re.sub(r"^(https?://)?(www\.)?", "", site.lower()).split("/")[0]
    return next((cid for cid, item in library.items() if item["site"].lower().removeprefix("www.") == bare), None)


def open_library(cid, library):
    """Show a company analysed ahead of time, at once."""
    item = library[cid]
    source = {"kind": "library", "id": cid, "name": item["name"], "url": item["url"], "date": item["fetched_on"]}
    text_path = PROJECT_ROOT / "data" / "library" / "texts" / f"{cid}.txt"  # only on the computer that collected it
    if text_path.exists():
        source["text"] = text_path.read_text(encoding="utf-8")
    st.session_state.update(gap_report=item["report"], gap_source=source, gap_error=None,
                            gap_cached=False, gap_saved_on=item["analysed_on"])
    add_history("gap", item["name"], {"source": source, "report": item["report"]})


def requirement_card(r, checklist, i=0):
    label, tone = STATUS.get(r["status"], (r["status"], "bad"))
    row = checklist.get(r["req_id"])
    title = (row.requirement_en if EN else row.requirement_ar) if row is not None else r["requirement"]
    lines = []
    if r["evidence"]:
        lines.append(f'<div class="evidence" dir="auto">"{esc(r["evidence"])}"</div>')
        if not r["evidence_found"]:
            lines.append(f'<div class="meta bad-text">{tr("تنبيه: النص المقتبس غير موجود في السياسة", "Warning: the quoted text is not in the policy")}</div>')
        if r.get("opp115"):
            category = r["opp115"] if EN else OPP115_AR.get(r["opp115"], r["opp115"])
            match = (tr("يطابق المتطلب", "matches the requirement") if r["opp115_match"]
                     else tr("لا يطابق تصنيفات المتطلب", "not one of the requirement's categories"))
            lines.append(f'<div class="meta">{tr("تصنيف المصنّف (OPP-115)", "Classifier label (OPP-115)")}: '
                         f'{esc(category)} · {match}</div>')
    if r.get("applies") == "conditional" and r["status"] == "missing" and row is not None:
        lines.append(f'<div class="meta">{tr("مطلوب فقط إذا", "Only required if")}: {esc(row.condition if EN else row.condition_ar)}</div>')
    lines.append(f'<div class="meta">{tr("المرجع", "Source")}: {esc(articles_short(r["articles"]))}</div>')
    return (f'<div class="req-card {tone}" style="animation-delay:{i * 50}ms">'
            f'<div class="req-head"><span class="req-id">{esc(r["req_id"])}</span>'
            f'<span class="req-title">{esc(title)}</span>'
            f'<span class="badge {tone}">{ICONS.get(tone, "")}{label}</span></div>'
            f'{"".join(lines)}</div>')


def compliance_gauge(percent):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=percent, number={"suffix": "%", "font": {"size": 44, "color": "#0F5132"}},
        title={"text": "مغطى بالكامل", "font": {"size": 15, "color": "#6B7280"}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 0, "tickfont": {"size": 11}},
               "bar": {"color": "#0F5132", "thickness": 0.75},
               "bgcolor": "#E3EBE6", "borderwidth": 0,
               "steps": [{"range": [0, 50], "color": "#FBEAEA"}, {"range": [50, 80], "color": "#FBF3E2"},
                         {"range": [80, 100], "color": "#E8F1EC"}]}))
    fig.update_layout(height=230, margin={"t": 40, "b": 10, "l": 40, "r": 40},
                      paper_bgcolor="rgba(0,0,0,0)", font={"family": "IBM Plex Sans Arabic"})
    return fig


def page_gap():
    policies = list_policies()
    library = list_library()
    today = date.today().isoformat()
    source = None
    with st.container(key="askbox_gap"):
        mode = st.segmented_control(tr("طريقة الإدخال", "Input"), list(GAP_MODES), format_func=GAP_MODES.get,
                                    default="search", key="gap_mode", label_visibility="collapsed") or "search"
        with st.form("gap_form", border=False):
            if mode == "search":
                text_col, button_col = st.columns([6, 1], vertical_alignment="center")
                query = text_col.text_input(tr("الشركة", "Company"), label_visibility="collapsed",
                                            placeholder=tr("اكتب اسم الشركة أو الصق رابط موقعها، مثل: الراجحي أو stc.com.sa",
                                                           "Type a company name or paste its website, e.g. Al Rajhi or stc.com.sa"))
                submitted = button_col.form_submit_button(tr("حلّل", "Check"), type="primary", use_container_width=True)
            else:
                upload = st.file_uploader(tr("ملف السياسة: PDF أو Word أو TXT", "Policy file: PDF, Word or TXT"), type=FILE_TYPES)
                submitted = st.form_submit_button(tr("حلّل", "Check"), type="primary")
    if submitted and mode == "file":
        if upload is None:
            st.warning(tr("اختر ملف السياسة أولاً.", "Choose the policy file first."))
        else:
            try:
                text = read_policy_file(upload.name, upload.getvalue())
                name = re.sub(r"[_-]+", " ", upload.name.rsplit(".", 1)[0]).strip() or tr("الملف المرفوع", "Uploaded file")
                source = {"kind": "file", "name": name, "text": text, "date": today}
            except ValueError as error:
                st.warning(message(str(error)))
    elif submitted and len(query.strip()) >= MIN_PASTED:  # the policy itself was pasted in the box
        source = {"kind": "paste", "name": tr("السياسة الملصوقة", "Pasted policy"), "text": query, "date": today}
    elif submitted and query.strip():
        matches = match_saved(query, policies, library)
        if len(matches) == 1 and matches[0][0] == "library":
            open_library(matches[0][1], library)
        elif len(matches) == 1:
            source = example_source(matches[0][1], policies)
        elif matches:
            names = tr("، ", ", ").join(name for _, _, name in matches)
            note = tr(f"لقينا أكثر من شركة تطابق بحثك: {esc(names)}. اكتب اسم الشركة كامل.",
                      f"More than one company matches: {esc(names)}. Type the full company name.")
            st.markdown(f'<div class="note">{note}</div>',
                        unsafe_allow_html=True)
        elif looks_like_site(query) and library_by_site(query, library):
            open_library(library_by_site(query, library), library)
        elif looks_like_site(query):
            site = query.strip()
            source = {"kind": "site", "site": site, "date": today,
                      "name": re.sub(r"^(https?://)?(www\.)?", "", site).split("/")[0]}
        else:  # a company that is not an example: find its website from the name
            source = {"kind": "site", "site": "", "date": today, "name": query.strip()}
    elif submitted:
        st.warning(tr("اكتب اسم الشركة أو موقعها أولاً.", "Type a company name or website first."))

    if source:
        run_analysis(source)

    error = st.session_state.get("gap_error")
    if error:
        show_error(error)
        if len(error) > 2:  # the site cannot be read for us: open it and paste
            pasted = paste_help(*error[2])
            if pasted:
                run_analysis(pasted)
                st.rerun()
    report = st.session_state.get("gap_report")
    source = st.session_state.get("gap_source")
    if not report or not source:
        return

    if st.session_state.get("gap_cached"):
        st.markdown(f'<div class="note">{tr("تحليل محفوظ مسبقاً: تعذر الاتصال بـ OpenAI.", "A saved analysis: OpenAI could not be reached.")}</div>',
                    unsafe_allow_html=True)
    counts = pd.Series([r["status"] for r in report]).value_counts()
    total = len(report)
    parts = [("ok", int(counts.get("covered", 0)), STATUS["covered"][0]),
             ("warn", int(counts.get("partial", 0)), STATUS["partial"][0]),
             ("bad", int(counts.get("missing", 0)), STATUS["missing"][0])]
    percent = round(100 * parts[0][1] / total)
    checklist = load_checklist_ar()
    st.markdown(verdict_card(source["name"], percent, parts, total, report, checklist)
                + source_note(source, st.session_state.get("gap_saved_on")), unsafe_allow_html=True)

    with st.expander(tr("عرض كل التفاصيل: المتطلبات الـ17 والاقتباسات من السياسة",
                        "Show all details: the 17 requirements and the quotes from the policy")):
        details(report, checklist, source)


VERDICTS = [(80, "ok", tr("جيدة", "Good"), tr("سياستهم توضح لك أغلب اللي يهمك عن بياناتك",
                                              "Their policy tells you most of what matters about your data")),
            (50, "warn", tr("متوسطة", "Fair"), tr("سياستهم توضح أشياء وتسكت عن أشياء مهمة عن بياناتك",
                                                  "Their policy explains some things and is silent on others that matter")),
            (0, "bad", tr("ضعيفة", "Weak"), tr("سياستهم ما توضح لك أشياء مهمة عن بياناتك",
                                               "Their policy leaves out important things about your data"))]
# Each requirement as the plain question a person would ask about their data.
QUESTION_AR = {"R01": "مين الجهة المسؤولة عن بياناتك وكيف تتواصل معها؟",
               "R02": "مين الشخص المسؤول عن حماية بياناتك عندهم؟",
               "R03": "وش البيانات اللي ياخذونها منك؟",
               "R04": "كيف ياخذون بياناتك؟",
               "R05": "ليش يبون بياناتك؟",
               "R06": "بأي حق يجمعون بياناتك؟",
               "R07": "وش اللي لازم تعطيهم، ووش اللي لك الخيار فيه؟",
               "R08": "وش يصير لو ما عطيتهم بياناتك؟",
               "R09": "مع مين يشاركون بياناتك؟",
               "R10": "هل بياناتك تطلع برا السعودية؟",
               "R11": "كم يحتفظون ببياناتك؟",
               "R12": "كيف يحفظون بياناتك وكيف يحذفونها؟",
               "R13": "وش حقوقك على بياناتك؟",
               "R14": "كيف تطلب حقوقك منهم؟",
               "R15": "كيف تسحب موافقتك؟",
               "R16": "كيف يحمون بياناتك؟",
               "R17": "هل فيه برنامج يقرر عنك بدون إنسان؟"}
QUESTION_EN = {"R01": "Who is responsible for your data, and how do you reach them?",
               "R02": "Who is their data protection officer?",
               "R03": "What data do they take from you?",
               "R04": "How do they collect your data?",
               "R05": "Why do they want your data?",
               "R06": "On what legal basis do they collect it?",
               "R07": "What must you give them, and what is your choice?",
               "R08": "What happens if you do not give them your data?",
               "R09": "Who do they share your data with?",
               "R10": "Does your data leave Saudi Arabia?",
               "R11": "How long do they keep your data?",
               "R12": "How do they store and delete your data?",
               "R13": "What are your rights over your data?",
               "R14": "How do you ask for your rights?",
               "R15": "How do you withdraw your consent?",
               "R16": "How do they protect your data?",
               "R17": "Does a program decide about you without a person?"}
# What can happen to the person when the policy leaves this out, in plain words.
WHY_AR = {"R01": "بدونها ما تعرف مين تكلم لو صار شي ببياناتك",
          "R02": "هو اللي ترجع له لو صار تسريب أو مشكلة ببياناتك",
          "R03": "بدونها ما تعرف وش بالضبط صار عندهم عنك",
          "R04": "ممكن ياخذون بياناتك من جهات ثانية بدون ما تدري",
          "R05": "بدونها ممكن يستخدمون بياناتك لأغراض ما وافقت عليها",
          "R06": "النظام ما يسمح بجمع بياناتك إلا بسبب واضح، مثل موافقتك",
          "R07": "تعرف وش اللي تقدر ترفضه بدون ما تخسر الخدمة",
          "R08": "تعرف وش تخسر لو رفضت، قبل ما تقرر",
          "R09": "ممكن توصل بياناتك لشركات ما تعرفها",
          "R10": "لو تطلع برا المملكة، ممكن ما تكون محمية بنفس الدرجة",
          "R11": "بدونها ممكن تبقى بياناتك عندهم بدون نهاية",
          "R12": "تعرف إن بياناتك تنحذف صح لما ما يحتاجونها",
          "R13": "من حقك تشوف بياناتك وتصححها وتطلب حذفها",
          "R14": "الحق ما ينفعك إذا ما تعرف كيف تطلبه",
          "R15": "من حقك توقفهم عن استخدام بياناتك متى ما بغيت",
          "R16": "تعرف هل بياناتك في أمان من التسريب",
          "R17": "ممكن برنامج يرفض طلبك، مثل طلب تمويل، بدون ما يراجعه إنسان"}
WHY_EN = {"R01": "Without it you do not know who to contact if something happens to your data",
          "R02": "This is who you turn to after a leak or a problem with your data",
          "R03": "Without it you do not know exactly what they hold about you",
          "R04": "They may get your data from others without you knowing",
          "R05": "Without it they may use your data for things you never agreed to",
          "R06": "The law allows collecting your data only for a clear reason, such as your consent",
          "R07": "You know what you can refuse without losing the service",
          "R08": "You know what you lose by refusing, before you decide",
          "R09": "Your data may reach companies you do not know",
          "R10": "Outside the Kingdom your data may not be protected as well",
          "R11": "Without it your data may stay with them forever",
          "R12": "You know your data is really deleted when they no longer need it",
          "R13": "You have the right to see, correct and ask to delete your data",
          "R14": "A right does not help if you do not know how to ask for it",
          "R15": "You can stop them using your data whenever you want",
          "R16": "You know whether your data is safe from leaks",
          "R17": "A program may turn down a request, like a loan, with no person reviewing it"}


def verdict_card(name, percent, parts, total, report, checklist):
    """The whole result in one look: a verdict word with its colour, and the
    three most important gaps in plain words."""
    tone, word, meaning = next((t, w, m) for limit, t, w, m in VERDICTS if percent >= limit)
    order = {("missing", "always"): 0, ("missing", "conditional"): 1, ("partial", "always"): 2,
             ("partial", "conditional"): 3}
    gaps = sorted((r for r in report if r["status"] in ("missing", "partial")),
                  key=lambda r: (order.get((r["status"], r.get("applies", "always")), 4), r["req_id"]))[:3]
    items = "".join(
        f'<li class="{"bad" if r["status"] == "missing" else "warn"}">'
        f'<span class="gap-mark">{"✗" if r["status"] == "missing" else "!"}</span>'
        f'<span>{esc((QUESTION_EN if EN else QUESTION_AR).get(r["req_id"], r["requirement"]))}'
        f'<small>{tr("سياستهم ما تجاوب عليه", "Their policy does not answer this") if r["status"] == "missing" else tr("سياستهم تجاوب عليه بشكل ناقص", "Their policy answers this only in part")}</small>'
        + (f'<em class="why">{esc((WHY_EN if EN else WHY_AR)[r["req_id"]])}</em>' if r["req_id"] in WHY_AR else "")
        + '</span></li>'
        for r in gaps)
    words = {"ok": tr("واضحة", "clear"), "warn": tr("ناقصة", "incomplete"), "bad": tr("ما ذكروها", "not mentioned")}
    segments = "".join(f'<i class="{t}"></i>' * n for t, n, _ in parts)
    legend = tr("، ", ", ").join(f'<b class="{t}">{n} {words[t]}</b>' for t, n, _ in parts if n)
    bar = (f'<div class="req-bar" title="{tr("كل مربع متطلب واحد", "Each square is one requirement")}">{segments}</div>'
           f'<div class="req-legend">{tr(f"من {total} شي يطلبه نظام حماية البيانات الشخصية", f"Of the {total} things the Personal Data Protection Law requires")}: {legend}</div>')
    gaps_html = (f'<div class="gaps-title">{tr("أهم أسئلة عن بياناتك ما تجاوب عليها سياستهم:", "Top questions about your data their policy leaves open:")}</div>'
                 f'<ul class="gaps">{items}</ul>'
                 if gaps else f'<div class="gaps-title">{tr("السياسة تغطي كل المتطلبات", "The policy covers every requirement")} 🎉</div>')
    return (f'<div class="verdict {tone}">'
            f'<div class="verdict-head"><div class="ring" style="--p:{percent}"><span>{percent}%</span></div>'
            f'<div><div class="verdict-name">{tr(f"سياسة {esc(name)}", f"{esc(name)} privacy policy")}</div>'
            f'<div class="verdict-word">{word}</div>'
            f'<div class="verdict-meaning">{tr("يعني", "Meaning")}: {meaning}</div>'
            f'{bar}</div></div>'
            f'{gaps_html}</div>')


def details(report, checklist, source):
    shown = st.radio(tr("عرض", "Show"), list(FILTERS), format_func=FILTERS.get, horizontal=True,
                     label_visibility="collapsed", key="gap_filter")
    order = {"missing": 0, "error": 0, "partial": 1, "covered": 2}
    rows = sorted(report, key=lambda r: (order.get(r["status"], 0), r["req_id"]))
    if shown != "all":
        rows = [r for r in rows if r["status"] == shown]
    st.markdown("".join(requirement_card(r, checklist, i) for i, r in enumerate(rows)) or
                f'<div class="note">{tr("لا توجد متطلبات بهذه الحالة.", "No requirements with this status.")}</div>',
                unsafe_allow_html=True)
    if source.get("text"):  # a library result on another computer has no text, only its link
        st.markdown(f'<div class="gaps-title">{tr("نص السياسة اللي تم تحليله", "The policy text that was checked")}</div>'
                    f'<div class="article-text policy-text" dir="auto">{esc(source["text"][:8000])}</div>',
                    unsafe_allow_html=True)


# ---------- History (side panel) ----------

def load_history():
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def add_history(kind, title, data):
    """Put a question ("qa") or an analysis ("gap") at the top of the history.
    The same question or company asked again replaces its older entry."""
    items = [i for i in load_history() if (i["kind"], i["title"]) != (kind, title)]
    items.insert(0, {"id": datetime.now().strftime("%Y%m%d%H%M%S%f"), "kind": kind, "title": title,
                     "date": datetime.now().strftime("%Y-%m-%d %H:%M"), **data})
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(items[:HISTORY_MAX], ensure_ascii=False, indent=1), encoding="utf-8")


def open_item(item_id):
    """Show a saved question or analysis on its page, without running it again."""
    item = next((i for i in load_history() if i["id"] == item_id), None)
    if item is None:
        return
    if item["kind"] == "qa":
        st.session_state.update(qa_result=item["result"], qa_error=None, question=item["title"])
    else:
        st.session_state.update(gap_report=item["report"], gap_source=item["source"], gap_error=None,
                                gap_cached=False, gap_saved_on=item["date"][:10])
    st.session_state.page_now = item["kind"]


def ask_delete(target):
    """Ask before deleting: target is an item id, or "all" for the whole history."""
    st.session_state.confirm_delete = target


def cancel_delete():
    st.session_state.confirm_delete = None


def delete_item(item_id):
    items = [i for i in load_history() if i["id"] != item_id]
    HISTORY_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    st.session_state.confirm_delete = None
    st.toast(tr("انحذف من السجل", "Removed from history"))


def clear_history():
    HISTORY_PATH.unlink(missing_ok=True)
    st.session_state.confirm_delete = None
    st.toast(tr("تم مسح السجل", "History cleared"))


def day_group(stamp):
    """'اليوم', 'أمس' or 'أقدم' for an item's date 'YYYY-MM-DD HH:MM'."""
    days = (date.today() - date.fromisoformat(stamp[:10])).days
    return tr("اليوم", "Today") if days <= 0 else tr("أمس", "Yesterday") if days == 1 else tr("أقدم", "Older")


def confirm_box(key, question, on_yes, args=()):
    """A small red box under an item: the question, then Delete and Cancel."""
    with st.container(key=key):
        st.markdown(f'<div class="confirm-q">{esc(question)}</div>', unsafe_allow_html=True)
        yes_col, no_col = st.columns(2, gap="small")
        yes_col.button(tr("احذف", "Delete"), key=f"{key}_yes", on_click=on_yes, args=args, type="primary",
                       use_container_width=True)
        no_col.button(tr("إلغاء", "Cancel"), key=f"{key}_no", on_click=cancel_delete, use_container_width=True)


def history_panel():
    with st.sidebar:
        items = load_history()
        count = f'<span class="hist-count">{len(items)}</span>' if items else ""
        st.markdown(f'<div class="hist-head"><div class="hist-title">{tr("سجلك", "Your history")} {count}</div>'
                    f'<div class="hist-hint">{tr("أسئلتك وتحليلاتك السابقة، محفوظة على هذا الجهاز فقط", "Your past questions and checks, kept on this computer only")}</div></div>',
                    unsafe_allow_html=True)
        if not items:
            st.markdown(f'<div class="hist-empty">{tr("ما فيه شي في السجل لسى. اسأل سؤالاً أو حلّل سياسة وبيطلع هنا.", "Nothing here yet. Ask a question or check a policy and it will show here.")}</div>',
                        unsafe_allow_html=True)
            return
        confirm = st.session_state.get("confirm_delete")
        with st.container(key="history"):
            group = None
            for item in items:
                if day_group(item["date"]) != group:
                    group = day_group(item["date"])
                    st.markdown(f'<div class="hist-group">{group}</div>', unsafe_allow_html=True)
                qa = item["kind"] == "qa"
                with st.container(key=f"item_{item['id']}"):
                    open_col, delete_col = st.columns([8, 1], gap="small", vertical_alignment="center")
                    open_col.button(f"{item['title']}  \n{tr('سؤال', 'Question') if qa else tr('تحليل سياسة', 'Policy check')} · {item['date'][11:]}",
                                    key=f"h_{item['id']}", on_click=open_item, args=(item["id"],),
                                    use_container_width=True,
                                    icon=":material/chat_bubble:" if qa else ":material/description:")
                    delete_col.button("", key=f"d_{item['id']}", on_click=ask_delete, args=(item["id"],),
                                      icon=":material/close:", help=tr("احذف من السجل", "Remove from history"))
                    if confirm == item["id"]:
                        confirm_box(f"confirm_{item['id']}", tr("تحذف هذا من السجل؟", "Remove this from history?"),
                                    delete_item, (item["id"],))
        if confirm == "all":
            confirm_box("confirm_all", tr(f"تمسح السجل كله ({len(items)})؟ ما تقدر ترجعه.",
                                         f"Clear all history ({len(items)})? This cannot be undone."), clear_history)
        else:
            st.button(tr("مسح السجل كله", "Clear all history"), key="clear_history", on_click=ask_delete, args=("all",),
                      icon=":material/delete:", use_container_width=True)


# ---------- Layout ----------

warm_up()
if "page_now" not in st.session_state:  # the first visit can open a page from the address (?page=gap)
    st.session_state.page_now = "gap" if st.query_params.get("page") == "gap" else "qa"
page = st.session_state.page_now
with st.container(key="navbar"):  # drawn on top of the hero (see .st-key-navbar in style.css)
    # the page switch is plain HTML: PAGE_SCRIPT shows the other page at once, in the browser,
    # then presses the hidden go_ button below so Python also knows which page is open
    st.markdown('<div class="page-nav" data-p="' + page + '">' + "".join(
        f'<button type="button" data-page="{key}">{NAV_ICONS[key]}<span>{esc(name)}</span></button>'
        for key, name in PAGES.items()) + "</div>", unsafe_allow_html=True)
    for key in PAGES:
        st.button(key, key=f"go_{key}", on_click=go_to, args=(key,))
    # moon / sun beside the page switch; PAGE_SCRIPT switches the look when it is clicked
    st.markdown(f'<button class="theme-btn" type="button" title="{tr("المظهر الليلي أو النهاري", "Dark or light look")}">'
                f'<span class="moon">{MOON}</span><span class="sun">{SUN}</span></button>', unsafe_allow_html=True)
    # English / Arabic: all the texts come from Python, so this one reruns the app
    st.button("ع" if EN else "EN", key="lang_toggle", on_click=switch_language,
              help="العربية" if EN else "English")
# Both pages are built on every run and CSS shows the open one (see .page-nav in style.css),
# so switching pages needs no rerun.
with st.container(key="page_qa"):
    hero("qa")
    page_qa()
with st.container(key="page_gap"):
    hero("gap")
    page_gap()
st.markdown('<div class="app-footer"><div class="footer-notice">'
            + tr('تنبيه: مشروع تعليمي، وليس استشارة قانونية. <span dir="ltr">Educational project, not legal advice.</span>',
                 'Note: educational project, not legal advice. <span dir="rtl">مشروع تعليمي، وليس استشارة قانونية.</span>')
            + '</div><div>' + tr("© 2026 فريق مشروع PDPL Compliance Copilot. جميع الحقوق محفوظة.",
                                 "© 2026 PDPL Compliance Copilot team. All rights reserved.")
            + '</div></div>', unsafe_allow_html=True)
history_panel()  # last, so a question or analysis from this run is already in the list
st.iframe(PAGE_SCRIPT, height=1)

