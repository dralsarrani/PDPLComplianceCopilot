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
from streamlit_option_menu import option_menu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.citation_check import load_articles  # noqa: E402
from src.gap.analyse_v0 import analyse  # noqa: E402
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
GAP_MODES = {"search": "ابحث عن شركة", "file": "ارفع ملف"}
MIN_PASTED = 300  # a search longer than this is a pasted policy, not a name
NAMES_AR = {"hungerstation": "هنقرستيشن", "careem": "كريم", "alrajhi": "مصرف الراجحي", "alinma": "مصرف الإنماء",
            "tamara": "تمارا", "jarir": "مكتبة جرير", "stc": "إس تي سي", "saudia": "الخطوط السعودية",
            "nahdi": "صيدليات النهدي", "ksu": "جامعة الملك سعود", "kfupm": "جامعة البترول والمعادن"}
SECTORS_AR = {"Banking": "بنوك", "Fintech (BNPL)": "تقنية مالية", "Telecom": "اتصالات",
              "Food delivery": "توصيل طعام", "Ride-hailing": "نقل ركاب", "Retail": "تجزئة",
              "Airline": "طيران", "Health / pharmacy": "صحة وصيدليات", "Education": "جامعات"}

PAGES = {"qa": "اسأل عن النظام", "gap": "تحليل سياسة الخصوصية"}
DOCUMENTS = {"PDPL": "نظام حماية البيانات الشخصية",
             "IR": "اللائحة التنفيذية",
             "TR": "لائحة نقل البيانات خارج المملكة"}
SHORT_DOCUMENTS = {"PDPL": "النظام", "IR": "اللائحة", "TR": "لائحة النقل"}
STATUS = {"covered": ("مغطى", "ok"),
          "partial": ("مغطى جزئياً", "warn"),
          "missing": ("غير مغطى", "bad"),
          "error": ("تعذر التحليل", "bad")}
FILTERS = {"all": "الكل", "missing": "غير مغطى", "partial": "مغطى جزئياً", "covered": "مغطى"}
PROBLEM_LABELS = {"not_in_corpus": "مادة غير موجودة في النظام",
                  "not_retrieved": "مادة لم يسترجعها البحث",
                  "unsupported": "المادة لا تدعم الجملة",
                  "uncited": "جملة بدون استشهاد",
                  "error": "تعذر الفحص"}
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
st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def esc(text):
    return html.escape(str(text))


def article_label(record_id):
    """'PDPL-A15-AR' -> 'نظام حماية البيانات الشخصية، المادة 15'"""
    doc, number, _ = record_id.split("-")
    return f"{DOCUMENTS.get(doc, doc)}، المادة {int(number[1:])}"


def articles_short(articles):
    """'IR-A04(1b); PDPL-A13(3)' -> 'اللائحة م4(1b)، النظام م13(3)'"""
    def one(ref):
        m = re.match(r"(PDPL|IR|TR)-A0*(\d+)(.*)", ref.strip())
        return f"{SHORT_DOCUMENTS[m[1]]} م{m[2]}{m[3]}" if m else ref.strip()
    return "، ".join(one(r) for r in articles.split(";") if r.strip())


def count_articles(n):
    return {1: "مادة واحدة", 2: "مادتين"}.get(n, f"{n} مواد" if n <= 10 else f"{n} مادة")


HERO = {"qa": ("وش تبي تعرف عن حماية بياناتك؟", "إجابات من نصوص النظام واللوائح، مع رقم المادة"),
        "gap": ("حلّل سياسة الخصوصية لأي شركة", "نقارن السياسة بمتطلبات النظام، ونعرض الدليل من نصها")}


def hero(page):
    """Big green banner: brand on the right, page headline in the middle (nav sits on its left)."""
    title, subtitle = HERO[page]
    st.markdown(
        f'<div class="hero"><div class="brand"><div class="app-logo">{ICONS["shield"]}</div><div>'
        '<div class="app-title">مساعد الامتثال لنظام حماية البيانات الشخصية</div>'
        '<div class="app-subtitle">PDPL Compliance Copilot</div></div></div>'
        f'<div class="hero-center"><h1>{title}</h1><p>{subtitle}</p></div></div>',
        unsafe_allow_html=True)


def show_error(error):
    message, detail = error[:2]
    st.error(message)
    if detail:  # only real technical errors; plain messages are already clear
        with st.expander("التفاصيل التقنية"):
            st.code(detail, language=None)


# ---------- Q&A ----------

@st.cache_resource(show_spinner=False)
def warm_up():
    """Start loading BM25, the embeddings and the reranker in the background when
    the app starts, so the page shows at once and the first question is not slower
    than the others. Returns the thread; ask() waits for it."""
    def load():
        from retrieval.retriever import retrieve
        retrieve("ما هي البيانات الشخصية؟")
    thread = threading.Thread(target=load, daemon=True)
    thread.start()
    return thread


ASK_STEPS = {"retrieve": "البحث في 180 مادة نظامية...",
             "answer": "كتابة الإجابة من نصوص المواد...",
             "check": "التحقق من كل استشهاد..."}


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
            st.session_state.qa_error = ("تعذر الاتصال بـ OpenAI، ولا توجد إجابة محفوظة لهذا السؤال.", str(error.__cause__ or error))
    progress.empty()


def short_article(record_id):
    """'PDPL-A29-AR' -> 'النظام م29'"""
    doc, number, _ = record_id.split("-")
    return f"{SHORT_DOCUMENTS.get(doc, doc)} م{int(number[1:])}"


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
                 f'<div class="evidence-meta">{ICONS["check"]}نص حرفي من {esc(article_label(evidence["id"]))}</div></div>'
                 if evidence else "")
        parts.append(f'<div class="answer-sentence"><div dir="auto">{esc(s["text"])} {tags}</div>{quote}</div>')
    return '<div class="answer-body">' + "".join(parts) + "</div>"


def page_qa():
    # a form, so pressing Enter sends the question like the button does
    with st.container(key="askbox"), st.form("ask_form", border=False):
        text_col, button_col = st.columns([6, 1], vertical_alignment="center")
        question = text_col.text_input("السؤال", key="question", label_visibility="collapsed",
                                       placeholder="اكتب سؤالك بالعربية أو الإنجليزية")
        clicked = button_col.form_submit_button("اسأل", type="primary", use_container_width=True)
    if clicked:
        if question.strip():
            ask(question)
        else:
            st.warning("الرجاء كتابة السؤال أولاً.")

    if st.session_state.get("qa_error"):
        show_error(st.session_state.qa_error)
    result = st.session_state.get("qa_result")
    if not result:
        return

    if result["from_cache"]:
        st.markdown('<div class="note">إجابة محفوظة مسبقاً: تعذر الاتصال بـ OpenAI.</div>',
                    unsafe_allow_html=True)
    if not result["covered"]:
        st.markdown('<div class="answer-card muted"><div class="answer-head">الإجابة</div>'
                    f'<div class="answer-body">{esc(result["answer"])}</div></div>', unsafe_allow_html=True)
        return

    n = len(result["citations"])
    articles = load_articles()
    sources = []
    for cid in result["citations"]:
        doc, number, _ = cid.split("-")
        text = articles[cid]["official_text"]
        sources.append(f'<div class="source"><div class="source-num"><small>مادة</small>{int(number[1:])}</div><div>'
                       f'<div class="source-title">{esc(DOCUMENTS.get(doc, doc))}</div>'
                       f'<div class="source-doc" dir="auto">{esc(text[:90])}...</div></div></div>')
    st.markdown('<div class="answer-card split"><div class="answer-main">'
                '<div class="answer-title">الإجابة</div>'
                f'{answer_sentences(result)}'
                f'<span class="verified">{ICONS["check"]}مستندة إلى {count_articles(n)}</span></div>'
                f'<div class="answer-side"><div class="side-title">المصادر</div>{"".join(sources)}</div></div>',
                unsafe_allow_html=True)
    with st.expander("عرض نص المواد كاملاً"):
        for cid in result["citations"]:
            st.markdown(f'<div class="label">{esc(article_label(cid))}</div>'
                        f'<div class="article-text" dir="auto">{esc(articles[cid]["official_text"])}</div>',
                        unsafe_allow_html=True)

    checked = sum(len(row["citations"]) for row in result.get("citation_check", []))
    problems = result.get("problems", [])
    summary = f"تم التحقق من {checked} استشهاد"
    if problems:
        summary += f"، واستُبعد {len(problems)} لا تظهر في الإجابة"
    st.markdown(f'<div class="check-line">{ICONS["check"]}{summary}</div>', unsafe_allow_html=True)
    if problems:
        with st.expander("تفاصيل الاستشهادات المستبعدة"):
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


GAP_STEPS = {"guess": "البحث عن موقع الشركة...",
             "find": "البحث عن صفحة سياسة الخصوصية في موقع الشركة...",
             "archive": "البحث عن نسخة السياسة في الأرشيف العام...",
             "web": "البحث عن السياسة في الإنترنت...",
             "analyse": "مقارنة السياسة بـ17 متطلباً من النظام...",
             "check": "التحقق من أن كل اقتباس موجود في نص السياسة..."}


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
        steps = ["افتح السياسة من الزر تحت", "انسخ نصها كله: <kbd>Ctrl+A</kbd> ثم <kbd>Ctrl+C</kbd>",
                 "الصقه في الخانة: <kbd>Ctrl+V</kbd>، واضغط حلّل"]
        st.markdown('<div class="paste-steps">' + "".join(f'<div><span class="num">{i}</span>{step}</div>'
                                                          for i, step in enumerate(steps, 1)) + "</div>",
                    unsafe_allow_html=True)
        st.link_button(f"افتح سياسة {name}", url, icon=":material/open_in_new:")
        with st.form("paste_help_form", border=False):
            text = st.text_area("نص السياسة", height=140, label_visibility="collapsed",
                                placeholder=f"الصق نص سياسة {name} هنا")
            submitted = st.form_submit_button("حلّل", type="primary")
    if not submitted:
        return None
    if len(text.strip()) < 300:
        st.warning("النص قصير جداً. الصق نص السياسة كاملاً.")
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
                    raise GapError("تعذر الاتصال بـ OpenAI للبحث عن موقع الشركة.", str(error))
                if not source["site"]:
                    raise GapError(f"ما عرفنا موقع \"{source['name']}\". الصق رابط موقعها (مثل company.com.sa) "
                                   "أو ارفع ملف السياسة.", "", policy_help(source))
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
                            raise GapError(f"{error} (الموقع: {source['site']}) تقدر تفتحها بنفسك وتلصقها هنا:",
                                           "", policy_help(source))
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
                        raise GapError("تعذر الاتصال بـ OpenAI، ولا يوجد تحليل محفوظ لهذه السياسة.", str(error))
                    report = json.loads(fallback.read_text(encoding="utf-8"))
                    st.session_state.gap_cached = True
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
    if source["kind"] in ("example", "library"):
        copy = f"لنسخة سياسة {esc(source['name'])} المحفوظة بتاريخ {esc(source['date'])}"
    elif source.get("via") == "web":
        copy = (f"لنص سياسة {esc(source['name'])} كما جمعه بحث OpenAI من الإنترنت بتاريخ {esc(source['date'])}"
                "، وقد يختلف قليلاً عن نصها الأصلي")
    elif source.get("archive"):
        copy = f"لنسخة السياسة المحفوظة في الأرشيف العام ({esc(source['archive'])}) بتاريخ {esc(source['date'])}"
    elif source["kind"] == "site":
        copy = f"لنص الصفحة كما جُلب بتاريخ {esc(source['date'])}"
    elif source["kind"] == "file":
        copy = f"للملف المرفوع ({esc(source['name'])}) بتاريخ {esc(source['date'])}"
    else:
        copy = f"للنص الملصوق بتاريخ {esc(source['date'])}"
    when = f" (نتيجة محفوظة من تحليل بتاريخ {esc(saved_on)})" if saved_on else ""
    link = (f' <a href="{esc(source["url"])}" target="_blank" dir="ltr">{esc(source["url"])}</a>'
            if source.get("url") else "")
    return f'<div class="source-note">{ICONS["info"]}تحليل آلي {copy}، وليس حكماً قانونياً{when}.{link}</div>'


def example_source(policy_id, policies):
    row = policies[policy_id]
    return {"kind": "example", "id": policy_id, "name": NAMES_AR.get(policy_id, row.name), "url": row.url,
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
    title = row.requirement_ar if row is not None else r["requirement"]
    lines = []
    if r["evidence"]:
        lines.append(f'<div class="evidence" dir="auto">"{esc(r["evidence"])}"</div>')
        if not r["evidence_found"]:
            lines.append('<div class="meta bad-text">تنبيه: النص المقتبس غير موجود في السياسة</div>')
    if r.get("applies") == "conditional" and r["status"] == "missing" and row is not None:
        lines.append(f'<div class="meta">مطلوب فقط إذا: {esc(row.condition_ar)}</div>')
    lines.append(f'<div class="meta">المرجع: {esc(articles_short(r["articles"]))}</div>')
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
    with st.container(key="askbox"):
        mode = st.segmented_control("طريقة الإدخال", list(GAP_MODES), format_func=GAP_MODES.get,
                                    default="search", key="gap_mode", label_visibility="collapsed") or "search"
        with st.form("gap_form", border=False):
            if mode == "search":
                text_col, button_col = st.columns([6, 1], vertical_alignment="center")
                query = text_col.text_input("الشركة", label_visibility="collapsed",
                                            placeholder="اكتب اسم الشركة أو الصق رابط موقعها، مثل: الراجحي أو stc.com.sa")
                submitted = button_col.form_submit_button("حلّل", type="primary", use_container_width=True)
            else:
                upload = st.file_uploader("ملف السياسة: PDF أو Word أو TXT", type=FILE_TYPES)
                submitted = st.form_submit_button("حلّل", type="primary")
    if submitted and mode == "file":
        if upload is None:
            st.warning("اختر ملف السياسة أولاً.")
        else:
            try:
                text = read_policy_file(upload.name, upload.getvalue())
                name = re.sub(r"[_-]+", " ", upload.name.rsplit(".", 1)[0]).strip() or "الملف المرفوع"
                source = {"kind": "file", "name": name, "text": text, "date": today}
            except ValueError as error:
                st.warning(str(error))
    elif submitted and len(query.strip()) >= MIN_PASTED:  # the policy itself was pasted in the box
        source = {"kind": "paste", "name": "السياسة الملصوقة", "text": query, "date": today}
    elif submitted and query.strip():
        matches = match_saved(query, policies, library)
        if len(matches) == 1 and matches[0][0] == "library":
            open_library(matches[0][1], library)
        elif len(matches) == 1:
            source = example_source(matches[0][1], policies)
        elif matches:
            names = "، ".join(name for _, _, name in matches)
            st.markdown(f'<div class="note">لقينا أكثر من شركة تطابق بحثك: {esc(names)}. اكتب اسم الشركة كامل.</div>',
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
        st.warning("اكتب اسم الشركة أو موقعها أولاً.")

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
        st.markdown('<div class="note">تحليل محفوظ مسبقاً: تعذر الاتصال بـ OpenAI.</div>', unsafe_allow_html=True)
    counts = pd.Series([r["status"] for r in report]).value_counts()
    total = len(report)
    parts = [("ok", int(counts.get("covered", 0)), "مغطى"),
             ("warn", int(counts.get("partial", 0)), "مغطى جزئياً"),
             ("bad", int(counts.get("missing", 0)), "غير مغطى")]
    percent = round(100 * parts[0][1] / total)
    checklist = load_checklist_ar()
    st.markdown(verdict_card(source["name"], percent, parts, total, report, checklist)
                + source_note(source, st.session_state.get("gap_saved_on")), unsafe_allow_html=True)

    with st.expander("عرض كل التفاصيل: المتطلبات الـ17 والاقتباسات من السياسة"):
        details(report, checklist, source)


VERDICTS = [(80, "ok", "جيدة", "سياستهم توضح لك أغلب اللي يهمك عن بياناتك"),
            (50, "warn", "متوسطة", "سياستهم توضح أشياء وتسكت عن أشياء مهمة عن بياناتك"),
            (0, "bad", "ضعيفة", "سياستهم ما توضح لك أشياء مهمة عن بياناتك")]
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
        f'<span>{esc(QUESTION_AR.get(r["req_id"], r["requirement"]))}'
        f'<small>{"سياستهم ما تجاوب عليه" if r["status"] == "missing" else "سياستهم تجاوب عليه بشكل ناقص"}</small>'
        + (f'<em class="why">{esc(WHY_AR[r["req_id"]])}</em>' if r["req_id"] in WHY_AR else "")
        + '</span></li>'
        for r in gaps)
    words = {"ok": "واضحة", "warn": "ناقصة", "bad": "ما ذكروها"}
    segments = "".join(f'<i class="{t}"></i>' * n for t, n, _ in parts)
    legend = "، ".join(f'<b class="{t}">{n} {words[t]}</b>' for t, n, _ in parts if n)
    bar = (f'<div class="req-bar" title="كل مربع متطلب واحد">{segments}</div>'
           f'<div class="req-legend">من {total} شي يطلبه نظام حماية البيانات الشخصية: {legend}</div>')
    gaps_html = (f'<div class="gaps-title">أهم أسئلة عن بياناتك ما تجاوب عليها سياستهم:</div><ul class="gaps">{items}</ul>'
                 if gaps else '<div class="gaps-title">السياسة تغطي كل المتطلبات 🎉</div>')
    return (f'<div class="verdict {tone}">'
            f'<div class="verdict-head"><div class="ring" style="--p:{percent}"><span>{percent}%</span></div>'
            f'<div><div class="verdict-name">سياسة {esc(name)}</div>'
            f'<div class="verdict-word">{word}</div>'
            f'<div class="verdict-meaning">يعني: {meaning}</div>'
            f'{bar}</div></div>'
            f'{gaps_html}</div>')


def details(report, checklist, source):
    shown = st.radio("عرض", list(FILTERS), format_func=FILTERS.get, horizontal=True,
                     label_visibility="collapsed", key="gap_filter")
    order = {"missing": 0, "error": 0, "partial": 1, "covered": 2}
    rows = sorted(report, key=lambda r: (order.get(r["status"], 0), r["req_id"]))
    if shown != "all":
        rows = [r for r in rows if r["status"] == shown]
    st.markdown("".join(requirement_card(r, checklist, i) for i, r in enumerate(rows)) or
                '<div class="note">لا توجد متطلبات بهذه الحالة.</div>', unsafe_allow_html=True)
    if source.get("text"):  # a library result on another computer has no text, only its link
        st.markdown('<div class="gaps-title">نص السياسة اللي تم تحليله</div>'
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
    st.session_state.nav_to = item["kind"]


def delete_item(item_id):
    items = [i for i in load_history() if i["id"] != item_id]
    HISTORY_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")


def clear_history():
    HISTORY_PATH.unlink(missing_ok=True)
    st.toast("تم مسح السجل")


def history_panel():
    with st.sidebar:
        st.markdown('<div class="side-title">سجلك</div>'
                    '<div class="side-hint">أسئلتك وتحليلاتك السابقة، محفوظة على هذا الجهاز فقط</div>',
                    unsafe_allow_html=True)
        items = load_history()
        if not items:
            st.markdown('<div class="side-empty">ما فيه شي في السجل لسى. اسأل سؤالاً أو حلّل سياسة وبيطلع هنا.</div>',
                        unsafe_allow_html=True)
            return
        with st.container(key="history"):
            for item in items:
                qa = item["kind"] == "qa"
                open_col, delete_col = st.columns([8, 1], gap="small", vertical_alignment="center")
                open_col.button(f"{item['title']}  \n{'سؤال' if qa else 'تحليل'} · {item['date'][5:]}",
                                key=f"h_{item['id']}", on_click=open_item, args=(item["id"],),
                                use_container_width=True,
                                icon=":material/chat_bubble:" if qa else ":material/description:")
                delete_col.button("", key=f"d_{item['id']}", on_click=delete_item, args=(item["id"],),
                                  icon=":material/close:", help="احذف من السجل")
        st.button("مسح السجل", key="clear_history", on_click=clear_history, icon=":material/delete:",
                  use_container_width=True)


# ---------- Layout ----------

warm_up()
default = st.query_params.get("page", "qa")
nav_to = st.session_state.pop("nav_to", None)  # set when an item in the history is opened
with st.container(key="navbar"):  # drawn on top of the hero (see .st-key-navbar in style.css)
    label = option_menu(None, list(PAGES.values()), icons=["chat-square-text", "file-earmark-check"],
                        orientation="horizontal", key="nav",
                        manual_select=list(PAGES).index(nav_to) if nav_to else None,
                        default_index=list(PAGES).index(default) if default in PAGES else 0,
                        styles={"container": {"padding": "4px", "background-color": "#fff", "border-radius": "999px",
                                              "direction": "rtl"},
                                "icon": {"font-size": "16px"},
                                "nav-link": {"font-size": "15px", "margin": "0 2px", "border-radius": "999px",
                                             "color": "#1F2937", "--hover-color": "#E8F1EC", "padding": "6px 14px",
                                             "white-space": "nowrap"},
                                "nav-link-selected": {"background-color": "#14573A", "color": "#fff",
                                                      "font-weight": "600"}})
page = nav_to or next(k for k, v in PAGES.items() if v == label)
hero(page)
if page == "qa":
    page_qa()
else:
    page_gap()
st.markdown('<div class="app-footer"><div class="footer-notice">تنبيه: مشروع تعليمي، وليس استشارة قانونية. '
            '<span dir="ltr">Educational project, not legal advice.</span></div>'
            '<div>© 2026 فريق مشروع PDPL Compliance Copilot. جميع الحقوق محفوظة.</div></div>', unsafe_allow_html=True)
history_panel()  # last, so a question or analysis from this run is already in the list

