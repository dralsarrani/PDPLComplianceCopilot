"""PDPL Compliance Copilot: Streamlit app.

Two pages:
    Q&A               question -> answer() (retrieval/answer.py): answer, cited
                      articles with their text, citation-check problems
    Policy gap check  a saved Saudi policy or pasted text -> analyse()
                      (src/gap/analyse_v0.py): one row per PDPL requirement

Fallback when OpenAI cannot be reached (demo day without network):
    Q&A               answer() returns the cached answer for the same question
    Policy gap check  the last saved result in results/gap_v0/<policy>.json

Run from the project root:
    python -m streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.answer import answer  # noqa: E402
from retrieval.citation_check import load_articles  # noqa: E402
from src.gap.analyse_v0 import analyse  # noqa: E402

POLICIES_DIR = PROJECT_ROOT / "data" / "policies"
GAP_RESULTS_DIR = PROJECT_ROOT / "results" / "gap_v0"
PASTE = "الصق نص سياسة"

DOCUMENTS = {"PDPL": "نظام حماية البيانات الشخصية",
             "IR": "اللائحة التنفيذية",
             "TR": "لائحة نقل البيانات خارج المملكة"}
STATUS_LABELS = {"covered": "✅ مغطى", "partial": "🟡 جزئي", "missing": "❌ ناقص", "error": "⚠️ خطأ"}
PROBLEM_LABELS = {"not_in_corpus": "مادة غير موجودة في النظام",
                  "not_retrieved": "مادة لم يسترجعها البحث",
                  "unsupported": "المادة لا تدعم الجملة",
                  "uncited": "جملة بدون استشهاد",
                  "error": "تعذر الفحص"}

st.set_page_config(page_title="PDPL Compliance Copilot", page_icon="⚖️", layout="wide")

DISCLAIMER = (
    "⚠️ هذا النظام مشروع تعليمي وليس استشارة قانونية. "
    "This tool is an educational project and is not legal advice."
)


def rtl(text):
    st.markdown(f'<div dir="auto" style="line-height:1.9">{text}</div>', unsafe_allow_html=True)


def article_label(record_id):
    """'PDPL-A15-AR' -> 'نظام حماية البيانات الشخصية، المادة 15'"""
    doc, number, _ = record_id.split("-")
    return f"{DOCUMENTS.get(doc, doc)}، المادة {int(number[1:])}"


def page_qa():
    st.header("اسأل عن نظام حماية البيانات الشخصية")
    st.caption("Ask the PDPL in Arabic or English. Answers cite the article.")
    question = st.text_area("سؤالك", placeholder="هل يجوز نقل بيانات العملاء خارج المملكة؟")
    if not st.button("اسأل", type="primary"):
        return
    if not question.strip():
        st.warning("اكتب سؤال أول.")
        return

    with st.spinner("أبحث في النظام وأكتب الجواب..."):
        try:
            result = answer(question)
        except RuntimeError:
            st.error("ما قدرت أوصل لـ OpenAI، وما فيه جواب محفوظ لهذا السؤال.")
            return

    if result["from_cache"]:
        st.info("💾 جواب محفوظ: ما قدرت أوصل لـ OpenAI، فهذا آخر جواب لنفس السؤال.")

    if not result["covered"]:
        st.warning(result["answer"])
    else:
        st.subheader("الجواب")
        rtl(result["answer"])
        st.subheader("المواد المستشهد بها")
        articles = load_articles()
        for cid in result["citations"]:
            with st.expander(f"{article_label(cid)}  ({cid})"):
                rtl(articles[cid]["official_text"])

    problems = result.get("problems", [])
    if problems:
        with st.expander(f"⚠️ فحص الاستشهاد: {len(problems)} استشهاد مرفوض (ما يظهر في الجواب)"):
            for p in problems:
                sentence = result["citation_check"][p["sentence"]]["text"]
                rtl(f"<b>{PROBLEM_LABELS.get(p['status'], p['status'])}</b>"
                    f"{' (' + p['id'] + ')' if p['id'] else ''}: {sentence}")


def list_policies():
    names = {}
    csv_path = POLICIES_DIR / "policies.csv"
    if csv_path.exists():
        for row in pd.read_csv(csv_path).itertuples():
            if (POLICIES_DIR / f"{row.policy_id}.txt").exists():
                names[row.policy_id] = row.name
    return names


def show_gap_report(report):
    counts = pd.Series([r["status"] for r in report]).value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ مغطى", int(counts.get("covered", 0)))
    c2.metric("🟡 جزئي", int(counts.get("partial", 0)))
    c3.metric("❌ ناقص", int(counts.get("missing", 0)))

    rows = []
    for r in report:
        note = ""
        if r["evidence"] and not r["evidence_found"]:
            note = "الجملة غير موجودة في السياسة"
        elif r.get("applies") == "conditional" and r["status"] == "missing":
            note = f"مطلوب فقط إذا: {r['condition']}"
        rows.append({"المتطلب": r["req_id"],
                     "الوصف": r["requirement"],
                     "الحكم": STATUS_LABELS.get(r["status"], r["status"]),
                     "الدليل من السياسة": r["evidence"],
                     "المادة": r["articles"],
                     "ملاحظة": note})
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def page_gap():
    st.header("تحليل سياسة الخصوصية")
    st.caption("Check a privacy policy against the PDPL requirements: Covered / Partially covered / Missing.")
    policies = list_policies()
    choice = st.selectbox("السياسة", [PASTE] + list(policies),
                          format_func=lambda p: policies.get(p, p))
    if choice == PASTE:
        policy = st.text_area("الصق سياسة الخصوصية هنا", height=300)
    else:
        policy = (POLICIES_DIR / f"{choice}.txt").read_text(encoding="utf-8")
        with st.expander("نص السياسة"):
            st.text(policy[:5000] + ("\n..." if len(policy) > 5000 else ""))

    if not st.button("حلّل السياسة", type="primary"):
        return
    if not policy.strip():
        st.warning("الصق نص السياسة أول.")
        return

    with st.spinner("أقارن السياسة بمتطلبات النظام..."):
        try:
            report = analyse(policy)
        except Exception:
            saved = GAP_RESULTS_DIR / f"{choice}.json"
            if choice == PASTE or not saved.exists():
                st.error("ما قدرت أوصل لـ OpenAI، وما فيه تحليل محفوظ لهذه السياسة.")
                return
            st.info("💾 تحليل محفوظ: ما قدرت أوصل لـ OpenAI، فهذا آخر تحليل لنفس السياسة.")
            report = json.loads(saved.read_text(encoding="utf-8"))
        else:
            if choice != PASTE:
                GAP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                (GAP_RESULTS_DIR / f"{choice}.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    show_gap_report(report)


st.sidebar.title("PDPL Compliance Copilot")
page = st.sidebar.radio("الصفحة", ["الأسئلة والأجوبة", "تحليل سياسة الخصوصية"])
st.warning(DISCLAIMER)

if page == "الأسئلة والأجوبة":
    page_qa()
else:
    page_gap()
