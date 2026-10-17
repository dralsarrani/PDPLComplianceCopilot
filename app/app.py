"""PDPL Compliance Copilot: Streamlit app (skeleton).

Run from the project root:
    streamlit run app/app.py
"""
import streamlit as st

st.set_page_config(page_title="PDPL Compliance Copilot", page_icon="⚖️", layout="wide")

DISCLAIMER = (
    "⚠️ هذا النظام مشروع تعليمي وليس استشارة قانونية. "
    "This tool is an educational project and is not legal advice."
)


def page_qa():
    st.header("اسأل عن نظام حماية البيانات الشخصية")
    st.caption("Ask the PDPL in Arabic or English. Answers cite the article.")
    question = st.text_area("سؤالك", placeholder="هل يجوز نقل بيانات العملاء خارج المملكة؟")
    if st.button("اسأل", type="primary"):
        if not question.strip():
            st.warning("اكتب سؤال أول.")
        else:
            # Next: connect answer(question) from person 2
            st.info("الإجابة بتظهر هنا لما نربط دالة answer() من الشخص 2.")


def page_gap():
    st.header("تحليل سياسة الخصوصية")
    st.caption("Paste a privacy policy to get a Covered / Partially covered / Missing report.")
    policy = st.text_area("الصق سياسة الخصوصية هنا", height=300)
    if st.button("حلّل السياسة", type="primary"):
        if not policy.strip():
            st.warning("الصق نص السياسة أول.")
        else:
            # Next: connect the gap analyser (3B)
            st.info("تقرير الفجوات بيظهر هنا لما نبني محلل الفجوات.")


st.sidebar.title("PDPL Compliance Copilot")
page = st.sidebar.radio("الصفحة", ["الأسئلة والأجوبة", "تحليل سياسة الخصوصية"])
st.warning(DISCLAIMER)

if page == "الأسئلة والأجوبة":
    page_qa()
else:
    page_gap()
