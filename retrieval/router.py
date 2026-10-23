"""
route(question): which corpus answers the question.

    "pdpl"  the Personal Data Protection Law and its two regulations (personal data,
            consent, rights, transfer outside the Kingdom, breaches of personal data ...)
    "ecc"   the NCA Essential Cybersecurity Controls (passwords, vulnerabilities,
            penetration tests, backups, firewalls, cybersecurity department ...)
    "both"  a question that needs both, e.g. how a company must protect personal data
            from leaks (PDPL security duty + ECC data protection controls)

gpt-4o-mini decides; without OpenAI a keyword rule does. A question about neither
still goes to "pdpl", whose answer then says "not covered".

Returns {"source": "pdpl" | "ecc" | "both", "by": "llm" | "keywords"}

Run from the project root:
    python -m retrieval.router "كم مرة يجب إجراء اختبار الاختراق؟"
"""
import json
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
SOURCES = ("pdpl", "ecc", "both")

ROUTER_PROMPT = """You route questions to the Saudi legal text that answers them.
- "pdpl": the Personal Data Protection Law (PDPL) and its Implementing and Transfer Regulations:
  personal data, consent, data subject rights, privacy notices, disclosure and sharing of personal data,
  transfer of personal data outside the Kingdom, retention and destruction, personal data breach
  notification, data protection officers, penalties.
- "ecc": the National Cybersecurity Authority's Essential Cybersecurity Controls (ECC-2:2024):
  cybersecurity strategy, the cybersecurity department and its roles, cybersecurity risk management,
  asset management, identity and access management, passwords and multi-factor authentication,
  email, network and mobile security, cryptography, backups, vulnerabilities, penetration testing,
  event logs and monitoring, cybersecurity incidents, physical security, web applications,
  business continuity, third-party and cloud security, cybersecurity awareness and training.
- "both": the question clearly needs both, e.g. the technical security measures a company must take
  to protect personal data.
If the question is about neither, answer "pdpl".
Reply in JSON: {"source": "pdpl" | "ecc" | "both"}"""

ECC_WORDS = re.compile(
    r"cyber|password|vulnerab|penetration|firewall|backup|malware|phishing|encrypt|cryptograph|"
    r"incident|multi-factor|mfa|access management|patch|soc\b|logs?\b|nca\b|ecc\b|"
    r"سيبران|كلمة المرور|كلمات المرور|ثغر|اختبار الاختراق|اختراق|جدار الحماية|نسخ احتياطي|النسخ الاحتياطي|"
    r"تشفير|برمجيات ضارة|تصيد|حوادث|التحقق متعدد|الهيئة الوطنية للأمن", re.IGNORECASE)
PDPL_WORDS = re.compile(
    r"personal data|privacy|consent|data subject|controller|transfer outside|pdpl|"
    r"بيانات شخصية|البيانات الشخصية|بياناتي|الخصوصية|موافق|صاحب البيانات|جهة التحكم|خارج المملكة", re.IGNORECASE)


def keyword_route(question):
    ecc, pdpl = bool(ECC_WORDS.search(question)), bool(PDPL_WORDS.search(question))
    return "both" if ecc and pdpl else "ecc" if ecc else "pdpl"


def route(question, model=None):
    try:
        load_dotenv()
        response = OpenAI().chat.completions.create(
            model=model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL), temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": ROUTER_PROMPT},
                      {"role": "user", "content": question}])
        reply = json.loads(response.choices[0].message.content)
        source = reply.get("source") if reply.get("source") in SOURCES else "pdpl"
        return {"source": source, "by": "llm"}
    except Exception as error:
        print(f"Router: OpenAI not reachable ({type(error).__name__}); using keywords.")
        return {"source": keyword_route(question), "by": "keywords"}


if __name__ == "__main__":
    print(route(" ".join(sys.argv[1:]) or input("Enter your question: ")))
