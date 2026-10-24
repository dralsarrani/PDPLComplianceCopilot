"""
The company a privacy policy belongs to, read from the policy text itself.

Used when the policy was pasted or uploaded, so the report can say "Tamara's policy"
instead of "the pasted policy". No OpenAI call: the text is searched in this order and
the first hit wins:
    1. the title                    -> "Privacy Notice | Careem", "سياسة الخصوصية من Apple"
    2. the sentence naming it       -> Tamara Financing Company ("Tamara", "we") -> Tamara
       or, in an Arabic policy, the Latin brand it repeats -> stc
    3. a link to its own site       -> www.jarir.com -> Jarir
    4. the copyright line           -> © 2025 Tamara. All rights reserved -> Tamara
    5. a privacy or company email   -> data.protection@hungerstation.com -> Hungerstation
    6. the uploaded file's name     -> careem.pdf -> Careem
Links and emails of authorities, browsers and platforms (sdaia.gov.sa, google.com ...)
are skipped. Returns (name, site): site is the domain found in steps 3 or 5, else "".
("", "") when nothing is found.

    python -m src.policies.company_name data/policies/*.txt
"""
import re
import sys
from collections import Counter
from pathlib import Path

OTHER_SITES = {"sdaia", "ncsc", "nca", "google", "googleapis", "gstatic", "apple", "microsoft", "mozilla",
               "opera", "facebook", "fb", "instagram", "twitter", "x", "linkedin", "youtube", "tiktok",
               "snapchat", "whatsapp", "w3", "wikipedia", "aboutcookies", "allaboutcookies",
               "youronlinechoices", "networkadvertising", "aboutads", "optout", "gmail", "hotmail",
               "outlook", "yahoo", "icloud", "example", "adobe", "amazonaws", "cloudflare", "hubspot",
               "onetrust", "cookiebot", "cookielaw", "europa", "ico"}
SECOND_LEVEL = {"com", "net", "org", "co", "gov", "edu", "ac", "sch", "med"}
GENERIC = re.compile(r"^(we|us|our|the\s+)?(company|bank|university|group|platform|app|website|services?|"
                     r"controller|data\s+controller|affiliates?)$|^(الشركة|المصرف|البنك|الجامعة|المنصة|"
                     r"التطبيق|نحن|المجموعة|الموقع|الجهة)$", re.I)

LINK = re.compile(r"(?:https?://|www\.)([a-z0-9][a-z0-9.-]*\.[a-z]{2,})", re.I)
EMAIL = re.compile(r"\b([a-z0-9._%+-]+)@([a-z0-9-]+(?:\.[a-z0-9-]+)+)", re.I)
QUOTED = r"[\"“”«»']\s*([^\"“”«»'\n]{2,40}?)\s*[\"“”«»']"
TITLE_EN = re.compile(r"privacy\s+(?:policy|notice|statement)(?:\s+\w+){0,2}?\s*(?:of|for|[-–|:])\s*(?:the\s+)?(.{2,80})$"
                      r"|^(.{2,60}?)\s*(?:[-–|:]\s*)?(?:'s\s+)?privacy\s+(?:policy|notice|statement)$", re.I)
TITLE_AR = re.compile(r"سياسة\s+(?:ال)?خصوصية\s*(?:الخاصة\s+ب|من\s+|لدى\s+|في\s+|ل|[-–|:])\s*(.{2,60})$"
                      r"|^(.{2,60}?)\s*[-–|:]\s*سياسة\s+(?:ال)?خصوصية$")
# a pasted policy can arrive as one long line: the Latin name right after the title
TITLE_LATIN = re.compile(r"^\s*(?:سياسة\s+(?:ال)?خصوصية|privacy\s+(?:policy|notice|statement))\s*"
                         r"(?:\s+\w+){0,2}?\s*(?:من|لدى|الخاصة\s+ب|ل|of|for|from|[-–|:])\s*(?:the\s+)?"
                         r"([A-Za-z][A-Za-z0-9&.\-]*(?:\s+(?:of|and|for|the|&)?\s*[A-Z][A-Za-z0-9&.\-]*){0,7}"
                         r"(?:\s*\([A-Z][A-Za-z0-9&]{1,7}\))?)",
                         re.I)
TITLE_STOP = re.compile(r"\s*\b(?:Last|Updated|Effective|Date|Version|Introduction|Who|We|This|Our|Welcome|"
                        r"January|February|March|April|May|June|July|August|September|October|November|"
                        r"December)\b.*$")
LATIN_WORD = re.compile(r"\b[A-Z][A-Za-z0-9&]{1,20}\b|\b[a-z]{2,5}\b(?=\s*[\u0600-\u06FF])")
NOT_NAMES = {"PDPL", "SDAIA", "Saudi", "Arabia", "Kingdom", "Google", "Apple ID", "App", "Store", "iOS", "Android",
             "Personal", "Data", "Privacy", "Policy", "IP", "SMS", "GPS", "ID", "URL", "Cookies", "Cookie", "Wi",
             "Fi", "EU", "GDPR", "USA", "UAE", "OTP", "PDF", "Inc", "Ltd", "LLC", "Co", "www", "http", "https",
             "com", "sa", "net", "org", "html", "Facebook", "Instagram", "Twitter", "WhatsApp", "Microsoft"}
NAMED_EN = re.compile(r"([A-Z][A-Za-z0-9&'.,\- ]{1,70}?)\s*\(\s*(?:hereinafter\s+(?:referred\s+to\s+as\s+)?|"
                      r"together\s+|collectively\s+)?(?:the\s+)?" + QUOTED)
NAMED_AR = re.compile(r"((?:شركة|مؤسسة|مصرف|بنك|جامعة|مجموعة|منصة)\s+[^\n(،.:]{2,60}?)\s*\(\s*"
                      r"(?:و?يشار\s+إلي(?:ها|ه)\s*(?:فيما\s+بعد\s*)?(?:بـ\s*|ب\s*)?)?" + QUOTED)
COPYRIGHT = re.compile(r"(?:©|copyright|حقوق\s+(?:الطبع|النشر)\s+(?:والتأليف\s+)?محفوظة\s+(?:ل|لـ))\s*(?:©)?\s*"
                       r"(?:\d{4}\s*[-–]?\s*(?:\d{4})?)?\s*[,.]?\s*([^\n|•]{2,70})", re.I)
COPYRIGHT_TAIL = re.compile(r"[.,،|]?\s*(all\s+rights\s+reserved|جميع\s+الحقوق\s+محفوظة|الحقوق\s+محفوظة|\d{4}).*$", re.I)
ACRONYM = re.compile(r"^(.{4,}?)\s*\(\s*([A-Z][A-Za-z0-9&]{1,7})\s*\)$")
FILE_WORDS = re.compile(r"\b(privacy|policy|policies|notice|statement|final|draft|copy|document|doc|file|terms|"
                        r"data|protection|v\d+|new|old|en|ar|english|arabic|scan)\b|سياسة|الخصوصية|خصوصية|نسخة|"
                        r"ملف|نهائي|\d+", re.I)


def tidy(name):
    """Trim spaces and punctuation; 'King Fahd University ... (KFUPM)' -> 'KFUPM'."""
    name = re.sub(r"\s+", " ", name).strip(" .,،;:-–|\"'“”«»")
    short = ACRONYM.match(name)
    if short:
        name = short.group(2)
    name = re.sub(r"\s*\([^)]*$", "", name).strip()  # a bracket left open
    return name if 2 <= len(name) <= 60 and not GENERIC.match(name) else ""


def name_from_site(site):
    """'www.jarir.com' -> 'Jarir', 'stc.com.sa' -> 'STC', 'al-rajhi.com' -> 'Al Rajhi'."""
    parts = site.lower().removeprefix("www.").split(".")
    while len(parts) > 2 and (parts[-1] in SECOND_LEVEL or len(parts[-1]) <= 2 or parts[-2] in SECOND_LEVEL):
        parts.pop()
    word = re.sub(r"[-_]+", " ", parts[0]).strip()
    return word.upper() if len(word) <= 4 else word.title()


def own_site(site):
    site = site.lower().removeprefix("www.").rstrip(".")
    parts = site.split(".")
    return "" if set(parts) & OTHER_SITES or site.endswith("gov.sa") else site


def from_title(text):
    for line in [l.strip() for l in text.splitlines() if l.strip()][:4]:
        if len(line) > 120:  # a pasted policy as one long line: only a Latin name right after the title
            latin = TITLE_LATIN.search(line[:200])
            name = tidy(TITLE_STOP.split(latin.group(1))[0]) if latin else ""
            if name and not re.search(r"privacy|policy", name, re.I):
                return name
            continue
        match = TITLE_EN.search(line) or TITLE_AR.search(line)
        if match:
            name = tidy(next(g for g in match.groups() if g))
            name = re.sub(r"^(?:لشركة|شركة)\s+", "", name) if name.startswith(("لشركة", "شركة")) else name
            if name and not re.search(r"privacy|سياسة", name, re.I):
                return name
    return ""


def from_sentence(text):
    """The company named with its short name in brackets, near the top of the policy."""
    top = text[:3000]
    for pattern in (NAMED_EN, NAMED_AR):
        for match in pattern.finditer(top):
            full, short = match.group(1), match.group(2)
            full = re.sub(r"^.*?\b(?:by|between|from|that|this\s+\w+\s+(?:is|explains\s+how))\s+(?=[A-Z])", "", full)
            name = tidy(short or "") or tidy(full)
            if name:
                return name
    return ""


def from_brand(text):
    """An Arabic policy that keeps its brand in Latin letters ('Apple', 'stc'): the Latin
    word it repeats most, when it repeats it at least 4 times."""
    if len(re.findall(r"[\u0600-\u06FF]", text[:5000])) < len(re.findall(r"[A-Za-z]", text[:5000])):
        return ""  # an English policy repeats many capitalised words
    words = Counter(w for w in LATIN_WORD.findall(text) if w not in NOT_NAMES and w.lower() not in NOT_NAMES)
    if not words:
        return ""
    word, count = words.most_common(1)[0]
    return word if count >= 4 else ""


def from_links(text):
    sites = Counter(s for s in (own_site(m) for m in LINK.findall(text)) if s)
    return sites.most_common(1)[0][0] if sites else ""


def from_copyright(text):
    for found in COPYRIGHT.findall(text):
        name = tidy(COPYRIGHT_TAIL.sub("", found))
        if name and not re.fullmatch(r"[\d\s]+", name):
            return name
    return ""


def from_emails(text):
    found = [(user.lower(), own_site(site)) for user, site in EMAIL.findall(text)]
    found = [(user, site) for user, site in found if site]
    for wanted in ("privacy", "data", "dpo", "dmo"):  # a privacy address first, then any company address
        for user, site in found:
            if user.startswith(wanted):
                return site
    return found[0][1] if found else ""


def from_file_name(file_name):
    stem = re.sub(r"[_\-.]+", " ", Path(file_name or "").stem).strip()
    left = re.sub(r"\s+", " ", FILE_WORDS.sub(" ", stem)).strip()
    return left.title() if len(left) >= 2 and re.search(r"[^\W\d_]", left) else ""


def company_name(text, file_name=""):
    """(name, site) of the policy's company, or ("", "") when the text does not say."""
    name = from_title(text) or from_sentence(text) or from_brand(text)
    if name:
        return name, ""
    site = from_links(text)
    if site:
        return name_from_site(site), site
    name = from_copyright(text)
    if name:
        return name, ""
    site = from_emails(text)
    if site:
        return name_from_site(site), site
    return from_file_name(file_name), ""


def main():
    for path in sys.argv[1:] or ["data/policies/tamara.txt"]:
        text = Path(path).read_text(encoding="utf-8")
        print(f"{Path(path).name:22} {company_name(text)}")


if __name__ == "__main__":
    main()
