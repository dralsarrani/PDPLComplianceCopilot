"""
find_policy(site): get a company's privacy policy text from its website.

    "stc.com.sa"                               ->  open the home page, follow the
                                                   link whose text or address says
                                                   privacy / الخصوصية
    "https://tamara.co/privacy-policy"         ->  the page itself
    "mobily.com.sa" (no readable link)         ->  the privacy page listed in the
                                                   site's sitemap, then common
                                                   addresses like /privacy-policy

guess_website("أرامكو") asks OpenAI for the company's official website
("aramco.com") when the user types only a name. The app shows the site it
used, and the privacy page must really be found there, so a wrong guess
ends in a clear message, not a wrong analysis.

The app reads only public privacy pages, says honestly who it is (USER_AGENT in
fetch_policies.py), follows each site's robots.txt, and never tries to get round a site that blocks
or ignores it: then the user is asked to paste the policy text instead.

Returns (policy_url, text). Raises ValueError with a message for the user when
no privacy link is found or the page has too little text (usually a site that
blocks scripts or builds the page with JavaScript: paste the text instead).
"""
import gzip
import json
import os
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

from src.policies.fetch_policies import HEADERS, MIN_CHARS, TIMEOUT, Blocked, allowed, html_to_text, robots

GUESS_PROMPT = """You get the name of a company or organisation, usually Saudi, in Arabic or English.
Reply in JSON with its official name, its official website domain and, if you know it,
the address of its privacy policy page, e.g.
{"name": "Saudi Aramco", "site": "aramco.com", "privacy_url": "https://www.aramco.com/en/privacy"}.
Leave privacy_url empty if you are not sure of it.
If you are not sure which company is meant or do not know its website, reply {"name": "", "site": "", "privacy_url": ""}."""
BLOCKED = "الموقع لا يسمح بقراءة صفحاته آلياً، ونحترم ذلك."
NOT_FOUND = "تعذّر العثور على صفحة سياسة الخصوصية في موقع الجهة."
NO_ANSWER = "لم يستجب موقع الجهة للطلب."
JS_PAGE = "لا تتضمن صفحة سياسة الخصوصية نصاً كافياً، ويُرجّح أن الموقع يعرضها باستخدام JavaScript."
PRIVACY_WORDS = re.compile(r"privacy|الخصوصية|خصوصية", re.IGNORECASE)
SITEMAP_FILES = 4  # sitemap files read at most per site
HUB_CHARS = 10000  # a privacy page shorter than this may be a hub that links to the full policy
FULL_POLICY = re.compile(r"privacy[ -]?(policy|statement|notice)|privacy-full|full|سياسة الخصوصية|بيان الخصوصية", re.IGNORECASE)
SITEMAP_PAGES = 3  # privacy pages from the sitemap tried at most
COMMON_PATHS = ["/privacy-policy", "/privacy", "/en/privacy-policy", "/ar/privacy-policy"]


def as_url(site):
    """'stc.com.sa' -> 'https://stc.com.sa'"""
    site = site.strip()
    return site if re.match(r"^https?://", site) else "https://" + site


def looks_like_site(text):
    """True for 'stc.com.sa', 'www.tamara.co', 'https://...'."""
    text = text.strip()
    return bool(re.match(r"^(https?://)?[\w-]+(\.[\w-]+)+(/\S*)?$", text))


def guess_website(company, model=None):
    """(official name, domain, privacy page address or "") of a company from its
    name, or ("", "", "") if unknown. The privacy address is only a hint to try first."""
    load_dotenv()
    response = OpenAI().chat.completions.create(
        model=model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": GUESS_PROMPT}, {"role": "user", "content": company}],
    )
    reply = json.loads(response.choices[0].message.content)
    name, site, hint = (str(reply.get(key) or "").strip() for key in ("name", "site", "privacy_url"))
    if not looks_like_site(site):
        return "", "", ""
    return name, site, hint if looks_like_site(hint) else ""


def get(url):
    if not allowed(url):
        raise Blocked(url)
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        r.encoding = r.apparent_encoding  # fixes garbled Arabic
    return r


def privacy_link(page):
    """The best privacy-policy link on a fetched page, or None. Links whose text
    says privacy beat links that only have it in the address."""
    soup = BeautifulSoup(page.text, "html.parser")
    best, best_score = None, 0
    for a in soup.find_all("a", href=True):
        text, href = a.get_text(" ", strip=True), a["href"]
        score = 2 * bool(PRIVACY_WORDS.search(text)) + bool(PRIVACY_WORDS.search(href))
        if score > best_score:
            best, best_score = urljoin(page.url, href), score
    return best


def full_policy(page, text):
    """(address, text) of the full policy when page is a privacy hub (a short page
    such as a "privacy center" that links to the policy itself), else page and text."""
    if len(text) >= HUB_CHARS:
        return page.url, text
    host = urlparse(page.url).netloc
    soup = BeautifulSoup(page.text, "html.parser")
    for a in soup.find_all("a", href=True):
        link = urljoin(page.url, a["href"]).split("#")[0]
        if link == page.url.split("#")[0] or urlparse(link).netloc != host:
            continue
        if FULL_POLICY.search(a.get_text(" ", strip=True)) or FULL_POLICY.search(urlparse(link).path):
            try:
                longer = html_to_text(get(link).text)
            except (Blocked, requests.RequestException):
                continue
            if len(longer) > len(text) * 1.5 and looks_like_policy(longer):
                return link, longer
    return page.url, text


POLICY_WORDS = re.compile(r"personal (data|information)|privacy|البيانات الشخصية|المعلومات الشخصية|الخصوصية",
                          re.IGNORECASE)


def looks_like_policy(text):
    """True when the text reads like a privacy policy, not a home page or a shop
    page that a privacy address redirected to (those say "privacy" once, in the footer)."""
    return len(POLICY_WORDS.findall(text)) >= 8


def home_pages(site):
    """'aramco.com' -> https://www.aramco.com and https://aramco.com"""
    host = urlparse(as_url(site)).netloc
    bare = host[4:] if host.startswith("www.") else host
    return [f"https://www.{bare}", f"https://{bare}"]


def sitemap_links(site):
    """Privacy pages listed in the site's sitemap, the list of its pages a site
    publishes for search engines. Sitemaps named in robots.txt come first."""
    home = home_pages(site)[0]
    rules = robots(home)
    queue = list((rules.site_maps() or []) if not isinstance(rules, bool) else [])
    queue += [home + "/sitemap.xml", home + "/sitemap_index.xml"]
    pages, read = [], 0
    while queue and read < SITEMAP_FILES:
        sitemap = queue.pop(0)
        try:
            r = get(sitemap)
        except (Blocked, requests.RequestException):
            continue
        read += 1
        body = r.text
        if r.content[:2] == b"\x1f\x8b":  # still gzipped; a ".gz" name alone does not mean it is
            try:
                body = gzip.decompress(r.content).decode("utf-8", "ignore")
            except (OSError, EOFError):
                continue
        for loc in re.findall(r"<loc>\s*(.*?)\s*</loc>", body, re.IGNORECASE):
            loc = loc.replace("&amp;", "&")
            if re.search(r"\.xml(\.gz)?$", loc, re.IGNORECASE):
                # a sitemap of sitemaps: read the ones about pages first
                (queue.insert(0, loc) if re.search(r"page|privacy|legal", loc, re.I) else queue.append(loc))
            elif PRIVACY_WORDS.search(urlparse(loc).path):
                pages.append(loc)
        if pages:
            break
    # the plainest address first: /privacy-policy before /blog/2021/privacy-tips
    return sorted(dict.fromkeys(pages), key=lambda u: len(urlparse(u).path))[:SITEMAP_PAGES]


def find_policy(site, hint=""):
    """(privacy page address, its text). Tries, in order: the site itself when it is
    already a privacy page, the hint, the privacy link on the home page (with and
    without www.), privacy pages in the site's sitemap, then common addresses.
    Every request respects robots.txt. Raises ValueError with a message for the user."""
    url = as_url(site)
    problem, blocked = NO_ANSWER, False

    def candidates():
        nonlocal problem, blocked
        if PRIVACY_WORDS.search(urlparse(url).path):
            yield url
        if hint:
            yield hint
        for home in home_pages(site):
            try:
                link = privacy_link(get(home))
            except Blocked:
                blocked = True
                continue
            except requests.RequestException:
                continue
            if problem == NO_ANSWER:
                problem = NOT_FOUND
            if link:
                yield link
        if problem == NO_ANSWER:
            return  # the site does not answer: no point waiting on more addresses
        yield from sitemap_links(site)
        for path in COMMON_PATHS:
            yield home_pages(site)[0] + path

    tried = set()
    for page in candidates():
        if page in tried:
            continue
        tried.add(page)
        try:
            fetched = get(page)
            text = html_to_text(fetched.text)
        except Blocked:
            blocked = True
            continue
        except requests.RequestException:
            continue
        if len(text) >= MIN_CHARS and looks_like_policy(text):
            return full_policy(fetched, text)
        if len(text) < MIN_CHARS:
            problem = JS_PAGE
        elif problem == NO_ANSWER:  # the page answered but is not the policy
            problem = NOT_FOUND
    raise ValueError(BLOCKED if blocked and problem != JS_PAGE else problem)
