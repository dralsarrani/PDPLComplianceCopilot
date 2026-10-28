"""
find_policy(site): get a company's privacy policy text from its website.

    "stc.com.sa"                               ->  open the home page, follow the
                                                   link whose text or address says
                                                   privacy / الخصوصية
    "https://tamara.co/privacy-policy"         ->  the page itself
    "mobily.com.sa" (no readable link)         ->  the privacy page listed in the
                                                   site's sitemap, then common
                                                   addresses like /privacy-policy

guess_website("جاهز") finds the company's website and privacy page when the user
types only a name, the way a person would: a web search for its privacy policy,
keeping only addresses the search really found. The search engine is Tavily
(TAVILY_API_KEY in .env, about a second); without it, OpenAI's search tool. Asked
from memory alone, the model can be wrong ("jahez.com" for jahez.net), so
memory is used only when the search cannot run. The policy text itself is then
read from that page by us, word for word, not written by the model.

The app reads only public privacy pages, says honestly who it is (USER_AGENT in
fetch_policies.py), follows each site's robots.txt, and never tries to get round a site that blocks
or ignores it: then the user is asked to paste the policy text instead.

A privacy page in one language may hold only the terms of use while the other
language holds the policy (Ninja's Arabic page shows its terms, the English page
"Terms and conditions & Privacy Policy"), so the other language of a privacy
address is tried too.

Returns (policy_url, text). Raises ValueError with a message for the user when
no privacy link is found or the page has too little text (usually a site that
blocks scripts or builds the page with JavaScript: paste the text instead).
"""
import gzip
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

from src.policies.fetch_policies import HEADERS, MIN_CHARS, TIMEOUT, Blocked, allowed, html_to_text, robots
from src.policies.web_search import ask, sources

GUESS_PROMPT = """You get the name of a company or organisation, usually Saudi, in Arabic or English.
Reply in JSON with its official name, its official website domain and, if you know it,
the address of its privacy policy page, e.g.
{"name": "Saudi Aramco", "site": "aramco.com", "privacy_url": "https://www.aramco.com/en/privacy"}.
Leave privacy_url empty if you are not sure of it.
If you are not sure which company is meant or do not know its website, reply {"name": "", "site": "", "privacy_url": ""}."""
SEARCH_PROMPT = """Search the web for the official privacy policy page of "{company}", a company,
app or organisation that users in Saudi Arabia know (the name may be in Arabic or English, and
may be just the short name of an app, such as a delivery, shopping or banking app). If several
companies share the name, the best-known one in Saudi Arabia comes first.
Give the company's OWN website, never a news, blog, app store, social media or directory site.
Prefer the policy page for users in Saudi Arabia. Reply ONLY with JSON, best match first, at most 3:
{{"name": "<official name>", "candidates": [{{"site": "<official website domain>", "privacy_url": "<address of its privacy policy page>"}}]}}
Use only addresses you found in the search results. Reply {{"name": "", "candidates": []}}
only if no such company can be found at all."""
TAVILY_URL = "https://api.tavily.com/search"
NOT_COMPANY = re.compile(r"(^|\.)(google|apple|facebook|instagram|x|twitter|linkedin|youtube|tiktok|snapchat|"
                         r"wikipedia|reddit|crunchbase|glassdoor|trustpilot|medium|zawya|argaam)\.", re.IGNORECASE)
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
POLICY_TITLE = re.compile(r"privacy[ -]?(policy|notice|statement)|سياسة الخصوصية|بيان الخصوصية", re.IGNORECASE)


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
    name, or ("", "", "") if unknown: the first of website_candidates()."""
    name, found = website_candidates(company, model)
    return (name, *found[0]) if found else ("", "", "")


def website_candidates(company, model=None):
    """(official name, [(domain, privacy page address or ""), ...]) of a company from
    its name, best first, to try one after another (a short name like "نينجا" can
    first bring up another site). Searches the web first; the model's memory only
    when the search cannot run or finds nothing."""
    load_dotenv()
    if os.environ.get("TAVILY_API_KEY"):
        try:
            found = tavily_candidates(company)
            if found:
                return "", found
        except Exception as error:
            print(f"[find_policy] Tavily search failed: {error}")
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    try:
        name, found = search_website(company, model)
        if found:
            return name, found
    except Exception as error:  # no web search on this account or no connection: memory instead
        print(f"[find_policy] web search for the website failed: {error}")
    response = OpenAI().chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": GUESS_PROMPT}, {"role": "user", "content": company}],
    )
    name, site, hint = site_reply(json.loads(response.choices[0].message.content))
    return name, [(site, hint)] if site else []


def tavily_candidates(company):
    """[(domain, privacy page address)] from a Tavily web search for the company's
    privacy policy, asked in Arabic and in English at once. Results that name the
    company come first (any Saudi site is not enough: "بنك الرياض" once brought up
    saib.com.sa), then Saudi sites (".sa", "/sa/", Arabic text or Saudi Arabia
    named), since "نينجا" is also the name of companies elsewhere; then privacy pages; never news, social media or
    app-store sites; at most one page per site."""
    queries = [f"{company} سياسة الخصوصية السعودية", f"{company} privacy policy Saudi Arabia"]
    with ThreadPoolExecutor(len(queries)) as pool:
        answers = list(pool.map(tavily_search, queries))
    results = [item for row in zip_longest(*answers) for item in row if item]  # 1st of each, 2nd of each...
    results = [item for item in results if item.get("url") and not NOT_COMPANY.search(bare_host(item["url"]))]
    if re.search(r"[\u0600-\u06FF]", company):  # a name typed in Arabic means a company serving Saudi Arabia
        results = [item for item in results if saudi_score(item)]
    order = {id(item): i for i, item in enumerate(results)}
    results.sort(key=lambda item: (not names_company(item, company), -saudi_score(item),
                                   not PRIVACY_WORDS.search(urlparse(item["url"]).path), order[id(item)]))
    found = []
    for item in results:
        url = item["url"]
        if all(bare_host(url) != site for site, _ in found):
            found.append((bare_host(url), url if PRIVACY_WORDS.search(urlparse(url).path) else ""))
    return found[:3]


def squash(text):
    """'بنك  الرياض' and 'بنك الرياض' -> 'بنكالرياض', 'Riyad Bank' -> 'riyadbank'."""
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text.lower())
    text = re.sub(r"[إأآٱ]", "ا", text).replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"[^\w]|_", "", text)


def names_company(item, company):
    """True when a search result's address, title or text has the company's name."""
    name = squash(company)
    return bool(name) and name in squash(f"{bare_host(item['url'])} {item.get('title', '')} {item.get('content', '')}")


def tavily_search(query):
    r = requests.post(TAVILY_URL, timeout=15,
                      headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
                      json={"query": query, "max_results": 8})
    r.raise_for_status()
    return r.json().get("results", [])


def saudi_score(item):
    """How surely a search result belongs to a company serving Saudi Arabia."""
    url, text = item["url"].lower(), f"{item.get('title', '')} {item.get('content', '')}"
    host, path = bare_host(url), urlparse(url).path
    return (2 * bool(host.endswith(".sa") or re.search(r"/(sa|ksa|sa-ar|sa-en|ar-sa|en-sa)(/|$)", path))
            + bool(re.search(r"saudi|ksa|السعودية|المملكة", text, re.IGNORECASE))
            + bool(re.search(r"[\u0600-\u06FF]{3}", text)))


def search_website(company, model):
    """(official name, [(domain, privacy page address)]) found by a web search: the
    model's candidates, then other privacy pages the search cited. Addresses the
    search did not cite are kept only when on the candidate's own site."""
    response = ask(OpenAI(), model, SEARCH_PROMPT.format(company=company))
    reply = re.search(r"\{.*\}", response.output_text or "", re.DOTALL)
    reply = json.loads(reply.group(0)) if reply else {}
    cited = [url for url in sources(response) if PRIVACY_WORDS.search(urlparse(url).path)]
    found = []
    for item in (reply.get("candidates") or [])[:3]:
        _, site, hint = site_reply({"name": "", **item} if isinstance(item, dict) else {})
        if not site:
            continue
        if hint not in cited:
            same_site = [url for url in cited if bare_host(url).endswith(bare_host(site))]
            hint = same_site[0] if same_site else hint if bare_host(hint).endswith(bare_host(site)) else ""
        found.append((site, hint))
    for url in cited:  # privacy pages the search read but the model did not list
        if all(not bare_host(url).endswith(bare_host(site)) for site, _ in found):
            found.append((bare_host(url), url))
    return str(reply.get("name") or "").strip(), found


def site_reply(reply):
    """(name, site, privacy address) from the model's JSON, empty when not a website."""
    name, site, hint = (str(reply.get(key) or "").strip() for key in ("name", "site", "privacy_url"))
    if not looks_like_site(site):
        return "", "", ""
    return name, site, hint if looks_like_site(hint) else ""


def bare_host(url):
    """'https://www.jahez.net/ar/privacy' -> 'jahez.net'"""
    return urlparse(as_url(url)).netloc.lower().removeprefix("www.")


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


POLICY_WORDS = re.compile(r"personal (data|information)|privacy|(البيانات|بيانات\w*) الشخصية|(المعلومات|معلومات\w*) الشخصية|الخصوصية|خصوصيت\w*",
                          re.IGNORECASE)


def looks_like_policy(text):
    """True when the text reads like a privacy policy, not a home page or a shop
    page that a privacy address redirected to (those say "privacy" once, in the footer).
    A page whose title says "Privacy Policy" needs fewer policy words: it may be a
    combined "Terms and conditions & Privacy Policy" page."""
    words = len(POLICY_WORDS.findall(text))
    titles = [line for line in text.splitlines() if len(line) > 15][:3]
    return words >= 8 or (words >= 4 and any(POLICY_TITLE.search(line) for line in titles))


def other_language(url):
    """'https://x.com/sa/ar/privacy-policy' -> 'https://x.com/sa/en/privacy-policy' (and back), or ''."""
    parts = urlparse(url)
    path = re.sub(r"(?<=/)(ar|en)(?=/|$)", lambda m: "en" if m.group(1) == "ar" else "ar", parts.path, count=1)
    return parts._replace(path=path).geturl() if path != parts.path else ""


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

    tried, other = set(), []

    def pages():
        for page in candidates():
            yield page
            while other:  # the same privacy page in the other language, tried at once
                yield other.pop(0)

    for page in pages():
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
        for alt in (other_language(page), other_language(fetched.url)):
            if alt and alt not in tried and PRIVACY_WORDS.search(urlparse(alt).path):
                other.append(alt)
        if len(text) < MIN_CHARS:
            problem = JS_PAGE
        elif problem == NO_ANSWER:  # the page answered but is not the policy
            problem = NOT_FOUND
    raise ValueError(BLOCKED if blocked and problem != JS_PAGE else problem)
