"""
from_archive(name, site, hint=""): a company's privacy policy from a public
archive, for when its own website cannot be read (it blocks scripts, does not
answer, or builds the page with JavaScript).

Tries, in order:
1. Open Terms Archive (opentermsarchive.org), an open project started in 2020
   by France's Ambassador for Digital Affairs. It keeps the current text of the
   privacy policies of big online services and publishes it in public GitHub
   repositories, checked every day. Read through the GitHub API.
2. Wayback Machine (web.archive.org) of the Internet Archive, a non-profit
   library: the latest saved copy of the company's privacy page. Found with its
   public search API.
3. Common Crawl (commoncrawl.org), a non-profit that saves billions of web
   pages every month and publishes them for free: the privacy page from its two
   latest monthly crawls. Found with its public index.

All three are public archives that publish these copies for anyone to read. Requests
use the same honest USER_AGENT, and archive pages follow robots.txt. APIs made
for programs (GitHub, the Wayback and Common Crawl indexes) are used as their
documentation describes, within their free limits.

Returns (address to show, text, date "YYYY-MM-DD" of the copy, archive name).
Raises ValueError if neither archive has a usable copy.

Try it from the project root:
    python -m src.policies.archive Netflix netflix.com
"""
import gzip
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import requests

from src.policies.fetch_policies import HEADERS, MIN_CHARS, TIMEOUT, Blocked, allowed, html_to_text
from src.policies.find_policy import looks_like_policy

OTA_REPOS = ["contrib-versions", "pga-versions"]  # Open Terms Archive collections with big services
OTA_FILE = "Privacy Policy.md"
INDEX_DIR = Path("results/archive_index")  # list of services per collection, kept one day
INDEX_DAYS = 1
COMMON_PATHS = ["/privacy", "/privacy-policy", "/legal/privacy", "/privacy-notice"]
NOT_ARCHIVED = "ما لقينا نسخة من السياسة في الأرشيف العام."


def squash(text):
    """'X (Twitter)' -> 'xtwitter', 'Netflix, Inc.' -> 'netflixinc'"""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def site_label(site):
    """'https://www.netflix.com/sa' -> 'netflix'"""
    host = urlparse(site if "://" in site else "https://" + site).netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return host.split(".")[0]


# ---------- 1. Open Terms Archive ----------

def ota_services(repo):
    """{service folder: path of its privacy policy file} in one collection.
    The list is saved for a day so the GitHub API is asked rarely."""
    saved = INDEX_DIR / f"{repo}.json"
    if saved.exists() and time.time() - saved.stat().st_mtime < INDEX_DAYS * 86400:
        return json.loads(saved.read_text(encoding="utf-8"))
    r = requests.get(f"https://api.github.com/repos/OpenTermsArchive/{repo}/git/trees/main?recursive=1",
                     headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    services = {path.split("/")[0]: path for path in (item["path"] for item in r.json().get("tree", []))
                if path.count("/") == 1 and path.endswith("/" + OTA_FILE)}
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(services, ensure_ascii=False), encoding="utf-8")
    return services


def ota_match(services, name, site):
    """The service folder for this company: same name, or same name as the site
    ('Amazon.com' for Amazon / amazon.com). None if there is none."""
    host = urlparse(site if "://" in site else "https://" + site).netloc.lower().removeprefix("www.")
    wanted = {squash(name), site_label(site), squash(host)} - {""}

    def key(folder):  # 'Amazon.com' -> 'amazon', 'X (Twitter)' -> 'xtwitter'
        return squash(re.sub(r"\.[a-z]{2,3}$", "", folder, flags=re.IGNORECASE))

    for folder in services:
        if squash(folder) in wanted or key(folder) in wanted:
            return folder
    for folder in services:  # 'Netflix, Inc.' still finds 'Netflix'
        if len(key(folder)) >= 4 and squash(name).startswith(key(folder)):
            return folder
    return None


def from_open_terms(name, site):
    for repo in OTA_REPOS:
        try:
            services = ota_services(repo)
        except (requests.RequestException, ValueError) as error:
            say(f"Open Terms Archive ({repo}) ما رد: {error}")
            continue
        folder = ota_match(services, name, site)
        if not folder:
            say(f"Open Terms Archive ({repo}): الشركة مو موجودة")
            continue
        path = quote(services[folder])
        try:
            r = requests.get(f"https://raw.githubusercontent.com/OpenTermsArchive/{repo}/main/{path}",
                             headers=HEADERS, timeout=TIMEOUT)
            r.encoding = "utf-8"
            if not r.ok or len(r.text) < MIN_CHARS:
                continue
            when = ""
            commits = requests.get(f"https://api.github.com/repos/OpenTermsArchive/{repo}/commits",
                                   params={"path": services[folder], "per_page": 1},
                                   headers=HEADERS, timeout=TIMEOUT)
            if commits.ok and commits.json():
                when = commits.json()[0]["commit"]["committer"]["date"][:10]
        except (requests.RequestException, ValueError, KeyError):
            continue
        page = f"https://github.com/OpenTermsArchive/{repo}/blob/main/{path}"
        return page, r.text, when, "Open Terms Archive"
    return None


# ---------- 2. Wayback Machine ----------

CDX = "https://web.archive.org/cdx/search/cdx"  # the Wayback Machine's public search API
CDX_TIMEOUT = (6, 20)
WAYBACK_SECONDS = 45  # the user should not wait longer than this
VERBOSE = False  # True from the command line: print what happened at each step


def say(message):
    if VERBOSE:
        print("  ·", message)


def wayback_pages(site, hint=""):
    """Addresses to look up in the Wayback Machine: the privacy page we were told
    about, then common privacy addresses on the site. Asking about exact pages
    is quick; searching a whole big site there times out."""
    host = urlparse(site if "://" in site else "https://" + site).netloc.lower().removeprefix("www.")
    pages = [hint] if hint else []
    for home in (f"https://www.{host}", f"https://{host}"):
        pages += [home + path for path in COMMON_PATHS]
    return pages


def wayback_latest(url):
    """Timestamp of the latest copy of url that loaded (or redirected), or None."""
    r = requests.get(CDX, params={"url": url, "fl": "timestamp,statuscode", "filter": "statuscode:[23]..",
                                  "limit": -1, "output": "json"}, headers=HEADERS, timeout=CDX_TIMEOUT)
    r.raise_for_status()
    rows = r.json()[1:]
    return rows[-1][0] if rows else None


def from_wayback(site, hint=""):
    deadline = time.time() + WAYBACK_SECONDS
    for url in dict.fromkeys(wayback_pages(site, hint)):
        if time.time() > deadline:
            say("Wayback بطيء اليوم، وقفنا البحث")
            break
        try:
            stamp = wayback_latest(url)
            if not stamp:
                say(f"ما فيه نسخة محفوظة: {url}")
                continue
            # "id_" asks for the page as it was saved, without the archive's toolbar
            copy = f"https://web.archive.org/web/{stamp}id_/{url}"
            if not allowed(copy):
                say("robots.txt حق الأرشيف ما يسمح")
                continue
            r = requests.get(copy, headers=HEADERS, timeout=CDX_TIMEOUT)
            r.raise_for_status()
            if not r.encoding or r.encoding.lower() == "iso-8859-1":
                r.encoding = r.apparent_encoding
            text = html_to_text(r.text)
        except (requests.RequestException, ValueError, Blocked) as error:
            say(f"خطأ في {url}: {error}")
            continue
        if len(text) < MIN_CHARS:  # pages built with JavaScript are saved nearly empty
            say(f"النسخة شبه فاضية ({len(text)} حرف، غالباً الصفحة مبنية بجافاسكربت): {url}")
            continue
        if not looks_like_policy(text):  # the address redirected to another page
            say(f"النسخة مو سياسة خصوصية: {url}")
            continue
        when = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
        return f"https://web.archive.org/web/{stamp}/{url}", text, when, "Wayback Machine"
    return None


# ---------- 3. Common Crawl ----------

CC_COLLECTIONS = "https://index.commoncrawl.org/collinfo.json"  # its monthly crawls, newest first
CC_DATA = "https://data.commoncrawl.org/"
CC_CRAWLS = 2  # latest monthly crawls searched
CC_SECONDS = 40


def cc_latest(url, crawl_api):
    """(timestamp, url, filename, offset, length) of this exact page in one crawl, or None.
    Asking about exact pages is quick; searching a whole big site there times out."""
    r = requests.get(crawl_api, params={"url": url, "output": "json", "filter": "=status:200"},
                     headers=HEADERS, timeout=CDX_TIMEOUT)
    if r.status_code == 404:  # "no captures found"
        return None
    r.raise_for_status()
    rows = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    if not rows:
        return None
    row = rows[-1]
    return row["timestamp"], row["url"], row["filename"], int(row["offset"]), int(row["length"])


def cc_text(filename, offset, length):
    """The page's text from its saved record (a slice of a compressed WARC file)."""
    r = requests.get(CC_DATA + filename, headers={**HEADERS, "Range": f"bytes={offset}-{offset + length - 1}"},
                     timeout=CDX_TIMEOUT)
    r.raise_for_status()
    record = gzip.decompress(r.content)
    # WARC headers, blank line, HTTP headers, blank line, the page
    _, _, http = record.partition(b"\r\n\r\n")
    head, _, body = http.partition(b"\r\n\r\n")
    charset = re.search(rb"charset=([\w-]+)", head, re.I)
    return html_to_text(body.decode(charset.group(1).decode() if charset else "utf-8", "ignore"))


def from_common_crawl(site, hint=""):
    deadline = time.time() + CC_SECONDS
    try:
        crawls = requests.get(CC_COLLECTIONS, headers=HEADERS, timeout=CDX_TIMEOUT).json()[:CC_CRAWLS]
    except (requests.RequestException, ValueError) as error:
        say(f"Common Crawl ما رد: {error}")
        return None
    for crawl in crawls:
        for url in dict.fromkeys(wayback_pages(site, hint)):  # the same addresses as for Wayback
            if time.time() > deadline:
                say("Common Crawl بطيء اليوم، وقفنا البحث")
                return None
            try:
                found = cc_latest(url, crawl["cdx-api"])
                if not found:
                    continue
                stamp, url, filename, offset, length = found
                text = cc_text(filename, offset, length)
            except (requests.RequestException, OSError, ValueError, LookupError) as error:
                say(f"Common Crawl ({crawl['id']}) خطأ في {url}: {error}")
                continue
            if len(text) < MIN_CHARS or not looks_like_policy(text):
                say(f"النسخة فاضية أو مو سياسة خصوصية: {url}")
                continue
            return url, text, f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}", "Common Crawl"
        say(f"Common Crawl ({crawl['id']}): ما فيه نسخة")
    return None


def from_archive(name, site, hint=""):
    """(address to show, text, date of the copy, archive name). Raises ValueError."""
    found = from_open_terms(name, site) or from_wayback(site, hint) or from_common_crawl(site, hint)
    if not found:
        raise ValueError(NOT_ARCHIVED)
    return found


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('مثال: python -m src.policies.archive Netflix netflix.com')
    VERBOSE = True
    from src.policies.find_policy import guess_website
    hint = guess_website(sys.argv[1])[2]  # the same privacy-page guess the app uses
    say(f"عنوان السياسة المتوقع: {hint or 'ما فيه'}")
    try:
        page, text, when, archive = from_archive(sys.argv[1], sys.argv[2], hint)
    except ValueError as error:
        sys.exit(str(error))
    print(f"المصدر: {archive}\nالتاريخ: {when}\nالرابط: {page}\nطول النص: {len(text)} حرف\n")
    print(text[:600])
