"""
from_web_search(name, site): a company's privacy policy found with OpenAI's
web search, the last try when neither the company's site nor the public
archives can give us the text.

OpenAI reads the page with its own search tool, which many sites allow, and
writes back what the policy says, keeping its wording as closely as it can.
The text is therefore written by the model, not copied from the page: it can
leave things out or word them differently. The app says so under the result.
One answer cannot hold a whole policy, so it is asked for in three parts at once.

Each search costs a little (a web search call plus the model's answer), so a
found policy is kept in results/web_policies/ for WEB_DAYS days.

Returns (policy page address or "", text, date "YYYY-MM-DD").
Raises ValueError with a message for the user.

Try it from the project root:
    python -m src.policies.web_search Shein shein.com
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.policies.fetch_policies import MIN_CHARS

SAVED_DIR = Path("results/web_policies")
WEB_DAYS = 30
TOOLS = ["web_search", "web_search_preview"]  # newer name first; older models know only the second
NOT_FOUND = "تعذّر العثور على سياسة الخصوصية حتى بالبحث في الإنترنت."

# One answer cannot hold a whole policy (it stops mid-way), so the policy is
# asked for in parts, each covering some of its sections, all at the same time.
PARTS = [
    "who the company is and how to contact it, any data protection officer, what personal data they "
    "collect and from where, why they use it and on what legal basis (consent etc.)",
    "who they share or sell data with, transfers of data to other countries, how long data is kept and "
    "how it is deleted or destroyed, how data is secured",
    "the user's rights (access, correction, deletion, objection, withdrawing consent) and how to use them, "
    "what happens if the user refuses to give data, automated decisions or profiling, children, "
    "cookies, and changes to the policy",
]
PROMPT = """Find the current official privacy policy (privacy notice) of {name} (website: {site}).
Use the English version, preferably the one for users in Saudi Arabia or the Middle East,
otherwise the global one. Open the policy page itself.
Write out ONLY the parts of the policy about: {part}.
Keep the policy's own wording as closely as you can, with its headings. Do not summarise,
comment or add anything that is not in the policy. If the policy says nothing about one of
these topics, skip it.
First line: SOURCE: <the address of the policy page you used>
If you cannot find the official policy, reply only: NOT_FOUND"""
MAX_TOKENS = 12000


def saved_path(name):
    return SAVED_DIR / (re.sub(r"[^\w]+", "_", name.lower()).strip("_") + ".json")


def ask(client, model, prompt):
    """The model's answer with web search on, trying the tool names this account knows."""
    last = None
    for tool in TOOLS:
        try:
            return client.responses.create(model=model, tools=[{"type": tool}], input=prompt,
                                           max_output_tokens=MAX_TOKENS)
        except Exception as error:  # an unknown tool name: try the next one
            last = error
    raise last


def read_part(client, model, prompt):
    """(source address, text) of one part, or ("", "") if not found."""
    response = ask(client, model, prompt)
    text = (response.output_text or "").strip()
    if not text or text.startswith("NOT_FOUND"):
        return "", ""
    first, _, rest = text.partition("\n")
    url = ""
    if first.upper().startswith("SOURCE:"):
        url, text = first.split(":", 1)[1].strip(" <>"), rest.strip()
    return (url if url.startswith("http") else (sources(response) or [""])[0]), text


def sources(response):
    """Addresses the answer cited, in order."""
    found = []
    for item in getattr(response, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            for note in getattr(part, "annotations", []) or []:
                url = getattr(note, "url", None)
                if url and url not in found:
                    found.append(url.split("?utm_source=")[0])
    return found


def from_web_search(name, site="", model=None):
    """(policy page address or "", text, date). Raises ValueError."""
    path = saved_path(name)
    if path.exists() and time.time() - path.stat().st_mtime < WEB_DAYS * 86400:
        saved = json.loads(path.read_text(encoding="utf-8"))
        return saved["url"], saved["text"], saved["date"]
    load_dotenv()
    client = OpenAI()
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    site = site or "unknown"
    try:
        with ThreadPoolExecutor(len(PARTS)) as pool:
            parts = list(pool.map(lambda part: read_part(client, model, PROMPT.format(name=name, site=site,
                                                                                      part=part)), PARTS))
    except Exception as error:
        raise ValueError(f"تعذر البحث في الإنترنت عن السياسة ({error.__class__.__name__}).")
    text = "\n\n".join(part_text for _, part_text in parts if part_text)
    if len(text) < MIN_CHARS:
        raise ValueError(NOT_FOUND)
    url = next((part_url for part_url, _ in parts if part_url), "")
    today = date.today().isoformat()
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": name, "url": url, "text": text, "date": today},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    return url, text, today


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("مثال: python -m src.policies.web_search Shein shein.com")
    saved_path(sys.argv[1]).unlink(missing_ok=True)  # from the command line, always search again
    try:
        page, text, when = from_web_search(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
    except ValueError as error:
        sys.exit(str(error))
    print(f"المصدر: {page or 'غير معروف'}\nالتاريخ: {when}\nطول النص: {len(text)} حرف\n")
    print(text[:800])
