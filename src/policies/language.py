"""
The language a policy text is written in, from its letters (no OpenAI).

    text_language("سياسة الخصوصية ...")   -> "ar"
    text_language("Privacy policy ...")     -> "en"

A text counts as Arabic when at least 30% of its letters are Arabic: an Arabic
policy often has English names and links in it, an English one almost no Arabic.
"""
import re

ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")
ARABIC_SHARE = 0.3


def text_language(text):
    arabic, latin = len(ARABIC.findall(text or "")), len(LATIN.findall(text or ""))
    if arabic + latin == 0:
        return ""
    return "ar" if arabic / (arabic + latin) >= ARABIC_SHARE else "en"
