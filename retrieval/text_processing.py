import re
from nltk.stem.isri import ISRIStemmer

stemmer = ISRIStemmer()

def tokenize_arabic(text):
    text = text.lower()

    # Remove Arabic diacritics
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # Normalize Arabic letters
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")

    # Extract words
    words = re.findall(r"[\u0621-\u064A]+|[a-zA-Z0-9]+", text)

    return [
        stemmer.stem(word) if re.fullmatch(r"[\u0621-\u064A]+", word) else word
        for word in words
    ]   