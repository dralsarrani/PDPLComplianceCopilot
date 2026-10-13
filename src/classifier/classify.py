"""classify(paragraphs): OPP-115 category labels for policy paragraphs.

English only. Arabic paragraphs must be translated first.
Model location: env CLASSIFIER_MODEL (HF repo id or local folder).
Private HF repo: set env HF_TOKEN.
"""
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.classifier.data import CATEGORIES, clean_text

MODEL_ID = os.environ.get("CLASSIFIER_MODEL", "dralsarrani/opp115-bert")
_cache = {}


def _load():
    if "model" not in _cache:
        token = os.environ.get("HF_TOKEN")
        _cache["tok"] = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=token)
        model.eval()
        _cache["model"] = model
    return _cache["tok"], _cache["model"]


def classify(paragraphs, threshold=0.5, batch_size=16, max_len=256):
    """Return one dict per paragraph:
    {"text": original, "labels": [category names], "scores": {category: prob}}"""
    if not paragraphs:
        return []
    tok, model = _load()
    results = []
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i + batch_size]
        enc = tok([clean_text(p) for p in batch], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        with torch.no_grad():
            probs = torch.sigmoid(model(**enc).logits).numpy()
        for text, p in zip(batch, probs):
            scores = {c: float(p[j]) for j, c in enumerate(CATEGORIES)}
            results.append({"text": text,
                            "labels": [c for c, s in scores.items() if s > threshold],
                            "scores": scores})
    return results