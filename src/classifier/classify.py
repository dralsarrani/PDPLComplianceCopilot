"""classify(paragraphs): OPP-115 category labels for policy paragraphs.

English only. Arabic paragraphs must be translated first.
Model location: env CLASSIFIER_MODEL (HF repo id or local folder).
Private HF repo: set env HF_TOKEN.
"""
import os, joblib
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from src.classifier.data import CATEGORIES, clean_text

MODEL_ID = os.environ.get("CLASSIFIER_MODEL", "dralsarrani/opp115-bert")
_cache = {}

BASELINE_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "classifier" / "baseline.joblib"
_baseline = None


def _load():
    if "model" not in _cache:
        _cache["tok"] = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        model.eval()
        _cache["model"] = model
    return _cache["tok"], _cache["model"]

def _bert_probs(texts, batch_size, max_len):
    tok, model = _load()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        with torch.no_grad():
            out.append(torch.sigmoid(model(**enc).logits).numpy())
    return np.vstack(out)


def _baseline_probs(texts):
    global _baseline
    if _baseline is None:
        _baseline = joblib.load(BASELINE_PATH)
    return _baseline.predict_proba(texts)


def classify(paragraphs, threshold=0.5, batch_size=16, max_len=256):
    """Return one dict per paragraph:
    {"text", "labels", "scores", "model"}  ("model" = "bert" or "baseline")"""
    if not paragraphs:
        return []
    texts = [clean_text(p) for p in paragraphs]
    try:
        probs = _bert_probs(texts, batch_size, max_len)
        used = "bert"
    except Exception as e:
        print(f"[classify] BERT unavailable ({type(e).__name__}); using TF-IDF baseline")
        probs = _baseline_probs(texts)
        used = "baseline"
    results = []
    for text, p in zip(paragraphs, probs):
        scores = {c: float(p[j]) for j, c in enumerate(CATEGORIES)}
        results.append({"text": text,
                        "labels": [c for c, s in scores.items() if s > threshold],
                        "scores": scores,
                        "model": used})
    return results
