import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

articles = json.loads(
    (ROOT / "data/processed/legal_articles_v1.0.json").read_text(encoding="utf-8")
)

questions = json.loads(
    (ROOT / "data/evaluation/gold_qa.json").read_text(encoding="utf-8")
)["questions"]

questions = [q for q in questions if q["gold_article_ids"]]

models = {
    "Monolingual": "sentence-transformers/all-MiniLM-L6-v2",
    "Multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

for model_type, model_name in models.items():

    print(f"\nLoading {model_type} model...")
    model = SentenceTransformer(model_name)

    for language in ["ar", "en"]:

        selected = [a for a in articles if a["language"] == language]
        texts = [a["normalized_text"] for a in selected]

        embeddings = model.encode(
            texts, normalize_embeddings=True
        )

        recalls = []
        mrrs = []

        for q in questions:
            if q["language"] != language:
                continue

            query_embedding = model.encode(
                [q["question"]], normalize_embeddings=True
            )[0]

            scores = embeddings @ query_embedding

            ranked = np.argsort(scores)[::-1]
            ids = [selected[i]["record_id"] for i in ranked]

            gold = set(q["gold_article_ids"])

            recall = len(set(ids[:5]) & gold) / len(gold)

            mrr = next(
                (1 / rank for rank, aid in enumerate(ids, 1) if aid in gold),
                0
            )

            recalls.append(recall)
            mrrs.append(mrr)

        print(
            f"{language}: Recall@5={100 * np.mean(recalls):.1f}% "
            f"| MRR={np.mean(mrrs):.3f}"
        )