import json 
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_articles_v1.0.json"
with open(DATA_PATH, "r", encoding="utf-8") as file:
     articles = json.load(file)

from rank_bm25 import BM25Okapi
texts = [article["normalized_text"] for article in articles]
tokenized_corpus = [text.split() for text in texts]
bm25 = BM25Okapi(tokenized_corpus)
query = input("Enter your question: ")
tokenized_query = query.split()
results = bm25.get_top_n(tokenized_query, articles, n=3)

for article in results:
    print("Article:", article["article_number"])
    print("ID:", article["record_id"])
    print("--------------------")