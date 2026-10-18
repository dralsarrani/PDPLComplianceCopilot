from text_processing import tokenize_arabic
import json
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
  


  

# 1. Load legal articles
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_articles_v1.0.json"

with open(DATA_PATH, "r", encoding="utf-8") as file:
    articles = json.load(file)

# 2. Prepare texts
texts = [article["normalized_text"] for article in articles]

# 3. BM25 Retrieval
tokenized_corpus = [tokenize_arabic(text) for text in texts]
bm25 = BM25Okapi(tokenized_corpus)

# 4. Dense Retrieval
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

embeddings = model.encode(texts, convert_to_tensor=True)

# 5. User question
query = input("Enter your question: ")

# 6. BM25 scores
bm25_scores = bm25.get_scores(tokenize_arabic(query))

# 7. Dense scores
query_embedding = model.encode(query, convert_to_tensor=True)
dense_scores = util.cos_sim(query_embedding, embeddings)[0]

# 8. Combine results using RRF
bm25_ranking = sorted(
    range(len(articles)),
    key=lambda i: bm25_scores[i],
    reverse=True
)

dense_ranking = sorted(
    range(len(articles)),
    key=lambda i: dense_scores[i].item(),
    reverse=True
)

rrf_scores = [0.0] * len(articles)

for ranking in [bm25_ranking, dense_ranking]:
    for rank, idx in enumerate(ranking):
        rrf_scores[idx] += 1 / (60 + rank + 1)

# 9. Get top 10  results for reranking  
top_indices = sorted(
    range(len(articles)),
    key=lambda i: rrf_scores[i],
    reverse=True
)[:10]
# Check Article 15 ranking

# 10. Display results
for idx in top_indices:
    article = articles[idx]

    print("Article:", article["article_number"])
    print("ID:", article["record_id"])
    print("Text:", article["official_text"])
    print("--------------------") 

    # Save Hybrid Retrieval results as HTML
import html
from pathlib import Path

output = """
<html>
<head>
<meta charset="UTF-8">
<style>
body {
    direction: rtl;
    text-align: right;
    font-family: Arial;
    line-height: 2;
    margin: 40px;
    background-color: #f8f9fa;
}
.article {
    background: white;
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 10px;
    border: 1px solid #ddd;
}
h1 { color: #234567; }
h3 { color: #176b87; }
</style>
</head>
<body>
<h1>نتائج البحث الهجين Hybrid Retrieval</h1>
"""

for idx in top_indices:
    article = articles[idx]
    output += '<div class="article">'
    output += f"<h3>المادة: {html.escape(str(article['article_number']))}</h3>"
    output += f"<p><b>المعرف:</b> {html.escape(str(article['record_id']))}</p>"
    output += f"<p>{html.escape(str(article['official_text']))}</p>"
    output += "</div>" 

output += "</body></html>"

output_path = Path("hybrid_results.html")
output_path.write_text(output, encoding="utf-8")

print("Hybrid results saved to:", output_path)

# Save top 10 Hybrid results for Reranking

import json

top_articles = [articles[i] for i in top_indices]

with open("hybrid_top10.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "query": query,
            "articles": top_articles
        },
        f,
        ensure_ascii=False,
        indent=4
    )

print("Top 10 Hybrid results saved successfully!")  
import subprocess
import sys

subprocess.run(
    [sys.executable, "retrieval/reranking.py"],
    check=True
) 