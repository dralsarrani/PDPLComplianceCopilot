from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)

print("Reranking model loaded successfully!") 
import json
 
with open("hybrid_top10.json", "r", encoding="utf-8") as f:
    data = json.load(f)

query = data["query"]
articles = data["articles"]

print("Question loaded successfully!")
print("Number of articles:", len(articles))
# 3. Prepare question-article pairs
pairs = [
    (
        query,
        f"عنوان المادة: {article['article_title']}\n"
        f"النص القانوني: {article['official_text']}"
    )
    for article in articles
]  

# 4. Calculate relevance scores
scores = model.predict(pairs)

# 5. Sort articles by relevance
ranked_results = sorted(
    zip(articles, scores),
    key=lambda x: float(x[1]),
    reverse=True
)
# Keep the original ranking before relevance filtering
all_ranked_results = ranked_results.copy()
# Prepare question-article pairs for relevance filtering
filter_pairs = [
    (query, article["normalized_text"])
    for article, score in all_ranked_results
]
   
# Save all reranking scores for evaluation
import csv  

with open("reranking_scores.csv", "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["rank", "article_id", "article_number", "score", "relevant"])

    for rank, (article, score) in enumerate(ranked_results, start=1):
        writer.writerow([
            rank,
            article["record_id"],
            article["article_number"],
            round(float(score), 4),
            ""
        ])

print("Evaluation scores saved successfully!")  

# 6. Display reranked results
print("\nReranking Results:")

for rank, (article, score) in enumerate(ranked_results[:5], start=1):   

    print(
        f"Rank {rank} | "
        f"Article {article['article_number']} | "
        f"Score: {float(score):.4f}"
    )   
    # 7. Save top 5 reranked results as HTML
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
<h1>نتائج البحث بعد إعادة الترتيب - Reranking</h1>
"""

output += f"<p><b>السؤال:</b> {html.escape(query)}</p>"

for rank, (article, score) in enumerate(ranked_results[:5], start=1):
    output += '<div class="article">'
    output += f"<h3>الترتيب {rank} - المادة {html.escape(str(article['article_number']))}</h3>"
    output += f"<p><b>المعرف:</b> {html.escape(str(article['record_id']))}</p>"
    output += f"<p>{html.escape(str(article['official_text']))}</p>"
    output += "</div>"

output += "</body></html>"

output_path = Path("reranked_results.html")
output_path.write_text(output, encoding="utf-8")

print("Reranked results saved to:", output_path)
 
 