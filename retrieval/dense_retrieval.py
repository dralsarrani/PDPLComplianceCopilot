import json
from pathlib import Path 
from sentence_transformers import SentenceTransformer, util

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_articles_v1.0.json"

with open(DATA_PATH, "r", encoding="utf-8") as file: 
    articles = json.load(file)

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

texts = [article["normalized_text"] for article in articles]

embeddings = model.encode(texts, convert_to_tensor=True)

query = "هل يحق للشركة مشاركة بياناتي الشخصية مع جهة أخرى دون موافقتي؟"
query_embedding = model.encode(query, convert_to_tensor=True)
scores = util.cos_sim(query_embedding, embeddings)[0] 
top_results = scores.topk(k=3)
for idx in top_results.indices:
     article = articles[idx.item()] 
     print("Article:", article["article_number"]) 
     print("ID:", article["record_id"]) 
     print("Text:", article["official_text"])
     print("--------------------")
     from pathlib import Path
import html

output = "<html><head><meta charset='UTF-8'></head>"
output += "<body dir='rtl' style='font-family:Arial; line-height:2;'>"
output += "<h2>نتائج البحث بالمعنى</h2>"

for idx in top_results.indices:
    article = articles[idx.item()]
    output += f"<h3>المادة: {html.escape(str(article['article_number']))}</h3>"
    output += f"<p>المعرف: {html.escape(str(article['record_id']))}</p>"
    output += f"<p>{html.escape(str(article['official_text']))}</p>"
    output += "<hr>"

output += "</body></html>"

output_path = Path("dense_results.html")
output_path.write_text(output, encoding="utf-8")

print("Results saved to:", output_path)
    



     
