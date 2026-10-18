import json
from pathlib import Path
from text_processing import tokenize_arabic
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder 
# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Gold evaluation dataset
GOLD_PATH = PROJECT_ROOT / "data" / "evaluation" / "gold_qa.json"

# Load evaluation questions
with open(GOLD_PATH, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

questions = gold_data["questions"]

print(f"Total evaluation questions: {len(questions)}")

for item in questions:
    print(f"\nQuestion ID: {item['id']}")
    print(f"Question: {item['question']}")
    print(f"Gold Articles: {item['gold_article_ids']}")  

    # Load legal articles
ARTICLES_PATH = PROJECT_ROOT / "data" / "processed" / "legal_articles_v1.0.json"

with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
    articles = json.load(f)

# Prepare article texts
texts = [article["normalized_text"] for article in articles]

print(f"\nTotal legal articles: {len(articles)}") 
  

# Prepare BM25 model
tokenized_corpus = [tokenize_arabic(text) for text in texts]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 model ready!")  
# Prepare Dense Retrieval model
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

embeddings = model.encode(texts, convert_to_tensor=True)

print("Dense model ready!")
# Prepare Reranking model
reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)

print("Reranking model ready!")
recall_scores = []
reciprocal_ranks = []
rerank_recall_scores = []
rerank_reciprocal_ranks = [] 
  
 # Evaluate Hybrid Retrieval using RRF

for item in questions:
    query = item["question"]

    # BM25 scores
    bm25_scores = bm25.get_scores(tokenize_arabic(query))

    # Dense scores
    query_embedding = model.encode(query, convert_to_tensor=True)
    dense_scores = util.cos_sim(query_embedding, embeddings)[0]

    # Rank articles
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

    # Reciprocal Rank Fusion
    rrf_scores = [0.0] * len(articles)

    for ranking in [bm25_ranking, dense_ranking]:
        for rank, idx in enumerate(ranking):
            rrf_scores[idx] += 1 / (60 + rank + 1)

    # Top 10 articles
    top_indices = sorted(
        range(len(articles)),
        key=lambda i: rrf_scores[i],
        reverse=True
    )[:10]

    print(f"\nHybrid results for {item['id']}:")

    for rank, idx in enumerate(top_indices, start=1):
        print(f"Rank {rank}: {articles[idx]['record_id']}")  

            # Prepare Hybrid top 10 articles for Reranking
    top_articles = [articles[idx] for idx in top_indices]

    pairs = [
        (
            query,
            f"عنوان المادة: {article['article_title']}\n"
            f"النص القانوني: {article['official_text']}"
        )
        for article in top_articles
    ]

    # Calculate Reranking scores
    reranking_scores = reranker.predict(pairs)

    # Sort articles by Reranking scores
    reranked_results = sorted(
        zip(top_articles, reranking_scores),
        key=lambda x: float(x[1]),
        reverse=True
    )

    print(f"\nReranking results for {item['id']}:")

    for rank, (article, score) in enumerate(reranked_results[:5], start=1):
        print(f"Rank {rank}: {article['record_id']} | Score: {float(score):.4f}")
            # Evaluate Reranking
    gold_ids = set(item["gold_article_ids"])

    reranked_ids = [
        article["record_id"]
        for article, score in reranked_results
    ]

    # Reranking Recall@5
    rerank_top5 = set(reranked_ids[:5])

    if gold_ids:
        rerank_recall = len(rerank_top5 & gold_ids) / len(gold_ids)
        rerank_recall_scores.append(rerank_recall)

    # Reranking Reciprocal Rank
    rerank_rr = 0.0

    for rank, article_id in enumerate(reranked_ids, start=1):
        if article_id in gold_ids:
            rerank_rr = 1 / rank
            break

    rerank_reciprocal_ranks.append(rerank_rr)

    print(f"\nReranking Evaluation for {item['id']}:")
    if gold_ids:
        print(f"Recall@5: {rerank_recall:.2%}")
    else:
        print("Recall@5: N/A")

    print(f"Reciprocal Rank: {rerank_rr:.4f}") 
            # Gold reference articles
    gold_ids = set(item["gold_article_ids"])

    # Retrieved article IDs
    retrieved_ids = [
        articles[idx]["record_id"]
        for idx in top_indices
    ]

    # Recall@5
    top5_ids = set(retrieved_ids[:5])

    recall_at_5 = (
        len(top5_ids & gold_ids) / len(gold_ids)
        if gold_ids else None
    )

    # Reciprocal Rank
    reciprocal_rank = 0.0

    for rank, article_id in enumerate(retrieved_ids, start=1):
        if article_id in gold_ids:
            reciprocal_rank = 1 / rank
            break
    print(f"\nEvaluation for {item['id']}:")
    print(f"Recall@5: {recall_at_5:.2%}" if recall_at_5 is not None else "Recall@5: N/A")
    print(f"Reciprocal Rank: {reciprocal_rank:.4f}")
    # Store evaluation scores
    if recall_at_5 is not None:
        recall_scores.append(recall_at_5)

    reciprocal_ranks.append(reciprocal_rank)


# Calculate average evaluation metrics
average_recall = (
    sum(recall_scores) / len(recall_scores)
    if recall_scores else 0.0
)

mrr = (
    sum(reciprocal_ranks) / len(reciprocal_ranks)
    if reciprocal_ranks else 0.0
)

print("\n===== Hybrid Evaluation Summary =====")
print(f"Total Questions: {len(questions)}")
print(f"Mean Recall@5: {average_recall:.2%}")
print(f"MRR: {mrr:.4f}")
# Calculate Reranking averages
rerank_average_recall = (
    sum(rerank_recall_scores) / len(rerank_recall_scores)
    if rerank_recall_scores else 0.0
)

rerank_mrr = (
    sum(rerank_reciprocal_ranks) / len(rerank_reciprocal_ranks)
    if rerank_reciprocal_ranks else 0.0
)

print("\n===== Hybrid vs Reranking =====")
print(f"Total Questions: {len(questions)}")
print(f"Hybrid Recall@5: {average_recall:.2%}")
print(f"Reranking Recall@5: {rerank_average_recall:.2%}")
print(f"Hybrid MRR: {mrr:.4f}")
print(f"Reranking MRR: {rerank_mrr:.4f}")    
    