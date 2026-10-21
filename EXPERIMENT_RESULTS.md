# PDPLComplianceCopilot – Experimental Evaluation

## 1. Overview

This report summarizes the experimental evaluation of the PDPLComplianceCopilot retrieval system using the gold evaluation dataset.

The experiments examine retrieval strategies, document chunking, multilingual performance, and out-of-scope detection.

**Evaluation dataset:**
- Total questions: 40
- Answerable questions: 34
- Unanswerable questions: 6
- Languages: Arabic and English

**Evaluation metrics:**
- Recall@5: Proportion of relevant gold articles retrieved within the top five results.
- MRR: Mean Reciprocal Rank of the first relevant retrieved article.
- Correct Abstention: Percentage of unanswerable questions correctly rejected.
- False Abstention: Percentage of answerable questions incorrectly rejected.

---

## 2. Experiment E1 – Retrieval Strategy Comparison

### Objective
Evaluate the effect of different retrieval strategies on legal article retrieval accuracy.

### Results

| Retrieval Method | Recall@5 | MRR |
|---|---:|---:|
| BM25 | 62.7% | 0.678 |
| Hybrid (BM25 + Dense) | 56.9% | 0.635 |
| Hybrid + Reranker | **83.3%** | **0.881** |

### Findings

The Hybrid + Reranker approach achieved the highest retrieval performance.

Adding the reranker improved Recall@5 by 26.4 percentage points compared with Hybrid retrieval alone.

Hybrid retrieval without reranking performed below the BM25 baseline, indicating that combining retrieval methods does not automatically improve ranking quality.

### Limitation

This experiment evaluates retrieval strategies only. It does not include the full comparison between LLM-only answering, simple RAG, and advanced RAG.

---

## 3. Experiment E2 – Chunking Strategy Comparison

### Objective
Compare article-based segmentation against fixed-length chunking using BM25 retrieval.

Fixed-length chunks contained up to 200 words.

### Results

| Language | Chunking Method | Recall@5 | MRR |
|---|---|---:|---:|
| Arabic | Article-based | 47.1% | 0.550 |
| Arabic | Fixed-length | **52.9%** | **0.608** |
| English | Article-based | 65.7% | 0.718 |
| English | Fixed-length | **68.6%** | **0.723** |

### Findings

Fixed-length chunking achieved higher Recall@5 and MRR in both languages.

For Arabic questions, Recall@5 improved by 5.8 percentage points.

For English questions, Recall@5 improved by 2.9 percentage points.

### Limitation

The comparison uses BM25 retrieval rather than the complete Hybrid + Reranker pipeline.

Each fixed-length chunk retains its original article ID, and duplicate article IDs are removed before evaluation.

---

## 4. Experiment E3 – Monolingual vs Multilingual Models

### Objective
Compare an English-focused embedding model with a multilingual embedding model for Arabic and English legal questions.

**Models:**
- Monolingual: `sentence-transformers/all-MiniLM-L6-v2`
- Multilingual: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

### Results

| Language | Model | Recall@5 | MRR |
|---|---|---:|---:|
| Arabic | Monolingual | 2.9% | 0.053 |
| Arabic | Multilingual | **32.4%** | **0.383** |
| English | Monolingual | **75.5%** | **0.869** |
| English | Multilingual | 72.5% | 0.703 |

### Findings

The multilingual model significantly outperformed the English-focused model on Arabic questions.

Arabic Recall@5 increased from 2.9% to 32.4%.

For English questions, the English-focused model achieved better performance.

These results demonstrate the importance of selecting an embedding model that supports the language of the legal corpus and user queries.

### Limitation

This experiment evaluates dense retrieval independently, without BM25 fusion or reranking.

---

## 5. Experiment E5 – Out-of-Scope Detection

### Objective
Evaluate whether applying a minimum retrieval-score threshold improves the detection of questions not covered by the legal corpus.

### Results

| Threshold | Correct Abstention | False Abstention |
|---|---|---|
| None | 0/6 (0%) | 0/34 (0%) |
| 0.3 | 6/6 (100%) | 8/34 (23.5%) |
| 0.5 | 6/6 (100%) | 8/34 (23.5%) |
| 0.7 | 6/6 (100%) | 8/34 (23.5%) |

### Findings

Applying a retrieval-score threshold correctly identified all six unanswerable questions in this evaluation.

However, the threshold also rejected eight answerable questions.

This demonstrates a trade-off between rejecting unsupported questions and preserving valid answers.

### Limitation

The experiment tests a retrieval-score-based abstention rule rather than the complete answer-generation and citation-verification pipeline.

The no-threshold condition is a retrieval-score baseline and does not represent the existing system's full abstention behavior.

The tested thresholds are experimental values and should not be interpreted as calibrated confidence probabilities.

---

## 6. Error Analysis

The experiments revealed several important limitations.

**Retrieval errors:** Hybrid retrieval without reranking performed worse than BM25 alone. This suggests that initial retrieval rankings may contain relevant articles at lower positions.

**Arabic language errors:** The English-focused embedding model performed poorly on Arabic legal questions, highlighting the importance of multilingual representations.

**Chunking errors:** Article-based segmentation performed below fixed-length chunking in the BM25 comparison. Long articles may contain relevant passages that are less prominent when the entire article is indexed as one unit.

**Abstention errors:** The retrieval-score threshold rejected eight answerable questions, demonstrating that score-based filtering can remove valid answers.

These are aggregate observations from the experiment results. A question-level error classification was not performed.

---

## 7. Conclusion

The experimental evaluation demonstrates that:

1. Hybrid retrieval combined with reranking achieved the strongest retrieval performance among the tested E1 configurations.
2. Fixed-length chunking outperformed article-based segmentation in the BM25-only comparison.
3. Multilingual embeddings substantially improved Arabic retrieval performance.
4. Retrieval-score thresholds improved out-of-scope detection but introduced false rejections.

The findings support the use of reranking and multilingual embeddings in the PDPLComplianceCopilot system.

However, the results should be interpreted within the limitations of the experimental configurations. The E1 and E5 evaluations do not fully cover the end-to-end comparisons described in the project requirements.

All experiments were evaluated on the available gold questions. Results may be optimistic if the system was previously tuned using the same dataset.

---

## 8. Reproducibility

The experimental scripts are available in the `retrieval` directory.

```bash
python -m retrieval.experiment_e1
python -m retrieval.experiment_e2
python -m retrieval.experiment_e3
python -m retrieval.experiment_e5
```

**Experiment files:**
- `retrieval/experiment_e1.py`
- `retrieval/experiment_e2.py`
- `retrieval/experiment_e3.py`
- `retrieval/experiment_e5.py`