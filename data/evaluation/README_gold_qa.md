# Gold Q&A — Development (40 questions)

Source: the team-reviewed 60-question Track B Gold Q&A set.
This directory contains only the 40 **development** questions. Keep the 20-question **final test holdout** outside this public repository to avoid leakage.

## Files
- `gold_qa.json`: **drop-in replacement** for the existing one-question sample. Compatible with `retrieval/batch_evaluation.py`: top-level `questions`; each record has `id`, `language` (`ar` or `en`), `question`, `answerable` boolean, `gold_article_ids` **array**, `reference_answer`. Other fields are documentation metadata.
- `gold_qa_development.csv`: human-readable export of the same 40 questions. It is not the direct input consumed by `batch_evaluation.py`.

The legal corpus already exists in `data/processed/legal_articles_v1.0.json`. No need to copy or overwrite it.

## From repository root
```bash
python retrieval/batch_evaluation.py
```

**Metric caveat:** The current `retrieval/batch_evaluation.py` appends zero reciprocal-rank scores for questions where `gold_article_ids` is empty (`answerable=false`), which depresses MRR. Exclude unanswerable questions from retrieval Recall@5 / MRR denominators; separately evaluate abstention/false refusal on answerable/unanswerable questions.

The old `gold_qa.json` contained one sample question Q001 about third-party disclosure; this update intentionally replaces it with the development set, where Q001 is a question on the definition of personal data.

`VERIFIED` is the team's review status, not an official legal certification.
