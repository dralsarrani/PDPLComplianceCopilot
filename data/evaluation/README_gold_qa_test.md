# Gold Q&A Test Set — 20 Questions

This dataset consists of 20 held-out evaluation questions (10 Arabic, 10 English); 14 are answerable from the scoped PDPL/IR/TR corpus and 6 are unanswerable.

- `gold_qa_test.json`: **repository-compatible** format (`{"questions": [...]}`), with lower-case `language` and list-valued `gold_article_ids`; intended for evaluation code.
- `gold_qa_test.csv`: table-friendly full gold set for inspecting IDs, references, quotes, and metadata.
- The existing `gold_qa.json` remains the 40-question development set; **do not overwrite it**.
- The existing `citation_cases.json` should not be changed.

**Important: Public exposure / leakage** — This repository is public. Publishing questions, gold article IDs, and reference answers makes this 20-question set visible to the whole team. If it is intended as an *unseen final holdout*, keep these files private and run them only after tuning is frozen. If you need this set on GitHub now, document that it was exposed before final evaluation and do not claim a truly blind held-out score.

**Current evaluation script** — `retrieval/batch_evaluation.py` currently hardcodes the path `data/evaluation/gold_qa.json`. Adding `gold_qa_test.json` to the repository will NOT automatically make the script evaluate it. Update the script to accept a gold-data-file parameter (e.g. `--gold data/evaluation/gold_qa_test.json`), then report test results separately; do not overwrite the development JSON.

`VERIFIED` denotes review/approval in the project, not independent legal certification. This corpus is a snapshot; verify any time-sensitive legal questions against official updates before claiming current legal validity.
