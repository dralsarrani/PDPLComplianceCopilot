# PDPL requirements checklist

`pdpl_checklist.csv` lists what the PDPL requires a privacy policy to state. The Gap Analyser checks each policy against it.

Each requirement is taken from the article text in `data/processed/legal_articles_v1.0.json`, mainly:
- PDPL Article 12: the privacy policy and what it must specify
- PDPL Article 13: what the data subject must be told at collection
- Implementing Regulation Article 4: the right to be informed

| Column | Meaning |
|---|---|
| req_id | R01 to R17 |
| requirement_en / requirement_ar | The requirement in English and Arabic |
| articles | Source articles, e.g. `PDPL-A13(4)` = PDPL Article 13, paragraph 4; `IR` = Implementing Regulation |
| applies | `always`, or `conditional` (only required when `condition` is true) |
| condition | When a conditional requirement applies |
| opp115_categories | OPP-115 categories from the 3A classifier whose paragraphs are checked for this requirement |
| condition_ar | The condition in Arabic, shown in the app |

Left out on purpose, because the law does not require them in the privacy policy itself: complaints to SDAIA (PDPL Art. 34), breach notification (PDPL Art. 20), direct marketing (IR Art. 29).
OPP-115 categories with no requirement: Policy Change, Do Not Track.

Draft by 3B, to be reviewed by person 1.
