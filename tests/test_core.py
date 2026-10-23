"""
Quick checks that the core parts of the system load and behave as expected.
No OpenAI calls and no models are needed, so it runs in a few seconds.

Run from the project root:
    python -m unittest discover tests -v
"""
import json
import unittest
from pathlib import Path

from retrieval.answer import locate_quote
from retrieval.citation_check import ID_PATTERN, check_citations, normalise_id
from src.gap.analyse_v0 import load_checklist, normalise, quote_in_policy
from src.gap.highlight import find_spans, highlight_html

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS = json.loads((PROJECT_ROOT / "data" / "processed" / "legal_articles_v1.0.json").read_text(encoding="utf-8"))
GOLD = json.loads((PROJECT_ROOT / "data" / "evaluation" / "gold_qa.json").read_text(encoding="utf-8"))["questions"]
ARTICLES = {a["record_id"]: a for a in CORPUS}


class TestData(unittest.TestCase):
    def test_corpus_ids(self):
        self.assertEqual(len(ARTICLES), len(CORPUS), "record ids must be unique")
        for rid, a in ARTICLES.items():
            self.assertRegex(rid, ID_PATTERN)
            self.assertTrue(a["official_text"].strip(), f"{rid} has no text")

    def test_corpus_has_both_languages(self):
        for rid in ARTICLES:
            other = rid[:-2] + ("EN" if rid.endswith("AR") else "AR")
            self.assertIn(other, ARTICLES, f"{rid} has no {other[-2:]} version")

    def test_gold_questions(self):
        self.assertGreaterEqual(len(GOLD), 40)
        self.assertGreaterEqual(sum(not q["gold_article_ids"] for q in GOLD), 6, "need unanswerable questions")
        for q in GOLD:
            for gid in q["gold_article_ids"]:
                self.assertIn(gid, ARTICLES, f"{q['id']} points to a missing article {gid}")

    def test_checklist(self):
        checklist = load_checklist()
        self.assertEqual(list(checklist["req_id"]), [f"R{i:02d}" for i in range(1, 18)])
        self.assertTrue(all(checklist["articles"].str.strip()), "every requirement needs its PDPL article")


class TestCitations(unittest.TestCase):
    def test_normalise_id(self):
        self.assertEqual(normalise_id("[pdpl-a4-en]"), "PDPL-A04-EN")
        self.assertEqual(normalise_id(" IR-A12-AR "), "IR-A12-AR")

    def test_check_citations_without_llm(self):
        sentences = [{"text": "Quoted and retrieved.", "citations": ["PDPL-A01-AR"]},
                     {"text": "Invented article.", "citations": ["PDPL-A99-AR"]},
                     {"text": "Real but not retrieved.", "citations": ["PDPL-A02-AR"]},
                     {"text": "No citation.", "citations": []}]
        result = check_citations(sentences, ["PDPL-A01-AR"], quoted={(0, "PDPL-A01-AR")})
        statuses = [[c["status"] for c in row["citations"]] for row in result["sentences"]]
        self.assertEqual(statuses, [["supported"], ["not_in_corpus"], ["not_retrieved"], []])
        self.assertEqual(result["valid_citations"], ["PDPL-A01-AR"])
        self.assertFalse(result["ok"])

    def test_quote_lookup(self):
        text = ARTICLES["PDPL-A01-AR"]["official_text"]
        self.assertIsNotNone(locate_quote("كل بيان -مهما كان مصدره أو شكله- من شأنه أن يؤدي إلى معرفة الفرد", text))
        self.assertIsNone(locate_quote("يجب حذف البيانات بعد سبعة أيام من جمعها", text))


class TestGapAnalyser(unittest.TestCase):
    def test_quote_in_policy(self):
        policy = normalise("We keep your data for 5 years. You can ask us to delete it.")
        self.assertTrue(quote_in_policy("We keep your data for 5 years.", policy))
        self.assertTrue(quote_in_policy("We keep your data ... ask us to delete it", policy))
        self.assertFalse(quote_in_policy("We never share your data.", policy))

    def test_highlight_quotes_in_policy(self):
        text = "We keep your data\nfor 5 years. You can ask us to <delete> it."
        self.assertEqual(len(find_spans(text, "we keep your data for 5 years ... ask us")), 2)
        self.assertEqual(find_spans(text, "We never share your data."), [])
        report = [{"req_id": "R11", "status": "covered", "evidence": "We keep your data for 5 years."},
                  {"req_id": "R02", "status": "missing", "evidence": ""}]
        marked = highlight_html(text, report)
        self.assertEqual(marked.count("<mark"), 1)
        self.assertIn("&lt;delete&gt;", marked)


if __name__ == "__main__":
    unittest.main()
