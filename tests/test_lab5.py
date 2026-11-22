import unittest
from copy import deepcopy

from src.dense_retrieval import load_corpus
from src.evaluate_lab5 import (
    check_citations, check_label, grade_gold, jaccard, load_gold,
    recall_at_5, summarize,
)
from src.hybrid_retrieval import RankedChunk


class GoldSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = {row["chunk_id"]: row for row in load_corpus()}
        cls.gold = load_gold(cls.corpus)

    def test_frozen_corpus_grounded_split(self):
        self.assertEqual(len(self.gold), 36)
        self.assertEqual(sum(r["expected_refusal"] for r in self.gold), 4)
        self.assertEqual(
            sum(r["case_type"] == "synthesis" for r in self.gold), 6
        )

    def test_gold_price_and_fact_are_cross_checked(self):
        row = deepcopy(self.gold[0])
        row["expected_answer"]["products"][0]["discounted_price"] += 1
        with self.assertRaisesRegex(ValueError, "discounted_price"):
            check_label(row, self.corpus)
        row = deepcopy(self.gold[0])
        fact = row["expected_answer"]["key_features"][0]
        fact["verbatim"] = "Fabricated fact"
        with self.assertRaisesRegex(ValueError, "Unsupported gold fact"):
            check_label(row, self.corpus)

    def test_synthesis_requires_multiple_sources(self):
        row = dict(self.gold[0], case_type="synthesis")
        with self.assertRaisesRegex(ValueError, "multiple chunks"):
            check_label(row, self.corpus)


class MetricTests(unittest.TestCase):
    def test_multilabel_recall_and_top_five_cutoff(self):
        self.assertEqual(recall_at_5(["a", "a", "c", "d", "e", "b"],
                                     ["a", "b"]), 0.5)
        self.assertIsNone(recall_at_5(["a"], []))

    def test_jaccard_does_not_treat_both_refusals_as_agreement(self):
        self.assertIsNone(jaccard([], []))
        self.assertEqual(jaccard(["a", "a", "b"], ["b", "c"]), 1 / 3)

    def test_denominators_for_refusals_and_citations(self):
        def item(status, labels, citations=None):
            observation = {
                "result": {"status": status, "recommendations": []},
                "retrieved": [{"chunk_id": "a"}], "citations_valid": citations,
            }
            facts = [{"verbatim": "known fact", "source_chunk_ids": labels}]
            gold = {
                "expected_refusal": not labels,
                "relevant_chunk_ids": labels,
                "expected_answer": {"key_features": facts} if labels else {},
            }
            return grade_gold(gold, observation)

        rows = [
            item("ok", ["a", "b"], True), item("insufficient_context", ["b"]),
            item("insufficient_context", []), item("NOT RUN", []),
        ]
        result = summarize(rows)
        self.assertEqual(result["recall_at_5"]["value"], 0.25)
        self.assertEqual(result["recall_at_5"]["denominator"], 2)
        self.assertEqual(result["correct_refusals"]["denominator"], 1)
        self.assertEqual(result["positive_refusals"]["value"], 0.5)
        self.assertEqual(result["valid_citations"]["denominator"], 1)


class CitationMetricTests(unittest.TestCase):
    def test_requires_original_context_quote_and_budget(self):
        hit = RankedChunk(
            "test:1", "test", "It supports salt and fresh water.",
            "Example Filter", "Pet Supplies", 698, "INR", 1, {},
        )
        output = {
            "status": "ok", "context_chunk_ids": [hit.chunk_id],
            "recommendations": [{
                "chunk_id": hit.chunk_id, "document_id": hit.document_id,
                "evidence": hit.text, "title": hit.title,
                "category": hit.category,
                "discounted_price": 698, "currency": "INR",
            }]
        }
        self.assertTrue(check_citations(output, [hit], 698))
        self.assertFalse(check_citations(output, [hit], 697))
        self.assertFalse(check_citations(
            dict(output, context_chunk_ids=["unknown"]), [hit], None
        ))
        forged = deepcopy(output)
        forged["recommendations"][0]["evidence"] = "Supports very hot water."
        self.assertFalse(check_citations(forged, [hit], 698))
        fabricated = deepcopy(output)
        fabricated["recommendations"][0]["chunk_id"] = "unknown"
        self.assertFalse(check_citations(fabricated, [hit], 698))


if __name__ == "__main__":
    unittest.main()
