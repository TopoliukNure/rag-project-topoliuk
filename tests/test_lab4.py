import json
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from src.hybrid_retrieval import RankedChunk, fuse_rankings
from src.product_budget import maximum_price
from src.rag_generation import (
    MODEL, Ollama, OllamaError, evidence_options, recommend,
    review_relevance, validate_output, validate_selection,
)
from src.sparse_retrieval import SparseIndex, tokenize

LAMP = RankedChunk(
    "flipkart:lamp:0000", "flipkart:lamp", "The LED lamp has a flexible neck.",
    "Sample lamp", "Computers", 500, "INR", 0.4, {},
)
QUOTE = "the LED lamp has a flexible neck"


def model_output(chunk_id=LAMP.chunk_id, evidence=LAMP.text):
    return json.dumps({
        "status": "ok",
        "recommendations": [{"chunk_id": chunk_id, "evidence": evidence}],
    })


def selection_output(chunk_id=LAMP.chunk_id, evidence_id="1.1"):
    return json.dumps({
        "status": "ok",
        "recommendations": [{"chunk_id": chunk_id, "evidence_id": evidence_id}],
    })


def review_output(*ids):
    return json.dumps({"accepted_chunk_ids": ids})


class RetrievalTests(unittest.TestCase):
    def test_tokenization(self):
        self.assertEqual(tokenize("ＵＳＢ-C / 26 L"), ["usb", "c", "26", "l"])

    def test_bm25_relevance_ties_filter_and_empty(self):
        rows = [
            {"chunk_id": "b", "text": "usb lamp"},
            {"chunk_id": "a", "text": "usb lamp"},
            {"chunk_id": "c", "text": "wooden chair"},
            {"chunk_id": "d", "text": "blue denims"},
            {"chunk_id": "e", "text": "garden hose"},
        ]
        index = SparseIndex(rows)
        hits = index.search("usb lamp", 5)
        self.assertEqual([h.chunk_id for h in hits], ["a", "b"])
        self.assertGreater(hits[0].score, 0)
        self.assertEqual(hits[0].score, hits[1].score)
        self.assertEqual(index.search("usb", 1, {"b"})[0].chunk_id, "b")
        self.assertEqual(index.search("unknown", 5), [])

    def test_rrf_ranks_and_deduplication(self):
        hits = fuse_rankings([["a", "a", "b"], ["b", "c"]], 3)
        self.assertEqual([key for key, _ in hits], ["b", "a", "c"])
        self.assertAlmostEqual(hits[0][1], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(hits[1][1], 1 / 61)

    def test_rrf_ties(self):
        hits = fuse_rankings([["z"], ["a"]], 1)
        self.assertEqual(hits, [("a", 1 / 61)])


class BudgetTests(unittest.TestCase):
    def test_maximum_and_rounding(self):
        cases = {
            "A lamp under INR 500": 499,
            "A lamp up to INR 500": 500,
            "A lamp below ₹500.50": 500,
            "A lamp for 500 INR or less": 500,
            "Lamp, budget of Rs. 1,200": 1200,
            "Shoes under 500": 499,
            "Budget of 250 for a lamp": 250,
            "Desk under INR 1,00,000": 99999,
            "Under INR 100, no more than 80 rupees": 80,
            "A bag weighing under 500 grams": None,
            "Rs Electrical aquarium filter": None,
        }
        for query, ceiling in cases.items():
            with self.subTest(query=query):
                self.assertEqual(maximum_price(query), ceiling)

    def test_unsupported_or_ambiguous_budget(self):
        for query in ("under USD 25", "budget INR five hundred",
                      "between INR 500 and INR 1000", "under INR 500oops",
                      "under INR 1,00"):
            with self.subTest(query=query), self.assertRaises(ValueError):
                maximum_price(query)

    def test_budget_filters_before_prompt(self):
        retriever, generator = Mock(), Mock()
        over_budget = replace(
            LAMP, chunk_id="over", document_id="over", discounted_price=501
        )
        retriever.hybrid.return_value = [over_budget, LAMP]
        generator.generate.side_effect = [
            selection_output(), review_output(LAMP.chunk_id),
        ]
        result = recommend("Lamp up to INR 500", retriever, generator)
        retriever.hybrid.assert_called_once_with("Lamp up to INR 500", 5, 500)
        self.assertEqual(result["context_chunk_ids"], [LAMP.chunk_id])
        self.assertEqual(result["recommendations"][0]["discounted_price"], 500)
        self.assertNotIn('"over"', str(generator.generate.call_args))

    def test_above_limit_rejected_after_generation(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_output(model_output(), {LAMP.chunk_id: LAMP}, 499)

    def test_no_candidates_skips_generation(self):
        retriever, generator = Mock(), Mock()
        retriever.hybrid.return_value = []
        result = recommend("Lamp under INR 100", retriever, generator)
        self.assertEqual(result["status"], "insufficient_context")
        generator.check.assert_not_called()
        generator.generate.assert_not_called()


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.context = {LAMP.chunk_id: LAMP}

    def test_verified_fields_and_no_model_price_override(self):
        result = validate_output(model_output(), self.context, None)
        self.assertEqual(result[0]["chunk_id"], LAMP.chunk_id)
        self.assertEqual(result[0]["discounted_price"], 500)
        self.assertEqual(result[0]["title"], "Sample lamp")
        with self.assertRaises(ValueError):
            validate_output(
                model_output().replace('"evidence"', '"price": 1, "evidence"'),
                self.context, None,
            )

    def test_rejects_bad_citations_and_fabricated_evidence(self):
        for raw in (model_output(chunk_id="flipkart:other:0000"),
                    model_output(chunk_id=1), model_output(evidence=QUOTE),
                    '{"status":"ok","recommendations":[]}', "not json"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_output(raw, self.context, None)

    def test_rejects_duplicates(self):
        data = json.loads(model_output())
        data["recommendations"].append(data["recommendations"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_output(json.dumps(data), self.context, None)

    def test_evidence_options_are_verbatim_and_exclude_prices(self):
        text = (
            "Aquarium filter. Mechanical filtration for salt water and fresh "
            "water. Includes a reusable sponge. Old price INR 698."
        )
        quotes = evidence_options(
            text, "Aquarium filter", "A filter for marine and fresh water"
        )
        self.assertTrue(any("salt water and fresh water" in q for q in quotes))
        for quote in quotes:
            self.assertIn(quote, text)
            self.assertNotIn("INR", quote)
            self.assertTrue(4 <= len(quote.split()) <= 32)

    def test_maximum_two_recommendations(self):
        selection = json.loads(model_output())
        selection["recommendations"] *= 3
        with self.assertRaisesRegex(ValueError, "recommendation count"):
            validate_output(json.dumps(selection), self.context, None)

    def test_title_only_evidence_rejected(self):
        title = "Sample adjustable desk lamp"
        lamp = replace(LAMP, title=title, text=title + ". " + LAMP.text)
        with self.assertRaisesRegex(ValueError, "beyond the title"):
            validate_output(
                model_output(evidence=title + "."), {lamp.chunk_id: lamp}, None
            )

    def test_review_discards_related_item_and_preserves_verified_quote(self):
        retriever, generator = Mock(), Mock()
        decoy = replace(
            LAMP, chunk_id="decoy", document_id="decoy",
            title="Floor Fan", text="The fan has a bendable stand.",
        )
        retriever.hybrid.return_value = [LAMP, decoy]
        selection = json.loads(selection_output())
        selection["recommendations"].append({
            "chunk_id": decoy.chunk_id, "evidence_id": "2.1",
        })
        generator.generate.side_effect = [
            json.dumps(selection), review_output(LAMP.chunk_id),
        ]
        result = recommend("A lamp with a flexible neck", retriever, generator)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["rejected_chunk_ids"], [decoy.chunk_id])
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertEqual(result["recommendations"][0]["evidence"], LAMP.text)

    def test_review_that_rejects_all_returns_insufficient_context(self):
        retriever, generator = Mock(), Mock()
        retriever.hybrid.return_value = [LAMP]
        generator.generate.side_effect = [selection_output(), review_output()]
        result = recommend("A lamp with Bluetooth", retriever, generator)
        self.assertEqual(result["status"], "insufficient_context")
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["rejected_chunk_ids"], [LAMP.chunk_id])

    def test_two_quotes_for_same_product_yield_one_recommendation(self):
        selection = json.loads(selection_output())
        selection["recommendations"].append({
            "chunk_id": LAMP.chunk_id, "evidence_id": "1.2",
        })
        verified = validate_selection(
            json.dumps(selection), self.context, None,
            {LAMP.chunk_id: {
                "1.1": LAMP.text, "1.2": "LED lamp has a flexible neck.",
            }},
        )
        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0]["evidence"], LAMP.text)

    def test_evidence_id_must_belong_to_cited_chunk(self):
        with self.assertRaisesRegex(ValueError, "belong to the cited chunk"):
            validate_selection(
                selection_output(evidence_id="2.1"), self.context, None,
                {LAMP.chunk_id: {"1.1": LAMP.text}},
            )

    def test_review_cannot_add_citations(self):
        proposals = validate_output(model_output(), self.context, None)
        generator = Mock()
        generator.generate.return_value = review_output("absent")
        with self.assertRaisesRegex(ValueError, "review citation"):
            review_relevance("A lamp", proposals, generator)

    def test_retry_then_success(self):
        retriever, generator = Mock(), Mock()
        retriever.hybrid.return_value = [LAMP]
        generator.generate.side_effect = [
            selection_output("missing"), selection_output(), review_output(LAMP.chunk_id)
        ]
        result = recommend("Find me a flexible lamp", retriever, generator)
        self.assertEqual(result["generation_attempts"], 2)
        self.assertEqual(len(result["validation_errors"]), 1)
        self.assertEqual(result["status"], "ok")

    def test_two_invalid_answers_fall_back(self):
        retriever, generator = Mock(), Mock()
        retriever.hybrid.return_value = [LAMP]
        generator.generate.return_value = selection_output("missing")
        result = recommend("Find me a flexible lamp", retriever, generator)
        self.assertEqual(result["status"], "unverified")
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(generator.generate.call_count, 2)

    def test_insufficient_context_contract(self):
        raw = '{"status":"insufficient_context","recommendations":[]}'
        self.assertEqual(validate_output(raw, self.context, None), [])


class OllamaTransportTests(unittest.TestCase):
    @patch("src.rag_generation.HTTPConnection")
    def test_preflight_and_chat_payload(self, connection):
        response = connection.return_value.getresponse.return_value
        response.status = 200
        response.read.side_effect = [
            json.dumps({"models": [{"name": MODEL}]}).encode(),
            json.dumps({"message": {"content": model_output()}}).encode(),
        ]
        client = Ollama()
        client.check()
        schema = {"type": "object"}
        self.assertEqual(client.generate([], schema), model_output())
        connection.assert_called_with("127.0.0.1", 11434, timeout=240)
        first, second = connection.return_value.request.call_args_list
        self.assertEqual(first.args, ("GET", "/api/tags"))
        self.assertEqual(second.args, ("POST", "/api/chat"))
        request = json.loads(second.kwargs["body"])
        self.assertEqual(request["model"], MODEL)
        self.assertEqual(request["format"], schema)
        self.assertFalse(request["stream"])

    @patch("src.rag_generation.HTTPConnection")
    def test_missing_local_model(self, connection):
        response = connection.return_value.getresponse.return_value
        response.status = 200
        response.read.return_value = b'{"models":[]}'
        with self.assertRaisesRegex(OllamaError, "ollama pull"):
            Ollama().check()


if __name__ == "__main__":
    unittest.main()

