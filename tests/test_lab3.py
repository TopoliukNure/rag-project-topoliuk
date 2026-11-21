import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from src.dense_retrieval import (
    DIMENSIONS,
    checked_vectors,
    connect,
    load_corpus,
    query_vector,
    read_jsonl,
    search,
    unique_records,
)
from src.evaluate_lab3 import EVAL_SET, load_eval_set, score


class RetrievalTests(unittest.TestCase):
    def test_corpus_and_labels(self):
        chunks = load_corpus()
        self.assertEqual(len(chunks), 1208)
        self.assertEqual(len({row["document_id"] for row in chunks}), 1000)
        labels = load_eval_set(EVAL_SET, {row["chunk_id"] for row in chunks})
        self.assertEqual(len(labels), 20)

    def test_recall(self):
        ranks = [1, 5, 6, 10, None]
        self.assertEqual(score(ranks, 5), {
            "recall": 0.4, "successful": 2, "unsuccessful": 3
        })
        self.assertEqual(score(ranks, 10), {
            "recall": 0.8, "successful": 4, "unsuccessful": 1
        })
        with self.assertRaises(ValueError):
            score([], 5)

    def test_vector_validation(self):
        vectors = np.eye(2, DIMENSIONS, dtype=np.float32)
        np.testing.assert_array_equal(checked_vectors(vectors, 2), vectors)
        for wrong in (vectors[:, :-1], vectors * 2, vectors * np.nan):
            with self.assertRaises(ValueError):
                checked_vectors(wrong, 2)

    def test_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            unique_records([{"id": "a"}, {"id": "a"}], "id")

    def test_bad_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl"
            path.write_text('{"id": 1}\n[]\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, ":2:"):
                read_jsonl(path)

    def test_invalid_or_missing_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, "Index not found"):
                connect(path, "ef10", "fake")

    def test_empty_query_precedes_encoding(self):
        encoder = Mock()
        with self.assertRaisesRegex(ValueError, "empty"):
            search(Mock(), encoder, " \n ")
        encoder.encode.assert_not_called()

    def test_distance_conversion_and_invalid_metadata(self):
        record = load_corpus()[0]
        metadata = {k: v for k, v in record.items() if k != "text"}
        collection = Mock()
        collection.query.return_value = {
            "ids": [[record["chunk_id"]]],
            "documents": [[record["text"]]],
            "metadatas": [[metadata]],
            "distances": [[0.25]],
        }
        vector = np.eye(1, DIMENSIONS, dtype=np.float32)[0]
        hit = query_vector(collection, vector, 1)[0]
        self.assertEqual(hit.similarity, 0.75)
        metadata["discounted_price"] = -1
        with self.assertRaisesRegex(ValueError, "invalid metadata"):
            query_vector(collection, vector, 1)


if __name__ == "__main__":
    unittest.main()

