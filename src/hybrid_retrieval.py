from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property, partial
from types import SimpleNamespace

from src.dense_retrieval import (
    DEFAULT_CONFIG,
    INDEX_ROOT,
    connect,
    fingerprint,
    load_corpus,
    load_encoder,
    search,
)
from src.sparse_retrieval import SparseIndex

RRF_K = 60
CANDIDATE_DEPTH = 30


@dataclass(frozen=True)
class RankedChunk:
    chunk_id: str
    document_id: str
    text: str
    title: str
    category: str
    discounted_price: int
    currency: str
    score: float
    metadata: dict


def fuse_rankings(rankings, limit):
    if limit < 1:
        raise ValueError("top-K must be positive")
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(dict.fromkeys(ranking), 1):
            scores[chunk_id] += 1 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


class ProductRetriever:
    def __init__(self, index_root=INDEX_ROOT):
        self.index_root = index_root
        self.rows = {row["chunk_id"]: row for row in load_corpus()}

    @cached_property
    def collection(self):
        collection = connect(
            self.index_root / DEFAULT_CONFIG, DEFAULT_CONFIG, fingerprint()
        )
        if collection.count() != len(self.rows):
            raise ValueError("Chroma count does not match the catalog")
        return collection

    @cached_property
    def encoder(self):
        return load_encoder()

    @cached_property
    def sparse_index(self):
        return SparseIndex(self.rows.values())

    def _hit(self, chunk_id, score):
        row = self.rows[chunk_id]
        return RankedChunk(
            chunk_id, row["document_id"], row["text"], row["title"],
            row["category"], row["discounted_price"], row["currency"],
            score, {key: value for key, value in row.items() if key != "text"},
        )

    def _eligible(self, max_price):
        if max_price is None:
            return None
        return {
            chunk_id for chunk_id, row in self.rows.items()
            if row["discounted_price"] <= max_price
        }

    def dense(self, query, limit=5, max_price=None):
        allowed = self._eligible(max_price)
        count = len(self.rows) if allowed is None else len(allowed)
        if not count:
            return []
        collection = self.collection
        if allowed is not None:
            collection = SimpleNamespace(
                count=collection.count,
                query=partial(
                    collection.query, where={
                        "discounted_price": {"$lte": max_price}
                    }
                ),
            )
        hits = search(collection, self.encoder, query, min(limit, count))
        if any(hit.chunk_id not in self.rows for hit in hits):
            raise ValueError("Chroma returned a chunk outside the catalog")
        if allowed is not None and any(
            hit.chunk_id not in allowed for hit in hits
        ):
            raise ValueError("Chroma returned a chunk outside the budget")
        return [
            self._hit(hit.chunk_id, hit.similarity)
            for hit in sorted(hits, key=lambda h: (-h.similarity, h.chunk_id))
        ]

    def bm25(self, query, limit=5, max_price=None):
        allowed = self._eligible(max_price)
        hits = self.sparse_index.search(query, limit, allowed)
        return [self._hit(hit.chunk_id, hit.score) for hit in hits]

    def hybrid(self, query, limit=5, max_price=None):
        depth = max(CANDIDATE_DEPTH, limit)
        dense = self.dense(query, depth, max_price)
        sparse = self.bm25(query, depth, max_price)
        rankings = [[hit.chunk_id for hit in hits] for hits in (dense, sparse)]
        return [
            self._hit(chunk_id, score)
            for chunk_id, score in fuse_rankings(rankings, limit)
        ]

