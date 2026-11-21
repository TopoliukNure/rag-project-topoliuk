import re
import unicodedata
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


@dataclass(frozen=True)
class SparseHit:
    chunk_id: str
    score: float


class SparseIndex:
    def __init__(self, rows):
        self.ids = [row["chunk_id"] for row in rows]
        if not self.ids or len(self.ids) != len(set(self.ids)):
            raise ValueError("BM25 requires a nonempty corpus with unique IDs")
        documents = [tokenize(row["text"]) for row in rows]
        if not any(documents):
            raise ValueError("BM25 corpus has no searchable terms")
        self.index = BM25Okapi(documents)
        self.terms = documents

    def search(self, query, limit, allowed=None):
        if limit < 1:
            raise ValueError("top-K must be positive")
        words = list(dict.fromkeys(tokenize(query)))
        if not words:
            return []
        scores = self.index.get_scores(words)
        matches = set(words)
        results = [
            SparseHit(chunk_id, float(score))
            for chunk_id, score, terms in zip(
                self.ids, scores, self.terms, strict=True
            )
            if (allowed is None or chunk_id in allowed)
            and not matches.isdisjoint(terms)
        ]
        results.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return results[:limit]

