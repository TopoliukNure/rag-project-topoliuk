import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.errors import ChromaError
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "data/lab-2/documents.jsonl"
CHUNKS = ROOT / "data/lab-2/chunks.jsonl"
LOCAL_CONFIG = ROOT / "config/lab3.local.json"
LOCAL_PATHS = (
    json.loads(LOCAL_CONFIG.read_text(encoding="utf-8"))
    if LOCAL_CONFIG.is_file()
    else {}
)
INDEX_ROOT = Path(
    os.environ.get("SMART_PRODUCT_INDEX_ROOT")
    or LOCAL_PATHS.get("index_root")
    or ROOT / "data/lab-3/chroma"
)
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIMENSIONS = 384
COLLECTION = "flipkart_chunks"
SEARCH_EF = {"ef10": 10, "ef100": 100}
DEFAULT_CONFIG = "ef10"
INDEX_ERRORS = (ValueError, OSError, ChromaError)


def read_jsonl(path):
    result = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            result.append(row)
    if not result:
        raise ValueError(f"{path}: no records")
    return result


def unique_records(rows, key):
    records = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in records:
            raise ValueError(f"Missing or duplicate {key}: {value!r}")
        records[value] = row
    return records


def load_corpus(documents=DOCUMENTS, chunks=CHUNKS):
    docs = unique_records(read_jsonl(documents), "document_id")
    records = unique_records(read_jsonl(chunks), "chunk_id")
    coverage = {key: [] for key in docs}
    for key, chunk in records.items():
        doc_id = chunk.get("document_id")
        if doc_id not in docs:
            raise ValueError(f"{key}: unknown document")
        doc = docs[doc_id]
        metadata = chunk_metadata(chunk)
        chunk_fields = {"chunk_id", "chunk_index", "word_start", "word_end"}
        if any(doc.get(k) != v for k, v in metadata.items()
               if k not in chunk_fields):
            raise ValueError(f"{key}: mismatched document metadata")
        index = chunk.get("chunk_index")
        start, end = chunk.get("word_start"), chunk.get("word_end")
        text = doc.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{doc_id}: missing document text")
        words = text.split()
        if (
            type(index) is not int or index < 0
            or key != f"{doc_id}:{index:04d}"
            or type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(words)
            or chunk.get("text") != " ".join(words[start:end])
            or not 200 <= end - start <= 400
        ):
            raise ValueError(f"{key}: invalid chunk ID, offsets or text")
        coverage[doc_id].append((index, start, end))

    for key, spans in coverage.items():
        spans.sort()
        if (
            not spans or spans[0][1] != 0
            or spans[-1][2] != len(docs[key]["text"].split())
        ):
            raise ValueError(f"{key}: incomplete chunk coverage")
        for index, (actual, start, _) in enumerate(spans):
            invalid_overlap = index > 0 and spans[index - 1][2] - start != 40
            if index != actual or invalid_overlap:
                raise ValueError(f"{key}: invalid ordering or overlap")
    return list(records.values())


def chunk_metadata(chunk):
    keys = (
        "chunk_id", "document_id", "pid", "uniq_id", "source_row",
        "source_url", "crawl_timestamp", "title", "category", "category_path",
        "language",
        "currency", "retail_price", "discounted_price", "chunk_index",
        "word_start", "word_end",
    )
    if any(key not in chunk for key in keys):
        raise ValueError(f"{chunk.get('chunk_id')}: incomplete metadata")
    result = {key: chunk[key] for key in keys}
    if (
        result["currency"] != "INR"
        or type(result["discounted_price"]) is not int
        or type(result["retail_price"]) is not int
        or not 0 < result["discounted_price"] <= result["retail_price"]
        or any(type(value) not in (str, int) for value in result.values())
    ):
        raise ValueError(f"{result['chunk_id']}: invalid metadata or prices")
    return result


def load_encoder():
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    cache_folder = None
    if not (os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")):
        cache_folder = LOCAL_PATHS.get("model_cache")
    return SentenceTransformer(
        MODEL, revision=REVISION, device="cpu", cache_folder=cache_folder
    )


def checked_vectors(vectors, count):
    array = np.asarray(vectors, dtype=np.float32)
    if (
        array.shape != (count, DIMENSIONS)
        or not np.isfinite(array).all()
        or not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Expected finite, normalized 384-dimensional vectors")
    return array


def encode(encoder, texts):
    return checked_vectors(
        encoder.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        len(texts),
    )


def fingerprint():
    return hashlib.sha256(CHUNKS.read_bytes()).hexdigest()


def identity(config, digest):
    return {
        "model": MODEL,
        "revision": REVISION,
        "corpus_sha256": digest,
        "normalized": True,
        "ef_search": SEARCH_EF[config],
    }


def connect(path, config, digest):
    if not (path / "chroma.sqlite3").is_file():
        raise ValueError(f"Index not found at {path}. Run src.index_lab3.")
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_collection(COLLECTION, embedding_function=None)
    hnsw = collection.configuration["hnsw"]
    if (
        collection.metadata != identity(config, digest)
        or hnsw["space"] != "cosine"
        or hnsw["ef_search"] != SEARCH_EF[config]
    ):
        raise ValueError(f"Incompatible index at {path}")
    return collection


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    text: str
    metadata: dict
    similarity: float


def query_vector(collection, vector, limit):
    result = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for chunk_id, text, meta, distance in zip(
        result["ids"][0], result["documents"][0],
        result["metadatas"][0], result["distances"][0], strict=True
    ):
        if (
            not isinstance(text, str) or not text
            or not math.isfinite(distance)
            or not -1e-5 <= distance <= 2.00001
        ):
            raise ValueError(f"{chunk_id}: invalid search result")
        if not isinstance(meta, dict) or meta.get("chunk_id") != chunk_id:
            raise ValueError(f"{chunk_id}: invalid result metadata")
        chunk_metadata(meta)
        similarity = 1.0 - max(0.0, min(2.0, distance))
        hits.append(SearchHit(chunk_id, text, meta, similarity))
    if len(hits) != limit:
        raise ValueError(f"Expected {limit} results, received {len(hits)}")
    return hits


def search(collection, encoder, query, limit=5):
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty")
    if not 1 <= limit <= collection.count():
        raise ValueError("Invalid result limit for this index")
    return query_vector(collection, encode(encoder, [query])[0], limit)

