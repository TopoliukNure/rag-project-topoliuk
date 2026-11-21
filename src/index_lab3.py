import argparse
import os
from pathlib import Path

import chromadb

from src.dense_retrieval import (
    COLLECTION,
    INDEX_ERRORS,
    INDEX_ROOT,
    SEARCH_EF,
    checked_vectors,
    chunk_metadata,
    connect,
    encode,
    fingerprint,
    identity,
    load_corpus,
    load_encoder,
)


def audit_index(collection, rows):
    stored = collection.get(include=["documents", "metadatas", "embeddings"])
    if collection.count() != len(rows) or len(stored["ids"]) != len(rows):
        raise ValueError("Index count does not match corpus")
    checked_vectors(stored["embeddings"], len(rows))
    expected = {row["chunk_id"]: row for row in rows}
    if set(stored["ids"]) != expected.keys():
        raise ValueError("Index IDs do not match corpus")
    for key, document, metadata in zip(
        stored["ids"], stored["documents"], stored["metadatas"], strict=True
    ):
        row = expected[key]
        if document != row["text"] or metadata != chunk_metadata(row):
            raise ValueError(f"{key}: stored data differs from corpus")


def index_size(path):
    files = (file for file in path.rglob("*") if file.is_file())
    return sum(file.stat().st_size for file in files)


def build_indexes(root, rows, digest):
    embeddings = None
    for config, ef_search in SEARCH_EF.items():
        path = root / config
        if (path / "chroma.sqlite3").is_file():
            collection = connect(path, config, digest)
            status = "verified existing"
        else:
            if embeddings is None:
                texts = [row["text"] for row in rows]
                embeddings = encode(load_encoder(), texts)
            client = chromadb.PersistentClient(path=str(path))
            collection = client.create_collection(
                COLLECTION,
                embedding_function=None,
                metadata=identity(config, digest),
                configuration={"hnsw": {
                    "space": "cosine",
                    "ef_construction": 100,
                    "ef_search": ef_search,
                    "num_threads": 1,
                    "sync_threshold": 100,
                }},
            )
            collection.upsert(
                ids=[row["chunk_id"] for row in rows],
                documents=[row["text"] for row in rows],
                metadatas=[chunk_metadata(row) for row in rows],
                embeddings=embeddings,
            )
            status = "built"
        audit_index(collection, rows)
        size = index_size(path)
        print(f"{config}: {status}, {len(rows)} vectors, {size} bytes")


def main():
    parser = argparse.ArgumentParser(description="Build and audit Lab 3")
    parser.add_argument("--index-root", type=Path, default=INDEX_ROOT)
    args = parser.parse_args()
    try:
        if os.name == "nt" and not str(args.index_root.resolve()).isascii():
            raise ValueError(
                "Chroma HNSW on Windows requires an ASCII index path for "
                "this version. Set --index-root or SMART_PRODUCT_INDEX_ROOT."
            )
        rows = load_corpus()
        if len(rows) != 1208 or len({r["document_id"] for r in rows}) != 1000:
            raise ValueError("Expected 1,000 documents and 1,208 chunks")
        build_indexes(args.index_root, rows, fingerprint())
    except INDEX_ERRORS as exc:
        parser.exit(1, f"Index error: {exc}\n")


if __name__ == "__main__":
    main()

