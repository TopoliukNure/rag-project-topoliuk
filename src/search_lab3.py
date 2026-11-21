import argparse
from pathlib import Path
from textwrap import shorten

from src.dense_retrieval import (
    DEFAULT_CONFIG,
    INDEX_ERRORS,
    INDEX_ROOT,
    SEARCH_EF,
    connect,
    fingerprint,
    load_encoder,
    search,
)


def main():
    parser = argparse.ArgumentParser(
        description="Top 5 chunks by cosine similarity (1 − cosine distance)"
    )
    parser.add_argument("query", help="English product request")
    parser.add_argument("--index-root", type=Path, default=INDEX_ROOT)
    parser.add_argument("--config", choices=SEARCH_EF, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("Query cannot be empty")
    try:
        collection = connect(
            args.index_root / args.config, args.config, fingerprint()
        )
        if collection.count() != 1208:
            raise ValueError("Incomplete index, expected 1,208 chunks")
        hits = search(collection, load_encoder(), args.query)
    except INDEX_ERRORS as exc:
        parser.exit(1, f"Search error: {exc}\n")

    print("Cosine similarity = 1 - Chroma cosine distance (higher is better)")
    for rank, hit in enumerate(hits, 1):
        item = hit.metadata
        print(f"{rank}. {hit.chunk_id} | {item['title']}")
        print(f"   {item['category']} | {item['discounted_price']} INR"
              f" | similarity={hit.similarity:.4f}")
        print(f"   {shorten(hit.text, width=180, placeholder='...')}")


if __name__ == "__main__":
    main()

