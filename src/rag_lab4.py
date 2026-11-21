import argparse
import json
from dataclasses import asdict
from pathlib import Path
from textwrap import shorten

from src.dense_retrieval import INDEX_ERRORS, INDEX_ROOT
from src.hybrid_retrieval import ProductRetriever
from src.product_budget import maximum_price
from src.rag_generation import Ollama, OllamaError, recommend

SCORES = {"dense": "cosine similarity", "bm25": "BM25", "hybrid": "RRF"}


def print_answer(result):
    print(result["catalog_notice"])
    if result["status"] != "ok":
        print(result["message"])
        return
    for row in result["recommendations"]:
        print(f"- {row['title']} | {row['category']} | "
              f"{row['discounted_price']} INR [{row['chunk_id']}]")
        print(f"  Catalog evidence: {row['evidence']}")


def run(query, mode, limit, retriever, generator):
    if not query.strip():
        raise ValueError("Query must not be empty")
    if mode == "rag":
        return recommend(query, retriever, generator)
    maximum = maximum_price(query)
    methods = tuple(SCORES) if mode == "compare" else (mode,)
    return {
        "query": query,
        "max_price_inr": maximum,
        "results": {
            name: {
                "score_type": SCORES[name],
                "hits": [asdict(hit) for hit in
                         getattr(retriever, name)(query, limit, maximum)],
            }
            for name in methods
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Smart Product Finder Lab 4")
    parser.add_argument("query", help="English product request")
    parser.add_argument(
        "--mode", choices=("rag", *SCORES, "compare"), default="rag"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-root", type=Path, default=INDEX_ROOT)
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.query.strip() or args.top_k < 1:
        parser.error("A nonempty query and positive --top-k are required")
    try:
        result = run(
            args.query, args.mode, args.top_k,
            ProductRetriever(args.index_root), Ollama(args.ollama_port),
        )
    except (*INDEX_ERRORS, OllamaError) as exc:
        if args.json:
            print(json.dumps({"status": "error", "message": str(exc)}))
            raise SystemExit(1)
        parser.exit(1, f"Smart Product Finder: {exc}\n")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.mode == "rag":
        print_answer(result)
    else:
        for method, data in result["results"].items():
            print(f"{method} ({data['score_type']})")
            for rank, hit in enumerate(data["hits"], 1):
                price = hit["discounted_price"]
                print(f"{rank}. {hit['title']} | {price} INR "
                      f"| {hit['score']:.6f} [{hit['chunk_id']}]")
                print(f"   {shorten(hit['text'], width=150)}")


if __name__ == "__main__":
    main()

