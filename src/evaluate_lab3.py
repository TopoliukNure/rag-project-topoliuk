import argparse
import hashlib
import json
from pathlib import Path

from src.dense_retrieval import (
    DIMENSIONS,
    INDEX_ERRORS,
    INDEX_ROOT,
    MODEL,
    REVISION,
    ROOT,
    SEARCH_EF,
    connect,
    encode,
    fingerprint,
    load_corpus,
    load_encoder,
    query_vector,
    read_jsonl,
    unique_records,
)
from src.index_lab3 import audit_index, index_size

EVAL_SET = ROOT / "tests/eval-set.jsonl"
RESULTS = ROOT / "data/lab-3/evaluation-results.json"


def load_eval_set(path, chunk_ids):
    rows = list(unique_records(read_jsonl(path), "query_id").values())
    if len(rows) != 20:
        raise ValueError("Expected exactly 20 queries")
    queries = set()
    for row in rows:
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{row['query_id']}: missing query")
        key = " ".join(query.casefold().split())
        if key in queries or row.get("relevant_chunk_id") not in chunk_ids:
            raise ValueError(f"{row['query_id']}: duplicate or unknown label")
        queries.add(key)
    return rows


def score(ranks, k):
    if not ranks or k < 1:
        raise ValueError("Recall requires nonempty ranks and positive k")
    successes = sum(rank is not None and 1 <= rank <= k for rank in ranks)
    return {
        "recall": successes / len(ranks),
        "successful": successes,
        "unsuccessful": len(ranks) - successes,
    }


def evaluate(root, rows, eval_set):
    digest = fingerprint()
    vectors = encode(load_encoder(), [row["query"] for row in eval_set])
    experiments = []
    for config, ef_search in SEARCH_EF.items():
        path = root / config
        collection = connect(path, config, digest)
        audit_index(collection, rows)
        details = []
        for row, vector in zip(eval_set, vectors, strict=True):
            hits = query_vector(collection, vector, 10)
            ids = [hit.chunk_id for hit in hits]
            target = row["relevant_chunk_id"]
            rank = ids.index(target) + 1 if target in ids else None
            details.append({
                **row, "relevant_rank": rank,
                "hit_at_5": rank is not None and rank <= 5,
                "hit_at_10": rank is not None,
                "retrieved_chunk_ids": ids,
            })
        ranks = [item["relevant_rank"] for item in details]
        experiments.append({
            "config": config,
            "ef_search": ef_search,
            "ef_construction": 100,
            "index_bytes": index_size(path),
            "at_5": score(ranks, 5),
            "at_10": score(ranks, 10),
            "queries": details,
        })
    best = max(experiments, key=lambda row: (
        row["at_5"]["recall"], row["at_10"]["recall"], -row["ef_search"]
    ))
    return {
        "model": MODEL,
        "revision": REVISION,
        "normalized": True,
        "dimension": DIMENSIONS,
        "vectors_per_index": len(rows),
        "metric": "cosine",
        "similarity": "1 - cosine distance",
        "corpus_sha256": digest,
        "eval_sha256": hashlib.sha256(EVAL_SET.read_bytes()).hexdigest(),
        "query_count": len(eval_set),
        "configs": experiments,
        "selected_config": best["config"],
        "selection_rule": (
            "Maximize Recall@5, then Recall@10, then lower ef_search"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval")
    parser.add_argument("--index-root", type=Path, default=INDEX_ROOT)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    try:
        rows = load_corpus()
        labels = load_eval_set(EVAL_SET, {row["chunk_id"] for row in rows})
        result = evaluate(args.index_root, rows, labels)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except INDEX_ERRORS as exc:
        parser.exit(1, f"Evaluation error: {exc}\n")

    for config in result["configs"]:
        print(f"{config['config']}: Recall@5={config['at_5']['recall']:.2f}, "
              f"Recall@10={config['at_10']['recall']:.2f}")
    print(f"Selected: {result['selected_config']}. Results: {args.output}")


if __name__ == "__main__":
    main()

