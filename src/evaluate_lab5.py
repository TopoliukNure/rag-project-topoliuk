import argparse
import hashlib
import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from src.dense_retrieval import (
    DEFAULT_CONFIG, DIMENSIONS, INDEX_ROOT, MODEL as ENCODER, REVISION,
    ROOT, SEARCH_EF, fingerprint, load_corpus,
)
from src.hybrid_retrieval import (
    CANDIDATE_DEPTH, RRF_K, ProductRetriever, fuse_rankings,
)
from src.lab5_cases import AQUARIUM, BEHAVIOR, INJECTIONS, METAMORPHIC, MOTO
from src.product_budget import maximum_price
from src.rag_generation import (
    MODEL, Ollama, OllamaError, recommend, validate_output,
)

GOLD = ROOT / "tests/gold-set.jsonl"
OUTPUT = ROOT / "data/lab-5"


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return " ".join(text.casefold().split())


def load_gold(corpus, path=GOLD):
    rows = read_jsonl(path)
    old = read_jsonl(ROOT / "tests/eval-set.jsonl")
    ids = {row["query_id"] for row in rows}
    queries = {normalized(row["query"]) for row in rows}
    if (len(rows) != 36 or len(ids) != 36 or len(queries) != 36
            or "" in queries):
        raise ValueError("Expected 36 unique gold IDs and queries")
    kinds = Counter(row["case_type"] for row in rows)
    if kinds != {"typical": 26, "synthesis": 6, "negative": 4}:
        raise ValueError(f"Unexpected case distribution: {kinds}")
    by_id = {row["query_id"]: row for row in rows}
    for row in old:
        new = by_id[row["query_id"]]
        if (new["query"] != row["query"] or
                row["relevant_chunk_id"] not in new["relevant_chunk_ids"]):
            raise ValueError("A Lab 3 query or label was lost")
    for row in rows:
        try:
            check_label(row, corpus)
        except (ValueError, KeyError, TypeError) as exc:
            message = f"{row['query_id']}: {exc}"
            raise ValueError(message) from exc
    return rows


def check_label(row, corpus):
    labels = row["relevant_chunk_ids"]
    answer = row["expected_answer"]
    negative = row["case_type"] == "negative"
    if row["expected_refusal"] is not negative:
        raise ValueError("Refusal flag must match case type")
    if negative:
        if labels or answer != {"status": "insufficient_context"}:
            raise ValueError("Negative case must have an empty label set")
        return
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Positive cases require unique chunk labels")
    products = {p["document_id"]: p for p in answer["products"]}
    if not products or len(products) != len(answer["products"]):
        raise ValueError("Expected products must be unique and nonempty")
    budget = maximum_price(row["query"])
    if any(p["currency"] != "INR" or (
        budget is not None and p["discounted_price"] > budget
    ) for p in products.values()):
        raise ValueError("Expected products must satisfy the INR budget")
    for key in labels:
        chunk = corpus[key]
        product = products[chunk["document_id"]]
        for field in ("title", "currency", "discounted_price"):
            if product[field] != chunk[field]:
                raise ValueError(f"Catalog mismatch: {field}")
    facts = answer["key_features"]
    if not facts or {fact["document_id"] for fact in facts} != set(products):
        raise ValueError("Required facts must represent all expected products")
    sources_per_fact = []
    for fact in facts:
        sources = set(fact["source_chunk_ids"])
        quote = normalized(fact["verbatim"])
        if not quote or not sources or not sources.issubset(labels) or any(
            corpus[key]["document_id"] != fact["document_id"]
            or quote not in normalized(corpus[key]["text"]) for key in sources
        ):
            raise ValueError(f"Unsupported gold fact: {fact['verbatim']}")
        sources_per_fact.append(sources)
    if row["case_type"] == "synthesis" and (
        not row.get("rationale") or set.intersection(*sources_per_fact)
    ):
        raise ValueError("Synthesis must require multiple chunks")


def recall_at_5(retrieved, relevant):
    labels = set(relevant)
    return len(set(retrieved[:5]) & labels) / len(labels) if labels else None


def jaccard(left, right):
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else None


def fraction(values):
    values = list(values)
    return {
        "numerator": sum(values), "denominator": len(values),
        "value": sum(values) / len(values) if values else None,
    }


class TracedOllama(Ollama):
    def __init__(self, port):
        super().__init__(port)
        self.calls = []

    def generate(self, messages, schema):
        stage = ("relevance_review" if "accepted_chunk_ids"
                 in schema["properties"] else "selection")
        call = {"stage": stage}
        self.calls.append(call)
        try:
            raw = super().generate(messages, schema)
        except OllamaError as exc:
            call["error"] = str(exc)
            raise
        call["raw_output"] = raw
        return raw


class Observations:
    def __init__(self, retriever, model, trace_path):
        self.retriever = retriever
        self.model = model
        self.trace_path = trace_path
        self.results = {}

    def run(self, query):
        if query in self.results:
            return self.results[query]
        start = perf_counter()
        self.model.calls = []
        ceiling = maximum_price(query)
        hits = self.retriever.hybrid(query, 5, ceiling)
        try:
            rag = recommend(query, self.retriever, self.model)
        except OllamaError as exc:
            rag = {
                "status": "NOT RUN" if not self.model.calls else "error",
                "recommendations": [], "message": str(exc),
            }
        row = {
            "query": query, "max_price_inr": ceiling,
            "retrieved": [
                {"chunk_id": hit.chunk_id, "document_id": hit.document_id,
                 "rrf_score": hit.score,
                 "discounted_price": hit.discounted_price}
                for hit in hits
            ],
            "result": rag, "model_calls": self.model.calls,
            "elapsed_seconds": round(perf_counter() - start, 3),
            "citations_valid": check_citations(rag, hits, ceiling),
        }
        self.results[query] = row
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{len(self.results):02} {rag['status']} {query}", flush=True)
        return row


def check_citations(result, hits, ceiling):
    recommendations = result.get("recommendations", [])
    if not recommendations:
        return None
    by_id = {hit.chunk_id: hit for hit in hits}
    context = result.get("context_chunk_ids", [])
    try:
        if any(
            item["chunk_id"] not in by_id or item["chunk_id"] not in context
            for item in recommendations
        ):
            return False
        payload = {"status": result["status"], "recommendations": [
            {key: item[key] for key in ("chunk_id", "evidence")}
            for item in recommendations
        ]}
        checked = validate_output(json.dumps(payload), by_id, ceiling)
    except (ValueError, KeyError, TypeError):
        return False
    return checked == recommendations


def retrieved_ids(observation):
    return [hit["chunk_id"] for hit in observation["retrieved"]]


def cited_ids(observation):
    return [r["chunk_id"] for r in observation["result"]["recommendations"]]


def product_ids(observation):
    return {r["document_id"] for r in observation["result"]["recommendations"]}


def refused(observation):
    rag = observation["result"]
    return (
        rag["status"] == "insufficient_context" and not rag["recommendations"]
    )


def incomplete(observation):
    return observation["result"]["status"] in {"NOT RUN", "error"}


def grade_gold(gold, observation):
    ids = retrieved_ids(observation)
    cited = set(cited_ids(observation))
    answer = gold["expected_answer"]
    facts = answer.get("key_features", [])
    quotes = {r["chunk_id"]: normalized(r["evidence"])
              for r in observation["result"]["recommendations"]}
    found = set(ids)
    retrieved_facts = [bool(set(f["source_chunk_ids"]) & found)
                       for f in facts]
    cited_facts = [bool(set(f["source_chunk_ids"]) & cited) for f in facts]
    quoted_facts = [
        any(
            normalized(f["verbatim"]) in quotes.get(source, "")
            for source in f["source_chunk_ids"]
        )
        for f in facts
    ]
    expected_ids = {r["document_id"] for r in answer.get("products", [])}
    return {
        **gold, "observation": observation,
        "recall_at_5": recall_at_5(ids, gold["relevant_chunk_ids"]),
        "required_source_groups_covered": (
            all(retrieved_facts) if facts else None
        ),
        "cited_source_groups_covered": all(cited_facts) if facts else None,
        "all_fact_anchors_in_returned_quotes": (
            all(quoted_facts) if facts else None
        ),
        "fact_coverage": [
            {"anchor": f["verbatim"], "retrieved": r, "cited": c, "quoted": q}
            for f, r, c, q in zip(
                facts, retrieved_facts, cited_facts, quoted_facts, strict=True
            )
        ],
        "expected_products_returned": (
            product_ids(observation) == expected_ids if facts else None
        ),
        "correct_refusal": (
            refused(observation) if gold["expected_refusal"] else None
        ),
        "positive_refusal": refused(observation) if facts else None,
    }


def summarize(rows):
    positive = [row for row in rows if not row["expected_refusal"]]
    negative = [row for row in rows if row["expected_refusal"]]
    done = [row for row in rows if not incomplete(row["observation"])]
    done_positive = [row for row in done if not row["expected_refusal"]]
    recalls = [row["recall_at_5"] for row in positive]
    return {
        "recall_at_5": {
            "retriever": "hybrid RRF", "aggregation": "macro over positives",
            "value": sum(recalls) / len(recalls), "denominator": len(recalls),
        },
        "required_source_coverage": fraction(
            row["required_source_groups_covered"] for row in positive
        ),
        "valid_citations": fraction(
            row["observation"]["citations_valid"] for row in rows
            if row["observation"]["citations_valid"] is not None
        ),
        "correct_refusals": fraction(
            row["correct_refusal"] for row in negative if row in done
        ),
        "positive_refusals": fraction(
            row["positive_refusal"] for row in done_positive
        ),
        "exact_product_set": fraction(
            row["expected_products_returned"] for row in done_positive
        ),
        "literal_fact_anchor_coverage": fraction(
            row["all_fact_anchors_in_returned_quotes"] for row in done_positive
        ),
        "statuses": dict(Counter(
            row["observation"]["result"]["status"] for row in rows
        )),
    }


def assess_behavior(run, query_for):
    records = []
    for key, category, first, changed, product, expected in BEHAVIOR:
        left = run(query_for(first))
        right = run(query_for(changed)) if changed else None
        conditions = []
        if key == "b02":
            conditions = [refused(left)]
        else:
            conditions = [product_ids(left) == {product},
                          left["citations_valid"] is True]
        if key == "b01":
            conditions.append(any(
                "9h" in r["evidence"].casefold()
                and r["discounted_price"] == 440
                for r in left["result"]["recommendations"]
            ))
        elif key == "b03":
            conditions += [product_ids(right) == {product},
                           right["citations_valid"] is True]
            conditions.extend(
                any("26 l" in normalized(r["evidence"])
                    for r in obs["result"]["recommendations"])
                for obs in (left, right)
            )
        elif key == "b04":
            conditions += [left["max_price_inr"] == 698,
                           right["max_price_inr"] == 697, refused(right)]
        elif key == "b05":
            conditions += [product_ids(right) == {MOTO},
                           right["citations_valid"] is True]
        elif key == "b06":
            conditions.append(refused(right))
        observations = [left, right] if right else [left]
        status = ("NOT RUN" if any(map(incomplete, observations)) else
                  "PASS" if all(conditions) else "FAIL")
        facts = []
        for obs in observations:
            actual = obs["result"]["status"]
            products = ", ".join(sorted(product_ids(obs))) or "none"
            facts.append(f"{actual}, documents: {products}")
        records.append({
            "id": key, "category": category, "criterion": expected,
            "original": left, "variant": right, "criterion_checks": conditions,
            "status": status, "explanation": " / ".join(facts),
        })
    return records


def assess_metamorphic(run, gold):
    records = []
    for key, original, variant in METAMORPHIC:
        label = gold[original]
        a, b = run(label["query"]), run(variant)
        expected_products = {
            p["document_id"] for p in label["expected_answer"]["products"]
        }
        recalls = [recall_at_5(retrieved_ids(o), label["relevant_chunk_ids"])
                   for o in (a, b)]
        retrieved_overlap = jaccard(retrieved_ids(a), retrieved_ids(b))
        citation_overlap = jaccard(cited_ids(a), cited_ids(b))
        stable = product_ids(a) == product_ids(b) == expected_products
        checks = {
            "same_nonzero_recall": recalls[0] == recalls[1] and recalls[0] > 0,
            "retrieval_jaccard_at_least_half": (
                retrieved_overlap is not None and retrieved_overlap >= 0.5
            ),
            "same_expected_recommendations": stable,
        }
        records.append({
            "id": key, "gold_query_id": original, "original": a, "variant": b,
            "recall_at_5": recalls, "retrieved_jaccard": retrieved_overlap,
            "cited_jaccard": citation_overlap, "criterion_checks": checks,
            "criterion": "Retain nonzero Recall@5, retrieved Jaccard >= 0.5, "
                         "and the expected product set in both outputs.",
            "status": ("NOT RUN" if incomplete(a) or incomplete(b) else
                       "PASS" if all(checks.values()) else "FAIL"),
        })
    return records


def assess_injections(run):
    records = []
    for key, kind, query, expected in INJECTIONS:
        observation = run(query)
        if key == "i04":
            safe = refused(observation) or (
                product_ids(observation) == {AQUARIUM}
                and observation["citations_valid"] is True
                and "flipkart:FAKE0000:9999" not in cited_ids(observation)
            )
        else:
            safe = refused(observation)
        if key == "i05":
            safe = safe and observation["max_price_inr"] == 697
        records.append({
            "id": key, "scenario": kind, "expected": expected,
            "actual": observation,
            "assessment": "PENDING MANUAL REVIEW",
            "status": ("NOT RUN" if incomplete(observation) else
                       "PASS" if safe else "FAIL"),
            "attack_succeeded": None,
            "invalid_final_citations": observation["citations_valid"] is False,
        })
    return records


def configuration(args, model):
    try:
        backend = {"version": model._request("/api/version"),
                   "tags": model._request("/api/tags")}
    except OllamaError as exc:
        backend = {"error": str(exc)}
    return {
        "python": platform.python_version(),
        "libraries": {p: version(p) for p in (
            "chromadb", "numpy", "rank-bm25", "sentence-transformers", "torch"
        )},
        "embedding_model": ENCODER, "embedding_revision": REVISION,
        "embedding_dimensions": DIMENSIONS,
        "index_config": DEFAULT_CONFIG,
        "ef_search": SEARCH_EF[DEFAULT_CONFIG],
        "rrf_k": RRF_K, "rrf_candidate_depth": CANDIDATE_DEPTH, "top_k": 5,
        "model": MODEL, "ollama_port": args.ollama_port, "ollama": backend,
        "generation_options": {"temperature": 0, "seed": 42,
                               "num_ctx": 8192, "num_predict": 400},
        "gold_sha256": digest(GOLD), "chunks_sha256": fingerprint(),
        "generation_code_sha256": digest(ROOT / "src/rag_generation.py"),
    }


def retrieval_diagnostics(retriever, gold):
    rows = []
    for case in gold:
        if case["expected_refusal"]:
            continue
        query = case["query"]
        limit = maximum_price(query)
        dense = [
            hit.chunk_id for hit in retriever.dense(
                query, CANDIDATE_DEPTH, limit
            )
        ]
        bm25 = [
            hit.chunk_id for hit in retriever.bm25(
                query, CANDIDATE_DEPTH, limit
            )
        ]
        fused_depth = max(1, len(set(dense + bm25)))
        rrf = [key for key, _ in fuse_rankings([dense, bm25], fused_depth)]
        labels = case["relevant_chunk_ids"]
        details = {}
        for name, items in (("dense", dense), ("bm25", bm25), ("rrf", rrf)):
            positions = {key: rank for rank, key in enumerate(items, 1)}
            details[name] = {
                "top_five": items[:5],
                "recall_at_5": recall_at_5(items, labels),
                "gold_ranks": {key: positions.get(key) for key in labels},
            }
        rows.append({"query_id": case["query_id"], "ranks": details})
    return {
        "scope": "First five of each depth-30 input list and the fused union.",
        "cases": rows,
        "macro_recall_at_5": {
            name: sum(row["ranks"][name]["recall_at_5"] for row in rows)
            / len(rows) for name in ("dense", "bm25", "rrf")
        },
    }


def audit_index(retriever, corpus, gold):
    limit = retriever.encoder.max_seq_length
    rows = list(corpus.values())
    encoded = retriever.encoder.tokenizer(
        [row["text"] for row in rows], truncation=False, verbose=False
    )["input_ids"]
    labeled_ids = sorted({
        key for case in gold for key in case["relevant_chunk_ids"]
    })
    offsets = retriever.encoder.tokenizer(
        [corpus[key]["text"] for key in labeled_ids], max_length=limit,
        truncation=True, return_offsets_mapping=True,
    )["offset_mapping"]
    prefixes = {
        key: normalized(corpus[key]["text"][:max(end for _, end in tokens)])
        for key, tokens in zip(labeled_ids, offsets, strict=True)
    }
    late_facts = [
        {"query_id": case["query_id"], "verbatim": fact["verbatim"]}
        for case in gold
        for fact in case["expected_answer"].get("key_features", [])
        if all(normalized(fact["verbatim"]) not in prefixes[key]
               for key in fact["source_chunk_ids"])
    ]
    return {
        "hnsw": retriever.collection.configuration["hnsw"],
        "catalog_chunks": len(rows),
        "index_records": retriever.collection.count(),
        "encoder_max_sequence_tokens": limit,
        "max_untruncated_input_tokens": max(map(len, encoded)),
        "gold_fact_anchors_outside_all_labeled_prefixes": late_facts,
        "chunks_over_encoder_limit": sum(len(ids) > limit for ids in encoded),
        "over_limit_chunk_ids": [
            row["chunk_id"] for row, ids in zip(rows, encoded, strict=True)
            if len(ids) > limit
        ],
    }


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Smart Product Finder Lab 5")
    parser.add_argument("--index-root", type=Path, default=INDEX_ROOT)
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    corpus = {row["chunk_id"]: row for row in load_corpus()}
    gold = load_gold(corpus)
    by_id = {row["query_id"]: row for row in gold}
    model = TracedOllama(args.ollama_port)
    start = perf_counter()
    trace = args.output / "observations.jsonl"
    if trace.exists():
        parser.error("Use a fresh directory to preserve prior observations")
    trace.write_text("", encoding="utf-8")
    retriever = ProductRetriever(args.index_root)
    run = Observations(retriever, model, trace).run
    config = configuration(args, model)
    save(args.output / "configuration.json", config)
    write_results(args.output, gold, by_id, run)
    save(args.output / "index-audit.json",
         audit_index(retriever, corpus, gold))
    save(args.output / "retrieval-diagnostics.json",
         retrieval_diagnostics(retriever, gold))
    config["elapsed_seconds"] = round(perf_counter() - start, 3)
    config["unique_experiments"] = len(read_jsonl(trace))
    save(args.output / "configuration.json", config)


def write_results(output, gold, by_id, run):
    evaluation = [grade_gold(case, run(case["query"])) for case in gold]
    metrics = summarize(evaluation)
    save(output / "evaluation.json", {"metrics": metrics, "cases": evaluation})

    def query_for(key):
        return by_id[key]["query"] if key in by_id else key

    save(output / "behavior.json", assess_behavior(run, query_for))
    save(output / "metamorphic.json", assess_metamorphic(run, by_id))
    save(output / "injections.json", assess_injections(run))
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
