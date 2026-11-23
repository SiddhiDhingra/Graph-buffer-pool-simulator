
"""Separate buffer-pool experiments for the Cora citation network."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from src.access_score import AccessScoreTracker
from src.buffer_pool import BufferPool
from src.cora_loader import load_cora
from src.graph_locality import build_page_graph
from src.lsh import LSHManager
from src.page_builder import build_pages
from src.proposed_policy import GraphLSHAwareEvictionPolicy
from src.replacement_policies import FIFOPolicy, LRUPolicy
from src.page_vectors import build_page_vectors
from src.usefulness_score import UsefulnessCalculator


def make_workload(kind, length, seed, page_ids, page_graph, lsh):
    rng = random.Random(seed)
    if not page_ids or length < 1:
        return []

    trace = []
    page_id_set = set(page_ids)
    current = rng.choice(page_ids)

    for _ in range(length):
        trace.append(current)

        request_type = (
            rng.choice(["random", "graph", "semantic"])
            if kind == "mixed" else kind
        )

        if request_type == "random":
            current = rng.choice(page_ids)

        elif request_type == "graph":
            neighbors = page_graph.neighbors(current)
            current = rng.choice(neighbors) if neighbors else rng.choice(page_ids)

        elif request_type == "semantic":
            candidates = [
                p for p in lsh.get_candidates(current)
                if p in page_id_set
            ]
            current = rng.choice(candidates) if candidates else rng.choice(page_ids)

        else:
            raise ValueError(f"Unknown workload: {kind}")

    return trace


def make_policy(name, page_graph, lsh, calculator):
    if name == "FIFO":
        return FIFOPolicy()

    if name == "LRU":
        return LRUPolicy()

    if name == "Proposed":
        tracker = AccessScoreTracker(
            recency_tau=20.0,
            frequency_tau=5.0,
        )
        return GraphLSHAwareEvictionPolicy(
            usefulness_calculator=calculator,
            page_graph=page_graph,
            lsh_manager=lsh,
            access_tracker=tracker,
            prepare_static=False,
        )

    raise ValueError(f"Unknown policy: {name}")


def run_trace(trace, storage, capacity, policy):
    pool = BufferPool(capacity, storage, policy)
    started = time.perf_counter()

    for page_id in trace:
        pool.request_page(page_id)

    elapsed = time.perf_counter() - started

    return {
        "Requests": pool.requests,
        "Capacity": capacity,
        "Hits": pool.hits,
        "Page Faults": pool.page_faults,
        "Hit Ratio (%)": round(pool.hit_ratio * 100, 4),
        "Miss Rate (%)": round(
            pool.page_faults / pool.requests * 100, 4
        ) if pool.requests else 0.0,
        "Storage Reads": pool.storage_reads,
        "Evictions": pool.evictions,
        "Policy Decision Time (ms)": round(
            pool.policy_decision_time * 1000, 4
        ),
        "Avg Decision Time (ms)": round(
            pool.policy_decision_time * 1000 / pool.policy_decisions, 6
        ) if pool.policy_decisions else 0.0,
        "Total Runtime (s)": round(elapsed, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/cora")
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--capacities", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 101])
    parser.add_argument("--lsh-tables", type=int, default=5)
    parser.add_argument("--lsh-planes", type=int, default=6)
    parser.add_argument("--probe-radius", type=int, default=1)
    parser.add_argument("--output-dir", default="results/cora")
    args = parser.parse_args()

    if args.page_size < 1 or args.requests < 1:
        parser.error("--page-size and --requests must be positive")
    if any(c < 1 for c in args.capacities):
        parser.error("All capacities must be positive")

    print("Loading Cora...", flush=True)
    graph = load_cora(args.data_dir)

    page_table = build_pages(graph, args.page_size)
    storage = page_table.pages
    page_ids = sorted(storage)

    print(
        f"Nodes: {graph.num_nodes}; pages: {len(page_ids)}",
        flush=True,
    )

    print("Building page vectors...", flush=True)
    page_vectors = build_page_vectors(graph, page_table)

    print("Building page graph...", flush=True)
    page_graph = build_page_graph(graph, page_table)

    print("Building LSH index...", flush=True)
    lsh = LSHManager(
        page_vectors,
        num_tables=args.lsh_tables,
        num_planes=args.lsh_planes,
        seed=42,
        probe_radius=args.probe_radius,
    )

    print("Preparing usefulness scores...", flush=True)
    calculator = UsefulnessCalculator(
        alpha=0.40,
        beta=0.40,
        gamma=0.20,
        graph_mode="directional",
        semantic_probe_radius=args.probe_radius,
        locality_gate=True,
    )
    calculator.prepare(page_ids, page_graph, lsh)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for workload in ("random", "mixed", "graph", "semantic"):
        for seed in args.seeds:
            print(
                f"Generating {workload} workload, seed={seed}",
                flush=True,
            )
            trace = make_workload(
                workload, args.requests, seed, page_ids, page_graph, lsh
            )

            for capacity in args.capacities:
                if capacity > len(page_ids):
                    print(f"Skipping capacity {capacity}: exceeds page count")
                    continue

                for policy_name in ("FIFO", "LRU", "Proposed"):
                    print(
                        f"  {workload} | capacity={capacity} | {policy_name}",
                        flush=True,
                    )

                    policy = make_policy(
                        policy_name, page_graph, lsh, calculator
                    )
                    metrics = run_trace(trace, storage, capacity, policy)

                    rows.append({
                        "Dataset": "Cora",
                        "Workload": workload,
                        "Seed": seed,
                        "Policy": policy_name,
                        "Page Size": args.page_size,
                        **metrics,
                    })

    csv_path = output_dir / "cora_results.csv"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    metadata = {
        "dataset": "Cora",
        "nodes": graph.num_nodes,
        "directed_citation_rows": 5429,
        "unique_undirected_edges": len(graph.edges),
        "feature_dimensions": int(graph.features.shape[1]),
        "logical_pages": len(page_ids),
        "page_size": args.page_size,
        "workloads": ["random", "mixed", "graph", "semantic"],
        "policies": ["FIFO", "LRU", "Proposed"],
        "capacities": args.capacities,
        "seeds": args.seeds,
        "requests_per_trace": args.requests,
        "lsh_tables": args.lsh_tables,
        "lsh_planes": args.lsh_planes,
        "probe_radius": args.probe_radius,
        "edge_assumption": "Citation edges are deduplicated and treated as undirected.",
        "layout_assumption": "Consecutive internal node IDs are grouped into logical pages.",
        "label_assumption": "Class labels are metadata and are excluded from feature vectors.",
        "results_csv": str(csv_path),
    }

    metadata_path = output_dir / "cora_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nCora experiments complete.", flush=True)
    print(f"Results: {csv_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Total experiment rows: {len(rows)}")


if __name__ == "__main__":
    main()
