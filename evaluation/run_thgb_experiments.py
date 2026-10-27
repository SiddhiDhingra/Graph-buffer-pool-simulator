"""Separate benchmark runner for THGB DBLP-TA.

Workloads are generated directly over logical pages:
- random: uniformly sampled page IDs
- graph: transitions over the collapsed undirected page graph
- semantic: transitions among LSH candidate pages

This is a simulator experiment, not a benchmark of a production DBMS.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

from src.access_score import AccessScoreTracker
from src.buffer_pool import BufferPool
from src.graph_locality import PageGraph
from src.lsh import LSHManager
from src.page import Page
from src.proposed_policy import GraphLSHAwareEvictionPolicy
from src.replacement_policies import FIFOPolicy, LRUPolicy
from src.thgb_graph_adapter import build_thgb_page_graph
from src.thgb_page_vectors import build_thgb_page_vectors
from src.usefulness_score import UsefulnessCalculator


def build_storage(num_nodes: int, page_size: int) -> dict[int, Page]:
    """Create contiguous logical pages, including a final partial page."""
    storage = {}
    for page_id, start in enumerate(range(0, num_nodes, page_size)):
        stop = min(start + page_size, num_nodes)
        storage[page_id] = Page(
            page_id=page_id,
            node_ids=tuple(range(start, stop)),
        )
    return storage


def count_nodes(node_path: Path) -> int:
    with node_path.open("r", encoding="utf-8") as stream:
        return sum(1 for _ in stream)


def make_workload(
    kind: str,
    length: int,
    seed: int,
    page_ids: list[int],
    page_graph: PageGraph,
    lsh: LSHManager,
) -> list[int]:
    """Generate deterministic page-ID requests independently of policy."""
    rng = random.Random(seed)
    if not page_ids or length < 1:
        return []

    trace = []
    page_id_set = set(page_ids)
    current = rng.choice(page_ids)

    for _ in range(length):
        trace.append(current)

        if kind == "random":
            current = rng.choice(page_ids)

        elif kind == "graph":
            neighbors = sorted(page_graph.neighbors(current))
            current = rng.choice(neighbors) if neighbors else rng.choice(page_ids)

        elif kind == "semantic":
            candidates = [
                p for p in lsh.get_candidates(current)
                if p != current and p in page_id_set
            ]
            current = rng.choice(candidates) if candidates else rng.choice(page_ids)

        else:
            raise ValueError(f"Unknown workload: {kind}")

    return trace


def make_policy(
    name: str,
    page_graph: PageGraph,
    lsh: LSHManager,
    shared_calculator=None,
):
    if name == "FIFO":
        return FIFOPolicy()

    if name == "LRU":
        return LRUPolicy()

    if name == "Proposed":
        calculator = shared_calculator

        if calculator is None:
            calculator = UsefulnessCalculator(
                alpha=0.40,
                beta=0.40,
                gamma=0.20,
                graph_mode="directional",
                semantic_probe_radius=1,
                locality_gate=True,
            )

        tracker = AccessScoreTracker(
            recency_tau=20.0,
            frequency_tau=5.0,
        )

        return GraphLSHAwareEvictionPolicy(
            usefulness_calculator=calculator,
            page_graph=page_graph,
            lsh_manager=lsh,
            access_tracker=tracker,
            prepare_static=(shared_calculator is None),
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
        "Hit Ratio (%)": round(pool.hit_ratio * 100.0, 4),
        "Miss Rate (%)": round(
            (pool.page_faults / pool.requests * 100.0)
            if pool.requests else 0.0, 4
        ),
        "Storage Reads": pool.storage_reads,
        "Evictions": pool.evictions,
        "Policy Decision Time (ms)": round(pool.policy_decision_time * 1000, 4),
        "Avg Decision Time (ms)": round(
            pool.policy_decision_time * 1000 / pool.policy_decisions
            if pool.policy_decisions else 0.0, 6
        ),
        "Total Runtime (s)": round(elapsed, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/thgb/DBLP-TA")
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--capacities", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 101])
    parser.add_argument("--lsh-tables", type=int, default=5)
    parser.add_argument("--lsh-planes", type=int, default=6)
    parser.add_argument("--probe-radius", type=int, default=1)
    parser.add_argument("--output-dir", default="results/thgb")
    args = parser.parse_args()

    if args.page_size < 1 or args.requests < 1:
        parser.error("--page-size and --requests must be positive")
    if any(c < 1 for c in args.capacities):
        parser.error("all capacities must be positive")

    root = Path(args.data_dir)
    node_path = root / "node.dat"
    if not node_path.is_file() or not (root / "link.dat").is_file():
        parser.error(f"Expected node.dat and link.dat under {root}")

    print("Counting DBLP-TA nodes...", flush=True)
    num_nodes = count_nodes(node_path)
    storage = build_storage(num_nodes, args.page_size)
    page_ids = sorted(storage)
    print(f"Nodes: {num_nodes:,}; logical pages: {len(page_ids):,}", flush=True)

    print("Building page vectors from node embeddings...", flush=True)
    vector_matrix = build_thgb_page_vectors(
        data_dir=str(root),
        page_size=args.page_size,
        normalize=True,
    )
    if vector_matrix.shape[0] != len(page_ids):
        raise ValueError(
            f"Page-vector count {vector_matrix.shape[0]} does not match "
            f"logical page count {len(page_ids)}"
        )
    page_vectors = {
        page_id: vector_matrix[page_id] for page_id in page_ids
    }

    print("Building collapsed undirected page graph...", flush=True)
    page_graph = build_thgb_page_graph(
        data_dir=str(root),
        page_size=args.page_size,
        num_nodes=num_nodes,
    )

    print("Building LSH index...", flush=True)
    lsh = LSHManager(
        page_vectors,
        num_tables=args.lsh_tables,
        num_planes=args.lsh_planes,
        seed=42,
        probe_radius=args.probe_radius,
    )

    print("Preparing shared usefulness scores...", flush=True)

    shared_calculator = UsefulnessCalculator(
        alpha=0.40,
        beta=0.40,
        gamma=0.20,
        graph_mode="directional",
        semantic_probe_radius=1,
        locality_gate=True,
    )

    shared_calculator.prepare(page_ids, page_graph, lsh)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for workload_name in ("random", "graph", "semantic"):
        for seed in args.seeds:
            print(
                f"Generating trace: workload={workload_name}, seed={seed}",
                flush=True,
            )
            trace = make_workload(
                workload_name, args.requests, seed, page_ids, page_graph, lsh
            )

            # Reuse exactly the same trace across policies and capacities.
            for capacity in args.capacities:
                for policy_name in ("FIFO", "LRU", "Proposed"):
                    print(
                        f"  {workload_name} seed={seed} capacity={capacity} "
                        f"policy={policy_name}",
                        flush=True,
                    )
                    policy = make_policy(
                        policy_name,
                        page_graph,
                        lsh,
                        shared_calculator=shared_calculator,
)
                    metrics = run_trace(trace, storage, capacity, policy)
                    rows.append({
                        "Dataset": "THGB DBLP-TA",
                        "Workload": workload_name,
                        "Seed": seed,
                        "Policy": policy_name,
                        "Page Size": args.page_size,
                        **metrics,
                    })

    csv_path = output_dir / "thgb_results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    metadata = {
        "dataset": "THGB DBLP-TA",
        "data_dir": str(root),
        "nodes": num_nodes,
        "logical_pages": len(page_ids),
        "page_size": args.page_size,
        "vector_shape": list(vector_matrix.shape),
        "workloads": ["random", "graph", "semantic"],
        "policies": ["FIFO", "LRU", "Proposed"],
        "capacities": args.capacities,
        "seeds": args.seeds,
        "requests_per_trace": args.requests,
        "lsh_tables": args.lsh_tables,
        "lsh_planes": args.lsh_planes,
        "probe_radius": args.probe_radius,
        "graph_assumption": (
            "Edge types are collapsed; page connections are symmetrized."
        ),
        "layout_assumption": (
            "Consecutive node IDs are grouped into logical pages; this does "
            "not establish the physical layout of the original database."
        ),
        "semantic_workload_assumption": (
            "Transitions use LSH candidate pages; this is a synthetic workload."
        ),
        "results_csv": str(csv_path),
    }
    metadata_path = output_dir / "thgb_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nExperiment complete.", flush=True)
    print(f"Results:  {csv_path}", flush=True)
    print(f"Metadata: {metadata_path}", flush=True)
    print(f"Rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
