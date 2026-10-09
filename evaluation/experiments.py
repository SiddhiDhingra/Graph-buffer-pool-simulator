"""
Research-grade Cora buffer-pool evaluation.

The console output is intentionally compact and easy to compare with
baseline implementations while the CSV files retain the full ablation study.

Policies:
    FIFO
    LRU
    Graph_Only
    LSH_Only
    Graph_LSH
    Proposed

All policies receive exactly the same request trace for every
(workload, seed, capacity) combination.
"""

from __future__ import annotations

import csv
import json
import os

from dataclasses import dataclass, asdict
from statistics import mean, stdev
from typing import Dict, Iterable, List

from src.access_score import AccessScoreTracker
from src.buffer_pool import BufferPool
from src.data_loader import load_cora
from src.graph_locality import build_page_graph
from src.lsh import LSHManager
from src.page_builder import build_pages
from src.page_vectors import build_page_vectors
from src.proposed_policy import GraphLSHAwareEvictionPolicy
from src.replacement_policies import FIFOPolicy, LRUPolicy
from src.usefulness_score import UsefulnessCalculator

from src.workload_generator import (
    graph_semantic_workload,
    graph_traversal_workload,
    mixed_workload,
    random_workload,
    semantic_page_workload,
    to_page_requests,
)

from evaluation.metrics import EvaluationMetrics
from evaluation.statistics import paired_effect


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ExperimentConfig:

    data_dir: str = "data/cora"

    page_size: int = 50

    requests: int = 5000

    capacities: tuple[int, ...] = (3, 5, 10)

    # 10 paired seeds
    seeds: tuple[int, ...] = (
        7,
        19,
        42,
        73,
        101,
        137,
        211,
        307,
        401,
        503,
    )

    # LSH configuration
    lsh_tables: int = 5
    lsh_planes: int = 6
    lsh_probe_radius: int = 1


# ============================================================
# PROPOSED POLICY
# ============================================================

def create_proposed_policy(
    graph,
    page_table,
    *,
    variant: str = "proposed",
    lsh_tables: int = 5,
    lsh_planes: int = 6,
    lsh_probe_radius: int = 1,
):

    page_graph = build_page_graph(graph, page_table)

    page_vectors = build_page_vectors(graph, page_table)

    lsh_manager = LSHManager(
        page_vectors,
        num_tables=lsh_tables,
        num_planes=lsh_planes,
        seed=42,
        probe_radius=lsh_probe_radius,
    )

    access_tracker = AccessScoreTracker(
        recency_tau=20.0,
        frequency_tau=5.0,
    )

    # --------------------------------------------------------
    # Ablation configurations
    # --------------------------------------------------------

    variants = {

        # Proposed method:
        # Graph + LSH + access history
        "proposed": dict(
            alpha=0.40,
            beta=0.40,
            gamma=0.20,
            gate=True,
        ),

        # Graph only
        "graph_only": dict(
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
            gate=False,
        ),

        # LSH only
        "lsh_only": dict(
            alpha=0.0,
            beta=1.0,
            gamma=0.0,
            gate=False,
        ),

        # Access-only ablation
        "access_only": dict(
            alpha=0.0,
            beta=0.0,
            gamma=1.0,
            gate=False,
        ),

        # Graph + LSH without access history
        "graph_lsh": dict(
            alpha=0.45,
            beta=0.55,
            gamma=0.0,
            gate=False,
        ),
    }

    if variant not in variants:
        raise ValueError(f"Unknown policy variant: {variant}")

    v = variants[variant]

    calculator = UsefulnessCalculator(
        alpha=v["alpha"],
        beta=v["beta"],
        gamma=v["gamma"],
        graph_mode="directional",
        semantic_probe_radius=lsh_probe_radius,
        locality_gate=v["gate"],
    )

    return GraphLSHAwareEvictionPolicy(
        usefulness_calculator=calculator,
        page_graph=page_graph,
        lsh_manager=lsh_manager,
        access_tracker=access_tracker,
    )


# ============================================================
# RUN ONE POLICY
# ============================================================

def run_policy(
    page_requests: Iterable[int],
    storage_pages,
    capacity: int,
    policy,
    buffer_cost: float = 1.0,
    storage_cost: float = 10.0,
) -> dict:

    buffer_pool = BufferPool(
        capacity,
        storage_pages,
        policy,
    )

    for page_id in page_requests:
        buffer_pool.request_page(page_id)

    metrics = EvaluationMetrics(
        buffer_cost=buffer_cost,
        storage_cost=storage_cost,
    )

    metrics.hits = buffer_pool.hits
    metrics.misses = buffer_pool.misses
    metrics.page_faults = buffer_pool.page_faults
    metrics.storage_reads = buffer_pool.storage_reads
    metrics.evictions = buffer_pool.evictions

    metrics.policy_decision_time = buffer_pool.policy_decision_time
    metrics.policy_decisions = buffer_pool.policy_decisions

    return metrics.summary()


# ============================================================
# WORKLOAD GENERATION
# ============================================================

def generate_workloads(
    graph,
    page_table,
    page_vectors,
    length: int,
    seed: int,
):

    """
    Generate all workload traces.

    IMPORTANT:
    Every policy receives exactly the same trace generated here.
    """

    # --------------------------------------------------------
    # Random workload
    # --------------------------------------------------------

    random_trace = to_page_requests(
        random_workload(
            graph,
            length,
            seed=seed,
        ),
        page_table,
    )

    # --------------------------------------------------------
    # Graph workload
    # --------------------------------------------------------

    graph_trace = to_page_requests(
        graph_traversal_workload(
            graph,
            length,
            restart_prob=0.05,
            seed=seed,
        ),
        page_table,
    )

    # --------------------------------------------------------
    # Vector / semantic workload
    #
    # This is the workload where LSH should have its strongest
    # opportunity because consecutive pages are selected using
    # semantic similarity.
    # --------------------------------------------------------

    vector_trace = to_page_requests(
        semantic_page_workload(
            graph,
            page_table,
            page_vectors,
            length,
            top_k=8,
            dwell=20,
            seed=seed,
        ),
        page_table,
    )

    # --------------------------------------------------------
    # Mixed workload
    #
    # Contains graph + temporal + semantic behavior.
    # --------------------------------------------------------

    mixed_trace = to_page_requests(
        graph_semantic_workload(
            graph,
            page_table,
            page_vectors,
            length,
            graph_probability=0.50,
            top_k_semantic=8,
            seed=seed,
        ),
        page_table,
    )

    return {
        "Random": random_trace,
        "Graph": graph_trace,
        "Vector": vector_trace,
        "Mixed": mixed_trace,
    }


# ============================================================
# POLICY FACTORY
# ============================================================

def _policy_factory(
    name: str,
    graph,
    page_table,
    config: ExperimentConfig,
):

    if name == "FIFO":
        return FIFOPolicy()

    if name == "LRU":
        return LRUPolicy()

    return create_proposed_policy(
        graph,
        page_table,
        variant=name.lower(),
        lsh_tables=config.lsh_tables,
        lsh_planes=config.lsh_planes,
        lsh_probe_radius=config.lsh_probe_radius,
    )


# ============================================================
# MAIN RESEARCH SUITE
# ============================================================

def run_research_suite(
    config: ExperimentConfig | None = None,
    output_dir: str = "results",
) -> List[dict]:

    config = config or ExperimentConfig()

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load Cora
    # --------------------------------------------------------

    graph = load_cora(
        config.data_dir,
        strict=True,
    )

    page_table = build_pages(
        graph,
        page_size=config.page_size,
    )

    page_vectors = build_page_vectors(
        graph,
        page_table,
    )

    storage_pages = page_table.pages

    # Full ablation set retained in CSV
    policy_names = [
        "FIFO",
        "LRU",
        "Graph_Only",
        "LSH_Only",
        "Graph_LSH",
        "Proposed",
    ]

    rows: List[dict] = []

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("Cora buffer-pool evaluation")
    print(
        f"Nodes: {graph.num_nodes} | "
        f"Edges: {len(graph.edges)}"
    )

    print(
        f"Logical pages: {page_table.num_pages}"
    )

    print(
        "Workloads: Random, Graph, Vector, Mixed"
    )

    print(
        f"Running {len(config.seeds)} paired seeds; "
        "each policy gets the same trace per seed."
    )

    # --------------------------------------------------------
    # Experiments
    # --------------------------------------------------------

    for workload_name in [
        "Random",
        "Graph",
        "Vector",
        "Mixed",
    ]:

        print("\n" + "=" * 72)
        print(f"WORKLOAD: {workload_name}")

        if workload_name == "Random":
            print(
                "No vector query is present; LSH has no query signal."
            )

        elif workload_name == "Graph":
            print(
                "Graph traversal locality is present."
            )

        elif workload_name == "Vector":
            print(
                "LSH semantic locality is present."
            )

        elif workload_name == "Mixed":
            print(
                "Graph + semantic locality is present."
            )

        # ----------------------------------------------------
        # Generate paired traces
        # ----------------------------------------------------

        traces_by_seed = {}

        for seed in config.seeds:

            workloads = generate_workloads(
                graph,
                page_table,
                page_vectors,
                config.requests,
                seed,
            )

            traces_by_seed[seed] = workloads[workload_name]

        # ----------------------------------------------------
        # Capacity loop
        # ----------------------------------------------------

        for capacity in config.capacities:

            print()
            print(f"Buffer capacity: {capacity}")

            print(
                "Policy                 "
                "Faults (mean +/- SD)     "
                "Hit ratio (mean +/- SD)"
            )

            print("-" * 72)

            # Store results by policy for summary printing
            capacity_results = {}

            # ------------------------------------------------
            # Every policy uses the exact same seed traces
            # ------------------------------------------------

            for policy_name in policy_names:

                seed_rows = []

                for seed in config.seeds:

                    trace = traces_by_seed[seed]

                    policy = _policy_factory(
                        policy_name,
                        graph,
                        page_table,
                        config,
                    )

                    summary = run_policy(
                        trace,
                        storage_pages,
                        capacity,
                        policy,
                    )

                    row = {
                        "Seed": seed,
                        "Workload": workload_name,
                        "Capacity": capacity,
                        "Policy": policy_name,
                        **summary,
                    }

                    rows.append(row)
                    seed_rows.append(row)

                capacity_results[policy_name] = seed_rows

            # ------------------------------------------------
            # Console output: primary comparison
            # ------------------------------------------------

            for policy_name in [
                "FIFO",
                "LRU",
                "Proposed",
            ]:

                items = capacity_results[policy_name]

                faults = [
                    float(x["Page Faults"])
                    for x in items
                ]

                hit_ratios = [
                    float(x["Hit Ratio"])
                    for x in items
                ]

                fault_mean = mean(faults)
                fault_sd = (
                    stdev(faults)
                    if len(faults) > 1
                    else 0.0
                )

                hr_mean = mean(hit_ratios)
                hr_sd = (
                    stdev(hit_ratios)
                    if len(hit_ratios) > 1
                    else 0.0
                )

                label = (
                    "Graph+LSH+Access"
                    if policy_name == "Proposed"
                    else policy_name
                )

                print(
                    f"{label:<22}"
                    f"{fault_mean:8.1f} +/- {fault_sd:5.1f}"
                    f"{hr_mean:13.2f}% +/- {hr_sd:5.2f}"
                )

            # ------------------------------------------------
            # Paired comparison: Proposed vs LRU
            # ------------------------------------------------

            lru_rows = capacity_results["LRU"]
            proposed_rows = capacity_results["Proposed"]

            lru_faults = [
                float(x["Page Faults"])
                for x in lru_rows
            ]

            proposed_faults = [
                float(x["Page Faults"])
                for x in proposed_rows
            ]

            fault_diffs = [
                p - l
                for p, l in zip(
                    proposed_faults,
                    lru_faults,
                )
            ]

            mean_fault_diff = mean(fault_diffs)

            if len(fault_diffs) > 1:
                diff_sd = stdev(fault_diffs)
                ci = 1.96 * diff_sd / (
                    len(fault_diffs) ** 0.5
                )
            else:
                ci = 0.0

            print(
                f"Proposed vs LRU: "
                f"{mean_fault_diff:+.1f} faults "
                f"(approx. 95% CI: "
                f"{mean_fault_diff - ci:+.1f} to "
                f"{mean_fault_diff + ci:+.1f})"
            )

            fifo_rows = capacity_results["FIFO"]

            fifo_faults = [
                float(x["Page Faults"])
                for x in fifo_rows
            ]

            fifo_diffs = [
                p - f
                for p, f in zip(
                    proposed_faults,
                    fifo_faults,
                )
            ]

            mean_fifo_diff = mean(fifo_diffs)

            if len(fifo_diffs) > 1:
                diff_sd = stdev(fifo_diffs)
                ci = 1.96 * diff_sd / (
                    len(fifo_diffs) ** 0.5
                )
            else:
                ci = 0.0

            print(
                f"Proposed vs FIFO: "
                f"{mean_fifo_diff:+.1f} faults "
                f"(approx. 95% CI: "
                f"{mean_fifo_diff - ci:+.1f} to "
                f"{mean_fifo_diff + ci:+.1f})"
            )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    raw_path = os.path.join(
        output_dir,
        "raw_results.csv",
    )

    with open(
        raw_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_rows = summarize_results(rows)

    summary_path = os.path.join(
        output_dir,
        "summary_results.csv",
    )

    with open(
        summary_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # ========================================================
    # PAIRED STATISTICS
    # ========================================================

    paired_rows = paired_comparisons(
        rows,
        baseline="LRU",
        target="Proposed",
    )

    if paired_rows:

        paired_path = os.path.join(
            output_dir,
            "paired_statistics.csv",
        )

        with open(
            paired_path,
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(paired_rows[0].keys()),
            )

            writer.writeheader()
            writer.writerows(paired_rows)

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "config": asdict(config),
        "dataset": "Cora",
        "nodes": graph.num_nodes,
        "edges": len(graph.edges),
        "pages": page_table.num_pages,
        "page_size": config.page_size,
        "policy_names": policy_names,
        "workloads": [
            "Random",
            "Graph",
            "Vector",
            "Mixed",
        ],
    }

    with open(
        os.path.join(
            output_dir,
            "experiment_metadata.json",
        ),
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    return rows


# ============================================================
# SUMMARY
# ============================================================

def summarize_results(
    rows: List[dict],
) -> List[dict]:

    groups: Dict[tuple, List[dict]] = {}

    for row in rows:

        key = (
            row["Workload"],
            row["Capacity"],
            row["Policy"],
        )

        groups.setdefault(
            key,
            [],
        ).append(row)

    out = []

    for (
        workload,
        capacity,
        policy,
    ), items in sorted(groups.items()):

        def agg(name):

            vals = [
                float(x[name])
                for x in items
            ]

            return (
                mean(vals),
                stdev(vals)
                if len(vals) > 1
                else 0.0,
            )

        hr, hr_sd = agg("Hit Ratio")
        faults, faults_sd = agg("Page Faults")
        io, io_sd = agg("Storage I/O")
        latency, latency_sd = agg(
            "Simulated Latency"
        )
        decision, decision_sd = agg(
            "Avg Decision Time (ms)"
        )

        out.append({

            "Workload": workload,

            "Capacity": capacity,

            "Policy": policy,

            "Hit Ratio Mean (%)":
                round(hr, 4),

            "Hit Ratio SD (%)":
                round(hr_sd, 4),

            "Page Faults Mean":
                round(faults, 4),

            "Page Faults SD":
                round(faults_sd, 4),

            "Storage I/O Mean":
                round(io, 4),

            "Storage I/O SD":
                round(io_sd, 4),

            "Simulated Latency Mean":
                round(latency, 4),

            "Simulated Latency SD":
                round(latency_sd, 4),

            "Avg Decision Time Mean (ms)":
                round(decision, 6),

            "Avg Decision Time SD (ms)":
                round(decision_sd, 6),
        })

    return out


# ============================================================
# PAIRED STATISTICS
# ============================================================

def paired_comparisons(
    rows: List[dict],
    baseline: str = "LRU",
    target: str = "Proposed",
) -> List[dict]:

    grouped = {}

    for row in rows:

        key = (
            row["Workload"],
            row["Capacity"],
        )

        grouped.setdefault(
            key,
            {},
        )

        grouped[key].setdefault(
            row["Policy"],
            {},
        )[row["Seed"]] = row

    out = []

    for (
        workload,
        capacity,
    ), policies in sorted(grouped.items()):

        if (
            baseline not in policies
            or target not in policies
        ):
            continue

        seeds = sorted(
            set(
                policies[baseline]
            )
            &
            set(
                policies[target]
            )
        )

        if not seeds:
            continue

        base_hr = [
            policies[baseline][s]["Hit Ratio"]
            for s in seeds
        ]

        target_hr = [
            policies[target][s]["Hit Ratio"]
            for s in seeds
        ]

        stats = paired_effect(
            target_hr,
            base_hr,
        )

        out.append({

            "Workload": workload,

            "Capacity": capacity,

            "Baseline": baseline,

            "Target": target,

            "Seeds": len(seeds),

            "Target HR Mean (%)":
                round(
                    mean(target_hr),
                    4,
                ),

            "Baseline HR Mean (%)":
                round(
                    mean(base_hr),
                    4,
                ),

            "Mean HR Improvement (percentage points)":
                round(
                    stats["mean_diff"],
                    4,
                ),

            "SD of Paired Difference":
                round(
                    stats["sd_diff"],
                    4,
                ),

            "Cohens dz":
                round(
                    stats["cohens_dz"],
                    4,
                )
                if stats["cohens_dz"]
                not in (
                    float("inf"),
                    float("-inf"),
                )
                else str(
                    stats["cohens_dz"]
                ),

            "Exact Paired p-value":
                round(
                    stats["p_value"],
                    6,
                ),
        })

    return out


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_research_suite()