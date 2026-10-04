from __future__ import annotations

from src.data_loader import load_cora
from src.page_builder import build_pages
from src.page_vectors import build_page_vectors
from src.graph_locality import build_page_graph
from src.workload_generator import (
    random_workload,
    graph_traversal_workload,
    local_graph_workload,
    mixed_workload,
    to_page_requests,
)

from src.buffer_pool import BufferPool
from src.replacement_policies import FIFOPolicy, LRUPolicy
from src.lsh import LSHManager
from src.access_score import AccessScoreTracker
from src.usefulness_score import UsefulnessCalculator
from src.proposed_policy import GraphLSHAwareEvictionPolicy

from evaluation.metrics import EvaluationMetrics


def create_proposed_policy(graph, page_table):
    """Create the proposed Graph + LSH + Access eviction policy."""

    page_graph = build_page_graph(graph, page_table)

    page_vectors = build_page_vectors(graph, page_table)

    lsh_manager = LSHManager(
        page_vectors,
        num_tables=5,
        num_planes=8,
        seed=42
    )

    access_tracker = AccessScoreTracker()

    usefulness_calculator = UsefulnessCalculator(
        alpha=0.2,
        beta=0.2,
        gamma=0.6
    )

    policy = GraphLSHAwareEvictionPolicy(
        usefulness_calculator=usefulness_calculator,
        page_graph=page_graph,
        lsh_manager=lsh_manager,
        access_tracker=access_tracker
    )

    return policy


def run_policy(page_requests, storage_pages, capacity, policy):
    """Run one policy on the same page-request workload."""

    buffer_pool = BufferPool(
        capacity=capacity,
        storage_pages=storage_pages,
        replacement_policy=policy
    )

    for page_id in page_requests:
        buffer_pool.request_page(page_id)

    metrics = EvaluationMetrics()

    metrics.hits = buffer_pool.hits
    metrics.misses = buffer_pool.misses
    metrics.page_faults = buffer_pool.page_faults
    metrics.storage_reads = buffer_pool.storage_reads
    metrics.evictions = buffer_pool.evictions

    return metrics.summary()


def generate_workloads(graph, page_table):
    """Generate the workloads used for comparison."""

    workloads = {}

    random_nodes = random_workload(
        graph,
        length=1000,
        seed=42
    )

    traversal_nodes = graph_traversal_workload(
        graph,
        length=1000,
        restart_prob=0.1,
        seed=42
    )

    local_nodes = local_graph_workload(
        graph,
        length=1000,
        radius=2,
        seed=42
    )

    mixed_nodes = mixed_workload(
        graph,
        length=1000,
        seed=42
    )

    workloads["Random"] = to_page_requests(
        random_nodes,
        page_table
    )

    workloads["Graph Traversal"] = to_page_requests(
        traversal_nodes,
        page_table
    )

    workloads["Local Graph"] = to_page_requests(
        local_nodes,
        page_table
    )

    workloads["Mixed"] = to_page_requests(
        mixed_nodes,
        page_table
    )

    return workloads

def run_comparison_experiment():
    print("Loading Cora dataset...")

    graph = load_cora()

    print(f"Nodes: {graph.num_nodes}")
    print(f"Edges: {len(graph.edges)}")

    page_table = build_pages(
        graph,
        page_size=50
    )

    print(f"Logical pages: {page_table.num_pages}")

    workloads = generate_workloads(
        graph,
        page_table
    )

    capacities = [3, 5, 10]

    policies = [
        "FIFO",
        "LRU",
        "Proposed"
    ]

    results = {}

    storage_pages = page_table.pages

    for workload_name, page_requests in workloads.items():

        print()
        print("=" * 60)
        print(f"WORKLOAD: {workload_name}")
        print(f"Requests: {len(page_requests)}")
        print("=" * 60)

        results[workload_name] = {}

        for capacity in capacities:

            print()
            print(f"Buffer Capacity: {capacity}")
            print("-" * 60)

            results[workload_name][capacity] = {}

            for policy_name in policies:

                if policy_name == "FIFO":
                    policy = FIFOPolicy()

                elif policy_name == "LRU":
                    policy = LRUPolicy()

                else:
                    policy = create_proposed_policy(
                        graph,
                        page_table
                    )

                summary = run_policy(
                    page_requests,
                    storage_pages,
                    capacity,
                    policy
                )

                results[workload_name][capacity][policy_name] = summary

                print(
                    f"{policy_name:10} | "
                    f"Hit Ratio: {summary['Hit Ratio']:6.2f}% | "
                    f"Faults: {summary['Page Faults']:4} | "
                    f"I/O: {summary['Storage I/O']:4} | "
                    f"Evictions: {summary['Evictions']:4} | "
                    f"Latency: {summary['Simulated Latency']}"
                )

    return results


if __name__ == "__main__":
    run_comparison_experiment()