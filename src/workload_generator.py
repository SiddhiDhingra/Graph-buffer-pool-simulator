"""Reproducible workload generators over the fixed Cora dataset.

Generators return node requests.  The buffer pool receives page requests after
``to_page_requests`` maps nodes to logical pages.
"""
from __future__ import annotations

import random
from typing import Dict, List

import numpy as np

from src.page_builder import PageTable


def to_page_requests(node_requests: List[int], page_table: PageTable) -> List[int]:
    return [page_table.get_page_for_node(n) for n in node_requests]


def random_workload(graph, length: int, seed: int = 0) -> List[int]:
    rng = random.Random(seed)
    return [rng.randrange(graph.num_nodes) for _ in range(length)]


def graph_traversal_workload(graph, length: int, restart_prob: float = 0.05,
                             seed: int = 0) -> List[int]:
    if not 0.0 <= restart_prob <= 1.0:
        raise ValueError("restart_prob must be in [0,1]")
    rng = random.Random(seed)
    cur = rng.randrange(graph.num_nodes)
    out: List[int] = []
    for _ in range(length):
        out.append(cur)
        nbrs = sorted(graph.neighbors(cur))
        if not nbrs or rng.random() < restart_prob:
            cur = rng.randrange(graph.num_nodes)
        else:
            cur = rng.choice(nbrs)
    return out


def _ball(graph, seed_node: int, radius: int) -> List[int]:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    seen, frontier = {seed_node}, [seed_node]
    for _ in range(radius):
        nxt = []
        for n in frontier:
            for m in graph.neighbors(n):
                if m not in seen:
                    seen.add(m)
                    nxt.append(m)
        frontier = nxt
    return sorted(seen)


def local_graph_workload(graph, length: int, radius: int = 2,
                         dwell: int = 200, seed: int = 0) -> List[int]:
    if dwell <= 0:
        raise ValueError("dwell must be positive")
    rng = random.Random(seed)
    out: List[int] = []
    while len(out) < length:
        region = _ball(graph, rng.randrange(graph.num_nodes), radius)
        region_set = set(region)
        cur = rng.choice(region)
        for _ in range(min(dwell, length - len(out))):
            out.append(cur)
            nbrs = sorted(m for m in graph.neighbors(cur) if m in region_set)
            cur = rng.choice(nbrs) if nbrs and rng.random() < 0.8 else rng.choice(region)
    return out


def _page_similarity_neighbors(page_vectors: Dict[int, np.ndarray], page_id: int,
                                top_k: int = 8) -> List[int]:
    target = page_vectors[page_id]
    norm = np.linalg.norm(target)
    if norm == 0:
        return []
    scored = []
    for pid, vector in page_vectors.items():
        if pid == page_id:
            continue
        vnorm = np.linalg.norm(vector)
        score = float(np.dot(target, vector) / (norm * vnorm)) if vnorm else 0.0
        scored.append((pid, score))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [pid for pid, _ in scored[:top_k]]


def semantic_page_workload(graph, page_table: PageTable,
                           page_vectors: Dict[int, np.ndarray],
                           length: int, top_k: int = 8,
                           dwell: int = 30, seed: int = 0) -> List[int]:
    """Generate node accesses that move among semantically similar pages."""
    if length <= 0:
        return []
    if top_k <= 0 or dwell <= 0:
        raise ValueError("top_k and dwell must be positive")

    rng = random.Random(seed)
    page_ids = sorted(page_table.pages)
    neighbors = {
        pid: _page_similarity_neighbors(page_vectors, pid, top_k)
        for pid in page_ids
    }

    out: List[int] = []
    current_page = rng.choice(page_ids)
    while len(out) < length:
        for _ in range(min(dwell, length - len(out))):
            nodes = page_table.get_nodes_in_page(current_page)
            out.append(rng.choice(nodes))
        options = neighbors.get(current_page, [])
        current_page = rng.choice(options) if options else rng.choice(page_ids)
    return out


def graph_semantic_workload(graph, page_table: PageTable,
                            page_vectors: Dict[int, np.ndarray], length: int,
                            graph_probability: float = 0.55,
                            top_k_semantic: int = 8, seed: int = 0) -> List[int]:
    """Combine graph-neighbor and semantic-neighbor page transitions.

    This workload is deliberately independent of the eviction policy.  It is
    used to test whether the proposed Graph+LSH method can exploit both forms
    of locality under the same request stream as LRU/FIFO.
    """
    if length <= 0:
        return []
    if not 0.0 <= graph_probability <= 1.0:
        raise ValueError("graph_probability must be in [0,1]")

    rng = random.Random(seed)
    page_ids = sorted(page_table.pages)
    semantic_neighbors = {
        pid: _page_similarity_neighbors(page_vectors, pid, top_k_semantic)
        for pid in page_ids
    }

    page_graph = None
    from src.graph_locality import build_page_graph
    page_graph = build_page_graph(graph, page_table)

    out: List[int] = []
    current_page = rng.choice(page_ids)
    for _ in range(length):
        nodes = page_table.get_nodes_in_page(current_page)
        out.append(rng.choice(nodes))

        graph_candidates = page_graph.neighbors(current_page)
        semantic_candidates = semantic_neighbors.get(current_page, [])
        if rng.random() < graph_probability and graph_candidates:
            current_page = rng.choice(sorted(graph_candidates))
        elif semantic_candidates:
            current_page = rng.choice(semantic_candidates)
        elif graph_candidates:
            current_page = rng.choice(sorted(graph_candidates))
        else:
            current_page = rng.choice(page_ids)
    return out


def mixed_workload(graph, length: int, mix=(0.2, 0.4, 0.4),
                   block: int = 100, seed: int = 0) -> List[int]:
    """Legacy mixed workload: random / graph traversal / local graph blocks."""
    if abs(sum(mix) - 1.0) > 1e-9:
        raise ValueError("mix must sum to 1")
    if block <= 0:
        raise ValueError("block must be positive")
    rng = random.Random(seed)
    out: List[int] = []
    i = 0
    while len(out) < length:
        n = min(block, length - len(out))
        kind = rng.choices(("random", "traversal", "local"), weights=mix)[0]
        s = seed * 1000 + i
        i += 1
        if kind == "random":
            out += random_workload(graph, n, s)
        elif kind == "traversal":
            out += graph_traversal_workload(graph, n, seed=s)
        else:
            out += local_graph_workload(graph, n, dwell=n, seed=s)
    return out[:length]


WORKLOADS = {
    "random": random_workload,
    "traversal": graph_traversal_workload,
    "local": local_graph_workload,
    "mixed": mixed_workload,
}
