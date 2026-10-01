"""Page-access workload generators.

Each generator returns a list of *node* requests; use to_page_requests()
to convert them to the page-access trace fed to the buffer pool.
"""
from __future__ import annotations

import random
from typing import List

from src.page_builder import PageTable


def to_page_requests(node_requests: List[int], page_table: PageTable) -> List[int]:
    return [page_table.get_page_for_node(n) for n in node_requests]


def random_workload(graph, length: int, seed: int = 0) -> List[int]:
    """Uniform random node accesses (no locality)."""
    rng = random.Random(seed)
    return [rng.randrange(graph.num_nodes) for _ in range(length)]


def graph_traversal_workload(graph, length: int, restart_prob: float = 0.05,
                             seed: int = 0) -> List[int]:
    """Random walk over graph edges, occasionally restarting elsewhere."""
    rng = random.Random(seed)
    cur = rng.randrange(graph.num_nodes)
    out = []
    for _ in range(length):
        out.append(cur)
        nbrs = sorted(graph.neighbors(cur))
        if not nbrs or rng.random() < restart_prob:
            cur = rng.randrange(graph.num_nodes)
        else:
            cur = rng.choice(nbrs)
    return out


def _ball(graph, seed_node: int, radius: int) -> List[int]:
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
    """Stay inside a small graph neighbourhood for `dwell` accesses, then
    move to a new region. Strong temporal + graph locality."""
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


def mixed_workload(graph, length: int, mix=(0.2, 0.4, 0.4),
                   block: int = 100, seed: int = 0) -> List[int]:
    """Interleave random / traversal / local blocks (mix must sum to 1)."""
    if abs(sum(mix) - 1.0) > 1e-9:
        raise ValueError("mix must sum to 1")
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
    return out


WORKLOADS = {
    "random": random_workload,
    "traversal": graph_traversal_workload,
    "local": local_graph_workload,
    "mixed": mixed_workload,
}
