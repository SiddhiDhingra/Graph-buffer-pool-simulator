"""Page-level structural locality derived from the Cora citation graph."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from src.page_builder import PageTable


class PageGraph:
    """Weighted page graph induced by node-level Cora edges."""

    def __init__(self):
        self.weights: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.internal: Dict[int, int] = defaultdict(int)

    def weight(self, a: int, b: int) -> int:
        return self.weights[a].get(b, 0)

    def cross_degree(self, page: int) -> int:
        return sum(self.weights[page].values())

    def neighbors(self, page: int) -> List[int]:
        return list(self.weights[page].keys())

    def transition_probability(self, source: int, target: int) -> float:
        """P(next page=target | current page=source), based on cross-page edges."""
        total = self.cross_degree(source)
        if total <= 0:
            return 0.0
        return self.weight(source, target) / total

    def normalized_neighbor_score(self, source: int, target: int) -> float:
        """Normalize a source's edge weights by its strongest neighbor."""
        max_weight = max(self.weights[source].values(), default=0)
        if max_weight <= 0:
            return 0.0
        return self.weight(source, target) / max_weight


def build_page_graph(graph, page_table: PageTable) -> PageGraph:
    pg = PageGraph()
    for u, v in graph.edges:
        pu = page_table.get_page_for_node(u)
        pv = page_table.get_page_for_node(v)
        if pu == pv:
            pg.internal[pu] += 1
        else:
            pg.weights[pu][pv] += 1
            pg.weights[pv][pu] += 1
    return pg


def calculate_graph_locality(page_a: int, page_b: int, page_graph: PageGraph,
                             mode: str = "jaccard") -> float:
    """Return a structural-locality score in [0,1] for normalized modes.

    Modes:
      count       raw cross-page edge count
      directional transition probability from A to B
      normalized  A->B edge count divided by A's strongest outgoing weight
      jaccard     legacy symmetric overlap score retained for compatibility
    """
    if page_a == page_b:
        return float(page_graph.internal.get(page_a, 0)) if mode == "count" else 1.0

    w = page_graph.weight(page_a, page_b)
    if mode == "count":
        return float(w)
    if mode == "directional":
        return page_graph.transition_probability(page_a, page_b)
    if mode == "normalized":
        return page_graph.normalized_neighbor_score(page_a, page_b)
    if mode == "jaccard":
        ext_a = page_graph.cross_degree(page_a)
        ext_b = page_graph.cross_degree(page_b)
        denom = ext_a + ext_b - w
        return w / denom if denom else 0.0
    raise ValueError(f"unknown mode {mode!r}")


def top_local_pages(page: int, page_graph: PageGraph, k: int = 5,
                    mode: str = "jaccard") -> List[Tuple[int, float]]:
    scored = [
        (p, calculate_graph_locality(page, p, page_graph, mode))
        for p in page_graph.neighbors(page)
    ]
    return sorted(scored, key=lambda x: (-x[1], x[0]))[:k]
