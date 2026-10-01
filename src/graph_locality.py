"""Page-level graph and graph-locality scores.

Node 10 -> Node 70 with Node 10 in Page 0 and Node 70 in Page 1 gives a
Page 0 -> Page 1 relationship. The weight is the number of such edges.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from src.page_builder import PageTable


class PageGraph:
    def __init__(self):
        # weights[a][b] = number of node-level edges between page a and b
        self.weights: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.internal: Dict[int, int] = defaultdict(int)   # edges inside a page

    def weight(self, a: int, b: int) -> int:
        return self.weights[a].get(b, 0) if a in self.weights else 0

    def cross_degree(self, page: int) -> int:
        return sum(self.weights[page].values()) if page in self.weights else 0

    def neighbors(self, page: int) -> List[int]:
        return list(self.weights[page].keys()) if page in self.weights else []


def build_page_graph(graph, page_table: PageTable) -> PageGraph:
    pg = PageGraph()
    for u, v in graph.edges:
        pu, pv = page_table.get_page_for_node(u), page_table.get_page_for_node(v)
        if pu == pv:
            pg.internal[pu] += 1
        else:
            pg.weights[pu][pv] += 1
            pg.weights[pv][pu] += 1
    return pg


def calculate_graph_locality(page_a: int, page_b: int, page_graph: PageGraph,
                             mode: str = "jaccard") -> float:
    """How strongly page_a and page_b are connected in the original graph.

    mode="count"       raw number of edges between the pages
    mode="jaccard"     w / (ext(a) + ext(b) - w), symmetric, in [0, 1]
    mode="directional" w / ext(a): chance an edge leaving A lands in B
                       (useful for prefetch decisions)
    """
    if page_a == page_b:
        return float(page_graph.internal.get(page_a, 0)) if mode == "count" else 1.0
    w = page_graph.weight(page_a, page_b)
    if mode == "count":
        return float(w)
    ext_a, ext_b = page_graph.cross_degree(page_a), page_graph.cross_degree(page_b)
    if mode == "directional":
        return w / ext_a if ext_a else 0.0
    if mode == "jaccard":
        denom = ext_a + ext_b - w
        return w / denom if denom else 0.0
    raise ValueError(f"unknown mode {mode!r}")


def top_local_pages(page: int, page_graph: PageGraph, k: int = 5,
                    mode: str = "jaccard") -> List[Tuple[int, float]]:
    scored = [(p, calculate_graph_locality(page, p, page_graph, mode))
              for p in page_graph.neighbors(page)]
    return sorted(scored, key=lambda x: -x[1])[:k]
