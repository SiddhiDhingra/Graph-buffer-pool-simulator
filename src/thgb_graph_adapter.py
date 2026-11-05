"""Build page-level graph signals from THGB DBLP-TA typed edges.

The current eviction scorer consumes an aggregate PageGraph. This adapter also
retains per-edge-type counts on ``PageGraph.edge_type_weights`` so typed
information is not discarded at ingestion. The current scoring function still
uses aggregate counts; type-aware scoring is a separately testable extension.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from src.graph_locality import PageGraph


def build_thgb_page_graph(data_dir: str = "data/thgb/DBLP-TA", page_size: int = 50, num_nodes: int | None = None) -> PageGraph:
    if page_size < 1:
        raise ValueError("page_size must be >= 1")
    root = Path(data_dir)
    node_path, link_path = root / "node.dat", root / "link.dat"
    if not node_path.is_file() or not link_path.is_file():
        raise FileNotFoundError(f"Expected node.dat and link.dat in {root}")
    if num_nodes is None:
        with node_path.open("r", encoding="utf-8") as f:
            num_nodes = sum(1 for _ in f)
    graph = PageGraph()
    graph.edge_type_weights = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    graph.edge_type_internal = defaultdict(lambda: defaultdict(int))
    with link_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"Malformed link.dat line {line_no}: expected 4 fields")
            source, target, edge_type = int(fields[0]), int(fields[1]), int(fields[2])
            if not (0 <= source < num_nodes and 0 <= target < num_nodes):
                raise ValueError(f"Out-of-range endpoint at link.dat line {line_no}")
            a, b = source // page_size, target // page_size
            if a == b:
                graph.internal[a] += 1
                graph.edge_type_internal[edge_type][a] += 1
            else:
                graph.weights[a][b] += 1
                graph.weights[b][a] += 1
                graph.edge_type_weights[edge_type][a][b] += 1
                graph.edge_type_weights[edge_type][b][a] += 1
    return graph
