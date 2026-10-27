"""Adapt THGB DBLP-TA edges to the simulator's PageGraph interface."""
from __future__ import annotations

from pathlib import Path

from src.graph_locality import PageGraph


def build_thgb_page_graph(
    data_dir: str = "data/thgb/DBLP-TA",
    page_size: int = 50,
    num_nodes: int | None = None,
) -> PageGraph:
    """Build a symmetric weighted PageGraph from DBLP-TA edge records."""
    if page_size < 1:
        raise ValueError("page_size must be >= 1")

    root = Path(data_dir)
    node_path = root / "node.dat"
    link_path = root / "link.dat"

    if not node_path.is_file() or not link_path.is_file():
        raise FileNotFoundError(f"Expected node.dat and link.dat in {root}")

    if num_nodes is None:
        with node_path.open("r", encoding="utf-8") as f:
            num_nodes = sum(1 for _ in f)

    page_graph = PageGraph()

    with link_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"Malformed link.dat line {line_no}")

            source, target, _edge_type = (
                int(fields[0]), int(fields[1]), int(fields[2])
            )

            if not (0 <= source < num_nodes and 0 <= target < num_nodes):
                raise ValueError(
                    f"Out-of-range endpoint at link.dat line {line_no}"
                )

            source_page = source // page_size
            target_page = target // page_size

            if source_page == target_page:
                page_graph.internal[source_page] += 1
            else:
                # Match the existing PageGraph convention: symmetric weights.
                page_graph.weights[source_page][target_page] += 1
                page_graph.weights[target_page][source_page] += 1

    return page_graph
