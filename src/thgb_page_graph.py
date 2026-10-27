"""Map THGB DBLP-TA node edges to logical buffer-pool pages."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np


def build_thgb_page_graph(
    data_dir: str = "data/thgb/DBLP-TA",
    page_size: int = 50,
    num_nodes: int | None = None,
) -> dict:
    """
    Stream link.dat and aggregate node edges into page-level connections.

    Returns directed page-edge counts by edge type, plus summary statistics.
    Original edge types are preserved. Self-page edges are counted separately.
    """
    if page_size < 1:
        raise ValueError("page_size must be >= 1")

    root = Path(data_dir)
    link_path = root / "link.dat"
    node_path = root / "node.dat"

    if not link_path.is_file() or not node_path.is_file():
        raise FileNotFoundError(f"Expected node.dat and link.dat in {root}")

    if num_nodes is None:
        # Count node records without loading their embeddings into memory.
        with node_path.open("r", encoding="utf-8") as f:
            num_nodes = sum(1 for _ in f)

    num_pages = (num_nodes + page_size - 1) // page_size
    page_edge_counts = Counter()
    self_page_counts = Counter()
    edge_type_counts = Counter()
    edge_records = 0

    with link_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"Malformed link.dat line {line_no}")

            source, target, edge_type = (
                int(fields[0]), int(fields[1]), int(fields[2])
            )
            edge_value = float(fields[3])

            if not (0 <= source < num_nodes and 0 <= target < num_nodes):
                raise ValueError(f"Out-of-range endpoint at link.dat line {line_no}")

            source_page = source // page_size
            target_page = target // page_size
            edge_records += 1
            edge_type_counts[edge_type] += 1

            if source_page == target_page:
                self_page_counts[edge_type] += 1
            else:
                page_edge_counts[
                    (source_page, target_page, edge_type)
                ] += 1

    return {
        "num_nodes": num_nodes,
        "num_pages": num_pages,
        "page_size": page_size,
        "edge_records": edge_records,
        "edge_type_counts": dict(edge_type_counts),
        "self_page_edge_counts": dict(self_page_counts),
        "cross_page_edge_counts": dict(page_edge_counts),
        "num_unique_cross_page_connections": len(page_edge_counts),
    }
