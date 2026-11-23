
"""Loader for the Cora citation network."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CoraGraph:
    num_nodes: int
    features: np.ndarray
    edges: list
    original_ids: list
    labels: list


def load_cora(data_dir: str = "data/cora") -> CoraGraph:
    root = Path(data_dir)
    content_path = root / "cora.content"
    cites_path = root / "cora.cites"

    if not content_path.is_file() or not cites_path.is_file():
        raise FileNotFoundError(f"Expected Cora files in {root}")

    original_ids = []
    labels = []
    feature_rows = []

    with content_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            fields = line.split()

            if len(fields) < 3:
                raise ValueError(f"Invalid content row: {line_no}")

            original_ids.append(fields[0])
            feature_rows.append([float(x) for x in fields[1:-1]])
            labels.append(fields[-1])

    if len(set(original_ids)) != len(original_ids):
        raise ValueError("Duplicate node IDs in Cora content file")

    id_to_index = {
        original_id: index
        for index, original_id in enumerate(original_ids)
    }

    features = np.asarray(feature_rows, dtype=np.float32)

    edges = set()

    with cites_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            fields = line.split()

            if len(fields) != 2:
                raise ValueError(f"Invalid citation row: {line_no}")

            source, target = fields

            if source not in id_to_index or target not in id_to_index:
                raise ValueError(f"Unknown node ID at citation row {line_no}")

            u = id_to_index[source]
            v = id_to_index[target]

            if u != v:
                edges.add((min(u, v), max(u, v)))

    return CoraGraph(
        num_nodes=len(original_ids),
        features=features,
        edges=sorted(edges),
        original_ids=original_ids,
        labels=labels
    )
