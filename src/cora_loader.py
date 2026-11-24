
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
            line = line.strip()
            if not line:
                continue

            if "," in line:
                fields = [x.strip() for x in line.split(",")]
                if len(fields) < 3:
                    raise ValueError(f"Invalid content row: {line_no}")

                node_id = fields[0]
                label = fields[1]
                feature_values = fields[2:]
            else:
                fields = line.split()
                if len(fields) < 3:
                    raise ValueError(f"Invalid content row: {line_no}")

                node_id = fields[0]
                label = fields[-1]
                feature_values = fields[1:-1]

            try:
                features = [float(x) for x in feature_values]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid feature value at content row {line_no}"
                ) from exc

            original_ids.append(node_id)
            labels.append(label)
            feature_rows.append(features)

    if len(set(original_ids)) != len(original_ids):
        raise ValueError("Duplicate node IDs in Cora content file")

    if not feature_rows or len(set(map(len, feature_rows))) != 1:
        raise ValueError("Inconsistent or empty Cora feature vectors")

    id_to_index = {
        original_id: index
        for index, original_id in enumerate(original_ids)
    }

    features = np.asarray(feature_rows, dtype=np.float32)

    edges = set()

    with cites_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue

            fields = (
                [x.strip() for x in line.split(",")]
                if "," in line
                else line.split()
            )

            if len(fields) != 2:
                raise ValueError(f"Invalid citation row: {line_no}")

            source, target = fields

            if source not in id_to_index or target not in id_to_index:
                raise ValueError(
                    f"Unknown node ID at citation row {line_no}: "
                    f"{source}, {target}"
                )

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
