"""Build page-level semantic vectors from THGB DBLP-TA embeddings."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def build_thgb_page_vectors(
    data_dir: str = "data/thgb/DBLP-TA",
    page_size: int = 50,
    normalize: bool = True,
) -> np.ndarray:
    """Stream node embeddings and average them into consecutive-ID pages."""
    if page_size < 1:
        raise ValueError("page_size must be >= 1")

    node_path = Path(data_dir) / "node.dat"
    if not node_path.is_file():
        raise FileNotFoundError(f"Cannot find {node_path.resolve()}")

    page_sums = []
    page_counts = []
    expected_id = 0
    embedding_dims = None

    with node_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            fields = line.rstrip("\r\n").split("\t", 3)
            if len(fields) != 4:
                raise ValueError(f"Malformed node.dat record at line {line_no}")

            node_id = int(fields[0])
            if node_id != expected_id:
                raise ValueError(
                    f"Expected node ID {expected_id}, got {node_id}"
                )

            vector = np.fromstring(fields[3], sep=",", dtype=np.float32)
            if vector.size == 0:
                raise ValueError(f"Empty embedding at line {line_no}")

            if embedding_dims is None:
                embedding_dims = vector.size
            elif vector.size != embedding_dims:
                raise ValueError(
                    f"Inconsistent embedding dimensions at line {line_no}"
                )

            page_id = node_id // page_size

            # Allocate a new accumulator when the next page begins.
            if page_id == len(page_sums):
                page_sums.append(np.zeros(embedding_dims, dtype=np.float64))
                page_counts.append(0)

            page_sums[page_id] += vector
            page_counts[page_id] += 1
            expected_id += 1

    if expected_id == 0:
        raise ValueError("node.dat contains no nodes")

    page_vectors = np.vstack([
        total / count
        for total, count in zip(page_sums, page_counts)
    ]).astype(np.float32)

    if normalize:
        # Center page vectors to reduce the common direction shared by pages.
        page_vectors -= page_vectors.mean(axis=0, keepdims=True)

        norms = np.linalg.norm(page_vectors, axis=1, keepdims=True)
        np.divide(
            page_vectors,
            norms,
            out=page_vectors,
            where=norms > 0,
        )

    return page_vectors
