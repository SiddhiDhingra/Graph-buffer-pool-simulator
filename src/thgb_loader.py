"""Loader for the THGB DBLP-TA heterogeneous graph."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


@dataclass
class THGBGraph:
    """DBLP-TA graph. Original node IDs are preserved and contiguous."""
    num_nodes: int
    node_types: np.ndarray
    texts: List[str]
    edges: List[Tuple[int, int, int, float]]
    embeddings: np.ndarray | None = None
    adjacency: Dict[int, Set[int]] = field(default_factory=dict)

    def build_adjacency(self, undirected: bool = True) -> Dict[int, Set[int]]:
        """Build neighbors while retaining typed edges in self.edges."""
        adj = defaultdict(set)
        for source, target, _edge_type, _edge_value in self.edges:
            adj[source].add(target)
            if undirected:
                adj[target].add(source)
        self.adjacency = dict(adj)
        return self.adjacency

    def neighbors(self, node_id: int) -> Set[int]:
        return self.adjacency.get(node_id, set())

    def degree(self, node_id: int) -> int:
        return len(self.neighbors(node_id))


def load_thgb_dblp(
    data_dir: str = "data/thgb/DBLP-TA",
    load_embeddings: bool = False,
    build_adjacency: bool = False,
) -> THGBGraph:
    """
    Load THGB DBLP-TA node.dat and link.dat.

    Node format: ID, text, node type, embedding (tab-separated).
    Edge format: source ID, target ID, edge type, edge value.

    Embeddings are optional because the full matrix is about 717 MiB
    as float32. Labels are intentionally not loaded in this first version.
    """
    root = Path(data_dir)
    node_path = root / "node.dat"
    link_path = root / "link.dat"

    if not node_path.is_file() or not link_path.is_file():
        raise FileNotFoundError(
            f"Expected node.dat and link.dat in {root.resolve()}"
        )

    texts: List[str] = []
    node_types: List[int] = []
    embeddings_list = [] if load_embeddings else None
    expected_dims = None

    with node_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            fields = line.rstrip("\r\n").split("\t", 3)
            if len(fields) != 4:
                raise ValueError(
                    f"Malformed node.dat line {line_no}: expected 4 fields"
                )
            try:
                node_id = int(fields[0])
                node_type = int(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid ID/type at node.dat line {line_no}"
                ) from exc

            if node_id != len(texts):
                raise ValueError(
                    f"Expected contiguous node ID {len(texts)}, got {node_id}"
                )

            texts.append(fields[1])
            node_types.append(node_type)

            if load_embeddings:
                try:
                    vector = np.fromstring(fields[3], sep=",", dtype=np.float32)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid embedding at node.dat line {line_no}"
                    ) from exc
                if vector.size == 0:
                    raise ValueError(
                        f"Empty embedding at node.dat line {line_no}"
                    )
                if expected_dims is None:
                    expected_dims = vector.size
                if vector.size != expected_dims:
                    raise ValueError(
                        f"Inconsistent embedding dimensions at line {line_no}"
                    )
                embeddings_list.append(vector)

    edges: List[Tuple[int, int, int, float]] = []
    num_nodes = len(texts)

    with link_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 4:
                raise ValueError(
                    f"Malformed link.dat line {line_no}: expected 4 fields"
                )
            try:
                source = int(fields[0])
                target = int(fields[1])
                edge_type = int(fields[2])
                edge_value = float(fields[3])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid edge value at link.dat line {line_no}"
                ) from exc

            if not (0 <= source < num_nodes and 0 <= target < num_nodes):
                raise ValueError(
                    f"Edge endpoint out of range at link.dat line {line_no}"
                )
            edges.append((source, target, edge_type, edge_value))

    embeddings = (
        np.stack(embeddings_list).astype(np.float32, copy=False)
        if load_embeddings and embeddings_list
        else None
    )

    graph = THGBGraph(
        num_nodes=num_nodes,
        node_types=np.asarray(node_types, dtype=np.int8),
        texts=texts,
        edges=edges,
        embeddings=embeddings,
    )

    if build_adjacency:
        graph.build_adjacency(undirected=True)

    return graph
