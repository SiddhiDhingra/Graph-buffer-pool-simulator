"""Cora dataset loader.

Supports the classic LINQS plain-text release (cora.content / cora.cites),
PyTorch Geometric (optional), and a synthetic Cora-like graph for testing
when the real files are not available.

cora.content : <paper_id> <1433 binary word features> <class_label>
cora.cites   : <cited_paper_id> <citing_paper_id>
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np


@dataclass
class CoraGraph:
    """Graph + vector data. Nodes are re-indexed to 0..num_nodes-1."""
    num_nodes: int
    features: np.ndarray                 # (N, F) float32
    labels: np.ndarray                   # (N,) int
    label_names: List[str]
    edges: List[Tuple[int, int]]         # unique undirected edges (u < v)
    adjacency: Dict[int, Set[int]] = field(default_factory=dict)
    raw_ids: List[str] = field(default_factory=list)  # original paper ids

    def neighbors(self, node_id: int) -> Set[int]:
        return self.adjacency.get(node_id, set())

    def degree(self, node_id: int) -> int:
        return len(self.neighbors(node_id))

    def to_networkx(self):
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(range(self.num_nodes))
        g.add_edges_from(self.edges)
        return g


def _build_adjacency(num_nodes: int, edges) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _finalize(features, labels, label_names, raw_edges, raw_ids) -> CoraGraph:
    n = len(labels)
    edges = sorted({(min(u, v), max(u, v)) for u, v in raw_edges if u != v})
    return CoraGraph(n, features, labels, label_names, edges,
                     _build_adjacency(n, edges), raw_ids)


def load_cora(data_dir: str = "data/cora", strict: bool = False) -> CoraGraph:
    """Load Cora from cora.content / cora.cites in `data_dir`."""
    content = os.path.join(data_dir, "cora.content")
    cites = os.path.join(data_dir, "cora.cites")
    if not (os.path.exists(content) and os.path.exists(cites)):
        raise FileNotFoundError(
            f"Expected {content} and {cites}. Download Cora from the LINQS "
            "page / Network Repository, or use load_cora_pyg() / "
            "make_synthetic_cora().")

    raw_ids, feats, label_strs = [], [], []
    with open(content) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            raw_ids.append(parts[0])
            feats.append([float(x) for x in parts[1:-1]])
            label_strs.append(parts[-1])

    id_to_idx = {rid: i for i, rid in enumerate(raw_ids)}
    label_names = sorted(set(label_strs))
    lab_idx = {name: i for i, name in enumerate(label_names)}
    labels = np.array([lab_idx[s] for s in label_strs], dtype=np.int64)

    raw_edges = []
    with open(cites) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            a, b = parts
            if a in id_to_idx and b in id_to_idx:   # skip dangling citations
                raw_edges.append((id_to_idx[a], id_to_idx[b]))

    result = _finalize(np.array(feats, dtype=np.float32), labels,
                       label_names, raw_edges, raw_ids)
    if strict:
        expected = {
            "nodes": 2708,
            "features": 1433,
            "classes": 7,
            "citation_rows": 5429,
        }
        actual = {
            "nodes": result.num_nodes,
            "features": int(result.features.shape[1]),
            "classes": len(result.label_names),
            "citation_rows": len(raw_edges),
        }
        mismatches = [
            f"{k}: expected {expected[k]}, found {actual[k]}"
            for k in expected if expected[k] != actual[k]
        ]
        if mismatches:
            raise ValueError(
                "The files in data/cora do not look like the standard Cora "
                "release. " + "; ".join(mismatches)
            )
    return result


def load_cora_pyg(root: str = "data/pyg") -> CoraGraph:
    """Load Cora through PyTorch Geometric (requires torch_geometric)."""
    from torch_geometric.datasets import Planetoid
    data = Planetoid(root=root, name="Cora")[0]
    ei = data.edge_index.numpy()
    raw_edges = list(zip(ei[0].tolist(), ei[1].tolist()))
    labels = data.y.numpy().astype(np.int64)
    names = [f"class_{i}" for i in range(int(labels.max()) + 1)]
    return _finalize(data.x.numpy().astype(np.float32), labels, names,
                     raw_edges, [str(i) for i in range(len(labels))])


def make_synthetic_cora(num_nodes: int = 2708, num_edges: int = 5429,
                        num_features: int = 1433, num_classes: int = 7,
                        seed: int = 0, local_frac: float = 0.7) -> CoraGraph:
    """Cora-sized synthetic graph (for tests / offline development).

    `local_frac` of edges connect nearby node ids (creating community
    structure); the rest connect random nodes.
    """
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    edges: Set[Tuple[int, int]] = set()
    while len(edges) < num_edges:
        u = rng.randrange(num_nodes)
        if rng.random() < local_frac:
            v = min(num_nodes - 1, max(0, u + rng.randint(-30, 30)))
        else:
            v = rng.randrange(num_nodes)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    feats = (nprng.random((num_nodes, num_features)) < 0.013).astype(np.float32)
    labels = (np.arange(num_nodes) * num_classes // num_nodes).astype(np.int64)
    return _finalize(feats, labels, [f"class_{i}" for i in range(num_classes)],
                     list(edges), [str(i) for i in range(num_nodes)])
