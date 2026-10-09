"""Random-hyperplane LSH for page-level semantic similarity.

The original implementation built an LSH index but the eviction policy later
called exact cosine similarity directly, meaning LSH did not actually affect
an eviction decision.  This implementation exposes an explicit approximate
candidate query and supports one-bit multi-probing.  Exact cosine is computed
only for LSH candidates, never as the candidate-generation mechanism.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple

import numpy as np


class LSHManager:
    def __init__(self, page_vectors: Dict[int, np.ndarray],
                 num_tables: int = 8, num_planes: int = 6,
                 seed: int = 42, probe_radius: int = 1):
        if not page_vectors:
            raise ValueError("page_vectors cannot be empty")
        if num_tables <= 0 or num_planes <= 0:
            raise ValueError("num_tables and num_planes must be positive")
        if probe_radius < 0 or probe_radius > 2:
            raise ValueError("probe_radius must be 0, 1, or 2")

        self.page_vectors = {
            pid: np.asarray(vec, dtype=np.float32).reshape(-1)
            for pid, vec in page_vectors.items()
        }
        self.num_tables = int(num_tables)
        self.num_planes = int(num_planes)
        self.probe_radius = int(probe_radius)

        first = next(iter(self.page_vectors.values()))
        self.dimension = int(first.shape[0])
        if self.dimension == 0:
            raise ValueError("page vectors must have non-zero dimension")
        for pid, vec in self.page_vectors.items():
            if vec.shape[0] != self.dimension:
                raise ValueError(f"Page {pid} has an incorrect vector dimension")

        # Normalize once.  Cora's word vectors are non-negative, so cosine is
        # naturally in [0,1].  We do not shift it to [0.5,1].
        self.normalized_vectors: Dict[int, np.ndarray] = {}
        for pid, vec in self.page_vectors.items():
            norm = float(np.linalg.norm(vec))
            self.normalized_vectors[pid] = vec / norm if norm > 0 else np.zeros_like(vec)

        rng = np.random.default_rng(seed)
        self.hyperplanes = rng.normal(
            size=(self.num_tables, self.num_planes, self.dimension)
        ).astype(np.float32)

        self.hash_tables: List[Dict[str, Set[int]]] = [
            defaultdict(set) for _ in range(self.num_tables)
        ]
        self._build_index()
        self._candidate_cache: Dict[Tuple[int, int], Set[int]] = {}
        self.semantic_cache: Dict[int, Dict[int, float]] = {}

    def _hash_vector(self, vector: np.ndarray, table_index: int) -> str:
        projections = np.dot(self.hyperplanes[table_index], vector)
        bits = np.where(projections >= 0.0, "1", "0")
        return "".join(bits.tolist())

    def _build_index(self) -> None:
        for page_id, vector in self.normalized_vectors.items():
            for table_index in range(self.num_tables):
                hv = self._hash_vector(vector, table_index)
                self.hash_tables[table_index][hv].add(page_id)

    def _probe_hashes(self, base_hash: str, radius: int | None = None) -> List[str]:
        radius = self.probe_radius if radius is None else radius
        hashes = {base_hash}
        if radius >= 1:
            for i in range(self.num_planes):
                bits = list(base_hash)
                bits[i] = "1" if bits[i] == "0" else "0"
                hashes.add("".join(bits))
        if radius >= 2:
            for i, j in combinations(range(self.num_planes), 2):
                bits = list(base_hash)
                bits[i] = "1" if bits[i] == "0" else "0"
                bits[j] = "1" if bits[j] == "0" else "0"
                hashes.add("".join(bits))
        return list(hashes)

    def get_candidates(self, page_id: int, probe_radius: int | None = None) -> Set[int]:
        if page_id not in self.normalized_vectors:
            raise KeyError(f"Page {page_id} does not exist")
        radius = self.probe_radius if probe_radius is None else int(probe_radius)
        if radius < 0 or radius > 2:
            raise ValueError("probe_radius must be 0, 1, or 2")

        key = (page_id, radius)
        if key in self._candidate_cache:
            return set(self._candidate_cache[key])

        candidates: Set[int] = set()
        vector = self.normalized_vectors[page_id]
        for table_index in range(self.num_tables):
            base = self._hash_vector(vector, table_index)
            hashes = self._probe_hashes(base, radius=radius)
            for hv in hashes:
                candidates.update(self.hash_tables[table_index].get(hv, ()))
        candidates.discard(page_id)
        self._candidate_cache[key] = set(candidates)
        return candidates

    def cosine_similarity(self, page_a: int, page_b: int) -> float:
        if page_a not in self.normalized_vectors or page_b not in self.normalized_vectors:
            raise KeyError("unknown page id")
        score = float(np.dot(self.normalized_vectors[page_a], self.normalized_vectors[page_b]))
        return float(np.clip(score, 0.0, 1.0))

    def get_lsh_similarity(self, page_a: int, page_b: int,
                           probe_radius: int | None = None) -> float:
        """Cosine similarity only if LSH retrieves page_b as a candidate."""
        if page_a == page_b:
            return 1.0
        if probe_radius is None and page_a in self.semantic_cache:
            return float(self.semantic_cache[page_a].get(page_b, 0.0))
        if page_b not in self.get_candidates(page_a, probe_radius=probe_radius):
            return 0.0
        return self.cosine_similarity(page_a, page_b)

    def get_similar_pages(self, page_id: int, top_k: int = 5,
                          probe_radius: int | None = None) -> List[tuple[int, float]]:
        if top_k <= 0:
            return []
        scored = [
            (candidate, self.cosine_similarity(page_id, candidate))
            for candidate in self.get_candidates(page_id, probe_radius=probe_radius)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]

    def get_page_similarity(self, page_a: int, page_b: int) -> float:
        """Backward-compatible exact cosine API."""
        return self.cosine_similarity(page_a, page_b)

    def build_semantic_cache(self, probe_radius: int | None = None) -> Dict[int, Dict[int, float]]:
        """Precompute LSH-retrieved semantic scores for fast eviction."""
        cache: Dict[int, Dict[int, float]] = {}
        for pid in self.page_vectors:
            cache[pid] = {
                q: self.cosine_similarity(pid, q)
                for q in self.get_candidates(pid, probe_radius=probe_radius)
            }
        self.semantic_cache = cache
        return cache
