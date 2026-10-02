"""Locality-Sensitive Hashing for page-level semantic similarity."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np


class LSHManager:
    """Random-hyperplane LSH for finding semantically similar pages."""

    def __init__(
        self,
        page_vectors: Dict[int, np.ndarray],
        num_tables: int = 5,
        num_planes: int = 8,
        seed: int = 42
    ):
        if not page_vectors:
            raise ValueError("page_vectors cannot be empty")

        if num_tables <= 0:
            raise ValueError("num_tables must be positive")

        if num_planes <= 0:
            raise ValueError("num_planes must be positive")

        self.page_vectors = page_vectors
        self.num_tables = num_tables
        self.num_planes = num_planes

        # Get feature dimension from the first page vector.
        first_vector = next(iter(page_vectors.values()))
        self.dimension = len(first_vector)

        # Random hyperplanes used to create LSH hashes.
        rng = np.random.default_rng(seed)

        self.hyperplanes = rng.normal(
            size=(num_tables, num_planes, self.dimension)
        )

        # One hash table for each set of hyperplanes.
        self.hash_tables: List[Dict[str, Set[int]]] = [
            defaultdict(set) for _ in range(num_tables)
        ]

        # Build the LSH index.
        self._build_index()

    def _hash_vector(self, vector: np.ndarray, table_index: int) -> str:
        """Convert a vector into a binary LSH hash."""

        planes = self.hyperplanes[table_index]

        projections = np.dot(planes, vector)

        bits = []

        for value in projections:
            if value >= 0:
                bits.append("1")
            else:
                bits.append("0")

        return "".join(bits)

    def _build_index(self) -> None:
        """Insert every page into all LSH hash tables."""

        for page_id, vector in self.page_vectors.items():
            vector = np.asarray(vector, dtype=float)

            if len(vector) != self.dimension:
                raise ValueError(
                    f"Page {page_id} has an incorrect vector dimension"
                )

            for table_index in range(self.num_tables):
                hash_value = self._hash_vector(vector, table_index)

                self.hash_tables[table_index][hash_value].add(page_id)

    def get_candidates(self, page_id: int) -> Set[int]:
        """Return pages that share an LSH bucket with the given page."""

        if page_id not in self.page_vectors:
            raise KeyError(f"Page {page_id} does not exist")

        candidates = set()

        vector = self.page_vectors[page_id]

        for table_index in range(self.num_tables):
            hash_value = self._hash_vector(vector, table_index)

            bucket = self.hash_tables[table_index].get(
                hash_value, set()
            )

            candidates.update(bucket)

        candidates.discard(page_id)

        return candidates

    def cosine_similarity(self, page_a: int, page_b: int) -> float:
        """Calculate cosine similarity between two page vectors."""

        if page_a not in self.page_vectors:
            raise KeyError(f"Page {page_a} does not exist")

        if page_b not in self.page_vectors:
            raise KeyError(f"Page {page_b} does not exist")

        vector_a = np.asarray(self.page_vectors[page_a], dtype=float)
        vector_b = np.asarray(self.page_vectors[page_b], dtype=float)

        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = np.dot(vector_a, vector_b) / (norm_a * norm_b)

        return float(similarity)

    def get_similar_pages(
        self,
        page_id: int,
        top_k: int = 5
    ) -> List[tuple[int, float]]:
        """Return the most similar LSH candidate pages."""

        candidates = self.get_candidates(page_id)

        scored_pages = []

        for candidate in candidates:
            similarity = self.cosine_similarity(page_id, candidate)
            scored_pages.append((candidate, similarity))

        scored_pages.sort(
            key=lambda item: item[1],
            reverse=True
        )

        return scored_pages[:top_k]

    def get_page_similarity(
        self,
        page_a: int,
        page_b: int
    ) -> float:
        """Return semantic similarity between two pages."""

        return self.cosine_similarity(page_a, page_b)