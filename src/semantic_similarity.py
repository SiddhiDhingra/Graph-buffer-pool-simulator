"""Page-level semantic similarity using LSH."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.lsh import LSHManager


class PageSemanticSimilarity:
    """Provides semantic similarity between logical database pages."""

    def __init__(
        self,
        page_vectors: Dict[int, np.ndarray],
        num_tables: int = 5,
        num_planes: int = 8,
        seed: int = 42
    ):
        self.page_vectors = page_vectors

        self.lsh_manager = LSHManager(
            page_vectors,
            num_tables=num_tables,
            num_planes=num_planes,
            seed=seed
        )

    def get_page_similarity(
        self,
        page_a: int,
        page_b: int
    ) -> float:
        """Return cosine similarity between two pages."""

        return self.lsh_manager.get_page_similarity(
            page_a,
            page_b
        )

    def get_similar_pages(
        self,
        page_id: int,
        top_k: int = 5
    ) -> List[tuple[int, float]]:
        """Return the most semantically similar pages."""

        return self.lsh_manager.get_similar_pages(
            page_id,
            top_k=top_k
        )