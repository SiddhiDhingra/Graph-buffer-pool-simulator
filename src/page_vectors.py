"""Create feature vectors for logical pages."""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.page_builder import PageTable


def build_page_vectors(graph, page_table: PageTable) -> Dict[int, np.ndarray]:
    """Create one feature vector for each logical page.

    The page vector is the average of the feature vectors
    of all nodes stored in that page.
    """

    page_vectors = {}

    for page_id in range(page_table.num_pages):
        node_ids = page_table.get_nodes_in_page(page_id)

        node_features = graph.features[node_ids]

        page_vector = np.mean(node_features, axis=0)

        page_vectors[page_id] = page_vector.astype(np.float32)

    return page_vectors