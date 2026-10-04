from __future__ import annotations

from typing import Dict

import numpy as np

from src.page_builder import PageTable


def build_page_vectors(
    graph,
    page_table: PageTable
) -> Dict[int, np.ndarray]:
    """Create TF-IDF-style semantic vectors for logical pages."""

    page_vectors = {}

    num_pages = page_table.num_pages

    # ------------------------------------------------------------
    # First calculate how many pages contain each feature.
    # ------------------------------------------------------------

    document_frequency = np.zeros(
        graph.features.shape[1],
        dtype=np.float32
    )

    page_features = {}

    for page_id in range(num_pages):

        node_ids = page_table.get_nodes_in_page(
            page_id
        )

        node_features = graph.features[node_ids]

        # Feature is considered present if at least one node
        # in this page contains it.
        feature_present = (
            np.any(node_features > 0, axis=0)
        )

        page_features[page_id] = feature_present

        document_frequency += feature_present.astype(
            np.float32
        )

    # ------------------------------------------------------------
    # IDF:
    #
    # Features appearing in many pages receive lower weight.
    # Features specific to fewer pages receive higher weight.
    # ------------------------------------------------------------

    idf = np.log(
        (num_pages + 1)
        /
        (document_frequency + 1)
    ) + 1.0

    # ------------------------------------------------------------
    # Build TF-IDF vector for every page.
    # ------------------------------------------------------------

    for page_id in range(num_pages):

        node_ids = page_table.get_nodes_in_page(
            page_id
        )

        node_features = graph.features[node_ids]

        # Term frequency = fraction of nodes in this page
        # containing each feature.
        tf = np.mean(
            node_features,
            axis=0
        )

        # TF-IDF weighting
        page_vector = tf * idf

        # L2 normalization
        norm = np.linalg.norm(
            page_vector
        )

        if norm > 0:

            page_vector = (
                page_vector / norm
            )

        page_vectors[page_id] = (
            page_vector.astype(np.float32)
        )

    return page_vectors