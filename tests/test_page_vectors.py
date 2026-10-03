import numpy as np

from src.data_loader import make_synthetic_cora
from src.page_builder import PageTable
from src.page_vectors import build_page_vectors


def test_page_vectors_are_created():
    graph = make_synthetic_cora(
        num_nodes=100,
        num_edges=150,
        num_features=10,
        seed=42
    )

    page_table = PageTable(
        num_nodes=graph.num_nodes,
        page_size=50
    ).build_pages()

    page_vectors = build_page_vectors(graph, page_table)

    assert len(page_vectors) == 2


def test_page_vector_has_correct_dimension():
    graph = make_synthetic_cora(
        num_nodes=100,
        num_edges=150,
        num_features=10,
        seed=42
    )

    page_table = PageTable(
        num_nodes=graph.num_nodes,
        page_size=50
    ).build_pages()

    page_vectors = build_page_vectors(graph, page_table)

    assert page_vectors[0].shape == (10,)
    assert page_vectors[1].shape == (10,)


def test_page_vector_is_average_of_node_features():
    graph = make_synthetic_cora(
        num_nodes=4,
        num_edges=2,
        num_features=3,
        seed=42
    )

    page_table = PageTable(
        num_nodes=graph.num_nodes,
        page_size=2
    ).build_pages()

    page_vectors = build_page_vectors(graph, page_table)

    expected = np.mean(
        graph.features[[0, 1]],
        axis=0
    )

    assert np.allclose(page_vectors[0], expected)