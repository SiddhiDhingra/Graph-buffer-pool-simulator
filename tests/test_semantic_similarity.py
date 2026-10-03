import numpy as np

from src.semantic_similarity import PageSemanticSimilarity


def test_page_similarity_returns_one_for_identical_pages():
    page_vectors = {
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([1.0, 2.0, 3.0]),
    }

    similarity = PageSemanticSimilarity(page_vectors)

    score = similarity.get_page_similarity(0, 1)

    assert abs(score - 1.0) < 1e-6


def test_page_similarity_returns_zero_for_orthogonal_pages():
    page_vectors = {
        0: np.array([1.0, 0.0]),
        1: np.array([0.0, 1.0]),
    }

    similarity = PageSemanticSimilarity(page_vectors)

    score = similarity.get_page_similarity(0, 1)

    assert abs(score) < 1e-6


def test_similar_pages_returns_list():
    page_vectors = {
        0: np.array([1.0, 0.0]),
        1: np.array([0.9, 0.1]),
        2: np.array([0.0, 1.0]),
    }

    similarity = PageSemanticSimilarity(
        page_vectors,
        num_tables=5,
        num_planes=4,
        seed=42
    )

    results = similarity.get_similar_pages(0)

    assert isinstance(results, list)


def test_similar_pages_do_not_include_query_page():
    page_vectors = {
        0: np.array([1.0, 0.0]),
        1: np.array([0.9, 0.1]),
        2: np.array([0.0, 1.0]),
    }

    similarity = PageSemanticSimilarity(
        page_vectors,
        num_tables=5,
        num_planes=4,
        seed=42
    )

    results = similarity.get_similar_pages(0)

    page_ids = [page_id for page_id, score in results]

    assert 0 not in page_ids