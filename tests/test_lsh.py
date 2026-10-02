import numpy as np

from src.lsh import LSHManager


def test_lsh_creates_index():
    page_vectors = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([0.9, 0.1, 0.0]),
        2: np.array([0.0, 1.0, 0.0]),
    }

    lsh = LSHManager(
        page_vectors,
        num_tables=3,
        num_planes=4,
        seed=42
    )

    assert len(lsh.hash_tables) == 3


def test_lsh_candidate_search():
    page_vectors = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([0.9, 0.1, 0.0]),
        2: np.array([0.0, 1.0, 0.0]),
    }

    lsh = LSHManager(
        page_vectors,
        num_tables=5,
        num_planes=4,
        seed=42
    )

    candidates = lsh.get_candidates(0)

    assert isinstance(candidates, set)
    assert 0 not in candidates


def test_cosine_similarity_identical_vectors():
    page_vectors = {
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([1.0, 2.0, 3.0]),
    }

    lsh = LSHManager(page_vectors)

    similarity = lsh.cosine_similarity(0, 1)

    assert abs(similarity - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    page_vectors = {
        0: np.array([1.0, 0.0]),
        1: np.array([0.0, 1.0]),
    }

    lsh = LSHManager(page_vectors)

    similarity = lsh.cosine_similarity(0, 1)

    assert abs(similarity) < 1e-6


def test_similar_pages_returns_sorted_results():
    page_vectors = {
        0: np.array([1.0, 0.0]),
        1: np.array([0.9, 0.1]),
        2: np.array([0.0, 1.0]),
    }

    lsh = LSHManager(
        page_vectors,
        num_tables=10,
        num_planes=4,
        seed=42
    )

    results = lsh.get_similar_pages(0, top_k=2)

    for i in range(len(results) - 1):
        assert results[i][1] >= results[i + 1][1]


def test_page_similarity_interface():
    page_vectors = {
        0: np.array([1.0, 0.0]),
        1: np.array([1.0, 0.0]),
    }

    lsh = LSHManager(page_vectors)

    similarity = lsh.get_page_similarity(0, 1)

    assert abs(similarity - 1.0) < 1e-6