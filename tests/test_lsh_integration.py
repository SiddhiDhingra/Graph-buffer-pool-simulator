from src.data_loader import make_synthetic_cora
from src.page_builder import PageTable
from src.page_vectors import build_page_vectors
from src.semantic_similarity import PageSemanticSimilarity


def test_cora_pages_can_use_lsh_similarity():
    graph = make_synthetic_cora(
        num_nodes=100,
        num_edges=150,
        num_features=20,
        seed=42
    )

    page_table = PageTable(
        num_nodes=graph.num_nodes,
        page_size=20
    ).build_pages()

    page_vectors = build_page_vectors(
        graph,
        page_table
    )

    similarity = PageSemanticSimilarity(
        page_vectors,
        num_tables=5,
        num_planes=8,
        seed=42
    )

    score = similarity.get_page_similarity(0, 1)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_cora_pages_can_find_similar_pages():
    graph = make_synthetic_cora(
        num_nodes=100,
        num_edges=150,
        num_features=20,
        seed=42
    )

    page_table = PageTable(
        num_nodes=graph.num_nodes,
        page_size=20
    ).build_pages()

    page_vectors = build_page_vectors(
        graph,
        page_table
    )

    similarity = PageSemanticSimilarity(
        page_vectors,
        num_tables=5,
        num_planes=8,
        seed=42
    )

    results = similarity.get_similar_pages(
        page_id=0,
        top_k=3
    )

    assert isinstance(results, list)

    for page_id, score in results:
        assert page_id != 0
        assert 0.0 <= score <= 1.0