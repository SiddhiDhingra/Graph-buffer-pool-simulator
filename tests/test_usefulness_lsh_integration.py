import numpy as np

from src.data_loader import make_synthetic_cora
from src.page_builder import PageTable
from src.page_vectors import build_page_vectors
from src.semantic_similarity import PageSemanticSimilarity
from src.usefulness_score import UsefulnessCalculator


class DummyAccessTracker:
    def get_access_score(self, page_id):
        return 0.5


def test_lsh_integrates_with_usefulness_calculator():
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

    lsh_manager = PageSemanticSimilarity(
        page_vectors,
        num_tables=5,
        num_planes=8,
        seed=42
    )

    calculator = UsefulnessCalculator()

    access_tracker = DummyAccessTracker()

    score = calculator.calculate_usefulness(
        page_id=0,
        reference_page=1,
        page_graph=None,
        lsh_manager=lsh_manager,
        access_tracker=access_tracker
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0