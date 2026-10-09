import numpy as np

from src.buffer_pool import BufferPool
from src.lsh import LSHManager
from src.page import Page
from src.proposed_policy import GraphLSHAwareEvictionPolicy
from src.usefulness_score import UsefulnessCalculator
from src.access_score import AccessScoreTracker
from src.graph_locality import PageGraph


def test_lsh_is_used_as_candidate_gate():
    vectors = {
        0: np.array([1.0, 0.0]),
        1: np.array([-1.0, 0.0]),
    }
    lsh = LSHManager(vectors, num_tables=2, num_planes=4, seed=42, probe_radius=0)
    assert lsh.get_lsh_similarity(0, 1) == 0.0
    assert lsh.cosine_similarity(0, 1) == 0.0


def test_access_score_has_no_hard_frequency_cap():
    tracker = AccessScoreTracker()
    for _ in range(10):
        tracker.record_access(1)
    score_10 = tracker.get_frequency_score(1)
    for _ in range(10):
        tracker.record_access(1)
    score_20 = tracker.get_frequency_score(1)
    assert score_20 > score_10
    assert score_20 < 1.0


def test_proposed_policy_never_exceeds_buffer_capacity():
    storage = {i: Page(i, tuple(range(i * 5, i * 5 + 5))) for i in range(6)}
    pg = PageGraph()
    pg.weights[0][1] = pg.weights[1][0] = 3
    pg.weights[1][2] = pg.weights[2][1] = 2
    vectors = {i: np.eye(6, dtype=float)[i] for i in range(6)}
    lsh = LSHManager(vectors, num_tables=5, num_planes=4, seed=42)
    policy = GraphLSHAwareEvictionPolicy(
        UsefulnessCalculator(), pg, lsh, AccessScoreTracker()
    )
    bp = BufferPool(2, storage, policy)
    for page_id in [0, 1, 2, 3, 4, 5, 0, 2]:
        bp.request_page(page_id)
        assert len(bp.current_pages()) <= 2
