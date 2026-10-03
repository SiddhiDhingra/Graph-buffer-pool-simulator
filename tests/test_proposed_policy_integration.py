import numpy as np

from src.access_score import AccessScoreTracker
from src.buffer_pool import BufferPool
from src.lsh import LSHManager
from src.page import Page
from src.proposed_policy import GraphLSHAwareEvictionPolicy
from src.usefulness_score import UsefulnessCalculator


def create_storage():
    storage = {}

    for page_id in range(4):
        nodes = list(range(page_id * 10, page_id * 10 + 10))
        storage[page_id] = Page(page_id, tuple(nodes))

    return storage


def test_proposed_policy_selects_victim():
    storage = create_storage()

    page_vectors = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([0.9, 0.1, 0.0]),
        2: np.array([0.0, 1.0, 0.0]),
        3: np.array([0.0, 0.0, 1.0]),
    }

    lsh_manager = LSHManager(
        page_vectors,
        num_tables=5,
        num_planes=8,
        seed=42
    )

    access_tracker = AccessScoreTracker()

    calculator = UsefulnessCalculator(
        alpha=0.4,
        beta=0.4,
        gamma=0.2
    )

    policy = GraphLSHAwareEvictionPolicy(
        usefulness_calculator=calculator,
        page_graph=None,
        lsh_manager=lsh_manager,
        access_tracker=access_tracker
    )

    buffer_pool = BufferPool(
        capacity=3,
        storage_pages=storage,
        replacement_policy=policy
    )

    buffer_pool.request_page(0)
    buffer_pool.request_page(1)
    buffer_pool.request_page(2)

    assert len(buffer_pool.current_pages()) == 3

    buffer_pool.request_page(3)

    assert len(buffer_pool.current_pages()) == 3
    assert buffer_pool.page_faults == 4
    assert buffer_pool.evictions == 1