import numpy as np

from src.access_score import AccessScoreTracker
from src.buffer_pool import BufferPool
from src.graph_locality import PageGraph
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


def create_page_graph():
    page_graph = PageGraph()

    # Pages 0 and 1 have a strong graph relationship.
    page_graph.weights[0][1] = 5
    page_graph.weights[1][0] = 5

    # Pages 1 and 2 have a weaker graph relationship.
    page_graph.weights[1][2] = 2
    page_graph.weights[2][1] = 2

    return page_graph


def test_proposed_policy_uses_graph_lsh_and_access():
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

    page_graph = create_page_graph()

    policy = GraphLSHAwareEvictionPolicy(
        usefulness_calculator=calculator,
        page_graph=page_graph,
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

    # Requesting page 3 fills the buffer and forces the proposed
    # policy to calculate usefulness and select a victim.
    buffer_pool.request_page(3)

    assert len(buffer_pool.current_pages()) == 3
    assert buffer_pool.page_faults == 4
    assert buffer_pool.evictions == 1

    # Page 3 must now be present because it was just requested.
    assert buffer_pool.contains(3)

    # Exactly one of the original pages was evicted.
    original_pages = {0, 1, 2}
    remaining_original_pages = original_pages.intersection(
        set(buffer_pool.current_pages())
    )

    assert len(remaining_original_pages) == 2