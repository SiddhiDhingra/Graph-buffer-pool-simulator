import numpy as np
import pytest

from src.buffer_pool import BufferPool
from src.lsh import LSHManager
from src.page import Page
from src.replacement_policies import FIFOPolicy, LRUPolicy


def create_storage(num_pages=3):
    storage = {}

    for page_id in range(num_pages):
        nodes = list(range(page_id * 10, page_id * 10 + 10))
        storage[page_id] = Page(page_id, tuple(nodes))

    return storage


def test_buffer_pool_capacity_one():
    storage = create_storage()

    policy = FIFOPolicy()

    buffer_pool = BufferPool(
        capacity=1,
        storage_pages=storage,
        replacement_policy=policy
    )

    buffer_pool.request_page(0)
    assert buffer_pool.current_pages() == [0]

    buffer_pool.request_page(1)

    assert buffer_pool.current_pages() == [1]
    assert buffer_pool.page_faults == 2
    assert buffer_pool.evictions == 1


def test_repeated_page_access_is_a_hit():
    storage = create_storage()

    policy = LRUPolicy()

    buffer_pool = BufferPool(
        capacity=2,
        storage_pages=storage,
        replacement_policy=policy
    )

    buffer_pool.request_page(0)
    buffer_pool.request_page(0)
    buffer_pool.request_page(0)

    assert buffer_pool.hits == 2
    assert buffer_pool.misses == 1
    assert buffer_pool.page_faults == 1
    assert buffer_pool.storage_reads == 1
    assert buffer_pool.current_pages() == [0]


def test_invalid_page_request_raises_error():
    storage = create_storage()

    buffer_pool = BufferPool(
        capacity=2,
        storage_pages=storage
    )

    with pytest.raises(KeyError):
        buffer_pool.request_page(99)


def test_fifo_ignores_accesses_when_selecting_victim():
    storage = create_storage()

    policy = FIFOPolicy()

    buffer_pool = BufferPool(
        capacity=2,
        storage_pages=storage,
        replacement_policy=policy
    )

    buffer_pool.request_page(0)
    buffer_pool.request_page(1)

    # Accessing page 0 should not change FIFO order.
    buffer_pool.request_page(0)

    buffer_pool.request_page(2)

    assert buffer_pool.current_pages() == [1, 2]


def test_lru_changes_victim_after_access():
    storage = create_storage()

    policy = LRUPolicy()

    buffer_pool = BufferPool(
        capacity=2,
        storage_pages=storage,
        replacement_policy=policy
    )

    buffer_pool.request_page(0)
    buffer_pool.request_page(1)

    # Page 0 becomes recently used.
    buffer_pool.request_page(0)

    buffer_pool.request_page(2)

    assert buffer_pool.current_pages() == [0, 2]


def test_lsh_rejects_empty_page_vectors():
    with pytest.raises(ValueError):
        LSHManager({})


def test_lsh_rejects_invalid_number_of_tables():
    page_vectors = {
        0: np.array([1.0, 0.0]),
        1: np.array([0.0, 1.0])
    }

    with pytest.raises(ValueError):
        LSHManager(
            page_vectors,
            num_tables=0
        )


def test_lsh_rejects_invalid_number_of_planes():
    page_vectors = {
        0: np.array([1.0, 0.0]),
        1: np.array([0.0, 1.0])
    }

    with pytest.raises(ValueError):
        LSHManager(
            page_vectors,
            num_planes=0
        )