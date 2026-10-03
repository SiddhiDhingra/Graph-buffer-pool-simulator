from src.buffer_pool import BufferPool
from src.page import Page
from src.replacement_policies import FIFOPolicy, LRUPolicy


def create_storage(num_pages=5):
    storage = {}

    for page_id in range(num_pages):
        nodes = list(range(page_id * 10, page_id * 10 + 10))
        storage[page_id] = Page(page_id, tuple(nodes))

    return storage


def test_buffer_pool_with_fifo():
    storage = create_storage()

    policy = FIFOPolicy()

    buffer_pool = BufferPool(
        capacity=3,
        storage_pages=storage,
        replacement_policy=policy
    )

    buffer_pool.request_page(0)
    buffer_pool.request_page(1)
    buffer_pool.request_page(2)

    assert buffer_pool.current_pages() == [0, 1, 2]

    buffer_pool.request_page(3)

    assert buffer_pool.current_pages() == [1, 2, 3]
    assert buffer_pool.page_faults == 4
    assert buffer_pool.evictions == 1


def test_buffer_pool_with_lru():
    storage = create_storage()

    policy = LRUPolicy()

    buffer_pool = BufferPool(
        capacity=3,
        storage_pages=storage,
        replacement_policy=policy
    )

    buffer_pool.request_page(0)
    buffer_pool.request_page(1)
    buffer_pool.request_page(2)

    # Access page 0 again, so page 1 becomes the least recently used.
    buffer_pool.request_page(0)

    buffer_pool.request_page(3)

    assert buffer_pool.current_pages() == [0, 2, 3]
    assert buffer_pool.page_faults == 4
    assert buffer_pool.evictions == 1


def test_buffer_pool_statistics():
    storage = create_storage()

    policy = LRUPolicy()

    buffer_pool = BufferPool(
        capacity=2,
        storage_pages=storage,
        replacement_policy=policy
    )

    buffer_pool.request_page(0)
    buffer_pool.request_page(1)
    buffer_pool.request_page(0)
    buffer_pool.request_page(2)

    assert buffer_pool.hits == 1
    assert buffer_pool.misses == 3
    assert buffer_pool.page_faults == 3
    assert buffer_pool.storage_reads == 3
    assert buffer_pool.evictions == 1
    assert buffer_pool.hit_ratio == 1 / 4