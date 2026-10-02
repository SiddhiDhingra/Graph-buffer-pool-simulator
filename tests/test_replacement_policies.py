from src.replacement_policies import FIFOPolicy, LRUPolicy


def test_fifo_selects_first_inserted_page():
    policy = FIFOPolicy()

    policy.record_insertion(1)
    policy.record_insertion(2)
    policy.record_insertion(3)

    victim = policy.select_victim([1, 2, 3])

    assert victim == 1


def test_fifo_access_does_not_change_order():
    policy = FIFOPolicy()

    policy.record_insertion(1)
    policy.record_insertion(2)
    policy.record_insertion(3)

    policy.record_access(1)

    victim = policy.select_victim([1, 2, 3])

    assert victim == 1


def test_fifo_removes_evicted_page():
    policy = FIFOPolicy()

    policy.record_insertion(1)
    policy.record_insertion(2)
    policy.record_insertion(3)

    policy.record_eviction(1)

    victim = policy.select_victim([2, 3])

    assert victim == 2


def test_lru_selects_least_recently_used_page():
    policy = LRUPolicy()

    policy.record_insertion(1)
    policy.record_insertion(2)
    policy.record_insertion(3)

    policy.record_access(1)
    policy.record_access(3)

    victim = policy.select_victim([1, 2, 3])

    assert victim == 2


def test_lru_updates_on_access():
    policy = LRUPolicy()

    policy.record_insertion(1)
    policy.record_insertion(2)
    policy.record_insertion(3)

    policy.record_access(1)

    victim = policy.select_victim([1, 2, 3])

    assert victim == 2


def test_lru_removes_evicted_page():
    policy = LRUPolicy()

    policy.record_insertion(1)
    policy.record_insertion(2)
    policy.record_insertion(3)

    policy.record_eviction(2)

    victim = policy.select_victim([1, 3])

    assert victim == 1