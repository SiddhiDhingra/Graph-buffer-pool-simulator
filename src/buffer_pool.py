"""Deterministic buffer-pool simulator for logical database pages."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from src.page import Page


class BufferPool:
    def __init__(self, capacity: int, storage_pages: Dict[int, Page],
                 replacement_policy=None):
        if capacity <= 0:
            raise ValueError("Buffer pool capacity must be positive")
        if storage_pages is None:
            raise ValueError("storage_pages cannot be None")

        self.capacity = int(capacity)
        self.storage_pages = storage_pages
        self.replacement_policy = replacement_policy
        self.pages: Dict[int, Page] = {}

        self.hits = 0
        self.misses = 0
        self.page_faults = 0
        self.storage_reads = 0
        self.evictions = 0
        self.requests = 0
        self.policy_decision_time = 0.0
        self.policy_decisions = 0

    def contains(self, page_id: int) -> bool:
        return page_id in self.pages

    def get_page(self, page_id: int) -> Optional[Page]:
        return self.pages.get(page_id)

    def request_page(self, page_id: int) -> Optional[Page]:
        if page_id not in self.storage_pages:
            raise KeyError(f"Page {page_id} does not exist in storage")

        self.requests += 1

        observer = getattr(self.replacement_policy, "observe_request", None)
        if observer is not None:
            observer(page_id)

        if page_id in self.pages:
            self.hits += 1
            if self.replacement_policy is not None:
                self.replacement_policy.record_access(page_id)
            return self.pages[page_id]

        self.misses += 1
        self.page_faults += 1
        self.storage_reads += 1

        if len(self.pages) >= self.capacity:
            start = time.perf_counter()
            victim = self._select_victim(page_id)
            self.policy_decision_time += time.perf_counter() - start
            self.policy_decisions += 1
            if victim is not None:
                self._evict_page(victim)

        page = self.storage_pages[page_id]
        self.pages[page_id] = page
        if self.replacement_policy is not None:
            self.replacement_policy.record_insertion(page_id)
        return page

    def _select_victim(self, requested_page: int) -> Optional[int]:
        if not self.pages:
            return None
        if self.replacement_policy is None:
            return next(iter(self.pages))

        selector = getattr(self.replacement_policy, "select_victim", None)
        if selector is None:
            return next(iter(self.pages))

        try:
            return selector(list(self.pages.keys()), requested_page)
        except TypeError:
            return selector(list(self.pages.keys()))

    def _evict_page(self, page_id: int) -> None:
        if page_id in self.pages:
            del self.pages[page_id]
            self.evictions += 1
            if self.replacement_policy is not None:
                self.replacement_policy.record_eviction(page_id)

    def current_pages(self) -> List[int]:
        return list(self.pages.keys())

    def reset_statistics(self) -> None:
        self.hits = self.misses = self.page_faults = 0
        self.storage_reads = self.evictions = 0
        self.requests = 0
        self.policy_decision_time = 0.0
        self.policy_decisions = 0

    def clear(self) -> None:
        self.pages.clear()
        self.reset_statistics()

    @property
    def hit_ratio(self) -> float:
        return self.hits / self.requests if self.requests else 0.0
