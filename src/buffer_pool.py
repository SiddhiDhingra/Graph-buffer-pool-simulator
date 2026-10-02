"""Buffer pool simulator for logical database pages."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.page import Page


class BufferPool:
    """Simulates a limited main-memory buffer containing database pages."""

    def __init__(self, capacity: int, storage_pages: Dict[int, Page],
                 replacement_policy=None):
        if capacity <= 0:
            raise ValueError("Buffer pool capacity must be positive")

        self.capacity = capacity
        self.storage_pages = storage_pages
        self.replacement_policy = replacement_policy

        # page_id -> Page currently present in memory
        self.pages: Dict[int, Page] = {}

        # Performance statistics
        self.hits = 0
        self.misses = 0
        self.page_faults = 0
        self.storage_reads = 0
        self.evictions = 0

    def contains(self, page_id: int) -> bool:
        """Return True if a page is currently in the buffer."""
        return page_id in self.pages

    def get_page(self, page_id: int) -> Optional[Page]:
        """Return a page from the buffer, or None if it is absent."""
        return self.pages.get(page_id)

    def request_page(self, page_id: int) -> Optional[Page]:
        """Request a page from the simulated storage."""

        if page_id not in self.storage_pages:
            raise KeyError(f"Page {page_id} does not exist in storage")

        # Case 1: page is already in memory
        if page_id in self.pages:
            self.hits += 1

            if self.replacement_policy is not None:
                self.replacement_policy.record_access(page_id)

            return self.pages[page_id]

        # Case 2: page is not in memory
        self.misses += 1
        self.page_faults += 1
        self.storage_reads += 1

        # Buffer is full, so select a victim
        if len(self.pages) >= self.capacity:
            victim = self._select_victim()

            if victim is not None:
                self._evict_page(victim)

        # Read page from simulated storage
        page = self.storage_pages[page_id]
        self.pages[page_id] = page

        if self.replacement_policy is not None:
            self.replacement_policy.record_insertion(page_id)

        return page

    def _select_victim(self) -> Optional[int]:
        """Ask the replacement policy to select a page for eviction."""

        if not self.pages:
            return None

        if self.replacement_policy is None:
            # Temporary fallback:
            # remove the first page if no policy is supplied.
            return next(iter(self.pages))

        return self.replacement_policy.select_victim(list(self.pages.keys()))

    def _evict_page(self, page_id: int) -> None:
        """Remove a page from the buffer."""

        if page_id in self.pages:
            del self.pages[page_id]
            self.evictions += 1

            if self.replacement_policy is not None:
                self.replacement_policy.record_eviction(page_id)

    def current_pages(self) -> List[int]:
        """Return page IDs currently stored in the buffer."""
        return list(self.pages.keys())

    def reset_statistics(self) -> None:
        """Reset performance counters without removing pages."""

        self.hits = 0
        self.misses = 0
        self.page_faults = 0
        self.storage_reads = 0
        self.evictions = 0

    def clear(self) -> None:
        """Remove all pages and reset statistics."""

        self.pages.clear()
        self.reset_statistics()

    @property
    def hit_ratio(self) -> float:
        """Return buffer hit ratio."""

        total_requests = self.hits + self.misses

        if total_requests == 0:
            return 0.0

        return self.hits / total_requests