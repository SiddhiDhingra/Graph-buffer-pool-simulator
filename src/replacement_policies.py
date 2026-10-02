"""Buffer pool page replacement policies."""

from __future__ import annotations

from collections import deque
from typing import Dict, List


class FIFOPolicy:
    """First-In-First-Out page replacement policy."""

    def __init__(self):
        self.order = deque()

    def record_insertion(self, page_id: int) -> None:
        """Record when a page enters the buffer."""

        if page_id not in self.order:
            self.order.append(page_id)

    def record_access(self, page_id: int) -> None:
        """FIFO does not change order when a page is accessed."""

        pass

    def record_eviction(self, page_id: int) -> None:
        """Remove an evicted page from the FIFO queue."""

        try:
            self.order.remove(page_id)
        except ValueError:
            pass

    def select_victim(self, buffer_pool_pages: List[int]) -> int:
        """Return the page that entered the buffer first."""

        # Remove stale page IDs that are no longer in the buffer.
        while self.order and self.order[0] not in buffer_pool_pages:
            self.order.popleft()

        if not self.order:
            return buffer_pool_pages[0]

        return self.order[0]


class LRUPolicy:
    """Least Recently Used page replacement policy."""

    def __init__(self):
        self.access_order: Dict[int, int] = {}
        self.counter = 0

    def record_insertion(self, page_id: int) -> None:
        """Record a newly inserted page."""

        self.counter += 1
        self.access_order[page_id] = self.counter

    def record_access(self, page_id: int) -> None:
        """Update the page's recent-access timestamp."""

        self.counter += 1
        self.access_order[page_id] = self.counter

    def record_eviction(self, page_id: int) -> None:
        """Remove an evicted page from the tracking structure."""

        self.access_order.pop(page_id, None)

    def select_victim(self, buffer_pool_pages: List[int]) -> int:
        """Return the least recently used page."""

        if not buffer_pool_pages:
            raise ValueError("Cannot select a victim from an empty buffer")

        return min(
            buffer_pool_pages,
            key=lambda page_id: self.access_order.get(page_id, -1)
        )
