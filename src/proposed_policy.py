from __future__ import annotations


class GraphLSHAwareEvictionPolicy:
    """Proposed policy using graph locality, semantic similarity, and access behaviour."""

    def __init__(self, usefulness_calculator, page_graph,
                 lsh_manager, access_tracker):
        self.calculator = usefulness_calculator
        self.page_graph = page_graph
        self.lsh_manager = lsh_manager
        self.access_tracker = access_tracker

    def record_access(self, page_id: int) -> None:
        """Record a page access for the access-behaviour component."""
        self.access_tracker.record_access(page_id)

    def record_insertion(self, page_id: int) -> None:
        """Record a newly inserted page access."""
        self.access_tracker.record_access(page_id)

    def record_eviction(self, page_id: int) -> None:
        """Remove an evicted page from access tracking."""
        self.access_tracker.access_counts.pop(page_id, None)
        self.access_tracker.last_access_time.pop(page_id, None)

    def select_victim(self, buffer_pool_pages: list[int]) -> int:
        """Select the page with the lowest combined usefulness score."""
        lowest_score = float('inf')
        victim_page = None

        for page_id in buffer_pool_pages:
            score = self.calculator.calculate_usefulness(
                page_id,
                buffer_pool_pages,
                self.page_graph,
                self.lsh_manager,
                self.access_tracker
            )

            if score < lowest_score:
                lowest_score = score
                victim_page = page_id

        return victim_page