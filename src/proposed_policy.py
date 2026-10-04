from __future__ import annotations

from src.graph_locality import calculate_graph_locality


class GraphLSHAwareEvictionPolicy:

    def __init__(
        self,
        usefulness_calculator,
        page_graph,
        lsh_manager,
        access_tracker
    ):
        self.calculator = usefulness_calculator
        self.page_graph = page_graph
        self.lsh_manager = lsh_manager
        self.access_tracker = access_tracker

    def record_access(self, page_id):
        self.access_tracker.record_access(page_id)

    def record_insertion(self, page_id):
        self.access_tracker.record_access(page_id)

    def record_eviction(self, page_id):
        self.access_tracker.remove_page(page_id)

    def select_victim(
        self,
        buffer_pool_pages: list[int],
        requested_page: int
    ) -> int:

        if not buffer_pool_pages:
            return None

        lowest_score = float("inf")
        victim_page = None

        for page_id in buffer_pool_pages:

            usefulness = self.calculator.calculate_usefulness(
                page_id,
                requested_page,
                self.page_graph,
                self.lsh_manager,
                self.access_tracker,
                candidate_pages=buffer_pool_pages
            )

            if usefulness < lowest_score:
                lowest_score = usefulness
                victim_page = page_id

        return victim_page