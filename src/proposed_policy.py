from __future__ import annotations

class GraphLSHAwareEvictionPolicy:
    """Proposed replacement policy that evicts the page with the lowest composite usefulness score."""
    def __init__(self, usefulness_calculator, page_graph, lsh_manager, access_tracker):
        self.calculator = usefulness_calculator
        self.page_graph = page_graph
        self.lsh_manager = lsh_manager
        self.access_tracker = access_tracker

    def select_victim(self, buffer_pool_pages: list[int]) -> int:
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