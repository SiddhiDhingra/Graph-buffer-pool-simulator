from __future__ import annotations
from src.graph_locality import calculate_graph_locality

class GraphPrefetcher:
    """Advanced prefetching engine to load highly connected pages proactively."""
    def __init__(self, page_graph, threshold: float = 0.3):
        self.page_graph = page_graph
        self.threshold = threshold

    def get_prefetch_candidates(self, current_page: int, buffer_pool_pages: list[int]) -> list[int]:
        candidates = []
        if not self.page_graph:
            return candidates
            
        neighbors = self.page_graph.neighbors(current_page)
        for nbr in neighbors:
            if nbr not in buffer_pool_pages:
                score = calculate_graph_locality(current_page, nbr, self.page_graph, mode="directional")
                if score >= self.threshold:
                    candidates.append((nbr, score))
                    
        candidates.sort(key=lambda x: -x[1])
        return [page for page, score in candidates[:2]]