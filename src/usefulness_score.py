from __future__ import annotations
from src.graph_locality import calculate_graph_locality

class UsefulnessCalculator:
    """Computes the composite usefulness score U(P) for buffer page eviction decisions."""
    def __init__(self, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2):
        self.alpha = alpha  # Graph locality weight
        self.beta = beta    # LSH semantic similarity weight
        self.gamma = gamma  # Access behavior weight

    def calculate_usefulness(self, page_id: int, current_pages_in_buffer: list[int], 
                             page_graph, lsh_manager, access_tracker) -> float:
        # 1. Graph Locality Score G(P)
        g_score = 0.0
        if page_graph:
            scores = []
            for p in current_pages_in_buffer:
                if p != page_id:
                    loc = calculate_graph_locality(page_id, p, page_graph, mode="jaccard")
                    scores.append(loc)
            if scores:
                g_score = max(scores)

        # 2. Semantic Similarity Score S(P) (integrates with Person 2's LSH module)
        s_score = 0.0
        if lsh_manager and hasattr(lsh_manager, 'get_page_similarity'):
            sim_scores = []
            for p in current_pages_in_buffer:
                if p != page_id:
                    sim_scores.append(lsh_manager.get_page_similarity(page_id, p))
            if sim_scores:
                s_score = max(sim_scores)

        # 3. Access Behavior Score A(P)
        a_score = access_tracker.get_access_score(page_id)

        # Composite Proposed Formula: U(P) = alpha*G + beta*S + gamma*A
        return (self.alpha * g_score) + (self.beta * s_score) + (self.gamma * a_score)