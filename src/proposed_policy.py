"""Adaptive Graph-LSH-aware page eviction."""
from __future__ import annotations

from src.graph_locality import calculate_graph_locality


class GraphLSHAwareEvictionPolicy:
    def __init__(self, usefulness_calculator, page_graph,
                 lsh_manager, access_tracker):
        self.calculator = usefulness_calculator
        self.page_graph = page_graph
        self.lsh_manager = lsh_manager
        self.access_tracker = access_tracker
        self.last_scores: dict[int, float] = {}
        self.decision_count = 0
        self.candidate_evaluations = 0
        self.previous_page = None

        # Static graph/semantic information is prepared once rather than at
        # every cache miss.
        page_ids = set()
        if page_graph is not None:
            page_ids.update(page_graph.weights.keys())
            for values in page_graph.weights.values():
                page_ids.update(values.keys())
        if lsh_manager is not None:
            page_ids.update(lsh_manager.page_vectors.keys())
        if page_ids:
            self.calculator.prepare(page_ids, page_graph, lsh_manager)

    def record_access(self, page_id: int) -> None:
        self.access_tracker.record_access(page_id)
        self.previous_page = page_id

    def record_insertion(self, page_id: int) -> None:
        self.access_tracker.record_access(page_id)
        self.previous_page = page_id

    def record_eviction(self, page_id: int) -> None:
        self.access_tracker.remove_page(page_id)

    def observe_request(self, page_id: int) -> None:
        """Observe a request before an eviction decision without updating history."""
        self._current_request = page_id

    def _context_confidence(self, requested_page: int) -> float:
        previous = getattr(self, "previous_page", None)
        if previous is None or previous == requested_page:
            return 0.0

        graph_signal = 0.0
        if self.page_graph is not None:
            graph_signal = calculate_graph_locality(
                previous, requested_page, self.page_graph, mode="directional"
            )

        semantic_signal = 0.0
        if self.lsh_manager is not None:
            if hasattr(self.lsh_manager, "get_lsh_similarity"):
                semantic_signal = self.lsh_manager.get_lsh_similarity(previous, requested_page)
            else:
                semantic_signal = self.lsh_manager.get_page_similarity(previous, requested_page)

        # Either observed transition is evidence of a locality-bearing stream.
        return max(0.0, min(1.0, 0.65 * min(1.0, graph_signal * 5.0)
                              + 0.35 * min(1.0, semantic_signal * 2.0)))

    def select_victim(self, buffer_pool_pages: list[int], requested_page: int) -> int:
        if not buffer_pool_pages:
            return None
        self.decision_count += 1
        self.candidate_evaluations += len(buffer_pool_pages)

        confidence = self._context_confidence(requested_page)
        scores = self.calculator.calculate_scores(
            buffer_pool_pages,
            requested_page,
            self.page_graph,
            self.lsh_manager,
            self.access_tracker,
            context_confidence=confidence,
        )
        self.last_scores = dict(scores)
        return min(buffer_pool_pages, key=lambda p: (scores.get(p, 0.0), p))
