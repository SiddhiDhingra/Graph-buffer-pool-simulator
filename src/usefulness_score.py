"""Composite usefulness scoring for the Graph-LSH-aware eviction policy.

The scorer is deliberately sparse: graph scores are built only for page-graph
neighbors and semantic scores only for LSH candidates. This avoids an O(P^2)
precomputation over logical pages while preserving the proposal's four signals:
recency, frequency, graph locality, and semantic similarity.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from src.graph_locality import calculate_graph_locality


class UsefulnessCalculator:
    """Score cached pages; larger means more useful to retain."""

    def __init__(
        self,
        alpha=0.40,
        beta=0.40,
        gamma=0.20,
        recency_weight=0.65,
        frequency_weight=0.35,
        graph_mode="directional",
        semantic_probe_radius=1,
        locality_gate=True,
    ):
        if min(alpha, beta, gamma) < 0 or alpha + beta + gamma <= 0:
            raise ValueError("invalid usefulness weights")

        if (
            min(recency_weight, frequency_weight) < 0
            or recency_weight + frequency_weight <= 0
        ):
            raise ValueError("invalid access weights")

        total = alpha + beta + gamma

        self.alpha = float(alpha) / total
        self.beta = float(beta) / total
        self.gamma = float(gamma) / total

        access_total = recency_weight + frequency_weight

        self.recency_weight = float(recency_weight) / access_total
        self.frequency_weight = float(frequency_weight) / access_total

        self.graph_mode = graph_mode
        self.semantic_probe_radius = int(semantic_probe_radius)
        self.locality_gate = bool(locality_gate)

        self.graph_rank_cache: dict[int, dict[int, float]] = {}
        self.semantic_rank_cache: dict[int, dict[int, float]] = {}

        self.graph_strength_cache: dict[int, float] = {}
        self.semantic_strength_cache: dict[int, float] = {}

    @staticmethod
    def _rank_normalize_pairs(
        pairs: Sequence[tuple[int, float]]
    ) -> dict[int, float]:
        """Rank only observed positive candidates; absent candidates remain 0."""

        if not pairs:
            return {}

        ordered = sorted(pairs, key=lambda x: (x[1], x[0]))

        if len(ordered) == 1:
            return {
                ordered[0][0]: 1.0 if ordered[0][1] > 0 else 0.0
            }

        values = [v for _, v in ordered]

        if max(values) - min(values) <= 1e-12:
            value = 1.0 if values[0] > 0 else 0.0
            return {
                pid: value
                for pid, _ in ordered
            }

        result: dict[int, float] = {}

        n = len(ordered)
        i = 0

        while i < n:
            j = i + 1

            while (
                j < n
                and abs(ordered[j][1] - ordered[i][1]) <= 1e-12
            ):
                j += 1

            rank = ((i + j - 1) / 2.0) / (n - 1)

            for k in range(i, j):
                result[ordered[k][0]] = rank

            i = j

        return result

    @staticmethod
    def _strength(values: Sequence[float]) -> float:
        """Estimate how strongly a locality signal varies across candidates."""

        positive = [
            float(v)
            for v in values
            if v > 0.0
        ]

        if not positive:
            return 0.0

        spread = max(positive) - min(positive)

        if spread > 1e-12:
            return min(1.0, spread * 5.0)

        return min(1.0, max(positive))

    def prepare(
        self,
        page_ids: Iterable[int],
        page_graph,
        lsh_manager,
    ) -> None:
        """Prepare sparse graph and semantic rankings once per experiment."""

        pages = sorted(set(page_ids))
        page_set = set(pages)

        self.graph_rank_cache.clear()
        self.semantic_rank_cache.clear()
        self.graph_strength_cache.clear()
        self.semantic_strength_cache.clear()

        # Sparse graph preparation.
        for reference in pages:

            graph_pairs: list[tuple[int, float]] = []

            if page_graph is not None:

                for p in page_graph.neighbors(reference):

                    if p in page_set and p != reference:

                        score = calculate_graph_locality(
                            reference,
                            p,
                            page_graph,
                            mode=self.graph_mode,
                        )

                        if score > 0:
                            graph_pairs.append(
                                (p, float(score))
                            )

            self.graph_rank_cache[reference] = (
                self._rank_normalize_pairs(graph_pairs)
            )

            self.graph_strength_cache[reference] = (
                self._strength(
                    [
                        score
                        for _, score in graph_pairs
                    ]
                )
            )

            # Sparse semantic preparation using LSH candidates.
            semantic_pairs: list[tuple[int, float]] = []

            if lsh_manager is not None:

                if hasattr(lsh_manager, "get_candidates"):

                    candidates = lsh_manager.get_candidates(
                        reference,
                        probe_radius=self.semantic_probe_radius,
                    )

                elif hasattr(lsh_manager, "lsh_manager"):

                    candidates = lsh_manager.lsh_manager.get_candidates(
                        reference,
                        probe_radius=self.semantic_probe_radius,
                    )

                else:
                    candidates = []

                for p in candidates:

                    if p in page_set and p != reference:

                        if hasattr(
                            lsh_manager,
                            "cosine_similarity",
                        ):
                            score = lsh_manager.cosine_similarity(
                                reference,
                                p,
                            )
                        else:
                            score = lsh_manager.get_page_similarity(
                                reference,
                                p,
                            )

                        if score > 0:
                            semantic_pairs.append(
                                (p, float(score))
                            )

            self.semantic_rank_cache[reference] = (
                self._rank_normalize_pairs(semantic_pairs)
            )

            self.semantic_strength_cache[reference] = (
                self._strength(
                    [
                        score
                        for _, score in semantic_pairs
                    ]
                )
            )

    def _access_score(self, tracker, page_id: int) -> float:

        try:

            return float(
                tracker.get_access_score(
                    page_id,
                    recency_weight=self.recency_weight,
                    frequency_weight=self.frequency_weight,
                )
            )

        except TypeError:

            return float(
                tracker.get_access_score(page_id)
            )

    def calculate_scores(
        self,
        candidate_pages: Iterable[int],
        reference_page: int,
        page_graph,
        lsh_manager,
        access_tracker,
        context_confidence: float = 0.0,
    ) -> dict[int, float]:

        candidates = list(candidate_pages)

        if not candidates:
            return {}

        if reference_page not in self.graph_rank_cache:

            self.prepare(
                set(candidates) | {reference_page},
                page_graph,
                lsh_manager,
            )

        graph_rank = self.graph_rank_cache.get(
            reference_page,
            {},
        )

        semantic_rank = self.semantic_rank_cache.get(
            reference_page,
            {},
        )

        graph_strength = self.graph_strength_cache.get(
            reference_page,
            0.0,
        )

        semantic_strength = self.semantic_strength_cache.get(
            reference_page,
            0.0,
        )

        confidence = max(
            0.0,
            min(1.0, context_confidence),
        )

        if self.locality_gate:

            # Base graph and semantic contributions remain active.
            # Strong contextual locality increases their influence.
            locality_factor = 0.25 + 0.75 * confidence

            graph_weight = (
                self.alpha
                * locality_factor
                * max(0.25, graph_strength)
            )

            semantic_weight = (
                self.beta
                * locality_factor
                * max(0.25, semantic_strength)
            )

            access_weight = self.gamma

        else:

            graph_weight = self.alpha
            semantic_weight = self.beta
            access_weight = self.gamma

        total = (
            graph_weight
            + semantic_weight
            + access_weight
        )

        if total <= 0:

            graph_weight = 0.0
            semantic_weight = 0.0
            access_weight = 1.0
            total = 1.0

        graph_weight /= total
        semantic_weight /= total
        access_weight /= total

        scores = {}

        for page_id in candidates:

            graph_component = graph_rank.get(
                page_id,
                0.0,
            )

            semantic_component = semantic_rank.get(
                page_id,
                0.0,
            )

            access_component = self._access_score(
                access_tracker,
                page_id,
            )

            scores[page_id] = (
                graph_weight * graph_component
                + semantic_weight * semantic_component
                + access_weight * access_component
            )

        return scores

    def calculate_usefulness(
        self,
        page_id,
        reference_page,
        page_graph,
        lsh_manager,
        access_tracker,
        candidate_pages=None,
    ):

        candidates = (
            list(candidate_pages)
            if candidate_pages is not None
            else [page_id]
        )

        if page_id not in candidates:
            candidates.append(page_id)

        scores = self.calculate_scores(
            candidates,
            reference_page,
            page_graph,
            lsh_manager,
            access_tracker,
        )

        return scores[page_id]