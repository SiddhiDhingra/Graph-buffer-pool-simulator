from __future__ import annotations

from src.graph_locality import calculate_graph_locality


class UsefulnessCalculator:

    def __init__(self, alpha=0.2, beta=0.2, gamma=0.6):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def calculate_usefulness(
        self,
        page_id,
        reference_page,
        page_graph,
        lsh_manager,
        access_tracker,
        candidate_pages=None
    ):
        graph_score = 0.0

        if page_graph is not None:
            graph_score = calculate_graph_locality(
                page_id,
                reference_page,
                page_graph,
                mode="jaccard"
            )

        semantic_score = 0.0

        if lsh_manager is not None:
            semantic_score = lsh_manager.get_page_similarity(
                page_id,
                reference_page
            )

        semantic_score = (semantic_score + 1.0) / 2.0

        access_score = access_tracker.get_access_score(page_id)

        if candidate_pages is not None:
            graph_scores = []
            semantic_scores = []
            access_scores = []

            for candidate in candidate_pages:
                candidate_graph = 0.0

                if page_graph is not None:
                    candidate_graph = calculate_graph_locality(
                        candidate,
                        reference_page,
                        page_graph,
                        mode="jaccard"
                    )

                candidate_semantic = 0.0

                if lsh_manager is not None:
                    candidate_semantic = (
                        lsh_manager.get_page_similarity(
                            candidate,
                            reference_page
                        )
                    )

                candidate_semantic = (
                    candidate_semantic + 1.0
                ) / 2.0

                candidate_access = access_tracker.get_access_score(
                    candidate
                )

                graph_scores.append(candidate_graph)
                semantic_scores.append(candidate_semantic)
                access_scores.append(candidate_access)

            def normalize(value, values):
                minimum = min(values)
                maximum = max(values)

                if maximum == minimum:
                    return 0.5

                return (value - minimum) / (maximum - minimum)

            graph_score = normalize(graph_score, graph_scores)
            semantic_score = normalize(
                semantic_score,
                semantic_scores
            )
            access_score = normalize(
                access_score,
                access_scores
            )

        usefulness = (
            self.alpha * graph_score
            + self.beta * semantic_score
            + self.gamma * access_score
        )

        return usefulness