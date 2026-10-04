

from __future__ import annotations


class AccessScoreTracker:
    """Tracks page access history using frequency and recency."""

    def __init__(self):
        self.access_counts: dict[int, int] = {}
        self.last_access_time: dict[int, int] = {}
        self.current_time = 0

    def record_access(self, page_id: int) -> None:
        self.current_time += 1

        self.access_counts[page_id] = (
            self.access_counts.get(page_id, 0) + 1
        )

        self.last_access_time[page_id] = self.current_time

    def get_access_score(self, page_id: int) -> float:
        """Return a normalized score based on frequency and recency."""

        if page_id not in self.access_counts:
            return 0.0

        frequency = self.access_counts[page_id]
        last_access = self.last_access_time[page_id]

        # Frequency score
        frequency_score = min(1.0, frequency / 10.0)

        # Recency score
        time_since_access = self.current_time - last_access
        recency_score = 1.0 / (1.0 + time_since_access)

        # Equal contribution from frequency and recency
        return (
            0.5 * frequency_score
            + 0.5 * recency_score
        )

    def remove_page(self, page_id: int) -> None:
        """Keep access history even after a page is evicted."""
        pass