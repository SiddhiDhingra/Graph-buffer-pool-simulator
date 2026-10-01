from __future__ import annotations

class AccessScoreTracker:
    """Tracks runtime access patterns (recency and frequency) for resident buffer pages."""
    def __init__(self):
        self.access_counts: dict[int, int] = {}
        self.last_access_time: dict[int, int] = {}
        self.current_time = 0

    def record_access(self, page_id: int):
        self.current_time += 1
        self.access_counts[page_id] = self.access_counts.get(page_id, 0) + 1
        self.last_access_time[page_id] = self.current_time

    def get_access_score(self, page_id: int) -> float:
        """Computes a normalized score balancing recency and frequency."""
        if page_id not in self.access_counts:
            return 0.0
        
        freq = self.access_counts[page_id]
        recency = self.last_access_time[page_id]
        
        # Normalized scores between 0.0 and 1.0
        recency_score = recency / max(1, self.current_time)
        freq_score = min(1.0, freq / 10.0)
        
        return 0.5 * recency_score + 0.5 * freq_score