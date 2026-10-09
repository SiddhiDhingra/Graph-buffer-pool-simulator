"""Online access-history features used by the proposed eviction policy.

The tracker deliberately keeps history after eviction.  A page can be
reloaded and should not look like a brand-new page merely because it left the
buffer once.  Recency is measured with exponential decay and frequency uses
saturation, which avoids the hard frequency cap used in the original version.
"""
from __future__ import annotations

import math
from typing import Dict


class AccessScoreTracker:
    """Track online recency/frequency without requiring the page to be cached."""

    def __init__(self, recency_tau: float = 20.0, frequency_tau: float = 5.0):
        if recency_tau <= 0 or frequency_tau <= 0:
            raise ValueError("recency_tau and frequency_tau must be positive")

        self.recency_tau = float(recency_tau)
        self.frequency_tau = float(frequency_tau)
        self.access_counts: Dict[int, int] = {}
        self.last_access_time: Dict[int, int] = {}
        self.current_time = 0
        self.eviction_counts: Dict[int, int] = {}

    def record_access(self, page_id: int) -> None:
        self.current_time += 1
        self.access_counts[page_id] = self.access_counts.get(page_id, 0) + 1
        self.last_access_time[page_id] = self.current_time

    def get_recency_score(self, page_id: int) -> float:
        """Return a [0,1] score; 1 means the page was just accessed."""
        last = self.last_access_time.get(page_id)
        if last is None:
            return 0.0
        age = max(0, self.current_time - last)
        return math.exp(-age / self.recency_tau)

    def get_frequency_score(self, page_id: int) -> float:
        """Return a saturated [0,1] frequency score."""
        count = self.access_counts.get(page_id, 0)
        if count <= 0:
            return 0.0
        return 1.0 - math.exp(-count / self.frequency_tau)

    def get_access_score(self, page_id: int,
                         recency_weight: float = 0.65,
                         frequency_weight: float = 0.35) -> float:
        """Combine recency and frequency into one [0,1] access score."""
        total = recency_weight + frequency_weight
        if total <= 0:
            raise ValueError("At least one access weight must be positive")
        r = self.get_recency_score(page_id)
        f = self.get_frequency_score(page_id)
        return (recency_weight * r + frequency_weight * f) / total

    def remove_page(self, page_id: int) -> None:
        """Record eviction while retaining historical access information."""
        self.eviction_counts[page_id] = self.eviction_counts.get(page_id, 0) + 1
