from __future__ import annotations

class EvaluationMetrics:
    """Tracks simulator performance metrics (hits, faults, disk I/O, latency)."""
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.page_faults = 0
        self.storage_reads = 0
        self.evictions = 0
        
        self.buffer_cost = 1
        self.storage_cost = 10

    def log_hit(self):
        self.hits += 1

    def log_fault(self):
        self.misses += 1
        self.page_faults += 1
        self.storage_reads += 1

    def log_eviction(self):
        self.evictions += 1

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

    @property
    def simulated_latency(self) -> int:
        total_accesses = self.hits + self.misses
        return (total_accesses * self.buffer_cost) + (self.storage_reads * self.storage_cost)

    def summary(self) -> dict:
        return {
            "Hit Ratio": round(self.hit_ratio * 100, 2),
            "Page Faults": self.page_faults,
            "Storage I/O": self.storage_reads,
            "Evictions": self.evictions,
            "Simulated Latency": self.simulated_latency
        }