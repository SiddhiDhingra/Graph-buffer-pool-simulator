from __future__ import annotations


class EvaluationMetrics:
    """Research metrics collected from one simulator run."""

    def __init__(self, buffer_cost: float = 1.0, storage_cost: float = 10.0):
        if buffer_cost < 0 or storage_cost < 0:
            raise ValueError("latency costs cannot be negative")
        self.hits = 0
        self.misses = 0
        self.page_faults = 0
        self.storage_reads = 0
        self.evictions = 0
        self.buffer_cost = float(buffer_cost)
        self.storage_cost = float(storage_cost)
        self.policy_decision_time = 0.0
        self.policy_decisions = 0

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_ratio

    @property
    def simulated_latency(self) -> float:
        total_accesses = self.hits + self.misses
        return total_accesses * self.buffer_cost + self.storage_reads * self.storage_cost

    @property
    def avg_policy_decision_time_ms(self) -> float:
        return (
            1000.0 * self.policy_decision_time / self.policy_decisions
            if self.policy_decisions else 0.0
        )

    def summary(self) -> dict:
        return {
            "Hit Ratio": round(self.hit_ratio * 100.0, 4),
            "Miss Rate": round(self.miss_rate * 100.0, 4),
            "Page Faults": int(self.page_faults),
            "Storage I/O": int(self.storage_reads),
            "Evictions": int(self.evictions),
            "Simulated Latency": round(self.simulated_latency, 4),
            "Policy Decision Time (ms)": round(self.policy_decision_time * 1000.0, 4),
            "Avg Decision Time (ms)": round(self.avg_policy_decision_time_ms, 6),
        }
