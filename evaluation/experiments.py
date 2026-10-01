from __future__ import annotations
from src.data_loader import make_synthetic_cora
from src.page_builder import build_pages
from src.graph_locality import build_page_graph
from src.workload_generator import WORKLOADS, to_page_requests

from src.access_score import AccessScoreTracker
from src.usefulness_score import UsefulnessCalculator
from src.proposed_policy import GraphLSHAwareEvictionPolicy
from evaluation.metrics import EvaluationMetrics


class SimulationEngine:
    def __init__(self, capacity: int, policy_type: str = "lru", policy_engine = None):
        self.capacity = capacity
        self.policy_type = policy_type
        self.policy_engine = policy_engine
        self.buffer: list[int] = []
        self.fifo_queue: list[int] = []
        self.lru_stack: list[int] = []

    def request_page(self, page_id: int) -> bool:
        if self.policy_engine and hasattr(self.policy_engine, 'access_tracker'):
            self.policy_engine.access_tracker.record_access(page_id)

        if page_id in self.buffer:
            if self.policy_type == "lru":
                self.lru_stack.remove(page_id)
                self.lru_stack.append(page_id)
            return True  # HIT

        if len(self.buffer) < self.capacity:
            self.buffer.append(page_id)
            if self.policy_type == "fifo":
                self.fifo_queue.append(page_id)
            elif self.policy_type == "lru":
                self.lru_stack.append(page_id)
        else:
            victim = self._select_victim()
            self._evict(victim)
            self.buffer.append(page_id)
            if self.policy_type == "fifo":
                self.fifo_queue.append(page_id)
            elif self.policy_type == "lru":
                self.lru_stack.append(page_id)

        return False  # PAGE FAULT

    def is_full(self) -> bool:
        return len(self.buffer) >= self.capacity

    def _select_victim(self) -> int:
        if self.policy_type == "fifo":
            return self.fifo_queue.pop(0)
        elif self.policy_type == "lru":
            return self.lru_stack.pop(0)
        elif self.policy_type == "proposed":
            return self.policy_engine.select_victim(self.buffer)
        return self.buffer[0]

    def _evict(self, victim_page: int):
        if victim_page in self.buffer:
            self.buffer.remove(victim_page)


def run_comparison_experiment() -> dict:
    print("🚀 Initializing Synthetic Cora Dataset for Evaluation...")
    g = make_synthetic_cora(num_nodes=300, num_edges=500, num_features=4, seed=42)
    t = build_pages(g, page_size=50)
    pg = build_page_graph(g, t)

    
    nodes = WORKLOADS["traversal"](g, length=1000, restart_prob=0.1, seed=42)
    page_requests = to_page_requests(nodes, t)

    buffer_capacity = 5
    policies = ["fifo", "lru", "proposed"]
    results = {}
    

    print(f"\n📊 Running comparative simulations (Capacity={buffer_capacity}, Workload Size={len(page_requests)})...\n")

    for pol in policies:
        access_tracker = AccessScoreTracker()
        usefulness_calc = UsefulnessCalculator(alpha=0.6, beta=0.0, gamma=0.4)
        
        proposed_engine = GraphLSHAwareEvictionPolicy(
            usefulness_calculator=usefulness_calc,
            page_graph=pg,
            lsh_manager=None,
            access_tracker=access_tracker
        )

        pool = SimulationEngine(
            capacity=buffer_capacity, 
            policy_type=pol, 
            policy_engine=proposed_engine
        )
        metrics = EvaluationMetrics()

        for page_id in page_requests:
            is_hit = pool.request_page(page_id)
            if is_hit:
                metrics.log_hit()
            else:
                metrics.log_fault()
                if pool.is_full():
                    metrics.log_eviction()

        results[pol] = metrics.summary()
        print(f"Policy: {pol.upper()} -> Hit Ratio: {results[pol]['Hit Ratio']}% | Page Faults: {results[pol]['Page Faults']} | Latency: {results[pol]['Simulated Latency']}")

    return results

if __name__ == "__main__":
    run_comparison_experiment()