import unittest
from src.data_loader import make_synthetic_cora
from src.page_builder import build_pages
from src.workload_generator import (random_workload, graph_traversal_workload,
    local_graph_workload, mixed_workload, to_page_requests)


class TestWorkloads(unittest.TestCase):
    def setUp(self):
        self.g = make_synthetic_cora(seed=3)
        self.t = build_pages(self.g)

    def test_length_range_and_determinism(self):
        for fn in (random_workload, graph_traversal_workload,
                   local_graph_workload, mixed_workload):
            w = fn(self.g, 1000, seed=7)
            self.assertEqual(len(w), 1000)
            self.assertTrue(all(0 <= n < self.g.num_nodes for n in w))
            self.assertEqual(w, fn(self.g, 1000, seed=7))

    def test_traversal_follows_edges(self):
        w = graph_traversal_workload(self.g, 2000, restart_prob=0.0, seed=1)
        hops = sum(b in self.g.neighbors(a) for a, b in zip(w, w[1:]))
        self.assertGreater(hops / (len(w) - 1), 0.95)

    def test_locality_ordering(self):
        distinct = lambda w: len(set(to_page_requests(w, self.t)))
        r = distinct(random_workload(self.g, 1000, seed=1))
        l = distinct(local_graph_workload(self.g, 1000, dwell=1000, seed=1))
        self.assertLess(l, r)

    def test_mixed_bad_mix(self):
        with self.assertRaises(ValueError):
            mixed_workload(self.g, 100, mix=(0.5, 0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
