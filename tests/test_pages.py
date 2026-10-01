import os, tempfile, unittest
from src.data_loader import load_cora, make_synthetic_cora
from src.page_builder import build_pages, get_page_for_node, PAGE_SIZE
from src.graph_locality import build_page_graph, calculate_graph_locality


class TestLoader(unittest.TestCase):
    def test_load_cora_files(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "cora.content"), "w") as f:
                f.write("100 1 0 1 A\n200 0 1 0 B\n300 1 1 0 A\n")
            with open(os.path.join(d, "cora.cites"), "w") as f:
                f.write("100 200\n200 100\n300 100\n999 100\n")  # dup + dangling
            g = load_cora(d)
            self.assertEqual(g.num_nodes, 3)
            self.assertEqual(g.features.shape, (3, 3))
            self.assertEqual(g.edges, [(0, 1), (0, 2)])
            self.assertEqual(g.label_names, ["A", "B"])

    def test_missing_files(self):
        with self.assertRaises(FileNotFoundError):
            load_cora("/nonexistent")


class TestPages(unittest.TestCase):
    def setUp(self):
        self.g = make_synthetic_cora(seed=1)
        self.t = build_pages(self.g, PAGE_SIZE)

    def test_layout(self):
        self.assertEqual(self.t.page_to_nodes[0], list(range(0, 50)))
        self.assertEqual(self.t.page_to_nodes[1], list(range(50, 100)))
        self.assertEqual(self.t.num_pages, 55)           # ceil(2708/50)
        self.assertEqual(len(self.t.page_to_nodes[54]), 8)

    def test_mappings_consistent(self):
        for n in range(self.g.num_nodes):
            p = get_page_for_node(n)
            self.assertIn(n, self.t.page_to_nodes[p])
        self.assertEqual(sum(len(v) for v in self.t.page_to_nodes.values()),
                         self.g.num_nodes)

    def test_unknown_node(self):
        with self.assertRaises(KeyError):
            self.t.get_page_for_node(10**6)


class TestLocality(unittest.TestCase):
    def test_cross_page_edge(self):
        g = make_synthetic_cora(num_nodes=200, num_edges=1, num_features=4, seed=0)
        g.edges = [(10, 70)]
        t = build_pages(g, 50)
        pg = build_page_graph(g, t)
        self.assertEqual(calculate_graph_locality(0, 1, pg, "count"), 1.0)
        self.assertEqual(calculate_graph_locality(0, 2, pg, "count"), 0.0)

    def test_symmetry_and_range(self):
        g = make_synthetic_cora(seed=2)
        t = build_pages(g)
        pg = build_page_graph(g, t)
        for a, b in [(0, 1), (3, 40), (10, 11)]:
            s1 = calculate_graph_locality(a, b, pg)
            self.assertAlmostEqual(s1, calculate_graph_locality(b, a, pg))
            self.assertTrue(0.0 <= s1 <= 1.0)
        self.assertGreater(calculate_graph_locality(10, 11, pg),
                           calculate_graph_locality(10, 50, pg))


if __name__ == "__main__":
    unittest.main()
