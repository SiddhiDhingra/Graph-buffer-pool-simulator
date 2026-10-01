import unittest

from src.data_loader import make_synthetic_cora
from src.page_builder import build_pages
from src.graph_locality import build_page_graph, calculate_graph_locality


class TestLocality(unittest.TestCase):

    def test_cross_page_edge(self):
        g = make_synthetic_cora(
            num_nodes=200,
            num_edges=1,
            num_features=4,
            seed=0
        )
        g.edges = [(10, 70)]

        t = build_pages(g, 50)
        pg = build_page_graph(g, t)

        self.assertEqual(
            calculate_graph_locality(0, 1, pg, "count"),
            1.0
        )

        self.assertEqual(
            calculate_graph_locality(0, 2, pg, "count"),
            0.0
        )

    def test_symmetry_and_range(self):
        g = make_synthetic_cora(seed=2)
        t = build_pages(g)
        pg = build_page_graph(g, t)

        for a, b in [(0, 1), (3, 40), (10, 11)]:
            s1 = calculate_graph_locality(a, b, pg)

            self.assertAlmostEqual(
                s1,
                calculate_graph_locality(b, a, pg)
            )

            self.assertTrue(0.0 <= s1 <= 1.0)

        self.assertGreater(
            calculate_graph_locality(10, 11, pg),
            calculate_graph_locality(10, 50, pg)
        )


if __name__ == "__main__":
    unittest.main()