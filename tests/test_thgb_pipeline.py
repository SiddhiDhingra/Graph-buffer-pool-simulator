from pathlib import Path
import numpy as np
import pytest
from src.thgb_loader import load_thgb_dblp
from src.thgb_page_vectors import build_thgb_page_vectors
from src.thgb_graph_adapter import build_thgb_page_graph


def make_dataset(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    # Four tab-separated fields: id, text, type, comma-separated embedding.
    (root / "node.dat").write_text(
        "0\talpha paper\t0\t1,0,0\n"
        "1\tbeta author\t1\t0,1,0\n"
        "2\tgamma paper\t0\t0,0,1\n"
        "3\tdelta venue\t2\t1,1,0\n", encoding="utf-8")
    (root / "link.dat").write_text(
        "0\t1\t0\t1\n1\t2\t1\t1\n3\t0\t2\t1\n", encoding="utf-8")


def test_thgb_loader_preserves_node_and_edge_types(tmp_path):
    make_dataset(tmp_path)
    graph = load_thgb_dblp(str(tmp_path), load_embeddings=True, build_adjacency=True)
    assert graph.num_nodes == 4
    assert graph.node_types.tolist() == [0, 1, 0, 2]
    assert [e[2] for e in graph.edges] == [0, 1, 2]
    assert graph.embeddings.shape == (4, 3)
    assert 1 in graph.neighbors(0)


def test_page_vectors_include_partial_final_page(tmp_path):
    make_dataset(tmp_path)
    vectors = build_thgb_page_vectors(str(tmp_path), page_size=3, normalize=False)
    assert vectors.shape == (2, 3)
    np.testing.assert_allclose(vectors[0], [1/3, 1/3, 1/3])
    np.testing.assert_allclose(vectors[1], [1, 1, 0])


def test_page_graph_preserves_edge_type_counts(tmp_path):
    make_dataset(tmp_path)
    graph = build_thgb_page_graph(str(tmp_path), page_size=2, num_nodes=4)
    assert graph.edge_type_internal[0][0] == 1
    assert graph.edge_type_weights[1][0][1] == 1
    assert graph.edge_type_weights[2][1][0] == 1


def test_missing_thgb_files_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_thgb_dblp(str(tmp_path))
