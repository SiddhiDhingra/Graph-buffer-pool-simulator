"""Streamlit dashboard for THGB DBLP-TA buffer-pool experiments.

This dashboard uses the repository's THGB page-level experiment pipeline.
The page layout is an explicit simulator assumption: consecutive node IDs are
packed into fixed-size logical pages; it is not claimed to be DBLP's physical
on-disk layout.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from evaluation.run_thgb_experiments import (
    build_storage,
    count_nodes,
    make_policy,
    make_workload,
    run_trace,
)
from src.thgb_graph_adapter import build_thgb_page_graph
from src.thgb_page_vectors import build_thgb_page_vectors
from src.lsh import LSHManager
from src.usefulness_score import UsefulnessCalculator

st.set_page_config(page_title="THGB Graph-Aware Buffer Pool Simulator", page_icon="🗄️", layout="wide")
DATA_DIR = Path("data/thgb/DBLP-TA")

st.title("🗄️ THGB DBLP-TA — Graph-Aware Buffer Pool Simulator")
st.caption("THGB node embeddings + typed graph data → logical pages → LSH/graph-aware eviction")
st.info("The dashboard uses THGB DBLP-TA only. Logical pages currently group consecutive node IDs; this is a simulator layout assumption, not a claim about the dataset's physical storage layout.")

@st.cache_resource(show_spinner="Reading THGB node records and building logical pages…")
def load_thgb_page_storage(data_dir: str, page_size: int):
    root = Path(data_dir)
    node_path, link_path = root / "node.dat", root / "link.dat"
    if not node_path.is_file() or not link_path.is_file():
        raise FileNotFoundError(f"THGB DBLP-TA files not found. Expected {node_path} and {link_path}.")
    n_nodes = count_nodes(node_path)
    storage = build_storage(n_nodes, page_size)
    return n_nodes, storage

@st.cache_resource(show_spinner="Building THGB page vectors and graph index…")
def load_thgb_indexes(data_dir: str, page_size: int, lsh_tables: int, lsh_planes: int, probe_radius: int):
    vectors = build_thgb_page_vectors(data_dir=data_dir, page_size=page_size, normalize=True)
    page_vectors = {i: vectors[i] for i in range(vectors.shape[0])}
    page_graph = build_thgb_page_graph(data_dir=data_dir, page_size=page_size)
    lsh = LSHManager(page_vectors, num_tables=lsh_tables, num_planes=lsh_planes, seed=42, probe_radius=probe_radius)
    calc = UsefulnessCalculator(alpha=0.40, beta=0.40, gamma=0.20, graph_mode="directional", semantic_probe_radius=probe_radius, locality_gate=True)
    calc.prepare(sorted(page_vectors), page_graph, lsh)
    return vectors, page_graph, lsh, calc

with st.sidebar:
    st.header("Experiment settings")
    data_dir = st.text_input("THGB dataset directory", str(DATA_DIR))
    page_size = st.number_input("Logical page size (nodes)", min_value=1, max_value=10000, value=50, step=1)
    capacity = st.number_input("Buffer capacity (pages)", min_value=1, max_value=10000, value=50, step=1)
    requests = st.number_input("Page requests", min_value=10, max_value=1000000, value=1000, step=100)
    seed = st.number_input("Random seed", min_value=0, max_value=2**31-1, value=42)
    workload_name = st.selectbox("Workload", ["random", "graph", "semantic"])
    lsh_tables = st.number_input("LSH tables", min_value=1, max_value=32, value=5)
    lsh_planes = st.number_input("LSH hyperplanes/table", min_value=1, max_value=16, value=6)
    probe_radius = st.selectbox("LSH probe radius", [0, 1, 2], index=1)

if st.button("Build THGB indexes", type="secondary"):
    try:
        n_nodes, storage = load_thgb_page_storage(data_dir, int(page_size))
        vectors, page_graph, lsh, calc = load_thgb_indexes(data_dir, int(page_size), int(lsh_tables), int(lsh_planes), int(probe_radius))
        st.session_state["thgb_ready"] = True
        st.success(f"THGB indexes ready: {n_nodes:,} nodes, {len(storage):,} logical pages, vector shape {vectors.shape}.")
    except Exception as exc:
        st.error(f"THGB initialization failed: {exc}")
        st.exception(exc)

if st.button("▶ Run LRU vs Proposed", type="primary"):
    try:
        n_nodes, storage = load_thgb_page_storage(data_dir, int(page_size))
        vectors, page_graph, lsh, calc = load_thgb_indexes(data_dir, int(page_size), int(lsh_tables), int(lsh_planes), int(probe_radius))
        page_ids = sorted(storage)
        trace = make_workload(workload_name, int(requests), int(seed), page_ids, page_graph, lsh)
        shared = calc
        rows = []
        for policy_name in ("LRU", "Proposed"):
            policy = make_policy(policy_name, page_graph, lsh, shared_calculator=shared if policy_name == "Proposed" else None)
            metrics = run_trace(trace, storage, int(capacity), policy)
            rows.append({"Dataset": "THGB DBLP-TA", "Workload": workload_name, "Policy": policy_name, **metrics})
        frame = pd.DataFrame(rows)
        st.session_state["thgb_results"] = frame
        st.session_state["thgb_summary"] = {"Nodes": n_nodes, "Edges": sum(1 for _ in (Path(data_dir)/"link.dat").open("r", encoding="utf-8")), "Pages": len(storage), "Vector dimensions": int(vectors.shape[1])}
    except Exception as exc:
        st.error(f"THGB simulation failed: {exc}")
        st.exception(exc)

if "thgb_results" in st.session_state:
    st.subheader("THGB experiment results")
    summary = st.session_state["thgb_summary"]
    a,b,c,d = st.columns(4)
    a.metric("THGB nodes", f"{summary['Nodes']:,}")
    b.metric("THGB edges", f"{summary['Edges']:,}")
    c.metric("Logical pages", f"{summary['Pages']:,}")
    d.metric("Embedding dimensions", summary["Vector dimensions"])
    st.dataframe(st.session_state["thgb_results"], use_container_width=True, hide_index=True)
    st.download_button("Download results CSV", st.session_state["thgb_results"].to_csv(index=False), "thgb_streamlit_results.csv", "text/csv")

st.divider()
st.caption("Research note: random, graph-transition, and LSH-neighbor traces are synthetic page-request workloads derived from THGB structures. They are not an end-to-end RAG evaluation. Compare policies on identical traces and report all workloads, seeds, and capacities.")
