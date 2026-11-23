# """Streamlit dashboard for THGB DBLP-TA buffer-pool experiments.

# This dashboard uses the repository's THGB page-level experiment pipeline.
# The page layout is an explicit simulator assumption: consecutive node IDs are
# packed into fixed-size logical pages; it is not claimed to be DBLP's physical
# on-disk layout.
# """
# from __future__ import annotations

# from pathlib import Path
# import pandas as pd
# import streamlit as st

# from evaluation.run_thgb_experiments import (
#     build_storage,
#     count_nodes,
#     make_policy,
#     make_workload,
#     run_trace,
# )
# from src.thgb_graph_adapter import build_thgb_page_graph
# from src.thgb_page_vectors import build_thgb_page_vectors
# from src.lsh import LSHManager
# from src.usefulness_score import UsefulnessCalculator

# st.set_page_config(page_title="THGB Graph-Aware Buffer Pool Simulator", page_icon="🗄️", layout="wide")
# DATA_DIR = Path("data/thgb/DBLP-TA")

# st.title("🗄️ THGB DBLP-TA — Graph-Aware Buffer Pool Simulator")
# st.caption("THGB node embeddings + typed graph data → logical pages → LSH/graph-aware eviction")
# st.info("The dashboard uses THGB DBLP-TA only. Logical pages currently group consecutive node IDs; this is a simulator layout assumption, not a claim about the dataset's physical storage layout.")

# @st.cache_resource(show_spinner="Reading THGB node records and building logical pages…")
# def load_thgb_page_storage(data_dir: str, page_size: int):
#     root = Path(data_dir)
#     node_path, link_path = root / "node.dat", root / "link.dat"
#     if not node_path.is_file() or not link_path.is_file():
#         raise FileNotFoundError(f"THGB DBLP-TA files not found. Expected {node_path} and {link_path}.")
#     n_nodes = count_nodes(node_path)
#     storage = build_storage(n_nodes, page_size)
#     return n_nodes, storage

# @st.cache_resource(show_spinner="Building THGB page vectors and graph index…")
# def load_thgb_indexes(data_dir: str, page_size: int, lsh_tables: int, lsh_planes: int, probe_radius: int):
#     vectors = build_thgb_page_vectors(data_dir=data_dir, page_size=page_size, normalize=True)
#     page_vectors = {i: vectors[i] for i in range(vectors.shape[0])}
#     page_graph = build_thgb_page_graph(data_dir=data_dir, page_size=page_size)
#     lsh = LSHManager(page_vectors, num_tables=lsh_tables, num_planes=lsh_planes, seed=42, probe_radius=probe_radius)
#     calc = UsefulnessCalculator(alpha=0.40, beta=0.40, gamma=0.20, graph_mode="directional", semantic_probe_radius=probe_radius, locality_gate=True)
#     calc.prepare(sorted(page_vectors), page_graph, lsh)
#     return vectors, page_graph, lsh, calc

# with st.sidebar:
#     st.header("Experiment settings")
#     data_dir = st.text_input("THGB dataset directory", str(DATA_DIR))
#     page_size = st.number_input("Logical page size (nodes)", min_value=1, max_value=10000, value=50, step=1)
#     capacity = st.number_input("Buffer capacity (pages)", min_value=1, max_value=10000, value=50, step=1)
#     requests = st.number_input("Page requests", min_value=10, max_value=1000000, value=1000, step=100)
#     seed = st.number_input("Random seed", min_value=0, max_value=2**31-1, value=42)
#     workload_name = st.selectbox("Workload", ["random", "mixed" , "graph", "semantic"])
#     lsh_tables = st.number_input("LSH tables", min_value=1, max_value=32, value=5)
#     lsh_planes = st.number_input("LSH hyperplanes/table", min_value=1, max_value=16, value=6)
#     probe_radius = st.selectbox("LSH probe radius", [0, 1, 2], index=1)

# if st.button("Build THGB indexes", type="secondary"):
#     try:
#         n_nodes, storage = load_thgb_page_storage(data_dir, int(page_size))
#         vectors, page_graph, lsh, calc = load_thgb_indexes(data_dir, int(page_size), int(lsh_tables), int(lsh_planes), int(probe_radius))
#         st.session_state["thgb_ready"] = True
#         st.success(f"THGB indexes ready: {n_nodes:,} nodes, {len(storage):,} logical pages, vector shape {vectors.shape}.")
#     except Exception as exc:
#         st.error(f"THGB initialization failed: {exc}")
#         st.exception(exc)

# if st.button("▶ Run FIFO vs LRU vs Proposed", type="primary"):
#     try:
#         n_nodes, storage = load_thgb_page_storage(data_dir, int(page_size))
#         vectors, page_graph, lsh, calc = load_thgb_indexes(data_dir, int(page_size), int(lsh_tables), int(lsh_planes), int(probe_radius))
#         page_ids = sorted(storage)
#         trace = make_workload(workload_name, int(requests), int(seed), page_ids, page_graph, lsh)
#         shared = calc
#         rows = []
#         for policy_name in ("FIFO", "LRU", "Proposed"):
#             policy = make_policy(
#                 policy_name,
#                 page_graph,
#                 lsh,
#                 shared_calculator=shared if policy_name == "Proposed" else None
#             )

#             metrics = run_trace(trace, storage, int(capacity), policy)

#             rows.append({
#                 "Dataset": "THGB DBLP-TA",
#                 "Workload": workload_name,
#                 "Policy": policy_name,
#                 **metrics
#             })
#         frame = pd.DataFrame(rows)
#         st.session_state["thgb_results"] = frame
#         st.session_state["thgb_summary"] = {"Nodes": n_nodes, "Edges": sum(1 for _ in (Path(data_dir)/"link.dat").open("r", encoding="utf-8")), "Pages": len(storage), "Vector dimensions": int(vectors.shape[1])}
#     except Exception as exc:
#         st.error(f"THGB simulation failed: {exc}")
#         st.exception(exc)

# if "thgb_results" in st.session_state:
#     st.subheader("THGB experiment results")
#     summary = st.session_state["thgb_summary"]
#     a,b,c,d = st.columns(4)
#     a.metric("THGB nodes", f"{summary['Nodes']:,}")
#     b.metric("THGB edges", f"{summary['Edges']:,}")
#     c.metric("Logical pages", f"{summary['Pages']:,}")
#     d.metric("Embedding dimensions", summary["Vector dimensions"])
#     st.dataframe(st.session_state["thgb_results"], use_container_width=True, hide_index=True)
#     st.download_button("Download results CSV", st.session_state["thgb_results"].to_csv(index=False), "thgb_streamlit_results.csv", "text/csv")

# st.divider()


"""Streamlit dashboard for THGB DBLP-TA and Cora buffer-pool experiments.

Both datasets support live FIFO, LRU and Proposed policy comparisons.
Cora also includes the historical results from its completed experiments.

Logical pages are simulator-level assumptions, not claims about physical
on-disk database page layouts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# THGB experiment pipeline
# ---------------------------------------------------------

from evaluation.run_thgb_experiments import (
    build_storage,
    count_nodes,
    make_policy as make_thgb_policy,
    make_workload as make_thgb_workload,
    run_trace as run_thgb_trace,
)

from src.thgb_graph_adapter import build_thgb_page_graph
from src.thgb_page_vectors import build_thgb_page_vectors


# ---------------------------------------------------------
# Cora experiment pipeline
# ---------------------------------------------------------

from evaluation.run_cora_experiments import (
    make_workload as make_cora_workload,
    make_policy as make_cora_policy,
    run_trace as run_cora_trace,
)

from src.cora_loader import load_cora
from src.page_builder import build_pages
from src.page_vectors import build_page_vectors
from src.graph_locality import build_page_graph


# ---------------------------------------------------------
# Shared components
# ---------------------------------------------------------

from src.lsh import LSHManager
from src.usefulness_score import UsefulnessCalculator


# ---------------------------------------------------------
# Application configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Graph-Aware Buffer Pool Simulator",
    page_icon="📊",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent

THGB_DATA_DIR = ROOT / "data" / "thgb" / "DBLP-TA"
CORA_DATA_DIR = ROOT / "data" / "cora"

CORA_RESULTS = ROOT / "results" / "cora" / "cora_results.csv"
CORA_METADATA = ROOT / "results" / "cora" / "cora_metadata.json"


st.title("Graph-Aware Buffer Pool Simulator")

st.caption(
    "Comparative evaluation of FIFO, LRU and "
    "Graph-LSH-aware Proposed eviction"
)


# =========================================================
# THGB: LOAD STORAGE
# =========================================================

@st.cache_resource(
    show_spinner="Reading THGB node records and building logical pages..."
)
def load_thgb_page_storage(data_dir: str, page_size: int):

    root = Path(data_dir)

    node_path = root / "node.dat"
    link_path = root / "link.dat"

    if not node_path.is_file() or not link_path.is_file():
        raise FileNotFoundError(
            f"THGB dataset files not found. Expected {node_path} "
            f"and {link_path}."
        )

    n_nodes = count_nodes(node_path)

    storage = build_storage(
        n_nodes,
        page_size
    )

    return n_nodes, storage


# =========================================================
# THGB: BUILD INDEXES
# =========================================================

@st.cache_resource(
    show_spinner="Building THGB page vectors and graph index..."
)
def load_thgb_indexes(
    data_dir: str,
    page_size: int,
    lsh_tables: int,
    lsh_planes: int,
    probe_radius: int
):

    vectors = build_thgb_page_vectors(
        data_dir=data_dir,
        page_size=page_size,
        normalize=True
    )

    page_vectors = {
        i: vectors[i]
        for i in range(vectors.shape[0])
    }

    page_graph = build_thgb_page_graph(
        data_dir=data_dir,
        page_size=page_size
    )

    lsh = LSHManager(
        page_vectors,
        num_tables=lsh_tables,
        num_planes=lsh_planes,
        seed=42,
        probe_radius=probe_radius
    )

    calculator = UsefulnessCalculator(
        alpha=0.40,
        beta=0.40,
        gamma=0.20,
        graph_mode="directional",
        semantic_probe_radius=probe_radius,
        locality_gate=True
    )

    calculator.prepare(
        sorted(page_vectors),
        page_graph,
        lsh
    )

    return vectors, page_graph, lsh, calculator


# =========================================================
# CORA: BUILD STORAGE AND INDEXES
# =========================================================

@st.cache_resource(
    show_spinner="Loading Cora and building graph/LSH indexes..."
)
def load_cora_indexes(
    data_dir: str,
    page_size: int,
    lsh_tables: int,
    lsh_planes: int,
    probe_radius: int
):

    # Load citation network
    graph = load_cora(data_dir)

    # Divide nodes into logical pages
    page_table = build_pages(
        graph,
        page_size
    )

    storage = page_table.pages
    page_ids = sorted(storage)

    # Generate page feature vectors
    page_vectors = build_page_vectors(
        graph,
        page_table
    )

    # Construct graph relationships between pages
    page_graph = build_page_graph(
        graph,
        page_table
    )

    # Build LSH index
    lsh = LSHManager(
        page_vectors,
        num_tables=lsh_tables,
        num_planes=lsh_planes,
        seed=42,
        probe_radius=probe_radius
    )

    # Prepare usefulness calculator
    calculator = UsefulnessCalculator(
        alpha=0.40,
        beta=0.40,
        gamma=0.20,
        graph_mode="directional",
        semantic_probe_radius=probe_radius,
        locality_gate=True
    )

    calculator.prepare(
        page_ids,
        page_graph,
        lsh
    )

    return (
        graph,
        storage,
        page_ids,
        page_vectors,
        page_graph,
        lsh,
        calculator
    )


# =========================================================
# CORA: LOAD HISTORICAL RESULTS
# =========================================================

@st.cache_data
def load_cora_results(csv_path: str):

    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Cora results file not found: {path}"
        )

    frame = pd.read_csv(path)

    required_columns = [
        "Dataset",
        "Workload",
        "Seed",
        "Policy",
        "Capacity",
        "Hit Ratio (%)",
        "Page Faults",
        "Evictions",
        "Hits",
        "Storage Reads",
        "Policy Decision Time (ms)",
        "Avg Decision Time (ms)",
        "Total Runtime (s)",
        "Requests",
        "Page Size"
    ]

    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Cora CSV is missing columns: {missing}"
        )

    return frame


# =========================================================
# THGB DASHBOARD
# =========================================================

def show_thgb_dashboard():

    st.header("THGB DBLP-TA")

    st.info(
        "THGB uses live simulation. Logical pages group consecutive "
        "node IDs. This is a simulator assumption, not a claim about "
        "physical database storage."
    )

    # -----------------------------------------------------
    # Sidebar settings
    # -----------------------------------------------------

    with st.sidebar:

        st.subheader("THGB experiment settings")

        data_dir = st.text_input(
            "THGB dataset directory",
            str(THGB_DATA_DIR),
            key="thgb_data_dir"
        )

        page_size = st.number_input(
            "Logical page size (nodes)",
            min_value=1,
            max_value=10000,
            value=50,
            step=1,
            key="thgb_page_size"
        )

        capacity = st.number_input(
            "Buffer capacity (pages)",
            min_value=1,
            max_value=10000,
            value=10,
            step=1,
            key="thgb_capacity"
        )

        requests = st.number_input(
            "Page requests",
            min_value=10,
            max_value=1000000,
            value=1000,
            step=100,
            key="thgb_requests"
        )

        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2**31 - 1,
            value=42,
            key="thgb_seed"
        )

        workload_name = st.selectbox(
            "Workload",
            ["random", "mixed", "graph", "semantic"],
            key="thgb_workload"
        )

        lsh_tables = st.number_input(
            "LSH tables",
            min_value=1,
            max_value=32,
            value=5,
            key="thgb_lsh_tables"
        )

        lsh_planes = st.number_input(
            "LSH hyperplanes/table",
            min_value=1,
            max_value=16,
            value=6,
            key="thgb_lsh_planes"
        )

        probe_radius = st.selectbox(
            "LSH probe radius",
            [0, 1, 2],
            index=1,
            key="thgb_probe_radius"
        )

    # -----------------------------------------------------
    # Build THGB indexes
    # -----------------------------------------------------

    if st.button(
        "Build THGB indexes",
        type="secondary",
        key="build_thgb"
    ):

        try:

            n_nodes, storage = load_thgb_page_storage(
                data_dir,
                int(page_size)
            )

            vectors, page_graph, lsh, calculator = load_thgb_indexes(
                data_dir,
                int(page_size),
                int(lsh_tables),
                int(lsh_planes),
                int(probe_radius)
            )

            st.success(
                f"THGB indexes ready: {n_nodes:,} nodes, "
                f"{len(storage):,} logical pages, "
                f"vector shape {vectors.shape}."
            )

        except Exception as exc:

            st.error(
                f"THGB initialization failed: {exc}"
            )

            st.exception(exc)

    # -----------------------------------------------------
    # Run THGB simulation
    # -----------------------------------------------------

    if st.button(
        "Run FIFO vs LRU vs Proposed",
        type="primary",
        key="run_thgb"
    ):

        try:

            n_nodes, storage = load_thgb_page_storage(
                data_dir,
                int(page_size)
            )

            vectors, page_graph, lsh, calculator = load_thgb_indexes(
                data_dir,
                int(page_size),
                int(lsh_tables),
                int(lsh_planes),
                int(probe_radius)
            )

            page_ids = sorted(storage)

            # Generate one common trace
            trace = make_thgb_workload(
                workload_name,
                int(requests),
                int(seed),
                page_ids,
                page_graph,
                lsh
            )

            rows = []

            for policy_name in ("FIFO", "LRU", "Proposed"):

                policy = make_thgb_policy(
                    policy_name,
                    page_graph,
                    lsh,
                    shared_calculator=(
                        calculator
                        if policy_name == "Proposed"
                        else None
                    )
                )

                metrics = run_thgb_trace(
                    trace,
                    storage,
                    int(capacity),
                    policy
                )

                rows.append({
                    "Dataset": "THGB DBLP-TA",
                    "Workload": workload_name,
                    "Seed": int(seed),
                    "Policy": policy_name,
                    "Page Size": int(page_size),
                    **metrics
                })

            frame = pd.DataFrame(rows)

            st.session_state["thgb_results"] = frame

            link_path = Path(data_dir) / "link.dat"

            st.session_state["thgb_summary"] = {
                "Nodes": n_nodes,
                "Edges": sum(
                    1
                    for _ in link_path.open(
                        "r",
                        encoding="utf-8"
                    )
                ),
                "Pages": len(storage),
                "Vector dimensions": int(vectors.shape[1])
            }

        except Exception as exc:

            st.error(
                f"THGB simulation failed: {exc}"
            )

            st.exception(exc)

    # -----------------------------------------------------
    # Display THGB results
    # -----------------------------------------------------

    if "thgb_results" in st.session_state:

        st.subheader("THGB live simulation results")

        summary = st.session_state["thgb_summary"]
        results = st.session_state["thgb_results"]

        a, b, c, d = st.columns(4)

        a.metric(
            "THGB nodes",
            f"{summary['Nodes']:,}"
        )

        b.metric(
            "THGB edges",
            f"{summary['Edges']:,}"
        )

        c.metric(
            "Logical pages",
            f"{summary['Pages']:,}"
        )

        d.metric(
            "Embedding dimensions",
            summary["Vector dimensions"]
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Performance comparison")

        chart1, chart2 = st.columns(2)

        with chart1:

            st.markdown("**Hit ratio (%)**")

            st.bar_chart(
                results.set_index("Policy")[["Hit Ratio (%)"]]
            )

        with chart2:

            st.markdown("**Page faults**")

            st.bar_chart(
                results.set_index("Policy")[["Page Faults"]]
            )

        st.download_button(
            "Download THGB results CSV",
            results.to_csv(index=False),
            "thgb_streamlit_results.csv",
            "text/csv"
        )


# =========================================================
# CORA: HISTORICAL RESULTS DASHBOARD
# =========================================================

def show_cora_historical_results():

    st.subheader("Completed Cora experiments")

    try:

        results = load_cora_results(
            str(CORA_RESULTS)
        )

        if not CORA_METADATA.is_file():
            raise FileNotFoundError(
                f"Cora metadata file not found: {CORA_METADATA}"
            )

        with CORA_METADATA.open(
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

        st.caption(
            "Historical experiment results averaged across "
            "the completed random seeds."
        )

        # -------------------------------------------------
        # Filters
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        workloads = sorted(
            results["Workload"].unique().tolist()
        )

        capacities = sorted(
            results["Capacity"].astype(int).unique().tolist()
        )

        with col1:

            selected_workload = st.selectbox(
                "Historical workload",
                workloads,
                key="cora_historical_workload"
            )

        with col2:

            selected_capacity = st.selectbox(
                "Historical buffer capacity",
                capacities,
                key="cora_historical_capacity"
            )

        filtered = results[
            (results["Workload"] == selected_workload)
            &
            (
                results["Capacity"].astype(int)
                == int(selected_capacity)
            )
        ].copy()

        # -------------------------------------------------
        # Average results across seeds
        # -------------------------------------------------

        summary = (
            filtered
            .groupby("Policy", as_index=False)
            .agg({
                "Hit Ratio (%)": "mean",
                "Page Faults": "mean",
                "Evictions": "mean",
                "Hits": "mean",
                "Storage Reads": "mean",
                "Policy Decision Time (ms)": "mean",
                "Avg Decision Time (ms)": "mean",
                "Total Runtime (s)": "mean"
            })
        )

        policy_order = [
            "FIFO",
            "LRU",
            "Proposed"
        ]

        summary["Policy"] = pd.Categorical(
            summary["Policy"],
            categories=policy_order,
            ordered=True
        )

        summary = (
            summary
            .sort_values("Policy")
            .reset_index(drop=True)
        )

        summary["Policy"] = summary["Policy"].astype(str)

        st.subheader(
            f"{selected_workload.capitalize()} workload — "
            f"Buffer capacity {selected_capacity}"
        )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        best_row = summary.loc[
            summary["Hit Ratio (%)"].idxmax()
        ]

        x, y, z = st.columns(3)

        x.metric(
            "Best hit ratio",
            f"{best_row['Hit Ratio (%)']:.2f}%",
            best_row["Policy"]
        )

        y.metric(
            "Average page faults",
            f"{summary['Page Faults'].mean():.2f}"
        )

        z.metric(
            "Average evictions",
            f"{summary['Evictions'].mean():.2f}"
        )

        # -------------------------------------------------
        # Comparison table
        # -------------------------------------------------

        st.subheader("Historical policy comparison")

        display_summary = summary.copy()

        numeric_columns = [
            "Hit Ratio (%)",
            "Page Faults",
            "Evictions",
            "Hits",
            "Storage Reads",
            "Policy Decision Time (ms)",
            "Avg Decision Time (ms)",
            "Total Runtime (s)"
        ]

        display_summary[numeric_columns] = (
            display_summary[numeric_columns].round(4)
        )

        st.dataframe(
            display_summary,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # Charts
        # -------------------------------------------------

        st.subheader("Historical performance comparison")

        chart1, chart2 = st.columns(2)

        with chart1:

            st.markdown("**Average hit ratio (%)**")

            st.bar_chart(
                summary.set_index("Policy")[["Hit Ratio (%)"]]
            )

        with chart2:

            st.markdown("**Average page faults**")

            st.bar_chart(
                summary.set_index("Policy")[["Page Faults"]]
            )

        # -------------------------------------------------
        # Individual seed results
        # -------------------------------------------------

        with st.expander("View individual seed results"):

            st.dataframe(
                filtered.sort_values(
                    ["Seed", "Policy"]
                ),
                use_container_width=True,
                hide_index=True
            )

        # -------------------------------------------------
        # Downloads
        # -------------------------------------------------

        st.download_button(
            "Download filtered Cora results",
            filtered.to_csv(index=False),
            "cora_filtered_results.csv",
            "text/csv"
        )

        st.download_button(
            "Download averaged comparison",
            summary.to_csv(index=False),
            "cora_average_comparison.csv",
            "text/csv"
        )

        # -------------------------------------------------
        # Experiment assumptions
        # -------------------------------------------------

        with st.expander("Dataset and experiment assumptions"):

            st.write(
                f"**Citation rows:** "
                f"{metadata['directed_citation_rows']:,}"
            )

            st.write(
                f"**Unique undirected edges:** "
                f"{metadata['unique_undirected_edges']:,}"
            )

            st.write(
                f"**Edge assumption:** "
                f"{metadata['edge_assumption']}"
            )

            st.write(
                f"**Logical page layout:** "
                f"{metadata['layout_assumption']}"
            )

            st.write(
                f"**Class labels:** "
                f"{metadata['label_assumption']}"
            )

            st.write(
                f"**LSH configuration:** "
                f"{metadata['lsh_tables']} tables, "
                f"{metadata['lsh_planes']} hyperplanes, "
                f"probe radius {metadata['probe_radius']}"
            )

    except Exception as exc:

        st.error(
            f"Could not load historical Cora results: {exc}"
        )

        st.exception(exc)


# =========================================================
# CORA: LIVE SIMULATION DASHBOARD
# =========================================================

def show_cora_dashboard():

    st.header("Cora — Live Buffer Pool Simulation")

    st.info(
        "Cora uses live simulation with FIFO, LRU and Proposed policies. "
        "The same request trace is evaluated by all three policies. "
        "Logical pages group consecutive internal node IDs."
    )

    # -----------------------------------------------------
    # Sidebar settings
    # -----------------------------------------------------

    with st.sidebar:

        st.subheader("Cora experiment settings")

        data_dir = st.text_input(
            "Cora dataset directory",
            str(CORA_DATA_DIR),
            key="cora_data_dir"
        )

        page_size = st.number_input(
            "Logical page size (nodes)",
            min_value=1,
            max_value=10000,
            value=50,
            step=1,
            key="cora_page_size"
        )

        capacity = st.number_input(
            "Buffer capacity (pages)",
            min_value=1,
            max_value=10000,
            value=10,
            step=1,
            key="cora_live_capacity"
        )

        requests = st.number_input(
            "Page requests",
            min_value=10,
            max_value=1000000,
            value=1000,
            step=100,
            key="cora_requests"
        )

        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2**31 - 1,
            value=42,
            key="cora_live_seed"
        )

        workload_name = st.selectbox(
            "Workload",
            ["random", "mixed", "graph", "semantic"],
            key="cora_live_workload"
        )

        lsh_tables = st.number_input(
            "LSH tables",
            min_value=1,
            max_value=32,
            value=5,
            key="cora_lsh_tables"
        )

        lsh_planes = st.number_input(
            "LSH hyperplanes/table",
            min_value=1,
            max_value=16,
            value=6,
            key="cora_lsh_planes"
        )

        probe_radius = st.selectbox(
            "LSH probe radius",
            [0, 1, 2],
            index=1,
            key="cora_probe_radius"
        )

    # -----------------------------------------------------
    # Build Cora indexes
    # -----------------------------------------------------

    if st.button(
        "Build Cora indexes",
        type="secondary",
        key="build_cora"
    ):

        try:

            (
                graph,
                storage,
                page_ids,
                page_vectors,
                page_graph,
                lsh,
                calculator
            ) = load_cora_indexes(
                data_dir,
                int(page_size),
                int(lsh_tables),
                int(lsh_planes),
                int(probe_radius)
            )

            st.success(
                f"Cora indexes ready: {graph.num_nodes:,} nodes, "
                f"{len(page_ids):,} logical pages, "
                f"{len(page_vectors)} page vectors."
            )

        except Exception as exc:

            st.error(
                f"Cora initialization failed: {exc}"
            )

            st.exception(exc)

    # -----------------------------------------------------
    # Run Cora live simulation
    # -----------------------------------------------------

    if st.button(
        "Run FIFO vs LRU vs Proposed",
        type="primary",
        key="run_cora"
    ):

        try:

            (
                graph,
                storage,
                page_ids,
                page_vectors,
                page_graph,
                lsh,
                calculator
            ) = load_cora_indexes(
                data_dir,
                int(page_size),
                int(lsh_tables),
                int(lsh_planes),
                int(probe_radius)
            )

            if int(capacity) > len(page_ids):

                raise ValueError(
                    f"Buffer capacity ({int(capacity)}) cannot exceed "
                    f"the number of logical pages ({len(page_ids)})."
                )

            # Generate one common request trace
            trace = make_cora_workload(
                workload_name,
                int(requests),
                int(seed),
                page_ids,
                page_graph,
                lsh
            )

            rows = []

            # Run all three policies on the same trace
            for policy_name in (
                "FIFO",
                "LRU",
                "Proposed"
            ):

                policy = make_cora_policy(
                    policy_name,
                    page_graph,
                    lsh,
                    calculator
                )

                metrics = run_cora_trace(
                    trace,
                    storage,
                    int(capacity),
                    policy
                )

                rows.append({
                    "Dataset": "Cora",
                    "Workload": workload_name,
                    "Seed": int(seed),
                    "Policy": policy_name,
                    "Page Size": int(page_size),
                    **metrics
                })

            frame = pd.DataFrame(rows)

            st.session_state["cora_live_results"] = frame

            st.session_state["cora_live_summary"] = {
                "Nodes": graph.num_nodes,
                "Edges": len(graph.edges),
                "Pages": len(page_ids),
                "Feature dimensions": int(
                    graph.features.shape[1]
                )
            }

        except Exception as exc:

            st.error(
                f"Cora simulation failed: {exc}"
            )

            st.exception(exc)

    # -----------------------------------------------------
    # Display Cora live results
    # -----------------------------------------------------

    if "cora_live_results" in st.session_state:

        st.subheader("Cora live simulation results")

        results = st.session_state["cora_live_results"]
        summary = st.session_state["cora_live_summary"]

        a, b, c, d = st.columns(4)

        a.metric(
            "Cora nodes",
            f"{summary['Nodes']:,}"
        )

        b.metric(
            "Undirected edges",
            f"{summary['Edges']:,}"
        )

        c.metric(
            "Logical pages",
            f"{summary['Pages']:,}"
        )

        d.metric(
            "Feature dimensions",
            summary["Feature dimensions"]
        )

        st.caption(
            f"Workload: {results.iloc[0]['Workload']} | "
            f"Seed: {results.iloc[0]['Seed']} | "
            f"Requests: {results.iloc[0]['Requests']} | "
            f"Buffer capacity: {results.iloc[0]['Capacity']} | "
            f"Page size: {results.iloc[0]['Page Size']}"
        )

        # -------------------------------------------------
        # Live comparison table
        # -------------------------------------------------

        st.subheader("Live policy comparison")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # Live performance charts
        # -------------------------------------------------

        st.subheader("Live performance comparison")

        chart1, chart2 = st.columns(2)

        with chart1:

            st.markdown("**Hit ratio (%)**")

            st.bar_chart(
                results.set_index("Policy")[["Hit Ratio (%)"]]
            )

        with chart2:

            st.markdown("**Page faults**")

            st.bar_chart(
                results.set_index("Policy")[["Page Faults"]]
            )

        chart3, chart4 = st.columns(2)

        with chart3:

            st.markdown("**Evictions**")

            st.bar_chart(
                results.set_index("Policy")[["Evictions"]]
            )

        with chart4:

            st.markdown("**Policy decision time (ms)**")

            st.bar_chart(
                results.set_index("Policy")[
                    ["Policy Decision Time (ms)"]
                ]
            )

        # -------------------------------------------------
        # Download live results
        # -------------------------------------------------

        st.download_button(
            "Download Cora live results CSV",
            results.to_csv(index=False),
            "cora_live_results.csv",
            "text/csv"
        )

    # -----------------------------------------------------
    # Historical Cora experiments
    # -----------------------------------------------------

    st.divider()

    with st.expander(
        "View completed Cora experiments and historical analysis"
    ):

        show_cora_historical_results()


# =========================================================
# DATASET SELECTION
# =========================================================

with st.sidebar:

    st.header("Dataset selection")

    selected_dataset = st.radio(
        "Choose dataset",
        ["THGB DBLP-TA", "Cora"],
        key="selected_dataset"
    )

    st.divider()


# =========================================================
# DISPLAY SELECTED DASHBOARD
# =========================================================

if selected_dataset == "THGB DBLP-TA":

    show_thgb_dashboard()

else:

    show_cora_dashboard()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Research simulator — results represent simulated logical-page "
    "replacement experiments, not production DBMS benchmarks."
)
