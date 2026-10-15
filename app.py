import streamlit as st
import pandas as pd

from evaluation.experiments import (
    ExperimentConfig,
    generate_workloads,
    create_proposed_policy,
    run_policy,
)

from src.data_loader import load_cora
from src.page_builder import build_pages
from src.page_vectors import build_page_vectors
from src.replacement_policies import LRUPolicy


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Graph-Aware Buffer Pool Simulator",
    page_icon="🗄️",
    layout="wide",
)

st.title("🗄️ Graph-Aware Buffer Pool Eviction")
st.caption(
    "Graph Locality + LSH Similarity + Access History"
)

st.divider()


# --------------------------------------------------
# SIMULATION FUNCTION
# --------------------------------------------------

def run_quick_simulation(workload_name, capacity, requests, seed):

    config = ExperimentConfig(
        requests=requests,
        capacities=(capacity,),
        seeds=(seed,),
    )

    # Load the actual Cora dataset
    graph = load_cora(config.data_dir, strict=True)

    # Build logical pages and page vectors
    page_table = build_pages(
        graph,
        page_size=config.page_size,
    )

    page_vectors = build_page_vectors(
        graph,
        page_table,
    )

    storage_pages = page_table.pages

    # Generate workloads using the existing project code
    workloads = generate_workloads(
        graph,
        page_table,
        page_vectors,
        config.requests,
        seed,
    )

    # Select ONLY the requested workload
    trace = workloads[workload_name]

    # Run LRU on the selected trace
    lru_result = run_policy(
        trace,
        storage_pages,
        capacity,
        LRUPolicy(),
    )

    # Run the actual proposed Graph + LSH + Access policy
    proposed_policy = create_proposed_policy(
        graph,
        page_table,
        variant="proposed",
        lsh_tables=config.lsh_tables,
        lsh_planes=config.lsh_planes,
        lsh_probe_radius=config.lsh_probe_radius,
    )

    proposed_result = run_policy(
        trace,
        storage_pages,
        capacity,
        proposed_policy,
    )

    results = pd.DataFrame([
        {"Policy": "LRU", **lru_result},
        {"Policy": "Proposed (Graph + LSH + Access)", **proposed_result},
    ])

    dataset_info = {
        "nodes": graph.num_nodes,
        "edges": len(graph.edges),
        "pages": page_table.num_pages,
    }

    return results, dataset_info


# --------------------------------------------------
# PROJECT OVERVIEW
# --------------------------------------------------

st.subheader("Project Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Dataset", "Cora")
col2.metric("Logical Page Size", "50 nodes")
col3.metric("Proposed Policy", "Graph + LSH + Access")

st.info(
    "Run a quick experiment using one selected workload "
    "and one buffer capacity. Results come from the actual simulator."
)


# --------------------------------------------------
# SIMULATION CONTROLS
# --------------------------------------------------

st.sidebar.header("Simulation Controls")

workload = st.sidebar.selectbox(
    "Select Workload",
    ["Random", "Graph", "Vector", "Mixed"],
)

capacity = st.sidebar.selectbox(
    "Buffer Capacity (pages)",
    [3, 5, 10],
    index=1,
)

requests = st.sidebar.selectbox(
    "Number of Page Requests",
    [100, 250, 500,5000],
    index=0,
)

seed = 42

st.sidebar.caption(
    "Quick demo: one workload, one capacity, one seed. "
    "The full research experiment remains separate."
)

st.subheader("Run Simulation")

st.write(f"**Selected workload:** {workload}")
st.write(f"**Buffer capacity:** {capacity} pages")
st.write(f"**Page requests:** {requests}")
st.write("**Policies:** LRU vs Proposed")

if st.button("▶ Run Selected Simulation", type="primary"):

    try:
        with st.spinner(
            "Running the selected experiment on Cora..."
        ):
            results, dataset_info = run_quick_simulation(
                workload,
                capacity,
                requests,
                seed,
            )

        st.session_state["quick_results"] = results
        st.session_state["dataset_info"] = dataset_info
        st.session_state["run_settings"] = {
            "workload": workload,
            "capacity": capacity,
            "requests": requests,
        }

    except Exception as error:
        st.error(f"Simulation failed: {error}")
        st.exception(error)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "quick_results" in st.session_state:

    results = st.session_state["quick_results"]
    info = st.session_state["dataset_info"]
    settings = st.session_state["run_settings"]

    st.divider()
    st.subheader("Simulation Results")

    st.caption(
        f"Completed run: {settings['workload']} workload | "
        f"Capacity: {settings['capacity']} pages | "
        f"Requests: {settings['requests']}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Cora Nodes", info["nodes"])
    col2.metric("Cora Edges", info["edges"])
    col3.metric("Logical Pages", info["pages"])

    st.markdown("### Policy Comparison")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
    )

    # Show the actual metrics when available
    proposed = results[
        results["Policy"] == "Proposed (Graph + LSH + Access)"
    ].iloc[0]

    lru = results[
        results["Policy"] == "LRU"
    ].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**LRU**")
        st.metric(
            "Page Faults",
            int(lru["Page Faults"]),
        )
        st.metric(
            "Hit Ratio",
            f"{float(lru['Hit Ratio']):.2f}%",
        )
        st.metric(
            "Evictions",
            int(lru["Evictions"]),
        )

    with col2:
        st.markdown("**Proposed Policy**")
        st.metric(
            "Page Faults",
            int(proposed["Page Faults"]),
        )
        st.metric(
            "Hit Ratio",
            f"{float(proposed['Hit Ratio']):.2f}%",
        )
        st.metric(
            "Evictions",
            int(proposed["Evictions"]),
        )

    st.markdown("### Page Fault Comparison")

    chart_data = results.set_index("Policy")[["Page Faults"]]
    st.bar_chart(chart_data)

    st.caption(
        "These results represent one run with seed 42. "
        "They are not multi-seed research averages. "
        "The proposed policy is not guaranteed to outperform LRU."
    )


# --------------------------------------------------
# PHASE 3 EXTENSIONS
# --------------------------------------------------

st.divider()
st.subheader("Research Components")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Implemented components**")
    st.write("- Cora citation graph")
    st.write("- Logical page construction")
    st.write("- LSH-based page similarity")
    st.write("- Graph locality")
    st.write("- Access-history-aware eviction")

