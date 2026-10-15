# Graph-Aware Buffer Pool Eviction using LSH for High-Performance Database Systems

## 1. Project Overview

This project develops a Python-based simulator to investigate a graph-aware buffer pool eviction strategy for high-performance database systems. It combines **graph locality, Locality-Sensitive Hashing (LSH)-based page similarity, and access history** to estimate page usefulness and guide eviction decisions.

The simulator uses the Cora Citation Network to model graph-structured data, constructs logical pages, simulates limited-capacity memory, and compares the proposed strategy with conventional and component-level replacement policies.

## 2. Problem Statement

Traditional buffer pool replacement policies, such as Least Recently Used (LRU) and First-In, First-Out (FIFO), primarily rely on access recency or page arrival order. They do not explicitly consider graph relationships or similarities between page contents.

For graph-structured workloads, these additional signals may provide useful information for page replacement. However, their effectiveness depends on the workload and buffer capacity.

**Objective:** Design and evaluate a graph-aware buffer pool eviction strategy that combines graph structure, LSH-based similarity, and access behaviour to investigate potential improvements in buffer performance.

## 3. Team Members

- Shefali Yadav
- Siddhi Dhingra
- Dishika Kanwal


## 4. Dataset

The project uses the **Cora Citation Network**, in which research papers are represented as nodes and citation relationships as edges.

| Dataset Property | Value |
|---|---|
| Nodes (research papers) | 2,708 |
| Features per node | 1,433 |
| Unique undirected edges | 5,278 |
| Classes | 7 |
| Logical page size | Up to 50 nodes |
| Total logical pages | 55 |

The dataset is processed into logical pages that can be loaded into and evicted from the simulated buffer pool. Page mappings and graph relationships support page lookup, locality analysis, and similarity-based processing.

## 5. Technologies and Tools

- **Python 3.11:** Core implementation and simulation.
- **NumPy:** Numerical computation and feature-vector operations.
- **NetworkX:** Graph construction and connectivity analysis.
- **Streamlit:** Interactive dashboard.
- **Matplotlib:** Performance visualization.
- **pytest:** Unit and integration testing.
- **Git and GitHub:** Version control and team collaboration.

## 6. Major Features and Modules

### Dataset Processing and Page Construction
Loads the Cora dataset and groups graph nodes into logical pages. Node-to-page mappings support efficient lookup and page-level graph construction.

### Page Feature Vectors
Generates page-level feature representations by aggregating node features and applying TF-IDF-style weighting and normalization.

### Locality-Sensitive Hashing (LSH)
Uses random-hyperplane hashing to organize page vectors into candidate buckets. This enables approximate similarity-based candidate retrieval without exhaustively comparing every page pair.

### Graph Locality
Builds a page-level graph from citation relationships and derives connectivity information to characterize relationships between logical pages.

### Access-History Tracking
Tracks page recency and frequency to represent recent and repeated access behaviour.

### Page Usefulness Estimation
Combines relevant graph, similarity, and access-history signals to estimate page usefulness and support eviction decisions.

### Buffer Pool Simulation
Models a limited-capacity memory buffer and simulated storage. It processes page requests, records hits and misses, loads pages, and evicts pages when the buffer is full.

### Replacement Policies
The evaluation framework includes:
- FIFO
- LRU
- Proposed graph-aware policy

### Workload Generation and Evaluation
Compares replacement policies across Random, Graph, Vector, and Mixed workloads using controlled request traces and repeated experiment seeds.

### Streamlit Dashboard
Provides a visual interface for presenting and exploring the simulator and its evaluation results.

## 7. System Workflow

1. Load and validate the Cora dataset.
2. Construct logical pages and node-to-page mappings.
3. Build the page-level graph and calculate locality information.
4. Generate page feature vectors and create the LSH index.
5. Track access recency and frequency during workload execution.
6. Process requests through the buffer pool simulator.
7. Estimate page usefulness and select a victim when eviction is required.
8. Record performance metrics and compare replacement policies.
9. Export results and generate visualizations for analysis.

## 8. Experimental Evaluation

The simulator evaluates the replacement strategies under different workloads and buffer capacities.

| Parameter | Configuration |
|---|---|
| Dataset | Cora Citation Network |
| Workloads | Random, Graph, Vector, Mixed |
| Buffer capacities | 3, 5, 10 pages |
| Requests per experiment | 5,000 |
| Paired random seeds | 10 |
| Policies | FIFO, LRU, component-level variants, proposed policy |

### Evaluation Metrics

- **Page faults:** Number of requested pages absent from the buffer.
- **Buffer hit ratio:** Proportion of requests served from memory.
- **Evictions:** Number of pages removed from the buffer.
- **Storage accesses:** Simulated storage-read activity.
- **Decision overhead:** Time required to select an eviction victim.
- **Latency:** Request or query latency, where measured.

## 9. Installation and Execution

### Prerequisites
Python 3.11 or a compatible version, Git, and pip.

### Clone the repository

```bash
git clone https://github.com/SiddhiDhingra/Graph-buffer-pool-simulator.git
cd Graph-buffer-pool-simulator
```

### Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies


```bash
python -m pip install numpy networkx matplotlib pytest streamlit
```

### Prepare the dataset

Place `cora.content` and `cora.cites` in the directory expected by the dataset loader.

### Run the research evaluation

```bash
python run_research.py
```

### Launch the dashboard

```bash
streamlit run app.py
```

### Run automated tests

```bash
python -m pytest -v
```


## 10. Current Project Status

The project has progressed from initial design to implementation and experimental evaluation.

**Implemented components include:**
- Cora dataset processing and logical page construction.
- Page feature-vector generation and LSH-based similarity.
- Page-level graph construction and locality analysis.
- Access-history tracking and page-usefulness estimation.
- Buffer pool simulation and replacement policies.
- Workload-based evaluation and automated tests.
- Development of a Streamlit dashboard.


## 12. Repository

**GitHub:** [SiddhiDhingra/Graph-buffer-pool-simulator](https://github.com/SiddhiDhingra/Graph-buffer-pool-simulator)

The repository contains the simulator implementation, supporting modules, evaluation code, tests, and dashboard components as developed.

---

**Project Title:** Graph-Aware Buffer Pool Eviction using LSH for High-Performance Database Systems

**Project Status:** Implementation and experimental evaluation in progress.