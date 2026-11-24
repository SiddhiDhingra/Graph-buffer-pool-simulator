# Graph-Aware Buffer Pool Eviction using LSH

A Python-based graph-aware buffer pool simulator that explores page replacement using **graph locality, semantic similarity, and access history**. The project uses the THGB DBLP-TA dataset and Locality-Sensitive Hashing (LSH) to support semantic candidate retrieval and evaluate different page replacement strategies.

## 1. Problem Statement & Objective

Traditional page replacement algorithms such as FIFO (First-In, First-Out) and LRU (Least Recently Used) primarily consider page insertion order or recent access history. However, graph-oriented and embedding-based database workloads may also contain structural relationships and semantic similarities between pages.

This project aims to develop a graph-aware buffer pool management simulator that considers these additional signals while making page eviction decisions.

**Main objectives:**
- Integrate and process the THGB DBLP-TA dataset.
- Construct logical pages and page-level graph relationships.
- Implement LSH-based semantic similarity retrieval.
- Simulate buffer pool management and page replacement.
- Compare FIFO, LRU, and the Proposed graph-aware policy across different workloads.
- Evaluate performance using a Streamlit dashboard.

## 2. Team Members

**Team Name:** Graph Pioneers  
**Team ID:** T083  
**Course:** OSDBMS-V | Semester V  
**Project Mentor:** Dr. Arun Chauhan

| Team Member | Primary Contribution |
|---|---|
| Shefali Yadav | THGB dataset integration, logical page modelling and graph construction |
| Siddhi Dhingra | Random Hyperplane LSH, semantic candidate retrieval, buffer pool simulator and FIFO/LRU policies |
| Dishika Kanwal | Access history tracking, usefulness scoring and graph-aware eviction |

All team members contributed to system integration, experiment development and dashboard implementation.

## 3. Technologies & Tools Used

- **Programming Language:** Python
- **Dashboard:** Streamlit
- **Numerical Processing:** NumPy
- **Data Analysis:** Pandas
- **Graph Processing:** NetworkX
- **Visualization:** Matplotlib
- **Testing:** pytest
- **Version Control:** Git and GitHub
- **Dataset:** THGB DBLP-TA

## 4. Project Setup & Installation

### Prerequisites

- Python 3.11 recommended
- Git
- pip

### Step 1: Clone the repository

```bash
git clone https://github.com/SiddhiDhingra/Graph-buffer-pool-simulator.git
```

### Step 2: Navigate to the project directory

```bash
cd Graph-buffer-pool-simulator
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Streamlit dashboard

```bash
python -m streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

### Running experiments from the command line

To execute the THGB experiment pipeline:

```bash
python -m evaluation.run_thgb_experiments
```

To run automated tests:

```bash
python -m pytest -v
```

**Note:** Ensure that the THGB DBLP-TA dataset files are available in the expected `data/thgb/DBLP-TA/` directory before running the experiments.

## 5. Major Features & Modules

### A. THGB Dataset Processing
- Loads node and link records from the THGB DBLP-TA dataset.
- Processes node embeddings and graph relationships.
- Constructs logical pages using a fixed page size of 50 nodes.

### B. Page-Level Graph Construction
- Represents relationships between logical pages.
- Provides structural information for graph locality calculations.

### C. LSH-Based Semantic Similarity
- Implements Random Hyperplane Hashing.
- Generates binary hash signatures from vector representations.
- Retrieves candidate pages with similar semantic representations.

### D. Buffer Pool Management
- Simulates a buffer pool with limited memory capacity.
- Handles page requests, cache hits, page faults and evictions.
- Tracks storage reads during simulation.

### E. Page Replacement Policies
- **FIFO:** Evicts the page that entered the buffer pool earliest.
- **LRU:** Evicts the least recently accessed page.
- **Proposed:** Uses graph locality, semantic similarity and access history for graph-aware page replacement.

### F. Workload Generation
The system supports multiple workload patterns:

- Random
- Graph
- Semantic
- Mixed

### G. Performance Evaluation & Dashboard
The Streamlit dashboard supports experiment configuration and policy comparison.

Important performance metrics include:
- Cache hits and hit ratio
- Page faults and miss rate
- Storage reads
- Page evictions
- Policy decision time

## 6. Project Structure

```text
Graph-buffer-pool-simulator/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── thgb/
│       └── DBLP-TA/
│
├── src/
│   ├── thgb_loader.py
│   ├── page.py
│   ├── page_builder.py
│   ├── page_vectors.py
│   ├── graph_locality.py
│   ├── lsh.py
│   ├── access_score.py
│   ├── usefulness_score.py
│   ├── proposed_policy.py
│   ├── replacement_policies.py
│   └── buffer_pool.py
│
├── evaluation/
│   └── run_thgb_experiments.py
│
├── results/
│   └── thgb/
│       ├── thgb_results.csv
│       └── thgb_metadata.json
│
└── tests/
```

## 7. Current Project Status & Progress

<details>
<summary>Completed project components</summary>

- THGB DBLP-TA dataset integration and preprocessing.
- Logical page construction and page-level graph generation.
- LSH-based semantic candidate retrieval.
- Buffer pool simulation and page replacement policies.
- Access history tracking and graph-aware usefulness scoring.
- Random, Graph, Semantic and Mixed workload generation.
- FIFO, LRU and Proposed policy evaluation.
- Streamlit dashboard for experiment configuration and comparison.
- Performance result generation and automated testing.

</details>


**Future Scope:**
- Evaluate additional workload and buffer-capacity combinations.
- Further improve replacement-policy performance.
- Extend performance analysis and result visualization.
- Explore validation with additional graph and embedding-based datasets.

---

**Project Type:** Academic Project-Based Learning (PBL)  
**Department:** Computer Science & Engineering  
**Institution:** Graphic Era (Deemed to be University), Dehradun  
**Academic Session:** 2026–27