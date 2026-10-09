# Graph-Aware Buffer Pool Eviction using LSH

Research-grade Python simulator for the submitted project:

> **Graph-Aware Buffer Pool Eviction using LSH for High-Performance Database Systems**

The implementation keeps the submitted Cora dataset and buffer-pool formulation. It extends the original simulator with:

- deterministic Cora loading and logical page modelling;
- LRU and FIFO baselines;
- page-level graph locality;
- random-hyperplane LSH with multi-probe candidate retrieval;
- online recency/frequency history;
- adaptive Graph + LSH + access usefulness scoring;
- locality-aware fallback toward access history when graph/semantic evidence is weak;
- semantic and graph+semantic workloads in addition to the original random/traversal/local/mixed workloads;
- ablation policies: Graph-only, LSH-only, Graph+LSH and Access-only;
- repeated-seed evaluation and paired statistical testing support;
- policy decision overhead measurement separately from simulated I/O latency;
- reproducible CSV, metadata and publication-ready plots.

## Dataset

Place the standard Cora files here:

```text
data/cora/cora.content
data/cora/cora.cites
```

The simulator does **not** silently substitute a synthetic graph for research runs.
Synthetic Cora remains available only for unit tests/offline development.

## Run tests

```bash
pip install -r requirements.txt
pytest -q
```

## Run the research evaluation

```bash
python run_research.py
```

Results are written to:

```text
results/raw_results.csv
results/summary_results.csv
results/experiment_metadata.json
results/plots/
```

## Important experimental rule

Every policy receives the exact same generated page-request trace for a given
(seed, workload, page size, buffer capacity) configuration. No policy gets a
separate random stream.

## Interpretation

The proposed method is **not expected to dominate LRU on every workload**.
LRU is a strong temporal-locality baseline and may win on random/recency-heavy
traces. The research question is whether graph and semantic locality allow the
proposed policy to reduce faults on graph/semantic workloads while remaining
competitive elsewhere and within a defensible decision-time overhead.
