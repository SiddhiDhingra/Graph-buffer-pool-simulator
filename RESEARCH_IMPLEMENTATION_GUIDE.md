# Research Implementation Guide — 12-Step Upgrade

This guide is an implementation extension of the submitted proposal. It keeps Cora, the Python buffer-pool simulator, LRU baseline, graph locality, LSH semantic similarity, access history, and the proposed keep/evict decision.

## 1. Install the exact environment

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Install the real Cora dataset

```bash
python scripts/prepare_cora.py
python scripts/validate_cora.py
python check_cora.py
```

Expected standard Cora: 2708 nodes, 1433 features, 7 classes, 5429 citation rows.

## 3. Verify page construction

Start with the submitted implementation's 50-node logical page size. Do not change it until the baseline run passes.

```bash
python check_cora.py
```

The check must report the expected node/feature/edge counts and a nonzero page count.

## 4. Verify the LSH path

The semantic signal must follow:

`page vector -> random hyperplane LSH -> candidate pages -> exact cosine only among candidates`.

The exact cosine operation must not be used to discover the candidates.

## 5. Verify graph locality

Graph relevance is computed at the page level from cross-page citation edges. The primary research signal is directional transition probability from the previous/requested context page.

## 6. Verify access history

Recency uses exponential decay and frequency uses smooth saturation. There is no hard `frequency >= 10 => 1.0` cutoff.

## 7. Verify composite usefulness

The proposed policy combines:

- graph locality;
- LSH semantic locality;
- recency;
- frequency.

A locality-confidence gate increases the contribution of graph/semantic signals only when the observed stream provides evidence of locality.

## 8. Verify identical traces

Generate one page-request trace per `(seed, workload)` and pass that exact list to FIFO, LRU, Graph-only, LSH-only, Graph+LSH, and Proposed.

Never generate a new random trace for each policy.

## 9. Run the complete baseline/ablation suite

```bash
pytest -q
python run_research.py
```

Policies:

- FIFO
- LRU
- Graph-only
- LSH-only
- Graph+LSH
- Proposed

Workloads:

- Random
- Graph Traversal
- Local Graph
- Semantic
- Mixed
- Graph + Semantic

## 10. Inspect the right metrics

Primary:

- hit ratio;
- miss rate;
- page faults;
- storage I/O;
- evictions;
- simulated I/O latency.

Additional systems metric:

- average replacement-decision time.

Never merge replacement CPU time into simulated disk latency.

## 11. Run robustness experiments

After the primary 50-node-page experiment passes, repeat the same protocol for page sizes such as 25 and 100 and for several buffer capacities. Keep the exact same seeds across policies.

Also test LSH parameters (`tables`, `planes`, and probe radius) without changing the dataset.

## 12. Statistical analysis and paper claims

Use at least 10 deterministic seeds for the final paper if runtime permits. Report mean ± standard deviation and paired comparisons against LRU.

Do not claim universal dominance. The correct research question is whether graph/semantic locality provides statistically and practically meaningful gains on workloads that contain those locality patterns while remaining competitive when temporal locality dominates.
