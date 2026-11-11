# THGB migration implementation notes

## Changes in this revision

- The Streamlit entry point (`app.py`) now uses THGB DBLP-TA node embeddings and links. It no longer imports or loads Cora.
- The default `run_research.py` entry point now delegates to `evaluation.run_thgb_experiments`.
- `requirements.txt` now includes Streamlit and pandas, which the dashboard imports.
- README now documents THGB as the primary dataset, its expected files, commands, and current modeling assumptions.
- Added four fixture-based THGB pipeline tests. They verify parsing of typed nodes/edges, page-vector aggregation, typed page-edge aggregation, and the symmetric adapter used by the current policy.
- Generic simulator components (buffer pool, LSH, access tracking, usefulness scoring) remain dataset-agnostic.

## Deliberately retained legacy code

`src/data_loader.py`, `src/page_vectors.py`, and parts of `evaluation/experiments.py` still contain the previous Cora-oriented path because existing tests and generic simulator examples depend on them. They are no longer called by the primary dashboard or `run_research.py` entry point. They should be treated as legacy compatibility code, not the primary THGB workflow.

## Known limitations (not silently solved)

1. `src/thgb_graph_adapter.py` collapses edge types and symmetrizes cross-page connections for compatibility with the current `PageGraph` interface. The separate `src/thgb_page_graph.py` aggregation preserves directed edge types, but those typed channels are not yet consumed by the eviction policy.
2. Logical pages are consecutive node-ID ranges. This is a simulator assumption, not evidence of the source database's physical page organization.
3. Semantic traces use LSH candidate transitions. This is synthetic semantic locality, not end-to-end RAG.
4. The full DBLP-TA files were not included in the uploaded code ZIP, so the full-data experiment and dashboard were not executed in this environment. The THGB pipeline tests use small synthetic fixtures.
5. The current THGB loader's optional `load_embeddings=True` path retains a full embedding matrix and should not be used casually on memory-constrained machines. The page-vector builder streams node embeddings and is the preferred path for experiments.

## Validation

- `python -m pytest -q`: **54 passed**.
- `python -m py_compile app.py run_research.py evaluation/run_thgb_experiments.py`: passed.
- `python -m evaluation.run_thgb_experiments --help`: passed.

This is a THGB-primary integration revision, not a claim that typed edge-aware eviction, physical page layout modeling, or end-to-end RAG have been completed.
