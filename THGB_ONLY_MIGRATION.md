# THGB-only migration

This source package removes Cora from the application and experiment entry points.
The Streamlit app reads `data/thgb/DBLP-TA/node.dat` and `link.dat`, creates logical
pages by consecutive node IDs, builds page vectors from the embeddings, and builds
page graph signals from THGB edges. Labels and text records are retained by the
THGB loader but are not yet used in the eviction score.

## Run

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
python run_research.py --data-dir data/thgb/DBLP-TA
```

## Research limitations to report

- Logical page grouping by consecutive node IDs is a simulator assumption, not a
  measured physical DBLP-TA storage layout.
- Current eviction scorer consumes aggregate graph locality. The adapter retains
  edge-type-specific counts, but edge types are not yet separately weighted in
  the scorer.
- Random, graph-transition and LSH-neighbor traces are synthetic page-level
  workloads, not a production RAG pipeline or end-to-end retrieval evaluation.
- THGB node text and labels are parsed/stored by the loader but are not yet
  incorporated into page-vector generation; vectors come from embeddings in
  `node.dat`.
- Validate runtime and memory on the full DBLP-TA files before large runs.
