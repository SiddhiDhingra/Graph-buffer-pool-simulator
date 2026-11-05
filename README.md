# Graph-Aware Buffer Pool Simulator — THGB DBLP-TA

This branch uses the THGB DBLP-TA heterogeneous graph dataset as its sole primary
dataset. The simulator evaluates LRU, FIFO and the proposed graph/LSH/access-aware
page replacement policy on reproducible page-request traces.

## Dataset layout

Place the Git LFS files here:

```text
data/thgb/DBLP-TA/
  node.dat
  link.dat
  label.dat
  label.dat.test
  text/
    paper.txt
    author.txt
    conf.txt
```

The application requires `node.dat` and `link.dat`. The THGB loader can also load
node embeddings and preserve node/edge types. The page-vector builder streams the
embedding field rather than retaining the entire node embedding matrix.

## Install and run (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```

Run the experiment suite:

```powershell
python run_research.py --data-dir data/thgb/DBLP-TA --page-size 50 --requests 1000 --capacities 10 50 100 --seeds 7 42 101
```

Outputs are written to `results/thgb/`.

## Current methodology and limitations

Logical pages are currently constructed by grouping consecutive THGB node IDs.
This is an explicit simulator assumption and must not be described as the real
physical layout of DBLP-TA. Page vectors are averages of node embeddings. Page
workloads are synthetic random, graph-transition, and LSH-neighbor request traces.
The graph adapter preserves per-edge-type page counts for analysis, while the
current eviction scorer still consumes aggregate page-graph weights. This is not
an end-to-end RAG benchmark. Report these limitations in any paper.
