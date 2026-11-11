# THGB DBLP-TA Experiment Protocol

## Dataset
Use THGB DBLP-TA from `data/thgb/DBLP-TA`. Record dataset file sizes, node count,
edge count, page size, and all preprocessing settings. Do not substitute Cora.

## Logical storage model
Group consecutive node IDs into pages of fixed size. This is a simulator assumption,
not a claim about the dataset's physical disk layout. Include the final partial page.

## Page representations
Average the precomputed node embedding vectors within each logical page. Apply
normalization consistently across all runs.

## Policies
Compare FIFO, LRU, and Proposed on exactly the same trace for each seed/capacity.
The proposed policy combines access history, aggregate graph locality, and LSH
candidate semantic similarity.

## Workloads
Use random, graph-transition, and LSH-neighbor page traces. These are synthetic
page-level workloads derived from THGB graph/embedding data; they are not end-to-end
RAG retrieval or answer-quality tests.

## Metrics
Report hits, page faults, hit ratio, miss rate, storage reads, evictions, policy
decision time, average decision time, and total runtime. Use paired seeds and
statistical comparisons. Do not claim the proposed method wins unless results
support that conclusion across workloads/capacities.
