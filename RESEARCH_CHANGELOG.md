# Research implementation changelog

## THGB DBLP-TA primary pipeline
- Set THGB DBLP-TA as the sole supported dataset in the application and experiment entry points.
- Stream node embeddings into logical-page vectors.
- Build page graph aggregates from THGB links and retain edge-type-specific page counts in the adapter.
- Keep LRU and FIFO as replacement-policy baselines.
- Replace the prior dataset-specific test suite with THGB fixture-based tests.

## Known limitations
- Consecutive node IDs are grouped into logical pages as a simulator assumption.
- Edge-type counts are retained, but the current usefulness scorer still consumes aggregate graph weights.
- Current workloads are synthetic page traces, not an end-to-end RAG evaluation.
