# Research-grade implementation changes

## 1. Critical issue in the original implementation

The original project instantiated `LSHManager`, but `UsefulnessCalculator` called `get_page_similarity()` directly. That method computes exact cosine similarity and does not use the LSH candidate index. Therefore, LSH was present in the architecture but was not actually participating in eviction decisions.

### Fix

The revised policy uses:

```text
Page vectors
   -> random-hyperplane LSH
   -> approximate candidate retrieval
   -> exact cosine only for retrieved candidates
   -> semantic usefulness signal
   -> composite eviction score
```

A one-bit multi-probe option improves candidate recall without turning the candidate generator into an exhaustive search.

## 2. Semantic-score calibration bug

The original implementation transformed cosine similarity as:

```text
(cosine + 1) / 2
```

For Cora's non-negative word-feature vectors, cosine similarity is naturally in [0,1]. The transformation incorrectly made an orthogonal pair receive 0.5 instead of 0.

### Fix

Cosine similarity is retained directly and clipped to [0,1].

## 3. Graph-locality signal

The original eviction score used page-level Jaccard overlap. For a page-transition prediction problem, a directional transition probability is more interpretable:

```text
P(next_page = q | current_page = p)
    = cross_edges(p,q) / total_cross_page_edges(p)
```

The Jaccard API remains available for compatibility, while the proposed eviction policy uses the directional mode.

## 4. Access-history signal

The original frequency feature saturated at ten accesses and recency used a reciprocal-age function. The revised tracker uses exponential decay for recency and a smooth saturation function for frequency. Historical information is retained after eviction.

## 5. Candidate-wise min-max normalization

The original implementation normalized each candidate set independently. This can make a weak signal appear strong simply because the current candidate set has a small numerical spread.

### Fix

Static graph and LSH scores are rank-normalized once per reference page. Online access history remains a direct [0,1] score.

## 6. Adaptive locality gate

The revised policy does not force graph/semantic signals to dominate on random traces. It estimates the locality confidence of the recent request stream and shifts weight toward access history when structural/semantic evidence is weak.

This is intended as a robustness mechanism, not as an attempt to make the proposed method win every workload.

## 7. Major performance optimization

The original policy repeatedly recalculated graph and semantic relationships at every eviction decision.

The revised implementation precomputes static graph ranks and LSH-derived semantic scores once. Online eviction therefore evaluates only the current buffer's access-history scores plus O(1) static lookups.

## 8. Workload coverage

The original suite contained Random, Graph Traversal, Local Graph and Mixed workloads.

The revised suite retains all of them and adds:

- Semantic locality
- Graph + Semantic locality

All policies receive exactly the same page-request trace for each experiment configuration.

## 9. Research ablations

The evaluation now supports:

- FIFO
- LRU
- Graph-only
- LSH-only
- Graph + LSH
- Full Proposed method

This makes it possible to attribute gains to individual components rather than reporting only one comparison.

## 10. Reproducibility

The evaluation records:

- seed
- workload
- buffer capacity
- page size
- policy
- hit ratio
- miss rate
- page faults
- storage I/O
- simulated latency
- policy decision time

Repeated seeds are summarized using mean and standard deviation. Paired exact sign-flip statistics are also produced for Proposed vs LRU.

## 11. Important interpretation rule

The project should **not** claim that Graph-LSH beats LRU on every workload. LRU is expected to remain strong when temporal locality dominates or when requests are genuinely random. The defensible research claim is that graph/semantic locality provides additional predictive information that can reduce faults on workloads exhibiting those locality structures, while the adaptive gate limits degradation elsewhere.
