# ACM/IEEE-style experiment protocol

This document is an implementation-level protocol and does not replace the submitted project proposal.

## Dataset

- Cora only.
- Standard `cora.content` and `cora.cites` files.
- No alternate dataset is required for the main evaluation.

## Storage model

- Cora nodes are mapped to fixed-size logical pages.
- Default primary page size: 50 nodes/page, matching the current project implementation.
- Page-size sensitivity can be reported separately if desired.

## Policies

1. FIFO baseline
2. LRU baseline
3. Graph-only ablation
4. LSH-only ablation
5. Graph+LSH ablation
6. Full Graph-LSH-aware proposed policy

## Workloads

1. Random
2. Graph Traversal
3. Local Graph
4. Semantic
5. Mixed
6. Graph + Semantic

## Fairness

For every `(seed, workload, capacity)` tuple:

- one trace is generated;
- every policy receives that exact trace;
- every policy starts with an empty buffer;
- storage/page layout is identical;
- buffer capacity is identical;
- page size is identical.

## Primary metrics

- Buffer hit ratio
- Miss rate
- Page faults
- Storage I/O accesses
- Evictions
- Simulated I/O latency

## Overhead metrics

- Total policy decision time
- Average decision time per eviction decision

Policy CPU overhead should be reported separately from simulated storage latency.

## Repetition

The default research runner uses five deterministic seeds. For the final paper, increase to at least ten independent seeds if runtime permits and report mean ± standard deviation.

## Statistical comparison

The runner generates paired comparisons between Proposed and LRU for every workload/capacity pair. Report:

- mean hit-ratio difference;
- standard deviation of paired differences;
- paired effect size (Cohen's dz);
- exact paired sign-flip p-value.

## What should be claimed

Do not claim universal dominance over LRU. Instead report where the proposed method provides statistically and practically meaningful improvement, where it is competitive, and where LRU remains preferable.
