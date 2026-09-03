# Benchmarks

Reproduce with:

```bash
uv run python benchmarks/benchmark.py --sizes 10000,100000 --out benchmarks/results/table.md
```

## Methodology

For each dataset size and regime the harness (`benchmarks/benchmark.py`) generates a
synthetic self-relation dataset (`n_entities = n_records / 20`, 3 labels) and measures:

- **time (s)** — wall-clock of `split_records` only (via `time.perf_counter`).
- **peak (MB)** — peak allocation during the split (via `tracemalloc`).
- **ratio dev** — L1 distance between achieved and requested split fractions (0 = exact).
- **label div** — ratio-weighted total-variation of the label distribution across splits
  (0 = identical).
- **reproducible** — whether re-running yields an identical manifest digest.

The `pair_cold_start` row is also run through a naive **group-shuffle** baseline (assign
whole canonical-pair groups to splits by *group* fraction, with no size balancing) to
contrast balance quality.

## Sample results

Measured on an Apple-silicon laptop (CPU only, Python 3.12). Absolute times vary by
machine; the relative story is the point.

| records | entities | method | regime | time (s) | peak (MB) | ratio dev | label div | reproducible |
|---:|---:|---|---|---:|---:|---:|---:|:--:|
| 10000 | 500 | heterosplit | random | 0.067 | 1.7 | 0.0000 | 0.0036 | yes |
| 10000 | 500 | heterosplit | pair_cold_start | 0.058 | 1.0 | 0.0000 | 0.0065 | yes |
| 10000 | 500 | heterosplit | source_cold_start | 0.052 | 0.1 | 0.0000 | 0.0092 | yes |
| 10000 | 500 | heterosplit | both_cold_start | 0.003 | 0.3 | 0.3409 | 0.0123 | yes |
| 10000 | 500 | group-shuffle | pair_cold_start | 0.002 | 1.0 | 0.0016 | 0.0058 | yes |
| 100000 | 5000 | heterosplit | random | 0.60 | 6.8 | 0.0000 | 0.0013 | yes |
| 100000 | 5000 | heterosplit | pair_cold_start | 0.61 | 10.5 | 0.0000 | 0.0012 | yes |
| 100000 | 5000 | heterosplit | source_cold_start | 0.54 | 1.1 | 0.0000 | 0.0015 | yes |
| 100000 | 5000 | heterosplit | both_cold_start | 0.03 | 3.2 | 0.3392 | 0.0025 | yes |
| 100000 | 5000 | group-shuffle | pair_cold_start | 0.03 | 10.5 | 0.0001 | 0.0017 | yes |

## Interpretation

- **Balance.** HeteroSplit hits the requested record ratios exactly (`ratio dev` = 0) for
  record-partition regimes, where the group-shuffle baseline drifts because it balances
  *group* count, not *record* count. The `both_cold_start` `ratio dev` of ~0.34 is the
  inherent distortion of entity-partition splitting (test records scale super-linearly in
  the held-out entity fraction) and is reported, not hidden.
- **Scale.** After capping the local-search refinement at 5k groups (a profiling-driven
  change), record-level `random`/`pair` splits on 100k records dropped from ~11s to ~0.6s
  (~18×) with unchanged ratio deviation and label divergence.
- **Reproducibility.** Every split re-runs to an identical manifest digest.

## Comparison to PyG `RandomLinkSplit`

`RandomLinkSplit` covers the transductive baseline only (no entity-disjoint / cold-start
regimes). HeteroSplit's `random` regime is the comparable operation; the cold-start
regimes and the leakage auditor have no direct PyG equivalent, which is the point of the
library.
