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

## Real data: DrugComb

Run on the full DrugComb v1.5 summary table (`summary_v_1_5.csv`, CC-BY-4.0) via
`heterosplit.datasets.drugcomb.load_drugcomb_csv`. After dropping mono-therapy rows
(`drug_col = NULL`) and rows missing a cell line or synergy score, the corpus is:

- **739,652** drug-combination records, **4,268** drugs, **288** cell lines
- label balance (synergy_loewe > 0): 224,847 synergistic / 514,805 antagonistic

Splitting the full corpus with `undirected_pairs=True`, `ratios=(0.8, 0.1, 0.1)`, seed 42
(one core, Python 3.12):

| regime | time (s) | peak (MB) | train | val | test | excluded | test % | leakage |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| random | 0.76 | 67 | 591,722 | 73,965 | 73,965 | 0 | 10.0% | clean |
| pair_cold_start | 0.67 | 67 | 591,722 | 73,965 | 73,965 | 0 | 10.0% | clean |
| either_cold_start | 0.03 | 18 | 506,619 | 99,784 | 133,249 | 0 | 18.0% | clean |
| both_cold_start | 0.03 | 24 | 506,619 | 16,629 | 14,681 | 201,723 | 2.7% | clean |
| joint_cold_start | 0.03 | 36 | 393,424 | 6,444 | 10,871 | 328,913 | 2.6% | clean |

Every split is audited leakage-free. The entity-partition regimes make the quadratic
effect concrete: `both`/`joint` exclude 200k–330k "bridge" records and shrink the test set
to ~2.7%, which is *reported*, not hidden.

### The split regime changes the answer

Training the small GraphSAGE link predictor from `examples/train_link_prediction.py` on a
200k-row DrugComb subset (98,671 records, 3,995 drugs), scoring held-out drug pairs against
negatives drawn from the same node pool:

| regime | test edges | test AUC | test AP |
|---|---:|---:|---:|
| random (transductive) | 9,867 | **0.839** | 0.894 |
| either_cold_start | 17,960 | **0.267** | 0.375 |
| both_cold_start | 1,774 | 0.472 | 0.500 |

A model that looks strong under a random split (AUC 0.84) is **worse than chance on genuinely
unseen drugs** (either-cold-start AUC 0.27) — the overestimation HeteroSplit exists to
prevent. Numbers vary with seed/subset; reproduce with:

```bash
HETEROSPLIT_DRUGCOMB_CSV=data/summary_v_1_5.csv uv run --extra pyg python examples/train_link_prediction.py
```
