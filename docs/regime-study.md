# Does the split regime change model rankings?

A comparative study across two real datasets and two models, quantifying how much the
**split regime** — not just the data or the model — determines measured link-prediction
performance. Reproduce with:

```bash
uv run --extra pyg python experiments/regime_study.py \
    --seeds 5 --epochs 40 --drugcomb-max-rows 150000 \
    --out experiments/results/regime_study.md
```

## Setup

- **Datasets.**
  - *DrugComb* (self-relation, `drug–drug` synergy): the first 150k summary rows →
    72,929 combination records, 3,995 drugs, 28 cell lines.
  - *MovieLens* `ml-latest-small` (bipartite, `user–movie`): 100,836 ratings.
- **Models.** `MF` = learnable node embeddings + dot product (no message passing);
  `SAGE` = the same embeddings through a 2-layer GraphSAGE encoder over the training
  message-passing graph + dot product. MF is a deliberately graph-free floor, so the
  `SAGE − MF` gap isolates *how much the graph structure helps*.
- **Protocol.** 80/10/10 splits via HeteroSplit; train on training supervision edges +
  sampled negatives; score held-out edges against **type-correct, same-node-pool**
  negatives (bipartite negatives are user–item; cold-start negatives are drawn from the
  held-out node pool, so the metric is not inflated by node-set artifacts). 5 seeds,
  40 epochs, embedding dim 32; mean ± std reported.

## Results (test ROC-AUC)

| dataset | regime | MF AUC | SAGE AUC | SAGE − MF |
|---|---|---:|---:|---:|
| DrugComb | random | 0.534 ± 0.008 | 0.787 ± 0.011 | **+0.254** |
| DrugComb | pair_cold_start | 0.534 ± 0.008 | 0.787 ± 0.011 | +0.254 |
| DrugComb | either_cold_start | 0.491 ± 0.020 | 0.247 ± 0.022 | **−0.244** |
| DrugComb | both_cold_start | 0.507 ± 0.024 | 0.481 ± 0.026 | −0.027 |
| MovieLens | random | 0.509 ± 0.004 | 0.830 ± 0.011 | **+0.321** |
| MovieLens | source_cold_start (cold user) | 0.505 ± 0.004 | 0.576 ± 0.023 | +0.072 |
| MovieLens | destination_cold_start (cold movie) | 0.501 ± 0.004 | 0.469 ± 0.088 | −0.032 |
| MovieLens | both_cold_start | 0.491 ± 0.028 | 0.495 ± 0.024 | +0.003 |

## Findings

1. **The split regime reorders model rankings.** On *both* datasets the graph model wins
   decisively under a random split (`SAGE − MF` = +0.25 / +0.32) but its advantage
   **collapses to ≤ 0** under the coldest regimes (DrugComb `either` −0.24, `both` −0.03;
   MovieLens cold-movie −0.03, `both` ≈ 0). A practitioner selecting a model on a random
   split would choose GraphSAGE — which is *no better than, or worse than*, a graph-free
   baseline exactly where it would be deployed (on unseen entities). On DrugComb `either`
   the flip is dramatic: GraphSAGE goes from 0.79 to **0.25 (below chance)** because an
   unseen drug has no neighbours, so message passing aggregates only noise.

2. **A monotone difficulty gradient.** Absolute SAGE performance falls as more of an
   edge's endpoints must be unseen: random/pair (0.79–0.83) → single-endpoint cold
   (0.25–0.58) → both-endpoint cold (≈ 0.48–0.50, i.e. chance). Random splitting
   systematically overstates cold-start performance.

3. **Source vs destination asymmetry (MovieLens).** Cold *users* retain some graph lift
   (+0.07) — a new user still links to *seen* movies, so neighbours exist — while cold
   *movies* reverse (−0.03), because a brand-new movie is isolated. HeteroSplit's separate
   `source`/`destination` regimes surface this directly; a single "inductive" split would
   hide it.

4. **`random` ≡ `pair_cold_start` for undirected drug pairs.** Both group by the canonical
   pair, so they produce identical splits and identical metrics here — an expected property
   of the symmetry policy, not a bug.

## Caveats

- **MF is a floor, not a strong baseline.** With no node features and sparse graphs, pure
  embeddings hover near chance; the informative quantity is the *contrast* with SAGE and
  how it changes across regimes, not MF's absolute value.
- **Both-cold-start rows are small and noisy.** Test sets shrink quadratically in the
  held-out fraction (few hundred to ~1–2k edges), so those rows have higher variance (see
  std, e.g. MovieLens cold-movie ±0.088) and should be read as directional.
- **DrugComb subset.** The first 150k summary rows are used for tractable runtime; the
  effect is structural and also holds on full MovieLens.
- Absolute numbers depend on model size, epochs, and seed; the robust, reproducible signal
  is the **sign and direction of `SAGE − MF` across regimes**, consistent on two datasets.

## Takeaway

Evaluate under the split regime that matches deployment. A random/transductive split can
**overstate cold-start performance by 0.2–0.5 AUC and invert model rankings** — which is
precisely the failure HeteroSplit's regimes and auditor are designed to prevent.
