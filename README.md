# HeteroSplit

**Leakage-safe entity-disjoint splits and audits for heterogeneous link-prediction datasets.**

HeteroSplit constructs, validates, and *reports* cold-start / inductive splits for
heterogeneous link-prediction tasks (drug–drug–cell-line synergy, drug–target
interaction, recommendation, knowledge graphs). It is designed to **complement**
PyTorch Geometric — it focuses narrowly on split *semantics* and leakage *auditing*,
and does not replace PyG's loaders, samplers, or training stack.

> **Status:** early development (`0.0.x`). APIs may change. This is positioned as a
> reusable implementation of entity-disjoint split policies and leakage audits for
> PyG `HeteroData` link-prediction tasks — not (yet) a research novelty claim.

## Why

Graph libraries make *random* node/edge splitting easy, but cold-start experiments
often rely on one-off scripts with ambiguous semantics. A nominally "cold" test set
can still leak information through:

- the same entity appearing in training under another edge or relation;
- reverse edges left in the message-passing graph;
- an unordered pair reversed across splits;
- context entities (e.g. cell lines) crossing a supposedly disjoint boundary;
- features / precomputed embeddings fit using held-out entities;
- negative sampling that pulls entities or pairs from the wrong regime.

HeteroSplit makes the split *contract* explicit, enforces it, and produces evidence
(a serializable **manifest** + **audit report**) that the contract holds.

## Install

The correctness core is pure Python + NumPy:

```bash
pip install heterosplit
```

PyTorch Geometric integration is an optional extra:

```bash
pip install "heterosplit[pyg]"
```

### From source (development)

This project uses [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ZubairQazi/heterosplit
cd heterosplit
uv sync                 # creates .venv and installs the dev toolchain
uv run pytest           # run the test suite
uv run --extra pyg pytest   # include the PyG adapter tests
```

## Quickstart

```python
from heterosplit import make_synthetic_dataset, split_records

# A DrugComb-shaped dataset: (drug, drug, cell-line) synergy observations.
# Swap in your own PredictionRecords / HeteroData for real data.
data = make_synthetic_dataset(
    n_records=5000, n_source_entities=120, n_context_entities=30,
    n_labels=2, source_type="drug", context_type="cell_line",
    relation="synergy", seed=42,
)

# Joint cold-start: a test triple must involve an unseen drug AND an unseen cell line.
spec = data.spec(
    "joint_cold_start",
    holdout={"drug": "either", "cell_line": "all"},
    ratios=(0.8, 0.1, 0.1),
    stratify_by="label",
    seed=42,
)

result = split_records(data.records, spec)
result.audit.raise_for_leakage()            # fails loudly on any leakage
result.manifest.save("split-manifest.json") # deterministic, reloadable

train_edges = result.message_passing_edge_index()   # leakage-safe training graph
```

Runnable versions live in [`examples/`](examples/) (`quickstart.py`,
`recommendation.py`, `corrupted_leakage.py`), or try the CLI:

```bash
uv run heterosplit demo --regime joint_cold_start
uv run heterosplit split --input data.csv --spec spec.json --out-dir out/
```

## Documentation

- [Architecture / design](docs/architecture.md)
- [Benchmarks & methodology](docs/benchmarks.md)

## Split taxonomy (v1 target)

| Regime | Test-set contract |
|---|---|
| Random / transductive | Test edges are unseen; entities may have appeared in training. |
| Pair cold-start | The `(s, d)` pair is unseen, but each entity may appear separately. |
| Source cold-start | Test source entities never appear as training sources. |
| Destination cold-start | Test destination entities never appear as training destinations. |
| Either-entity cold-start | At least one endpoint of each test edge is unseen. |
| Both-entity cold-start | Both endpoints of every test edge are unseen. |
| Context cold-start | Test context entities never occur in training. |
| Joint cold-start | A configured combination of endpoint and context disjointness. |

## Leakage audit & manifests

Every split ships with an **audit** and a reproducible **manifest**:

- `result.audit` turns the regime's contract into machine-checkable findings — entity /
  pair / context overlap (across *all* splits, including val-vs-test), reversed unordered
  pairs, message-passing leakage, duplicate observations, and optional negative-sample /
  feature-provenance checks. Each finding has a count, severity, and concrete offending
  ids; `raise_for_leakage()` fails on any violation.
- `result.manifest` records library/schema versions, a collision-resistant input
  fingerprint, the normalized spec, per-split counts, and hashes of the split indices.
  `manifest.digest()` is a stable reproducibility key; runtime/memory measurements are
  kept separate so a fixed input + spec + seed always produce the same digest.

## License

[MIT](LICENSE)
