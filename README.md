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
git clone https://github.com/zubairqazi/heterosplit
cd heterosplit
uv sync            # creates .venv and installs the dev toolchain
uv run pytest      # run the test suite
```

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

## License

[MIT](LICENSE)
