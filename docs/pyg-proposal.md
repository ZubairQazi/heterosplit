# Upstream PyG proposal (draft)

This is a **draft** feature request to post at
<https://github.com/pyg-team/pytorch_geometric/issues/new> (choose the *feature request*
template, or open under *Discussions → Ideas* if maintainers prefer). It is kept in the
repo so it can be reviewed before posting. Nothing is posted automatically.

---

**Title:** Add an entity-/group-disjoint (cold-start / inductive) link splitter

### Motivation

`RandomLinkSplit` performs a purely *random, edge-level* split. Many link-prediction
studies instead need **inductive / cold-start** splits, where test edges must involve
**unseen entities** — new users, new items, new drugs — rather than merely unseen edges
between seen nodes. Today users hand-roll this per project (e.g. discussions
[#8238](https://github.com/pyg-team/pytorch_geometric/discussions/8238),
[#4453](https://github.com/pyg-team/pytorch_geometric/discussions/4453), and community
`inductive_train_test_split()` helpers), usually without disjointness guarantees, reverse-/
symmetric-pair handling, or any check that the split is actually leakage-free.

A first-class, framework-level splitter for these regimes would remove a recurring source
of subtle evaluation leakage.

### Proposal

A transform that mirrors `RandomLinkSplit`'s output (so it is drop-in for existing
training loops) but assigns edges to splits by **holding out entities** rather than
edges. Sketch:

```python
from torch_geometric.transforms import EntityLinkSplit  # name TBD

transform = EntityLinkSplit(
    num_val=0.1,
    num_test=0.1,
    regime="both",  # source | destination | either | both | pair
    edge_types=("user", "buys", "item"),
    rev_edge_types=(...),
    is_undirected=False,
    add_negative_train_samples=True,
)
train, val, test = transform(data)  # Data or HeteroData, like RandomLinkSplit
```

Regimes (test-set contract):

- **source / destination** — test source (or destination) entities never appear as such
  in training.
- **either** — at least one endpoint of every test edge is unseen in training.
- **both** — both endpoints of every test edge are unseen (bridge edges are dropped and
  their count reported).
- **pair** — the `(s, d)` pair is unseen, though each endpoint may recur separately.

The training message-passing graph excludes held-out supervision edges *and their
reverses*, and unordered relations are canonicalized so `(A, B)` / `(B, A)` cannot land in
different splits.

### Relationship to `RandomLinkSplit`

Complementary, not a replacement — same output shape (`edge_label_index` / `edge_label`,
optional negative samples, `disjoint_train_ratio`-style MP/supervision separation), so it
slots into `LinkNeighborLoader` and existing examples unchanged. Could ship as a new
transform or as a `split_by={"random"|"source"|...}` option on `RandomLinkSplit`.

### Reference implementation

I've built and published a standalone reference implementation,
[HeteroSplit](https://github.com/ZubairQazi/heterosplit) (`pip install heterosplit`), which
implements these regimes plus context/joint cold-start, a leakage auditor, and
deterministic manifests, with a PyG `HeteroData` adapter. **I'd be happy to contribute a
focused transform PR** if the maintainers are open to it — starting with the endpoint
regimes (`source`/`destination`/`either`/`both`/`pair`) to keep scope tight.

### Open questions

- Preferred surface: a new transform vs. an option on `RandomLinkSplit`?
- Naming (`EntityLinkSplit`? `InductiveLinkSplit`?).
- How much of the auditing (leakage checks) belongs upstream vs. in a companion package.

### Non-goals

Negative-sampling strategies, training code, and dataset-specific preprocessing remain out
of scope; this is purely about producing leakage-safe inductive splits.
