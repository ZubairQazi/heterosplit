# Positioning & feature matrix

HeteroSplit **complements** PyTorch Geometric; it does not replace its loaders, samplers,
or training stack. This page states precisely what PyG already provides for
link-prediction splitting and what HeteroSplit adds, so any claim is engineering-scoped
(not a novelty claim) and backed by a concrete comparison.

## What PyG provides today

PyG's link splitter is [`RandomLinkSplit`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.RandomLinkSplit.html),
a transform that performs a **purely random, edge-level** split of a `Data`/`HeteroData`
object into train/val/test. Its parameters are `num_val`, `num_test`, `is_undirected`,
`key`, `split_labels`, `add_negative_train_samples`, `neg_sampling_ratio`,
`disjoint_train_ratio`, `edge_types`, `rev_edge_types`. It handles reverse edges
(`is_undirected` / `rev_edge_types`), separates message-passing from supervision edges
(`disjoint_train_ratio`), and does negative sampling.

It has **no** notion of holding out *entities* so that test edges involve unseen nodes
(inductive / cold-start), no source/destination/either/both entity-disjoint policies, no
context/hyperedge disjointness, no leakage auditing, and no serializable manifest. That
gap is felt in the community — e.g. discussions
[#8238 (inductive train/test splitting)](https://github.com/pyg-team/pytorch_geometric/discussions/8238)
and [#4453](https://github.com/pyg-team/pytorch_geometric/discussions/4453) — where users
hand-roll one-off `inductive_train_test_split()` helpers with no disjointness guarantees
or auditing.

## Feature matrix

| Capability | PyG `RandomLinkSplit` | Hand-rolled scripts | **HeteroSplit** |
|---|:--:|:--:|:--:|
| Random / transductive edge split | ✅ | ✅ | ✅ |
| Reverse-edge handling | ✅ | ~ | ✅ |
| Message-passing vs supervision separation | ✅ | ~ | ✅ |
| Built-in negative sampling | ✅ | ~ | audited¹ |
| `HeteroData` support | ✅ | ~ | ✅ (adapter) |
| Pair cold-start (held-out `(s,d)` pairs) | ❌ | ~ | ✅ |
| Source / destination entity cold-start | ❌ | ~ | ✅ |
| Either-entity cold-start (≥1 endpoint unseen) | ❌ | ~ | ✅ |
| Both-entity cold-start (both endpoints unseen) | ❌ | ~ | ✅ |
| Context / hyperedge cold-start (e.g. cell line) | ❌ | ❌ | ✅ |
| Joint (endpoint × context) cold-start | ❌ | ❌ | ✅ |
| Symmetry / canonical-pair policy | partial² | ~ | ✅ |
| Leakage auditing (contract → findings) | ❌ | ❌ | ✅ |
| Deterministic, serializable manifest | ❌ | ❌ | ✅ |
| Distribution balance / stratification | ❌ | ~ | ✅ (best-effort³) |

✅ = supported · ~ = possible but ad hoc, no guarantees · ❌ = not supported

1. HeteroSplit does not sample negatives; it *audits* a provided negative set for
   regime violations (false negatives / wrong-regime pairs).
2. `RandomLinkSplit` avoids reverse-edge leakage via `is_undirected`, but does not expose
   a canonical-pair grouping contract.
3. Entity-partition regimes cannot hit record ratios exactly (test records scale
   super-linearly in the held-out entity fraction); HeteroSplit reports achieved ratios
   and exclusions rather than silently relaxing constraints.

## Ecosystem prior art

Inductive / cold-start splitting is common in research but tends to live in
dataset-specific code rather than a reusable, framework-level tool: OGB ships
fixed inductive splits per dataset; recommender and drug-discovery pipelines hand-roll
cold-user / cold-item / cold-drug splits; scikit-learn offers `GroupShuffleSplit` (group
disjointness, but not the two-endpoint / context semantics of link prediction). HeteroSplit
generalizes these into one policy vocabulary with auditing and manifests.

## The upstream opportunity

A generic entity-/group-disjoint (cold-start) link splitter is a natural addition to PyG's
`transforms`. See [`pyg-proposal.md`](pyg-proposal.md) for a ready-to-post feature request.
