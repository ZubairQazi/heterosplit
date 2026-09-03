# HeteroSplit architecture

HeteroSplit constructs, validates, and reports leakage-safe cold-start / inductive
splits for heterogeneous link-prediction datasets. It focuses narrowly on **split
semantics** and **leakage auditing** and is designed to complement PyTorch Geometric,
not replace its loaders, samplers, or training stack.

## The central abstraction: prediction records

The *first design decision* (writeup §21) is to treat a heterogeneous observation — e.g.
a drug–drug–cell-line synergy measurement — as a **prediction record** with a source
entity `s`, destination entity `d`, optional context entities `c`, an optional
relation/label `r`, and an optional timestamp `t`, rather than a first-class hyperedge.
This maps cleanly onto ordinary link prediction plus contextual columns and integrates
naturally with PyG.

The key normalization (`records.py`) is a **per-entity-type codebook**: every column
referring to the same entity type is factorized against one shared codebook, so an
entity appearing as a source in one record and a destination in another gets the *same*
integer code. Every disjointness property then reduces to a statement about integer
code sets — and, crucially, a *bipartite* relation keeps its source and destination in
**independent** codebooks, so `user` code `k` and `item` code `k` are never conflated.

## Module map

```
heterosplit/
  schema.py        EntityRole / RoleKind / RelationMeta / TaskSchema
  canonical.py     order-independent pair canonicalization (symmetry policy)
  records.py       Codebook + PredictionRecords (normalized internal table)
  spec.py          SplitSpec: regime + ratios + seed + options, validated + serializable
  synthetic.py     deterministic synthetic dataset generator
  splitters/       one module per regime family + the shared assignment core
    assignment.py    seeded LPT greedy + size-preserving local-search refinement
    base.py          Splitter ABC, split_by_groups, shared helpers
    random.py, pair.py, entity_disjoint.py, context_disjoint.py, joint.py
  objective.py     size deviation + distribution divergence (balance metrics)
  result.py        SplitResult: index-centric assignment + derived views
  message_passing.py  leakage-safe supervision / message-passing edge reconstruction
  audit/           the leakage auditor (contract -> findings)
  manifest.py      deterministic, reloadable split manifest
  report/          JSON + Markdown distribution/audit report
  adapters/        tabular + PyG HeteroData (optional [pyg] extra)
  datasets/        narrow real-dataset adapters (DrugComb)
  cli.py           `heterosplit demo` / `heterosplit split`
```

## Split taxonomy and algorithm

Regimes fall into two families:

- **Record-partition** (`random`, `pair`, `source`, `destination`, `context`): a grouping
  key is derived per record (the record itself, the canonical pair, or an entity code);
  whole groups are assigned to splits. Disjointness is automatic because each group lands
  in exactly one split.
- **Entity-partition** (`either`, `both`, `joint`): endpoint (and context) entities are
  *labeled* train/val/test (degree-weighted), and each record's split is derived from its
  entities' labels. With split indices ordered `train < val < test`, `either` is the
  element-wise `max` of the endpoint labels (no exclusions) and `both` is the shared label
  else excluded (bridge records are dropped and *reported*). `joint` intersects several
  such axes: a record joins split `t` only if every configured axis agrees on tier `t`.

Assignment (`assignment.py`) is a seeded **longest-processing-time greedy** (largest group
first, random tie-break) that fills the split with the largest size deficit, optionally
followed by a **size-preserving local search** that improves label balance via
lexicographic acceptance (size deviation never worsens). Entity-partition ratios are
inherently approximate — achieved ratios and exclusions are surfaced, never silently
relaxed.

### Symmetry policy

For undirected/unordered relations, pairs are canonicalized to `(min, max)` so `(A, B)`
and `(B, A)` can never land in different splits. Undirected pairs are only valid for a
self-relation (same source/destination type).

## Leakage auditor

The auditor (`audit/`) turns each regime's contract into machine-checkable findings.
`contract_for` maps a regime (and joint holdout) to the disjointness properties that must
hold; the auditors verify them **across every pair of splits** (train/val/test), so
val-vs-test leakage is caught, not just train-vs-rest. Checks include entity overlap by
role, pair overlap (with reversed-pair awareness for self-relations), context overlap,
either-endpoint-unseen, duplicate observations, and a message-passing check that verifies
the reconstructed training graph contains no held-out edge or reverse. Optional auditors
cover negative-sample collisions and feature provenance. Every finding carries a count,
severity, and a few concrete offending ids; `result.audit.raise_for_leakage()` fails on
any error-severity violation.

## Reproducibility

`manifest.py` emits a deterministic, JSON-serializable manifest: library/schema versions,
a collision-resistant input fingerprint, the normalized spec, per-split counts, and hashes
of the split indices. Runtime/peak-memory *measurements* live in a separate section
excluded from `digest()`, reconciling "a fixed input, spec, and seed produce the same
manifest" with recording measurements. Manifests reload without re-running the split.

## PyG integration

The correctness core never imports torch. `adapters/pyg.py` (optional `[pyg]` extra)
converts a `HeteroData` supervision edge to `PredictionRecords`, runs a split, and
reconstructs per-split `HeteroData` in PyG's link-prediction convention (`edge_index` =
leakage-safe training message-passing graph; `edge_label_index` / `edge_label` =
supervision).

## Non-goals (v1)

No GNN architectures, no replacement for `NeighborLoader`/`LinkNeighborLoader`, no
hyperparameter-sweep platform, no CUDA sampler (profile first), and no claim that one
split policy is correct for every scientific question.
