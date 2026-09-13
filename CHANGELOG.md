# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-12

First public release: a feature-complete v1 of the correctness core, auditor, and
integrations.

### Added

- **Split regimes (8):** random/transductive, pair, source, destination, either, both,
  context, and joint cold-start, over a prediction-record model with per-entity-type
  shared codebooks.
- **Leakage audit suite:** regime-aware contract checks — entity/pair/context overlap
  across all splits (incl. val-vs-test), reversed-pair awareness, message-passing
  reconstruction leakage, duplicate observations, and optional negative-sample and
  feature-provenance auditors; `raise_for_leakage()` fails on any violation.
- **Deterministic manifests:** collision-resistant input fingerprint, normalized spec,
  per-split counts, and index hashes; `digest()` reproducibility key with measurements
  kept separate. Reloadable without re-running.
- **Constrained assignment:** seeded longest-processing-time greedy plus a
  size-preserving local-search refinement (capped for scale).
- **Message passing:** leakage-safe training-graph reconstruction (held-out edges and,
  for self-relations, their reverses removed).
- **Adapters:** PyTorch Geometric `HeteroData` (optional `[pyg]` extra) and a
  dependency-free tabular adapter.
- **Datasets:** real DrugComb drug--drug--cell-line loader (streaming CSV + Zenodo
  downloader).
- **Reporting & CLI:** JSON/Markdown distribution + audit report; `heterosplit demo` /
  `heterosplit split`.
- **Examples & benchmarks:** synthetic generator, corrupted-leakage demo, benchmark
  harness with a group-shuffle baseline, and an end-to-end GraphSAGE link-prediction
  example showing the regime changes measured performance.
- **Docs:** architecture and benchmark methodology; property-based and adversarial-review
  test coverage (~94%).

[Unreleased]: https://github.com/ZubairQazi/heterosplit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ZubairQazi/heterosplit/releases/tag/v0.1.0
