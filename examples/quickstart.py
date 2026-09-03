"""HeteroSplit quickstart: a leakage-safe cold-start split for drug--drug--cell-line synergy.

Run::

    uv run python examples/quickstart.py

This mirrors the DrugComb setting: each record is a (drug, drug, cell-line) synergy
observation. We build a *joint* cold-start split — a test triple must involve at least
one unseen drug **and** an unseen cell line — then prove the split is leakage-free and
write a reproducible manifest.

Real DrugComb data slots in unchanged: build ``PredictionRecords`` (or a ``HeteroData``)
with drug/drug/cell-line columns instead of the synthetic generator below; everything
after ``split_records`` is identical.
"""

from __future__ import annotations

from pathlib import Path

from heterosplit import make_synthetic_dataset, split_records
from heterosplit.report import to_markdown

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    # A DrugComb-shaped dataset: drugs (self-relation) observed under cell-line contexts,
    # labelled synergy vs antagonism.
    dataset = make_synthetic_dataset(
        n_records=5000,
        n_source_entities=120,  # drugs
        n_context_entities=30,  # cell lines
        n_labels=2,  # synergy / antagonism
        self_relation=True,
        source_type="drug",
        context_type="cell_line",
        relation="synergy",
        seed=42,
    )

    spec = dataset.spec(
        "joint_cold_start",
        holdout={"drug": "either", "cell_line": "all"},
        ratios=(0.8, 0.1, 0.1),
        stratify_by="label",
        seed=42,
    )

    result = split_records(dataset.records, spec)

    # Fail loudly if any contract is violated; here it passes.
    result.audit.raise_for_leakage()
    print(to_markdown(result))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = result.build_manifest()
    manifest.save(OUTPUT_DIR / "quickstart-manifest.json")
    print(f"\nWrote manifest to {OUTPUT_DIR / 'quickstart-manifest.json'}")
    print(f"Manifest digest (reproducibility key): {manifest.digest()[:16]}...")

    # The message-passing graph you would train on, with held-out edges + reverses removed.
    mp = result.message_passing_edge_index()
    print(f"Training message-passing edges (leakage-safe): {mp.shape[1]}")


if __name__ == "__main__":
    main()
