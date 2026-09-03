"""Second domain: an inductive user--item recommendation split (non-biomedical).

Run::

    uv run python examples/recommendation.py

Demonstrates that HeteroSplit is generic: the same API handles a *bipartite* relation
(user buys item) with a ``destination_cold_start`` regime — evaluating how the model
recommends to items it never saw during training (a classic cold-item setting).
"""

from __future__ import annotations

from heterosplit import make_synthetic_dataset, split_records
from heterosplit.report import to_markdown


def main() -> None:
    dataset = make_synthetic_dataset(
        n_records=8000,
        n_source_entities=500,  # users
        n_destination_entities=800,  # items
        n_labels=0,  # implicit feedback (edge existence)
        self_relation=False,
        source_type="user",
        destination_type="item",
        relation="buys",
        seed=7,
    )

    spec = dataset.spec("destination_cold_start", ratios=(0.8, 0.1, 0.1), seed=7)
    result = split_records(dataset.records, spec)

    result.audit.raise_for_leakage()
    print(to_markdown(result))

    src, dst = dataset.records.source_codes, dataset.records.destination_codes
    train_items = set(dst[result.train_indices].tolist())
    test_items = set(dst[result.test_indices].tolist())
    train_users = set(src[result.train_indices].tolist())
    test_users = set(src[result.test_indices].tolist())
    print(f"\nItems disjoint (cold): {train_items.isdisjoint(test_items)}")
    print(f"Users may recur (warm): {bool(train_users & test_users)}")


if __name__ == "__main__":
    main()
