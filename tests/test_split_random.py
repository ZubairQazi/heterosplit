"""Tests for the random / transductive splitter."""

from __future__ import annotations

import numpy as np
import pytest

from heterosplit import EntityRole, SplitSpec, TaskSchema, split_records
from heterosplit.canonical import pair_group_ids
from heterosplit.records import PredictionRecords
from heterosplit.synthetic import make_synthetic_dataset


def drug_schema() -> TaskSchema:
    return TaskSchema(
        ("drug", "synergy", "drug"),
        {"source": EntityRole.source("drug"), "destination": EntityRole.destination("drug")},
    )


class TestRandomSplitter:
    def test_covers_every_record_exactly_once(self) -> None:
        ds = make_synthetic_dataset(n_records=300, seed=0)
        result = split_records(ds.records, ds.spec("random"))
        assert result.covers_all_records()
        assert result.n_excluded == 0
        total = sum(result.counts.values())
        assert total == 300

    def test_deterministic(self) -> None:
        ds = make_synthetic_dataset(n_records=200, seed=0)
        a = split_records(ds.records, ds.spec("random", seed=5))
        b = split_records(ds.records, ds.spec("random", seed=5))
        np.testing.assert_array_equal(a.record_split, b.record_split)

    def test_ratios_approximately_met(self) -> None:
        ds = make_synthetic_dataset(n_records=2000, seed=0)
        result = split_records(ds.records, ds.spec("random", ratios=(0.7, 0.2, 0.1)))
        ratios = result.achieved_ratios()
        assert ratios["train"] == pytest.approx(0.7, abs=0.03)
        assert ratios["val"] == pytest.approx(0.2, abs=0.03)
        assert ratios["test"] == pytest.approx(0.1, abs=0.03)

    def test_two_way_split(self) -> None:
        ds = make_synthetic_dataset(n_records=100, seed=0)
        result = split_records(ds.records, ds.spec("random", ratios=(0.8, 0.2)))
        assert set(result.counts) == {"train", "test"}

    def test_undirected_reversed_pairs_share_split(self) -> None:
        # (A,B) and (B,A) are the same undirected edge -> must not cross splits.
        records = PredictionRecords.from_columns(
            drug_schema(),
            {"source": ["A", "B", "C", "E", "G"], "destination": ["B", "A", "D", "F", "H"]},
        )
        spec = SplitSpec(
            supervision_edge=("drug", "synergy", "drug"),
            roles=dict(drug_schema().roles),
            regime="random",
            ratios=(0.5, 0.5),
            undirected_pairs=True,
            seed=0,
        )
        result = split_records(records, spec)
        assert result.record_split[0] == result.record_split[1]

    def test_undirected_no_canonical_pair_crosses_splits(self) -> None:
        ds = make_synthetic_dataset(n_records=1000, n_source_entities=25, seed=3)
        result = split_records(ds.records, ds.spec("random", undirected_pairs=True))
        groups, _ = pair_group_ids(
            ds.records.source_codes, ds.records.destination_codes, undirected=True
        )
        # every canonical pair maps to a single split
        for g in np.unique(groups):
            splits = np.unique(result.record_split[groups == g])
            assert splits.size == 1

    def test_stratify_by_label_balances_distribution(self) -> None:
        ds = make_synthetic_dataset(n_records=3000, n_labels=3, seed=0)
        result = split_records(ds.records, ds.spec("random", stratify_by="label"))
        labels = ds.records.labels
        assert labels is not None
        # label distribution in train should match test within a small tolerance
        train = np.bincount(labels[result.train_indices], minlength=3) / result.counts["train"]
        test = np.bincount(labels[result.test_indices], minlength=3) / result.counts["test"]
        np.testing.assert_allclose(train, test, atol=0.05)

    def test_stratify_without_labels_warns(self) -> None:
        ds = make_synthetic_dataset(n_records=50, n_labels=0, seed=0)
        result = split_records(ds.records, ds.spec("random", stratify_by="label"))
        assert any("no labels" in w for w in result.warnings)
