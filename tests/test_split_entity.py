"""Invariant tests for the entity-disjoint (endpoint cold-start) splitters."""

from __future__ import annotations

import numpy as np
import pytest

from heterosplit import SpecError, SplitSpec, split_records
from heterosplit.result import SplitResult
from heterosplit.synthetic import make_synthetic_dataset


def _codes(result: SplitResult) -> tuple[np.ndarray, np.ndarray]:
    return result.records.source_codes, result.records.destination_codes


def _entity_set(src: np.ndarray, dst: np.ndarray, indices: np.ndarray) -> set[int]:
    return set(src[indices].tolist()) | set(dst[indices].tolist())


class TestSourceDestinationColdStart:
    def test_source_disjoint(self) -> None:
        ds = make_synthetic_dataset(n_records=2000, n_source_entities=60, seed=0)
        result = split_records(ds.records, ds.spec("source_cold_start"))
        assert result.covers_all_records()
        src, _ = _codes(result)
        train_src = set(src[result.train_indices].tolist())
        test_src = set(src[result.test_indices].tolist())
        assert train_src.isdisjoint(test_src)

    def test_destination_disjoint(self) -> None:
        ds = make_synthetic_dataset(n_records=2000, n_source_entities=60, seed=1)
        result = split_records(ds.records, ds.spec("destination_cold_start"))
        _, dst = _codes(result)
        train_dst = set(dst[result.train_indices].tolist())
        test_dst = set(dst[result.test_indices].tolist())
        assert train_dst.isdisjoint(test_dst)

    def test_deterministic(self) -> None:
        ds = make_synthetic_dataset(n_records=500, seed=0)
        a = split_records(ds.records, ds.spec("source_cold_start", seed=3))
        b = split_records(ds.records, ds.spec("source_cold_start", seed=3))
        np.testing.assert_array_equal(a.record_split, b.record_split)


class TestEitherColdStart:
    def test_at_least_one_endpoint_unseen_in_training(self) -> None:
        ds = make_synthetic_dataset(n_records=3000, n_source_entities=50, seed=2)
        result = split_records(ds.records, ds.spec("either_cold_start"))
        assert result.covers_all_records()
        assert result.n_excluded == 0
        src, dst = _codes(result)
        train_entities = np.array(sorted(_entity_set(src, dst, result.train_indices)))
        test = result.test_indices
        src_unseen = ~np.isin(src[test], train_entities)
        dst_unseen = ~np.isin(dst[test], train_entities)
        assert np.all(src_unseen | dst_unseen)

    def test_bipartite_either(self) -> None:
        ds = make_synthetic_dataset(
            n_records=3000,
            self_relation=False,
            source_type="user",
            destination_type="item",
            n_source_entities=40,
            n_destination_entities=40,
            seed=3,
        )
        result = split_records(ds.records, ds.spec("either_cold_start"))
        src, dst = _codes(result)
        train_src = set(src[result.train_indices].tolist())
        train_dst = set(dst[result.train_indices].tolist())
        test = result.test_indices
        src_unseen = ~np.isin(src[test], np.array(sorted(train_src)))
        dst_unseen = ~np.isin(dst[test], np.array(sorted(train_dst)))
        assert np.all(src_unseen | dst_unseen)


class TestBothColdStart:
    def test_train_and_test_entities_disjoint(self) -> None:
        ds = make_synthetic_dataset(n_records=3000, n_source_entities=50, seed=4)
        result = split_records(ds.records, ds.spec("both_cold_start"))
        src, dst = _codes(result)
        train_entities = _entity_set(src, dst, result.train_indices)
        test_entities = _entity_set(src, dst, result.test_indices)
        assert train_entities.isdisjoint(test_entities)

    def test_both_endpoints_unseen_in_training(self) -> None:
        ds = make_synthetic_dataset(n_records=3000, n_source_entities=50, seed=5)
        result = split_records(ds.records, ds.spec("both_cold_start"))
        src, dst = _codes(result)
        train_entities = np.array(sorted(_entity_set(src, dst, result.train_indices)))
        test = result.test_indices
        assert np.all(~np.isin(src[test], train_entities))
        assert np.all(~np.isin(dst[test], train_entities))

    def test_bridge_records_excluded_and_accounted(self) -> None:
        ds = make_synthetic_dataset(n_records=3000, n_source_entities=50, seed=6)
        result = split_records(ds.records, ds.spec("both_cold_start"))
        # every record is assigned or explicitly excluded; nothing is lost
        assert result.covers_all_records()
        assigned = sum(result.counts.values())
        assert assigned + result.n_excluded == ds.records.n_records
        assert result.n_excluded > 0  # bridges are expected with a 3-way split
        assert any("bridge" in w for w in result.warnings)

    def test_deterministic(self) -> None:
        ds = make_synthetic_dataset(n_records=1000, seed=0)
        a = split_records(ds.records, ds.spec("both_cold_start", seed=8))
        b = split_records(ds.records, ds.spec("both_cold_start", seed=8))
        np.testing.assert_array_equal(a.record_split, b.record_split)


class TestSpecGuards:
    def test_undirected_incompatible_with_source(self) -> None:
        with pytest.raises(SpecError, match="incompatible with regime"):
            SplitSpec(
                supervision_edge=("drug", "synergy", "drug"),
                roles=dict(make_synthetic_dataset(n_records=1, seed=0).records.schema.roles),
                regime="source_cold_start",
                undirected_pairs=True,
            )
