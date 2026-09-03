"""Tests for the pair cold-start splitter."""

from __future__ import annotations

import numpy as np

from heterosplit import split_records
from heterosplit.canonical import canonicalize_pairs
from heterosplit.synthetic import make_synthetic_dataset


def _pair_keys(codes_src: np.ndarray, codes_dst: np.ndarray, *, undirected: bool) -> set[tuple]:
    lo, hi = canonicalize_pairs(codes_src, codes_dst, undirected=undirected)
    return set(zip(lo.tolist(), hi.tolist(), strict=True))


class TestPairDisjointSplitter:
    def test_covers_all_records(self) -> None:
        ds = make_synthetic_dataset(n_records=400, seed=0)
        result = split_records(ds.records, ds.spec("pair_cold_start"))
        assert result.covers_all_records()
        assert result.n_excluded == 0

    def test_deterministic(self) -> None:
        ds = make_synthetic_dataset(n_records=300, seed=1)
        a = split_records(ds.records, ds.spec("pair_cold_start", seed=9))
        b = split_records(ds.records, ds.spec("pair_cold_start", seed=9))
        np.testing.assert_array_equal(a.record_split, b.record_split)

    def test_train_test_pairs_are_disjoint_directed(self) -> None:
        ds = make_synthetic_dataset(n_records=1500, n_source_entities=30, seed=2)
        result = split_records(ds.records, ds.spec("pair_cold_start"))
        src, dst = ds.records.source_codes, ds.records.destination_codes
        train = _pair_keys(src[result.train_indices], dst[result.train_indices], undirected=False)
        test = _pair_keys(src[result.test_indices], dst[result.test_indices], undirected=False)
        assert train.isdisjoint(test)

    def test_train_test_pairs_disjoint_including_reversed_when_undirected(self) -> None:
        ds = make_synthetic_dataset(n_records=1500, n_source_entities=30, seed=4)
        result = split_records(ds.records, ds.spec("pair_cold_start", undirected_pairs=True))
        src, dst = ds.records.source_codes, ds.records.destination_codes
        train = _pair_keys(src[result.train_indices], dst[result.train_indices], undirected=True)
        test = _pair_keys(src[result.test_indices], dst[result.test_indices], undirected=True)
        # canonical pairs disjoint => reversed pairs cannot cross splits
        assert train.isdisjoint(test)

    def test_entities_may_appear_across_splits(self) -> None:
        # Pair cold-start holds out pairs, not entities: endpoints can recur.
        ds = make_synthetic_dataset(n_records=2000, n_source_entities=20, seed=5)
        result = split_records(ds.records, ds.spec("pair_cold_start"))
        src = ds.records.source_codes
        train_src = set(src[result.train_indices].tolist())
        test_src = set(src[result.test_indices].tolist())
        assert train_src & test_src  # overlap expected and allowed

    def test_multi_record_pairs_stay_together(self) -> None:
        # Records sharing a pair must land in the same split.
        ds = make_synthetic_dataset(
            n_records=2000, n_source_entities=8, n_context_entities=5, seed=6
        )
        result = split_records(ds.records, ds.spec("pair_cold_start"))
        src, dst = ds.records.source_codes, ds.records.destination_codes
        from heterosplit.canonical import pair_group_ids

        groups, _ = pair_group_ids(src, dst, undirected=False)
        for g in np.unique(groups):
            assert np.unique(result.record_split[groups == g]).size == 1
