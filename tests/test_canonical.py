"""Tests for canonical pair representation and the symmetry policy."""

from __future__ import annotations

import numpy as np
import pytest

from heterosplit.canonical import canonical_pair, canonicalize_pairs, pair_group_ids


class TestCanonicalPairScalar:
    def test_undirected_is_order_independent(self) -> None:
        assert canonical_pair(3, 7) == canonical_pair(7, 3) == (3, 7)

    def test_directed_preserves_order(self) -> None:
        assert canonical_pair(7, 3, undirected=False) == (7, 3)
        assert canonical_pair(3, 7, undirected=False) == (3, 7)

    def test_self_loop(self) -> None:
        assert canonical_pair(5, 5) == (5, 5)


class TestCanonicalizePairs:
    def test_undirected_sorts_endpoints(self) -> None:
        src = [3, 7, 5, 1]
        dst = [7, 3, 5, 9]
        lo, hi = canonicalize_pairs(src, dst, undirected=True)
        np.testing.assert_array_equal(lo, [3, 3, 5, 1])
        np.testing.assert_array_equal(hi, [7, 7, 5, 9])

    def test_directed_is_identity(self) -> None:
        src = np.array([3, 7])
        dst = np.array([7, 3])
        lo, hi = canonicalize_pairs(src, dst, undirected=False)
        np.testing.assert_array_equal(lo, src)
        np.testing.assert_array_equal(hi, dst)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            canonicalize_pairs([1, 2], [1], undirected=True)

    def test_empty(self) -> None:
        lo, hi = canonicalize_pairs([], [], undirected=True)
        assert lo.shape == (0,) and hi.shape == (0,)


class TestPairGroupIds:
    def test_reversed_pairs_share_a_group_when_undirected(self) -> None:
        # (3,7) and (7,3) are the same undirected edge -> same group.
        src = np.array([3, 7, 1, 3])
        dst = np.array([7, 3, 2, 7])
        groups, unique_pairs = pair_group_ids(src, dst, undirected=True)
        assert groups[0] == groups[1] == groups[3]
        assert groups[2] != groups[0]
        assert len(unique_pairs) == 2

    def test_reversed_pairs_are_distinct_when_directed(self) -> None:
        src = np.array([3, 7])
        dst = np.array([7, 3])
        groups, unique_pairs = pair_group_ids(src, dst, undirected=False)
        assert groups[0] != groups[1]
        assert len(unique_pairs) == 2

    def test_group_ids_are_dense_and_deterministic(self) -> None:
        src = np.array([10, 2, 10, 2, 5])
        dst = np.array([1, 3, 1, 3, 5])
        groups, unique_pairs = pair_group_ids(src, dst, undirected=True)
        # dense: ids cover 0..G-1 exactly
        assert set(groups.tolist()) == set(range(len(unique_pairs)))
        # deterministic: re-running yields identical ids
        groups2, _ = pair_group_ids(src, dst, undirected=True)
        np.testing.assert_array_equal(groups, groups2)

    def test_unique_pairs_are_canonical(self) -> None:
        src = np.array([7, 9])
        dst = np.array([3, 1])
        _, unique_pairs = pair_group_ids(src, dst, undirected=True)
        # every stored pair has lo <= hi
        assert np.all(unique_pairs[:, 0] <= unique_pairs[:, 1])

    def test_empty_input(self) -> None:
        groups, unique_pairs = pair_group_ids([], [], undirected=True)
        assert groups.shape == (0,)
        assert unique_pairs.shape[0] == 0
