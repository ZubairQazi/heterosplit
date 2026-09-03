"""Tests for the seeded group-to-split assignment core."""

from __future__ import annotations

import numpy as np

from heterosplit.splitters.assignment import assign_groups


class TestAssignGroups:
    def test_empty(self) -> None:
        assert assign_groups([], (0.8, 0.1, 0.1), seed=0).shape == (0,)

    def test_ratio_adherence_equal_groups(self) -> None:
        sizes = np.ones(1000, dtype=np.int64)
        assignment = assign_groups(sizes, (0.7, 0.2, 0.1), seed=0)
        counts = np.bincount(assignment, minlength=3) / sizes.sum()
        np.testing.assert_allclose(counts, [0.7, 0.2, 0.1], atol=0.02)

    def test_variable_group_sizes_track_records_not_groups(self) -> None:
        # One huge group + many tiny ones: ratios are over *records*, not group count.
        sizes = np.array([500] + [1] * 500, dtype=np.int64)
        assignment = assign_groups(sizes, (0.5, 0.5), seed=1)
        per_split = np.array([sizes[assignment == s].sum() for s in range(2)])
        np.testing.assert_allclose(per_split / sizes.sum(), [0.5, 0.5], atol=0.05)

    def test_deterministic(self) -> None:
        sizes = np.ones(200, dtype=np.int64)
        a = assign_groups(sizes, (0.8, 0.1, 0.1), seed=42)
        b = assign_groups(sizes, (0.8, 0.1, 0.1), seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        sizes = np.ones(200, dtype=np.int64)
        a = assign_groups(sizes, (0.8, 0.1, 0.1), seed=1)
        b = assign_groups(sizes, (0.8, 0.1, 0.1), seed=2)
        assert not np.array_equal(a, b)

    def test_stratification_balances_each_stratum(self) -> None:
        # 600 groups, two strata; each stratum should split ~50/50.
        sizes = np.ones(600, dtype=np.int64)
        strata = np.array([0] * 300 + [1] * 300)
        assignment = assign_groups(sizes, (0.5, 0.5), seed=0, strata=strata)
        for stratum in (0, 1):
            mask = strata == stratum
            counts = np.bincount(assignment[mask], minlength=2) / mask.sum()
            np.testing.assert_allclose(counts, [0.5, 0.5], atol=0.05)

    def test_assignment_in_range(self) -> None:
        sizes = np.ones(50, dtype=np.int64)
        assignment = assign_groups(sizes, (0.8, 0.1, 0.1), seed=0)
        assert assignment.min() >= 0
        assert assignment.max() <= 2
