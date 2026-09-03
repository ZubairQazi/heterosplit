"""Tests for balance objective metrics."""

from __future__ import annotations

import numpy as np

from heterosplit.objective import (
    count_missing_values,
    distribution_divergence,
    size_deviation,
    value_counts_by_split,
)


class TestSizeDeviation:
    def test_perfect_is_zero(self) -> None:
        assert size_deviation([70, 20, 10], (0.7, 0.2, 0.1)) == 0.0

    def test_skewed_is_positive(self) -> None:
        assert size_deviation([100, 0, 0], (0.7, 0.2, 0.1)) > 0

    def test_empty(self) -> None:
        assert size_deviation([0, 0], (0.5, 0.5)) == 0.0


class TestValueCountsBySplit:
    def test_counts_and_excluded_ignored(self) -> None:
        values = np.array([0, 1, 0, 1, 0])
        record_split = np.array([0, 0, 1, 1, -1])  # last is EXCLUDED
        counts = value_counts_by_split(values, record_split, n_values=2, n_splits=2)
        np.testing.assert_array_equal(counts, [[1, 1], [1, 1]])


class TestDistributionDivergence:
    def test_identical_distributions_zero(self) -> None:
        counts = np.array([[5.0, 5.0], [5.0, 5.0]])
        assert distribution_divergence(counts, (0.5, 0.5)) == 0.0

    def test_disjoint_distributions_positive(self) -> None:
        counts = np.array([[10.0, 0.0], [0.0, 10.0]])
        assert distribution_divergence(counts, (0.5, 0.5)) > 0


class TestMissingValues:
    def test_counts_absent_cells(self) -> None:
        counts = np.array([[10.0, 0.0], [5.0, 5.0]])  # value 1 missing from split 0
        assert count_missing_values(counts) == 1

    def test_none_missing(self) -> None:
        counts = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert count_missing_values(counts) == 0
