"""Balance objectives: size deviation and distribution divergence.

These metrics quantify how well a split matches its requested ratios (size deviation)
and how similar a field's distribution is across splits (distribution divergence, e.g.
of labels, relations, or degree buckets). They are used both to *refine* an assignment
(local search) and to *report* balance in the manifest and distribution report.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "count_missing_values",
    "distribution_divergence",
    "size_deviation",
    "value_counts_by_split",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def size_deviation(split_sizes: npt.ArrayLike, ratios: npt.ArrayLike) -> float:
    """Total-variation-style L1 distance between achieved and requested split fractions."""
    sizes = np.asarray(split_sizes, dtype=np.float64)
    total = sizes.sum()
    if total == 0:
        return 0.0
    return float(np.abs(sizes / total - np.asarray(ratios, dtype=np.float64)).sum())


def value_counts_by_split(
    values: IntArray, record_split: IntArray, n_values: int, n_splits: int
) -> FloatArray:
    """``(n_splits, n_values)`` counts of each value within each split (ignores EXCLUDED)."""
    counts = np.zeros((n_splits, n_values), dtype=np.float64)
    valid = record_split >= 0
    np.add.at(counts, (record_split[valid], values[valid]), 1.0)
    return counts


def distribution_divergence(counts_by_split: npt.ArrayLike, ratios: npt.ArrayLike) -> float:
    """Ratio-weighted mean total-variation distance of each split's value distribution
    from the overall distribution. ``0`` means every split matches the whole.
    """
    counts = np.asarray(counts_by_split, dtype=np.float64)
    ratio_arr = np.asarray(ratios, dtype=np.float64)
    overall = counts.sum(axis=0)
    overall_total = overall.sum()
    if overall_total == 0:
        return 0.0
    overall_dist = overall / overall_total
    per_split_total = counts.sum(axis=1)
    divergence = 0.0
    for s in range(counts.shape[0]):
        if per_split_total[s] == 0:
            continue
        dist = counts[s] / per_split_total[s]
        tv = 0.5 * float(np.abs(dist - overall_dist).sum())
        divergence += float(ratio_arr[s]) * tv
    return divergence


def count_missing_values(counts_by_split: npt.ArrayLike) -> int:
    """Number of (split, value) cells absent in a split though present overall."""
    counts = np.asarray(counts_by_split, dtype=np.float64)
    present_overall = counts.sum(axis=0) > 0
    absent_in_split = (counts == 0) & present_overall[None, :]
    return int(absent_in_split.sum())
