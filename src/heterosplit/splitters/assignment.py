"""Seeded group-to-split assignment.

The baseline objective is *ratio adherence*: assign atomic groups (records, pairs, or
entities — whatever the regime deems indivisible) to splits so the per-split record
counts land near the requested ratios, while never dividing a group. Assignment is a
deterministic function of the group sizes, ratios, seed, and optional strata.

Groups are placed largest-first (a longest-processing-time heuristic) with random,
seeded tie-breaking among equal sizes, then each is assigned to the split with the
largest remaining size deficit. Largest-first keeps balance near-optimal even when a
few groups dominate; because record-level splits have all-equal group sizes, that case
degenerates to a pure seeded shuffle, preserving randomness where it matters.

When ``strata`` is supplied, each stratum (e.g. a label value) is distributed to the
splits independently and in proportion to the ratios, which keeps that field's
distribution similar across splits. This is the seed of the richer constrained
objective added later; it already guarantees the *constraint* (group atomicity) and a
good *balance* on size and one stratifying field.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

__all__ = ["assign_groups"]

IntArray = npt.NDArray[np.int64]


def assign_groups(
    group_sizes: npt.ArrayLike,
    ratios: Sequence[float],
    seed: int,
    *,
    strata: npt.ArrayLike | None = None,
) -> IntArray:
    """Assign each group to a split index in ``0..len(ratios)-1``.

    Args:
        group_sizes: Number of records in each group (length ``G``).
        ratios: Target fraction per split.
        seed: Seed controlling the within-stratum shuffle.
        strata: Optional per-group stratum label; groups are balanced within each
            stratum independently.

    Returns:
        Length-``G`` array mapping group index to split index. Empty input yields an
        empty array.
    """
    sizes = np.asarray(group_sizes, dtype=np.int64)
    if sizes.ndim != 1:
        raise ValueError(f"group_sizes must be 1-D, got ndim={sizes.ndim}")
    n_groups = sizes.shape[0]
    assignment = np.full(n_groups, -1, dtype=np.int64)
    if n_groups == 0:
        return assignment

    ratio_arr = np.asarray(ratios, dtype=np.float64)
    k = ratio_arr.shape[0]
    rng = np.random.default_rng(seed)

    if strata is None:
        strata_arr = np.zeros(n_groups, dtype=np.int64)
    else:
        strata_arr = np.asarray(strata, dtype=np.int64)
        if strata_arr.shape != (n_groups,):
            raise ValueError("strata must have the same length as group_sizes")

    for stratum in np.unique(strata_arr):
        members = np.flatnonzero(strata_arr == stratum)
        # Largest group first; ties broken by a seeded random key so equal-size
        # groups (the record-level case) are shuffled rather than index-ordered.
        tie_break = rng.permutation(members.shape[0])
        order = members[np.lexsort((tie_break, -sizes[members]))]
        targets = ratio_arr * float(sizes[members].sum())
        current = np.zeros(k, dtype=np.float64)
        for g in order:
            # Fill the split with the largest remaining deficit; ties -> lowest index.
            split = int(np.argmax(targets - current))
            assignment[g] = split
            current[split] += float(sizes[g])
    return assignment
