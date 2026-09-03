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

from ..objective import count_missing_values, distribution_divergence, size_deviation

__all__ = ["assign_groups", "refine_assignment"]

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


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


def refine_assignment(
    assignment: IntArray,
    group_sizes: npt.ArrayLike,
    group_value_counts: npt.ArrayLike,
    ratios: Sequence[float],
    *,
    max_passes: int = 6,
    eps: float = 1e-9,
) -> IntArray:
    """Improve label/relation balance without worsening the size ratios.

    Bounded local search that moves whole groups between splits. Acceptance is
    *lexicographic*: a move is taken only if it strictly reduces the size deviation, or
    leaves it unchanged (within ``eps``) while reducing distribution divergence plus a
    missing-value penalty. Because size deviation never increases, this can only make
    ratio adherence at least as good as the greedy seed while improving distribution
    balance. Deterministic: groups are visited largest-first each pass.

    Args:
        assignment: Greedy seed assignment (group -> split index); not mutated.
        group_sizes: Records per group.
        group_value_counts: ``(n_groups, n_values)`` per-group counts of the field to
            balance (e.g. label codes).
        ratios: Target fractions per split.
        max_passes: Maximum improvement sweeps.
        eps: Numerical tolerance for "no worse".

    Returns:
        The refined assignment (a new array).
    """
    result = np.asarray(assignment, dtype=np.int64).copy()
    sizes = np.asarray(group_sizes, dtype=np.float64)
    value_counts = np.asarray(group_value_counts, dtype=np.float64)
    ratio_arr = np.asarray(ratios, dtype=np.float64)
    k = ratio_arr.shape[0]
    n_groups = result.shape[0]
    if n_groups == 0 or value_counts.shape[1] == 0:
        return result

    split_sizes = np.zeros(k, dtype=np.float64)
    split_vc = np.zeros((k, value_counts.shape[1]), dtype=np.float64)
    for g in range(n_groups):
        split_sizes[result[g]] += sizes[g]
        split_vc[result[g]] += value_counts[g]

    def secondary() -> float:
        return distribution_divergence(split_vc, ratio_arr) + count_missing_values(split_vc)

    order = np.argsort(-sizes, kind="stable")
    for _ in range(max_passes):
        changed = False
        cur_primary = size_deviation(split_sizes, ratio_arr)
        cur_secondary = secondary()
        for g in order:
            s0 = int(result[g])
            best_split, best_primary, best_secondary = s0, cur_primary, cur_secondary
            for s1 in range(k):
                if s1 == s0:
                    continue
                split_sizes[s0] -= sizes[g]
                split_sizes[s1] += sizes[g]
                split_vc[s0] -= value_counts[g]
                split_vc[s1] += value_counts[g]
                primary = size_deviation(split_sizes, ratio_arr)
                sec = secondary()
                split_sizes[s0] += sizes[g]
                split_sizes[s1] -= sizes[g]
                split_vc[s0] += value_counts[g]
                split_vc[s1] -= value_counts[g]
                improves = primary < best_primary - eps or (
                    abs(primary - best_primary) <= eps and sec < best_secondary - eps
                )
                if improves:
                    best_split, best_primary, best_secondary = s1, primary, sec
            if best_split != s0:
                split_sizes[s0] -= sizes[g]
                split_sizes[best_split] += sizes[g]
                split_vc[s0] -= value_counts[g]
                split_vc[best_split] += value_counts[g]
                result[g] = best_split
                cur_primary, cur_secondary = best_primary, best_secondary
                changed = True
        if not changed:
            break
    return result
