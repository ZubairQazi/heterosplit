"""Canonical pair representation for unordered / undirected relations.

The *symmetry policy*: for an undirected or unordered relation, ``(A, B)`` and
``(B, A)`` denote the same edge and must therefore be treated identically by every
splitter. Canonicalizing each pair to ``(min, max)`` guarantees the two orderings
share a group and can never land in different splits.

Endpoints are represented as integer entity *codes* (the per-entity-type
factorization produced by the records table). Undirected canonicalization is only
meaningful when both endpoints draw from the same code space, i.e. a self-relation
such as ``(drug, synergy, drug)``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["canonical_pair", "canonicalize_pairs", "pair_group_ids"]

IntArray = npt.NDArray[np.int64]


def canonical_pair(a: int, b: int, *, undirected: bool = True) -> tuple[int, int]:
    """Order-independent representation of a single ``(a, b)`` pair.

    For undirected relations both ``(A, B)`` and ``(B, A)`` canonicalize to
    ``(min, max)``; for directed relations the pair is returned unchanged.
    """
    if undirected and b < a:
        return (b, a)
    return (a, b)


def canonicalize_pairs(
    src: npt.ArrayLike, dst: npt.ArrayLike, *, undirected: bool
) -> tuple[IntArray, IntArray]:
    """Vectorized canonicalization of parallel ``src``/``dst`` code arrays.

    Returns a ``(lo, hi)`` pair of arrays. When ``undirected`` is true these are the
    element-wise min and max; otherwise they are copies of the inputs unchanged.
    """
    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    if src_arr.shape != dst_arr.shape:
        raise ValueError(
            f"src and dst must have the same shape, got {src_arr.shape} and {dst_arr.shape}"
        )
    if src_arr.ndim != 1:
        raise ValueError(f"src and dst must be 1-D, got ndim={src_arr.ndim}")
    if not undirected:
        return src_arr.copy(), dst_arr.copy()
    lo = np.minimum(src_arr, dst_arr)
    hi = np.maximum(src_arr, dst_arr)
    return lo, hi


def pair_group_ids(
    src: npt.ArrayLike, dst: npt.ArrayLike, *, undirected: bool
) -> tuple[IntArray, IntArray]:
    """Assign each ``(src, dst)`` record a canonical-pair group id.

    Records sharing a canonical pair receive the same group id. Group ids are
    assigned in lexicographic order of the canonical pair, so the mapping is
    deterministic for a fixed input (independent of record order only up to that
    stable ordering).

    Returns:
        A tuple ``(group_ids, unique_pairs)`` where ``group_ids[i]`` is the group of
        record ``i`` and ``unique_pairs[g]`` is the ``(lo, hi)`` pair for group ``g``.
    """
    lo, hi = canonicalize_pairs(src, dst, undirected=undirected)
    pairs = np.stack([lo, hi], axis=1)
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    # numpy has changed the shape of ``return_inverse`` with ``axis`` across versions;
    # flatten defensively so callers always get a 1-D array of group ids.
    group_ids = np.asarray(inverse, dtype=np.int64).reshape(-1)
    return group_ids, np.asarray(unique_pairs, dtype=np.int64)
