"""Reconstruction of supervision and message-passing edge sets from a split.

Two edge sets matter for link prediction:

* **supervision edges** — the ``(source, destination)`` pairs to score in a split.
* **message-passing edges** — the edges the GNN may propagate over while training. The
  core invariant is that the *training* message-passing graph must not contain any
  held-out (validation/test) supervision edge **or its reverse**; otherwise the model
  observes the answer. Reverse forms are caught by canonicalizing every pair before the
  membership test, so ``(A, B)`` in training is dropped when ``(B, A)`` is a test edge.

Everything here works on entity *codes* and returns ``(2, E)`` integer edge-index arrays
in PyG's layout. The PyG adapter converts these to tensors; the auditors verify them.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .canonical import canonicalize_pairs
from .result import SplitResult

__all__ = ["message_passing_edge_index", "supervision_edge_index"]

IntArray = npt.NDArray[np.int64]


def _indices_for(result: SplitResult, splits: tuple[str, ...]) -> IntArray:
    if not splits:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate([result.indices(s) for s in splits])).astype(np.int64)


def _rows_in(rows: IntArray, reference: IntArray) -> npt.NDArray[np.bool_]:
    """Boolean mask of which ``rows`` (as 2-col int arrays) appear in ``reference``."""
    if rows.shape[0] == 0 or reference.shape[0] == 0:
        return np.zeros(rows.shape[0], dtype=bool)
    combined = np.concatenate([reference, rows], axis=0)
    _, inverse = np.unique(combined, axis=0, return_inverse=True)
    inverse = np.asarray(inverse, dtype=np.int64).reshape(-1)
    reference_ids = np.unique(inverse[: reference.shape[0]])
    return np.isin(inverse[reference.shape[0] :], reference_ids)


def _unique_columns(edge_index: IntArray) -> IntArray:
    if edge_index.shape[1] == 0:
        return edge_index
    unique = np.unique(edge_index.T, axis=0)
    return unique.T.astype(np.int64)


def supervision_edge_index(result: SplitResult, split: str) -> IntArray:
    """The ``(2, E)`` source/destination code edge index of ``split``'s records."""
    idx = result.indices(split)
    return np.stack(
        [result.records.source_codes[idx], result.records.destination_codes[idx]]
    ).astype(np.int64)


def message_passing_edge_index(
    result: SplitResult,
    *,
    mp_splits: tuple[str, ...] = ("train",),
    add_reverse: bool | None = None,
    remove_heldout: bool = True,
) -> IntArray:
    """Reconstruct the leakage-safe message-passing edge index for the supervision edge.

    Args:
        result: The split.
        mp_splits: Splits whose supervision edges form the message-passing graph
            (default: training only).
        add_reverse: Add both orderings of each edge. Defaults to the spec's
            ``undirected_pairs`` (undirected relations need both directions to propagate).
        remove_heldout: Drop any message-passing edge whose pair coincides with a
            held-out (non-``mp_splits``) supervision edge — and, for self-relations, its
            reverse. This is the core message-passing invariant and is on by default.
            Note that any split you place in ``mp_splits`` is, by definition, not
            held-out and so is *not* removed: only pass a held-out split here if you
            deliberately want its edges in the graph (e.g. ``("train", "val")`` for the
            test-time graph).

    Returns:
        A deduplicated ``(2, M)`` edge index of entity codes.
    """
    records = result.records
    src, dst = records.source_codes, records.destination_codes
    undirected = result.spec.undirected_pairs
    # Reverse-form collapsing is only valid for a self-relation, where both endpoints
    # share a code space. For a bipartite relation source-code k and destination-code k
    # are different entities, so (min,max) across the two codebooks must NOT be used.
    reverse_aware = records.schema.is_self_relation
    if add_reverse is None:
        add_reverse = undirected

    mp_idx = _indices_for(result, mp_splits)
    mp_src, mp_dst = src[mp_idx], dst[mp_idx]

    if remove_heldout:
        heldout_splits = tuple(s for s in result.split_names if s not in mp_splits)
        heldout_idx = _indices_for(result, heldout_splits)
        if heldout_idx.size and mp_idx.size:
            h_lo, h_hi = canonicalize_pairs(
                src[heldout_idx], dst[heldout_idx], undirected=reverse_aware
            )
            m_lo, m_hi = canonicalize_pairs(mp_src, mp_dst, undirected=reverse_aware)
            held_rows = np.stack([h_lo, h_hi], axis=1)
            mp_rows = np.stack([m_lo, m_hi], axis=1)
            keep = ~_rows_in(mp_rows, held_rows)
            mp_src, mp_dst = mp_src[keep], mp_dst[keep]

    if undirected:
        lo, hi = canonicalize_pairs(mp_src, mp_dst, undirected=True)
        pairs = (
            np.unique(np.stack([lo, hi], axis=1), axis=0) if lo.size else np.empty((0, 2), np.int64)
        )
        edge_index = np.stack([pairs[:, 0], pairs[:, 1]]).astype(np.int64)
        if add_reverse:
            edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
        return _unique_columns(edge_index)

    edge_index = np.stack([mp_src, mp_dst]).astype(np.int64)
    if add_reverse:
        edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    return _unique_columns(edge_index)
