"""Shared helpers for the auditors."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..canonical import canonicalize_pairs
from ..records import PredictionRecords
from ..result import SplitResult

MAX_EXAMPLES = 5


def native(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def held_out_indices(result: SplitResult) -> np.ndarray:
    """Record indices of every non-training split, concatenated."""
    held = [result.indices(s) for s in result.split_names if s != "train"]
    return np.concatenate(held) if held else np.empty(0, dtype=np.int64)


def pair_set(
    src: np.ndarray, dst: np.ndarray, idx: np.ndarray, *, undirected: bool
) -> set[tuple[int, int]]:
    lo, hi = canonicalize_pairs(src[idx], dst[idx], undirected=undirected)
    return set(zip(lo.tolist(), hi.tolist(), strict=True))


def decode_role_examples(records: PredictionRecords, role_name: str, codes: list[int]) -> list[Any]:
    if not codes:
        return []
    book = records.codebook_for(role_name)
    return [native(v) for v in book.decode(np.array(codes[:MAX_EXAMPLES]))]
