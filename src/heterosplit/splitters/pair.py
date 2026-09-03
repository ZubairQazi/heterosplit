"""Pair cold-start splitter.

Contract: the ``(s, d)`` *pair* of every test record is unseen in training, but each
endpoint entity may appear separately (in other pairs) in any split. Records are
grouped by their (optionally canonical) pair and each group is assigned to a single
split, so the train and test pair sets are disjoint by construction.

Pairs are canonicalized when ``undirected_pairs`` is set, so ``(A, B)`` and ``(B, A)``
count as the same pair; otherwise the ordered pair is used and the reversed pair is a
different pair (its potential leakage is reported by the reverse-pair auditor).
"""

from __future__ import annotations

from typing import ClassVar

from ..canonical import pair_group_ids
from ..records import PredictionRecords
from ..result import SplitResult
from ..spec import Regime, SplitSpec
from .base import Splitter, ensure_compatible, split_by_groups


class PairDisjointSplitter(Splitter):
    regime: ClassVar[Regime] = Regime.PAIR

    def split(self, records: PredictionRecords, spec: SplitSpec) -> SplitResult:
        ensure_compatible(records, spec)
        group_ids, unique_pairs = pair_group_ids(
            records.source_codes,
            records.destination_codes,
            undirected=spec.undirected_pairs,
        )
        return split_by_groups(records, spec, group_ids, int(unique_pairs.shape[0]))
