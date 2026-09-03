"""Random / transductive splitter.

Contract: test edges are *unseen instances*, but entities (and, for directed
relations, pairs) may reappear in training. Concretely:

* **Directed** relations split at the *record* level — each record is its own group,
  so this is a pure transductive edge split. Reverse-edge leakage into the
  message-passing graph is a separate concern handled during message-passing
  reconstruction and surfaced by the auditors.
* **Undirected** relations (``undirected_pairs=True``) group by *canonical pair* so an
  edge ``{A, B}`` cannot be a supervision target in two different splits via its two
  orderings. When there is one record per undirected pair this coincides with a record
  split; with multiple records per pair it is stricter (and leakage-safe) by design.

Callers wanting pair-level disjointness for directed data should use
``pair_cold_start`` instead.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ..canonical import pair_group_ids
from ..records import PredictionRecords
from ..result import SplitResult
from ..spec import Regime, SplitSpec
from .base import Splitter, ensure_compatible, split_by_groups


class RandomSplitter(Splitter):
    regime: ClassVar[Regime] = Regime.RANDOM

    def split(self, records: PredictionRecords, spec: SplitSpec) -> SplitResult:
        ensure_compatible(records, spec)
        if spec.undirected_pairs:
            group_ids, unique_pairs = pair_group_ids(
                records.source_codes, records.destination_codes, undirected=True
            )
            n_groups = int(unique_pairs.shape[0])
        else:
            n_groups = records.n_records
            group_ids = np.arange(n_groups, dtype=np.int64)
        return split_by_groups(records, spec, group_ids, n_groups)
