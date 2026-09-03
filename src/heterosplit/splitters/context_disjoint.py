"""Context cold-start splitter.

Contract: test *context* entities (e.g. cell lines) never occur in training. This is a
record-partition regime — records are grouped by their single context entity's code and
each group is assigned to one split, so the context entity sets are disjoint by
construction. Endpoint entities and pairs may recur freely across splits.

Exactly one context role is required (validated on the spec); use ``joint_cold_start``
to hold out several context types at once.
"""

from __future__ import annotations

from typing import ClassVar

from ..records import PredictionRecords
from ..result import SplitResult
from ..spec import Regime, SplitSpec
from .base import Splitter, ensure_compatible, split_by_groups


class ContextColdStartSplitter(Splitter):
    regime: ClassVar[Regime] = Regime.CONTEXT

    def split(self, records: PredictionRecords, spec: SplitSpec) -> SplitResult:
        ensure_compatible(records, spec)
        context_name = spec.schema.context_names[0]
        context_type = spec.schema.roles[context_name].entity_type
        return split_by_groups(
            records, spec, records.codes(context_name), records.n_entities(context_type)
        )
