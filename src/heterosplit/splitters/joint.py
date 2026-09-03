"""Joint cold-start splitter.

Contract: a configured *combination* of endpoint and context disjointness holds
simultaneously. The ``holdout`` mapping assigns each held-out entity type a mode:

* endpoint modes ``either`` / ``both`` (self-relation only), ``source``, ``destination``;
* context mode ``all``.

Each held-out entity type is labeled train/val/test (degree-weighted). For every record
each active axis independently yields a *tier* in ``{train, val, test}`` (or "inconsistent"
for a ``both`` axis whose endpoints straddle splits). A record joins split ``t`` only if
**every** axis agrees on tier ``t``; otherwise it is excluded. This makes the test set
"cold" along all configured axes at once (e.g. a *new* drug pair *and* a *new* cell line),
which is intentionally strict — achieved ratios and exclusions are reported rather than
relaxed.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ..records import PredictionRecords
from ..result import EXCLUDED, SplitResult
from ..schema import TaskSchema
from ..spec import Regime, SplitSpec
from .assignment import assign_groups
from .base import (
    Splitter,
    degree,
    empty_split_warnings,
    ensure_compatible,
    ratio_deviation_warnings,
)

IntArray = npt.NDArray[np.int64]

# Per-record axis tier sentinel: a ``both`` axis whose endpoints fell in different splits.
_INCONSISTENT = -2


class JointColdStartSplitter(Splitter):
    regime: ClassVar[Regime] = Regime.JOINT

    def split(self, records: PredictionRecords, spec: SplitSpec) -> SplitResult:
        ensure_compatible(records, spec)
        warnings: list[str] = []
        if spec.stratify_by is not None:
            warnings.append("stratify_by is not applied for joint_cold_start; ignoring")

        tiers = _axis_tiers(records, spec)
        stacked = np.stack(tiers)  # (n_axes, n_records)
        agree = np.all(stacked == stacked[0], axis=0) & np.all(stacked != _INCONSISTENT, axis=0)
        record_split = np.where(agree, stacked[0], EXCLUDED).astype(np.int64)

        n_excluded = int(np.count_nonzero(record_split == EXCLUDED))
        if n_excluded:
            warnings.append(
                f"{n_excluded} record(s) excluded (not cold along every configured axis)"
            )
        warnings.extend(empty_split_warnings(record_split, spec.split_names))
        warnings.extend(ratio_deviation_warnings(record_split, spec))
        return SplitResult(spec=spec, records=records, record_split=record_split, warnings=warnings)


def _axis_tiers(records: PredictionRecords, spec: SplitSpec) -> list[IntArray]:
    """One per-record tier array per active holdout axis."""
    assert spec.holdout is not None  # guaranteed by spec validation  # noqa: S101
    active = {t: m for t, m in spec.holdout.items() if m != "none"}
    labels = _label_held_out_types(records, spec, sorted(active))

    src_codes = records.source_codes
    dst_codes = records.destination_codes
    tiers: list[IntArray] = []
    for entity_type, mode in active.items():
        label = labels[entity_type]
        if mode == "either":
            tiers.append(np.maximum(label[src_codes], label[dst_codes]))
        elif mode == "both":
            src_l, dst_l = label[src_codes], label[dst_codes]
            tiers.append(np.where(src_l == dst_l, src_l, _INCONSISTENT).astype(np.int64))
        elif mode == "source":
            tiers.append(label[src_codes])
        elif mode == "destination":
            tiers.append(label[dst_codes])
        elif mode == "all":  # context
            tiers.append(_context_axis(records, spec.schema, entity_type, label))
    return tiers


def _label_held_out_types(
    records: PredictionRecords, spec: SplitSpec, types_sorted: list[str]
) -> dict[str, IntArray]:
    """Train/val/test labeling of each held-out type's entities (degree-weighted)."""
    child_seeds = np.random.SeedSequence(spec.seed).generate_state(len(types_sorted))
    labels: dict[str, IntArray] = {}
    for i, entity_type in enumerate(types_sorted):
        code_arrays = [
            records.codes(name)
            for name, role in spec.schema.roles.items()
            if role.entity_type == entity_type
        ]
        n = records.n_entities(entity_type)
        labels[entity_type] = assign_groups(
            degree(n, *code_arrays), spec.ratios, int(child_seeds[i])
        )
    return labels


def _context_axis(
    records: PredictionRecords, schema: TaskSchema, entity_type: str, label: IntArray
) -> IntArray:
    """Context tier: agreement across all context columns of ``entity_type``."""
    names = [n for n in schema.context_names if schema.roles[n].entity_type == entity_type]
    combined = label[records.codes(names[0])]
    for name in names[1:]:
        other = label[records.codes(name)]
        combined = np.where(combined == other, combined, _INCONSISTENT).astype(np.int64)
    return combined
