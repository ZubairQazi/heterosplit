"""Entity-disjoint (endpoint cold-start) splitters.

Two families live here:

* **Record-partition** regimes — *source* and *destination* cold-start. Each record is
  grouped by the entity code of the relevant endpoint and assigned via the shared
  :func:`split_by_groups`. Because every entity lands in exactly one split *as that
  role*, the endpoint intersection is empty by construction.

* **Entity-partition** regimes — *either* and *both* endpoint cold-start. Each endpoint
  entity is labeled train/val/test (weighted by its incident-record degree), then each
  record's split is derived from its two endpoint labels. With split indices ordered
  ``train < val < test``:

  - *either* = element-wise ``max`` of the endpoint labels — a record is training only
    if both endpoints are training, otherwise it flows to the most-held-out tier. No
    records are excluded.
  - *both* = the shared label when both endpoints agree, else :data:`EXCLUDED` — the
    "bridge" records whose endpoints straddle splits are dropped, and their count is
    reported.

Record ratios are only approximate for the entity-partition regimes (test records scale
super-linearly in the held-out entity fraction), so achieved ratios and large
deviations are surfaced rather than silently corrected.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ..records import PredictionRecords
from ..result import EXCLUDED, SplitResult
from ..spec import Regime, SplitSpec
from .assignment import assign_groups
from .base import Splitter, empty_split_warnings, ensure_compatible, split_by_groups

IntArray = npt.NDArray[np.int64]

_RATIO_DEVIATION_WARN = 0.15


class SourceColdStartSplitter(Splitter):
    regime: ClassVar[Regime] = Regime.SOURCE

    def split(self, records: PredictionRecords, spec: SplitSpec) -> SplitResult:
        ensure_compatible(records, spec)
        n_groups = records.n_entities(spec.schema.source_type)
        return split_by_groups(records, spec, records.source_codes, n_groups)


class DestinationColdStartSplitter(Splitter):
    regime: ClassVar[Regime] = Regime.DESTINATION

    def split(self, records: PredictionRecords, spec: SplitSpec) -> SplitResult:
        ensure_compatible(records, spec)
        n_groups = records.n_entities(spec.schema.destination_type)
        return split_by_groups(records, spec, records.destination_codes, n_groups)


class EitherColdStartSplitter(Splitter):
    regime: ClassVar[Regime] = Regime.EITHER

    def split(self, records: PredictionRecords, spec: SplitSpec) -> SplitResult:
        ensure_compatible(records, spec)
        src_label, dst_label, warnings = _endpoint_labels(records, spec)
        record_split = np.maximum(src_label, dst_label)
        warnings.extend(empty_split_warnings(record_split, spec.split_names))
        warnings.extend(_ratio_warnings(record_split, spec))
        return SplitResult(spec=spec, records=records, record_split=record_split, warnings=warnings)


class BothColdStartSplitter(Splitter):
    regime: ClassVar[Regime] = Regime.BOTH

    def split(self, records: PredictionRecords, spec: SplitSpec) -> SplitResult:
        ensure_compatible(records, spec)
        src_label, dst_label, warnings = _endpoint_labels(records, spec)
        record_split = np.where(src_label == dst_label, src_label, EXCLUDED).astype(np.int64)
        n_excluded = int(np.count_nonzero(record_split == EXCLUDED))
        if n_excluded:
            warnings.append(
                f"{n_excluded} bridge record(s) excluded (endpoints fell in different splits)"
            )
        warnings.extend(empty_split_warnings(record_split, spec.split_names))
        warnings.extend(_ratio_warnings(record_split, spec))
        return SplitResult(spec=spec, records=records, record_split=record_split, warnings=warnings)


# -- helpers -----------------------------------------------------------------


def _degree(n_entities: int, *code_arrays: IntArray) -> IntArray:
    degree = np.zeros(n_entities, dtype=np.int64)
    for codes in code_arrays:
        degree += np.bincount(codes, minlength=n_entities).astype(np.int64)
    return degree


def _endpoint_labels(
    records: PredictionRecords, spec: SplitSpec
) -> tuple[IntArray, IntArray, list[str]]:
    """Label each endpoint entity by split and project onto the two endpoint columns.

    For a self-relation both endpoints share one labeling; for a bipartite relation the
    source and destination types are labeled independently with derived child seeds.
    """
    warnings: list[str] = []
    if spec.stratify_by is not None:
        warnings.append(
            "stratify_by is not applied for entity-disjoint (either/both) regimes; ignoring"
        )

    schema = spec.schema
    src_codes = records.source_codes
    dst_codes = records.destination_codes

    if schema.is_self_relation:
        n = records.n_entities(schema.source_type)
        labels = assign_groups(_degree(n, src_codes, dst_codes), spec.ratios, spec.seed)
        return labels[src_codes], labels[dst_codes], warnings

    seed_s, seed_d = (int(s) for s in np.random.SeedSequence(spec.seed).generate_state(2))
    n_s = records.n_entities(schema.source_type)
    n_d = records.n_entities(schema.destination_type)
    labels_s = assign_groups(_degree(n_s, src_codes), spec.ratios, seed_s)
    labels_d = assign_groups(_degree(n_d, dst_codes), spec.ratios, seed_d)
    return labels_s[src_codes], labels_d[dst_codes], warnings


def _ratio_warnings(record_split: IntArray, spec: SplitSpec) -> list[str]:
    counts = np.array(
        [np.count_nonzero(record_split == i) for i in range(len(spec.ratios))], dtype=np.float64
    )
    total = counts.sum()
    if total == 0:
        return []
    warnings: list[str] = []
    for i, name in enumerate(spec.split_names):
        achieved = counts[i] / total
        if abs(achieved - spec.ratios[i]) > _RATIO_DEVIATION_WARN:
            warnings.append(
                f"split {name!r} record ratio {achieved:.2f} deviates from requested "
                f"{spec.ratios[i]:.2f} (inherent to entity-disjoint splitting)"
            )
    return warnings
