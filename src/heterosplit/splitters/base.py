"""Splitter base class and helpers shared across regimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ..records import PredictionRecords
from ..result import SplitResult
from ..spec import Regime, SplitSpec
from .assignment import assign_groups, refine_assignment

__all__ = [
    "Splitter",
    "degree",
    "empty_split_warnings",
    "ensure_compatible",
    "ratio_deviation_warnings",
    "resolve_strata",
    "split_by_groups",
]

IntArray = npt.NDArray[np.int64]

#: Report a warning when a split's achieved record ratio deviates from the request by
#: more than this (entity-disjoint / joint ratios are inherently approximate).
_RATIO_DEVIATION_WARN = 0.15

#: Skip the (super-linear) local-search refinement above this many groups so large
#: benchmark runs stay fast; the greedy assignment alone is already well-balanced.
REFINE_MAX_GROUPS = 100_000


class Splitter(ABC):
    """Base class for split-policy implementations.

    A splitter maps validated :class:`PredictionRecords` and a :class:`SplitSpec` to a
    :class:`SplitResult`. Subclasses declare the :attr:`regime` they implement.
    """

    regime: ClassVar[Regime]

    @abstractmethod
    def split(self, records: PredictionRecords, spec: SplitSpec) -> SplitResult:
        """Assign every record in ``records`` to a split according to ``spec``."""
        raise NotImplementedError


def ensure_compatible(records: PredictionRecords, spec: SplitSpec) -> None:
    """Check that ``records`` can be split under ``spec``.

    The spec's role names must exist as columns in the records and refer to the same
    entity type, and the supervision edge must match. This guards against pairing a
    spec with records built from a different schema.
    """
    if tuple(records.schema.supervision_edge) != tuple(spec.supervision_edge):
        raise ValueError(
            f"records supervision edge {records.schema.supervision_edge} does not match "
            f"spec {spec.supervision_edge}"
        )
    for name, role in spec.roles.items():
        if name not in records.columns:
            raise ValueError(f"spec role {name!r} has no matching column in records")
        got = records.schema.roles[name].entity_type
        if got != role.entity_type:
            raise ValueError(
                f"role {name!r} entity type mismatch: records={got!r} spec={role.entity_type!r}"
            )


def resolve_strata(
    records: PredictionRecords,
    spec: SplitSpec,
    group_ids: IntArray,
    n_groups: int,
) -> tuple[IntArray | None, list[str]]:
    """Compute a per-group stratum label for ``spec.stratify_by`` (or ``None``).

    A group's stratum is the minimum per-record stratum value among its records, a
    deterministic representative used when a group spans multiple values. Returns the
    strata array (or ``None`` when stratification is disabled or infeasible) plus any
    warnings.
    """
    warnings: list[str] = []
    key = spec.stratify_by
    if key is None:
        return None, warnings

    if key == "label":
        if records.labels is None:
            warnings.append("stratify_by='label' requested but records have no labels; ignoring")
            return None, warnings
        per_record = records.labels
    else:
        per_record = records.codes(key)

    rep = np.full(n_groups, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(rep, group_ids, per_record.astype(np.int64))
    return rep, warnings


def empty_split_warnings(record_split: IntArray, split_names: tuple[str, ...]) -> list[str]:
    """Warn for any split that received no records (a likely infeasibility)."""
    warnings: list[str] = []
    for i, name in enumerate(split_names):
        if not np.any(record_split == i):
            warnings.append(f"split {name!r} received 0 records")
    return warnings


def degree(n_entities: int, *code_arrays: IntArray) -> IntArray:
    """Incident-record count per entity across the given role code arrays."""
    result = np.zeros(n_entities, dtype=np.int64)
    for codes in code_arrays:
        result += np.bincount(codes, minlength=n_entities).astype(np.int64)
    return result


def ratio_deviation_warnings(record_split: IntArray, spec: SplitSpec) -> list[str]:
    """Warn when achieved record ratios deviate materially from the request.

    Used by regimes (entity-disjoint, joint) where the requested ratios are targets over
    an entity partition and cannot be met exactly at the record level.
    """
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


def split_by_groups(
    records: PredictionRecords,
    spec: SplitSpec,
    group_ids: IntArray,
    n_groups: int,
    *,
    extra_warnings: list[str] | None = None,
) -> SplitResult:
    """Assign atomic groups to splits and expand back to a per-record result.

    This is the shared tail of every record-partition regime: resolve strata, size the
    groups, run the seeded assignment, map the group assignment back onto records, and
    collect warnings. ``group_ids`` must be dense (``0..n_groups-1``).
    """
    warnings: list[str] = list(extra_warnings or [])
    n_splits = len(spec.ratios)
    if 0 < n_groups < n_splits:
        warnings.append(
            f"only {n_groups} group(s) for {n_splits} splits; some splits must be empty"
        )
    strata, strata_warnings = resolve_strata(records, spec, group_ids, n_groups)
    warnings.extend(strata_warnings)

    sizes = np.bincount(group_ids, minlength=n_groups).astype(np.int64)
    group_split = assign_groups(sizes, spec.ratios, spec.seed, strata=strata)

    # When the user has not requested hard stratification, softly balance the label
    # distribution across splits via bounded, size-preserving local search.
    if strata is None and records.labels is not None and 0 < n_groups <= REFINE_MAX_GROUPS:
        group_value_counts = np.zeros((n_groups, records.n_labels), dtype=np.float64)
        np.add.at(group_value_counts, (group_ids, records.labels), 1.0)
        group_split = refine_assignment(group_split, sizes, group_value_counts, spec.ratios)

    record_split = group_split[group_ids] if n_groups else np.empty(0, dtype=np.int64)

    warnings.extend(empty_split_warnings(record_split, spec.split_names))
    return SplitResult(spec=spec, records=records, record_split=record_split, warnings=warnings)
