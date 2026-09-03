"""Splitter base class and helpers shared across regimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ..records import PredictionRecords
from ..result import SplitResult
from ..spec import Regime, SplitSpec

__all__ = ["Splitter", "empty_split_warnings", "ensure_compatible", "resolve_strata"]

IntArray = npt.NDArray[np.int64]


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
