"""The output of a split: per-record assignment plus derived views.

A :class:`SplitResult` is deliberately index-centric — it stores a single integer per
record naming its split (or marking it excluded) and derives everything else on
demand. Message-passing edges, audits, and the serializable manifest are layered on
top in later modules; keeping the core small makes the invariants easy to state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .records import PredictionRecords
from .spec import SplitSpec

if TYPE_CHECKING:
    from .audit import AuditReport
    from .manifest import Manifest

__all__ = ["EXCLUDED", "SplitResult"]

IntArray = npt.NDArray[np.int64]

#: Sentinel value in ``record_split`` for records that belong to no split (e.g. the
#: "bridge" records dropped by both-entity cold-start).
EXCLUDED = -1


@dataclass
class SplitResult:
    """Assignment of every record to a split, with convenience accessors.

    Attributes:
        spec: The :class:`SplitSpec` that produced this result.
        records: The records that were split.
        record_split: Length-``n`` array; entry ``i`` is the split index of record
            ``i`` (``0..k-1`` following :pyattr:`SplitSpec.split_names`) or
            :data:`EXCLUDED`.
        warnings: Non-fatal diagnostics gathered while splitting.
    """

    spec: SplitSpec
    records: PredictionRecords
    record_split: IntArray
    warnings: list[str] = field(default_factory=list)
    _audit: AuditReport | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def split_names(self) -> tuple[str, ...]:
        return self.spec.split_names

    def _split_index(self, split: str) -> int:
        try:
            return self.split_names.index(split)
        except ValueError:
            raise KeyError(f"unknown split {split!r}; have {self.split_names}") from None

    def indices(self, split: str) -> IntArray:
        """Record indices assigned to ``split``, in ascending order."""
        return np.flatnonzero(self.record_split == self._split_index(split)).astype(np.int64)

    @property
    def train_indices(self) -> IntArray:
        return self.indices("train")

    @property
    def val_indices(self) -> IntArray:
        return self.indices("val")

    @property
    def test_indices(self) -> IntArray:
        return self.indices("test")

    @property
    def excluded_indices(self) -> IntArray:
        return np.flatnonzero(self.record_split == EXCLUDED).astype(np.int64)

    @property
    def counts(self) -> dict[str, int]:
        """Number of records in each split (excludes :data:`EXCLUDED`)."""
        return {
            name: int(np.count_nonzero(self.record_split == i))
            for i, name in enumerate(self.split_names)
        }

    @property
    def n_excluded(self) -> int:
        return int(np.count_nonzero(self.record_split == EXCLUDED))

    def achieved_ratios(self) -> dict[str, float]:
        """Fraction of *assigned* records in each split (excludes excluded records)."""
        assigned = self.records.n_records - self.n_excluded
        if assigned == 0:
            return dict.fromkeys(self.split_names, 0.0)
        return {name: count / assigned for name, count in self.counts.items()}

    def split_codes(self, role_name: str, split: str) -> IntArray:
        """Entity codes of ``role_name`` for the records in ``split``."""
        return self.records.codes(role_name)[self.indices(split)]

    def covers_all_records(self) -> bool:
        """Every record is assigned to exactly one split or explicitly excluded."""
        return bool(np.all(self.record_split >= EXCLUDED)) and self.record_split.shape == (
            self.records.n_records,
        )

    def supervision_edge_index(self, split: str) -> IntArray:
        """The ``(2, E)`` source/destination code edge index of ``split``'s records."""
        from .message_passing import supervision_edge_index

        return supervision_edge_index(self, split)

    def message_passing_edge_index(self, **kwargs: Any) -> IntArray:
        """Leakage-safe training message-passing edge index (see :mod:`message_passing`)."""
        from .message_passing import message_passing_edge_index

        return message_passing_edge_index(self, **kwargs)

    @property
    def audit(self) -> AuditReport:
        """The leakage :class:`~heterosplit.audit.AuditReport` for this split (cached).

        Call ``result.audit.raise_for_leakage()`` to fail on any contract violation.
        """
        if self._audit is None:
            from .audit import audit_split

            self._audit = audit_split(self)
        return self._audit

    @property
    def manifest(self) -> Manifest:
        """A deterministic :class:`~heterosplit.manifest.Manifest` for this result."""
        from .manifest import Manifest

        return Manifest.from_result(self)

    def build_manifest(self, **kwargs: Any) -> Manifest:
        """Build a manifest, optionally attaching ``audit``/``distributions``/``measurements``."""
        from .manifest import Manifest

        return Manifest.from_result(self, **kwargs)
