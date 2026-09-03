"""Exception hierarchy shared across HeteroSplit.

All library-raised exceptions derive from :class:`HeteroSplitError` so callers can
catch everything from HeteroSplit with a single ``except`` clause while still being
able to distinguish the failure mode.
"""

from __future__ import annotations

__all__ = [
    "HeteroSplitError",
    "InfeasibleSplitError",
    "LeakageError",
    "SchemaError",
    "SpecError",
]


class HeteroSplitError(Exception):
    """Base class for every exception raised by HeteroSplit."""


class SchemaError(HeteroSplitError):
    """Raised when a task schema, entity role, or relation metadata is invalid."""


class SpecError(HeteroSplitError):
    """Raised when a :class:`~heterosplit.spec.SplitSpec` is internally inconsistent."""


class InfeasibleSplitError(HeteroSplitError):
    """Raised when a split cannot satisfy the requested constraints.

    Carries diagnostics describing *why* the constraints could not be met rather
    than silently relaxing them.
    """


class LeakageError(HeteroSplitError):
    """Raised by :meth:`AuditReport.raise_for_leakage` when leakage is detected."""
