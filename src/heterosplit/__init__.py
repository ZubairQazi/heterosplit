"""HeteroSplit: leakage-safe entity-disjoint splits for heterogeneous link prediction.

HeteroSplit constructs, validates, and reports cold-start / inductive splits for
heterogeneous link-prediction datasets. It focuses narrowly on *split semantics*
and *leakage auditing*, and is designed to complement PyTorch Geometric rather
than replace its loaders, samplers, or training stack.

The correctness core is pure Python + NumPy. PyTorch Geometric integration is an
optional extra (``pip install heterosplit[pyg]``).
"""

from __future__ import annotations

# Defined before the sub-module imports below: manifest.py reads it via
# ``from . import __version__`` during package initialization.
__version__ = "0.1.0"

from .api import split_records
from .audit import AuditFinding, AuditReport, Severity, audit_split
from .errors import (
    HeteroSplitError,
    InfeasibleSplitError,
    LeakageError,
    SchemaError,
    SpecError,
)
from .manifest import Manifest
from .records import PredictionRecords
from .result import SplitResult
from .schema import EdgeType, EntityRole, RelationMeta, RoleKind, TaskSchema
from .spec import Regime, SplitSpec
from .synthetic import SyntheticDataset, make_synthetic_dataset

__all__ = [
    "AuditFinding",
    "AuditReport",
    "EdgeType",
    "EntityRole",
    "HeteroSplitError",
    "InfeasibleSplitError",
    "LeakageError",
    "Manifest",
    "PredictionRecords",
    "Regime",
    "RelationMeta",
    "RoleKind",
    "SchemaError",
    "Severity",
    "SpecError",
    "SplitResult",
    "SplitSpec",
    "SyntheticDataset",
    "TaskSchema",
    "__version__",
    "audit_split",
    "make_synthetic_dataset",
    "split_records",
]
