"""HeteroSplit: leakage-safe entity-disjoint splits for heterogeneous link prediction.

HeteroSplit constructs, validates, and reports cold-start / inductive splits for
heterogeneous link-prediction datasets. It focuses narrowly on *split semantics*
and *leakage auditing*, and is designed to complement PyTorch Geometric rather
than replace its loaders, samplers, or training stack.

The correctness core is pure Python + NumPy. PyTorch Geometric integration is an
optional extra (``pip install heterosplit[pyg]``).
"""

from __future__ import annotations

from .errors import (
    HeteroSplitError,
    InfeasibleSplitError,
    LeakageError,
    SchemaError,
    SpecError,
)
from .records import PredictionRecords
from .schema import EdgeType, EntityRole, RelationMeta, RoleKind, TaskSchema
from .spec import Regime, SplitSpec
from .synthetic import SyntheticDataset, make_synthetic_dataset

__version__ = "0.0.1.dev0"

__all__ = [
    "EdgeType",
    "EntityRole",
    "HeteroSplitError",
    "InfeasibleSplitError",
    "LeakageError",
    "PredictionRecords",
    "Regime",
    "RelationMeta",
    "RoleKind",
    "SchemaError",
    "SpecError",
    "SplitSpec",
    "SyntheticDataset",
    "TaskSchema",
    "__version__",
    "make_synthetic_dataset",
]
