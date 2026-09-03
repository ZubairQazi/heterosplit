"""PyTorch Geometric ``HeteroData`` adapter (optional ``[pyg]`` extra).

Converts the supervision edge of a :class:`HeteroData` into
:class:`~heterosplit.records.PredictionRecords`, runs a split, and reconstructs
per-split ``HeteroData`` objects in PyG's link-prediction convention
(``edge_index`` = leakage-safe message-passing graph, ``edge_label_index`` /
``edge_label`` = supervision). The correctness core never imports torch; importing
*this* module without the extra installed raises a clear, actionable error.

Since a heterogeneous observation may carry *context* (e.g. a cell line) that
``HeteroData`` has no standard slot for, context and label columns are supplied
explicitly (as arrays or as keys on the supervision edge store).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ..api import split_records
from ..records import PredictionRecords
from ..result import SplitResult
from ..spec import SplitSpec

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

__all__ = [
    "records_from_heterodata",
    "split_heterodata",
    "to_link_split",
]


def _require_pyg() -> tuple[Any, Any]:
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "PyG integration requires torch and torch-geometric; "
            "install with `pip install heterosplit[pyg]`"
        ) from exc
    return torch, HeteroData


def _edge_store(data: Any, edge_type: tuple[str, str, str]) -> Any:
    if edge_type not in data.edge_types:
        raise KeyError(
            f"supervision edge {edge_type} not found in HeteroData; "
            f"available edge types: {list(data.edge_types)}"
        )
    return data[edge_type]


def _as_numpy(tensor: Any) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().numpy())


def records_from_heterodata(
    data: HeteroData,
    spec: SplitSpec,
    *,
    context: Mapping[str, Any] | Any | None = None,
    label: Any | None = None,
    timestamp: Any | None = None,
) -> PredictionRecords:
    """Extract :class:`PredictionRecords` from a ``HeteroData`` supervision edge.

    Args:
        data: The heterogeneous graph.
        spec: The split spec; its ``supervision_edge`` selects the edge store.
        context: Per-edge context. If the schema has one context role, an array; if
            several, a mapping of role name to array. A string is read as a key on the
            supervision edge store (e.g. ``"cell_line"`` -> ``store["cell_line"]``).
        label: Per-edge relation/label array, or a key on the edge store; ``None`` reads
            ``store.edge_label`` when present.
        timestamp: Per-edge timestamp array or edge-store key.

    Returns:
        Records whose source/destination raw ids are the endpoint node indices.
    """
    _require_pyg()
    schema = spec.schema
    store = _edge_store(data, spec.supervision_edge)
    edge_index = _as_numpy(store.edge_index)
    columns: dict[str, Any] = {
        schema.source_name: edge_index[0],
        schema.destination_name: edge_index[1],
    }

    context_names = schema.context_names
    if context_names:
        resolved = _resolve_context(store, context, context_names)
        columns.update(resolved)

    labels = _resolve_edge_field(store, label, default_key="edge_label")
    timestamps = _resolve_edge_field(store, timestamp, default_key=None)
    return PredictionRecords.from_columns(schema, columns, labels=labels, timestamps=timestamps)


def _resolve_context(
    store: Any, context: Mapping[str, Any] | Any | None, context_names: list[str]
) -> dict[str, Any]:
    if context is None:
        raise ValueError(f"schema has context roles {context_names} but no `context` was provided")
    if isinstance(context, Mapping):
        missing = set(context_names) - set(context)
        if missing:
            raise ValueError(f"context missing entries for {sorted(missing)}")
        return {name: _resolve_edge_field(store, context[name]) for name in context_names}
    if len(context_names) != 1:
        raise ValueError(
            f"schema has {len(context_names)} context roles; pass a mapping of role -> values"
        )
    return {context_names[0]: _resolve_edge_field(store, context)}


def _resolve_edge_field(store: Any, ref: Any, *, default_key: str | None = None) -> Any:
    if ref is None:
        if default_key is not None and default_key in store:
            return _as_numpy(store[default_key])
        return None
    if isinstance(ref, str):
        return _as_numpy(store[ref])
    if hasattr(ref, "detach"):
        return _as_numpy(ref)
    return ref


def split_heterodata(
    data: HeteroData,
    spec: SplitSpec,
    *,
    context: Mapping[str, Any] | Any | None = None,
    label: Any | None = None,
    timestamp: Any | None = None,
) -> SplitResult:
    """Convenience: build records from ``data`` and split them under ``spec``."""
    records = records_from_heterodata(data, spec, context=context, label=label, timestamp=timestamp)
    return split_records(records, spec)


def to_link_split(result: SplitResult, data: HeteroData) -> dict[str, HeteroData]:
    """Build per-split ``HeteroData`` in PyG's link-prediction convention.

    Every split shares the leakage-safe *training* message-passing graph as
    ``edge_index`` for the supervision edge type; each split's supervision edges become
    ``edge_label_index`` (and ``edge_label`` when the records carry labels). Node stores
    and other edge types are copied from ``data`` unchanged.
    """
    torch, _ = _require_pyg()
    edge_type = result.spec.supervision_edge
    message_passing = torch.as_tensor(result.message_passing_edge_index(), dtype=torch.long)

    splits: dict[str, HeteroData] = {}
    for split in result.split_names:
        clone = data.clone()
        store = clone[edge_type]
        store.edge_index = message_passing
        store.edge_label_index = torch.as_tensor(
            result.supervision_edge_index(split), dtype=torch.long
        )
        if result.records.labels is not None:
            labels = result.records.labels[result.indices(split)]
            store.edge_label = torch.as_tensor(labels, dtype=torch.long)
        splits[split] = clone
    return splits
