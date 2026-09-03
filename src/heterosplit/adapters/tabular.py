"""Tabular adapter: build records from a dict or DataFrame-like table.

This has no third-party dependencies — it duck-types on ``table[column]`` so plain
dicts of arrays, ``pandas.DataFrame``, and similar column stores all work.
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt

from ..records import PredictionRecords
from ..schema import TaskSchema

__all__ = ["records_from_table"]

# A column value may be given directly (an array) or by name (a column in the table).
ColumnRef = str | npt.ArrayLike | None


def _resolve(table: Any, ref: ColumnRef) -> npt.ArrayLike | None:
    if ref is None:
        return None
    if isinstance(ref, str):
        return table[ref]  # type: ignore[no-any-return]
    return ref


def records_from_table(
    schema: TaskSchema,
    table: Any,
    *,
    label: ColumnRef = None,
    timestamp: ColumnRef = None,
) -> PredictionRecords:
    """Build :class:`PredictionRecords` from a column store.

    Each role name in ``schema.roles`` is read as ``table[name]``. ``label`` and
    ``timestamp`` may each be a column name or an array (or ``None``).
    """
    columns = {name: table[name] for name in schema.roles}
    return PredictionRecords.from_columns(
        schema,
        columns,
        labels=_resolve(table, label),
        timestamps=_resolve(table, timestamp),
    )
