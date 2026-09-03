"""Adapters converting external representations to :class:`PredictionRecords`.

* :mod:`heterosplit.adapters.tabular` — dicts / DataFrame-like tables (no dependencies).
* :mod:`heterosplit.adapters.pyg` — PyTorch Geometric ``HeteroData`` (optional ``[pyg]``
  extra; importing the module without torch installed raises a clear error).
"""

from __future__ import annotations

from .tabular import records_from_table

__all__ = ["records_from_table"]
