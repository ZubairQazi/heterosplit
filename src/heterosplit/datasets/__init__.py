"""Dataset adapters for real link-prediction corpora.

These are narrow, optional adapters that map an existing dataset's columns onto
:class:`~heterosplit.records.PredictionRecords`; they deliberately do not embed a data
platform. See :mod:`heterosplit.datasets.drugcomb`.
"""

from __future__ import annotations

from .drugcomb import (
    DRUGCOMB_SUMMARY_URL,
    download_drugcomb_summary,
    load_drugcomb_csv,
    records_from_drugcomb,
)
from .movielens import (
    MOVIELENS_SMALL_URL,
    download_movielens,
    load_movielens_csv,
    records_from_movielens,
)

__all__ = [
    "DRUGCOMB_SUMMARY_URL",
    "MOVIELENS_SMALL_URL",
    "download_drugcomb_summary",
    "download_movielens",
    "load_drugcomb_csv",
    "load_movielens_csv",
    "records_from_drugcomb",
    "records_from_movielens",
]
