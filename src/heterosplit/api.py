"""Top-level entry points for splitting prediction records."""

from __future__ import annotations

from .records import PredictionRecords
from .result import SplitResult
from .spec import SplitSpec
from .splitters import get_splitter

__all__ = ["split_records"]


def split_records(records: PredictionRecords, spec: SplitSpec) -> SplitResult:
    """Split ``records`` according to ``spec`` and return a :class:`SplitResult`.

    The regime named by the spec selects the splitter. Framework adapters (e.g. the PyG
    ``HeteroData`` adapter) convert their input to :class:`PredictionRecords` and call
    this function.
    """
    splitter = get_splitter(spec.regime)
    return splitter.split(records, spec)
