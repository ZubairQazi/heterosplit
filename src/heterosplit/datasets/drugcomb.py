"""DrugComb adapter: map the drug--drug--cell-line synergy corpus onto records.

`DrugComb <https://drugcomb.org>`_ is an integrative drug-combination screening portal.
Its summary table has one row per (drug_row, drug_col, cell_line) block with several
synergy metrics. This module maps those columns onto
:class:`~heterosplit.records.PredictionRecords` — a self-relation ``(drug, synergy,
drug)`` with a ``cell_line`` context and a binarized synergy label — filtering out
mono-therapy rows (``drug_col`` null) and rows with a missing cell line or synergy score.

Drug pairs are **unordered**, so split with ``undirected_pairs=True``.

The full summary (``summary_v_1_5.csv``, ~1.4 GB, CC-BY-4.0) is not embedded here; download
it once with :func:`download_drugcomb_summary` (or from the portal) and point
:func:`load_drugcomb_csv` at the file. Cite: Zheng et al., *DrugComb update*, NAR 2021,
doi:10.1093/nar/gkab438.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..errors import SchemaError
from ..records import PredictionRecords
from ..schema import EntityRole, TaskSchema

__all__ = [
    "DRUGCOMB_SUMMARY_URL",
    "SYNERGY_METRICS",
    "download_drugcomb_summary",
    "drugcomb_schema",
    "load_drugcomb_csv",
    "records_from_drugcomb",
]

#: Direct download URL for the DrugComb v1.5 summary table on Zenodo (~1.4 GB).
DRUGCOMB_SUMMARY_URL = "https://zenodo.org/api/records/15235991/files/summary_v_1_5.csv/content"

#: Synergy metrics available in the summary table.
SYNERGY_METRICS = ("synergy_zip", "synergy_bliss", "synergy_loewe", "synergy_hsa")

_NULL_STRINGS = frozenset({"", "null", "na", "nan", "none", "\\n"})


def drugcomb_schema() -> TaskSchema:
    """The ``(drug, synergy, drug)`` + ``cell_line`` context schema DrugComb maps to."""
    return TaskSchema(
        ("drug", "synergy", "drug"),
        {
            "drug_row": EntityRole.source("drug"),
            "drug_col": EntityRole.destination("drug"),
            "cell_line": EntityRole.context("cell_line"),
        },
    )


def _column(table: Any, name: str) -> Any:
    try:
        return table[name]
    except (KeyError, TypeError, IndexError) as exc:
        raise SchemaError(f"DrugComb column {name!r} not found in table") from exc


def _is_null(values: npt.NDArray[np.generic]) -> npt.NDArray[np.bool_]:
    out = np.zeros(values.shape, dtype=bool)
    for i, value in enumerate(values.tolist()):
        out[i] = value is None or (
            isinstance(value, str) and value.strip().lower() in _NULL_STRINGS
        )
    return out


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def records_from_drugcomb(
    table: Any,
    *,
    drug_row_col: str = "drug_row",
    drug_col_col: str = "drug_col",
    cell_line_col: str = "cell_line_name",
    synergy_metric: str = "synergy_loewe",
    synergy_threshold: float = 0.0,
    with_label: bool = True,
) -> PredictionRecords:
    """Map a DrugComb-formatted column store to :class:`PredictionRecords`.

    Rows are dropped when they are mono-therapy (``drug_col`` null), miss a drug or cell
    line, or (when ``with_label``) have a non-finite synergy score. The label is
    ``"synergistic"`` when the chosen synergy metric exceeds ``synergy_threshold``, else
    ``"antagonistic"``.
    """
    drug_row = np.asarray(_column(table, drug_row_col), dtype=object)
    drug_col = np.asarray(_column(table, drug_col_col), dtype=object)
    cell_line = np.asarray(_column(table, cell_line_col), dtype=object)
    synergy = np.array([_to_float(v) for v in _column(table, synergy_metric)], dtype=float)

    if not (drug_row.shape == drug_col.shape == cell_line.shape == synergy.shape):
        raise SchemaError("DrugComb columns have inconsistent lengths")

    keep = ~_is_null(drug_row) & ~_is_null(drug_col) & ~_is_null(cell_line)
    if with_label:
        keep &= np.isfinite(synergy)

    columns = {
        "drug_row": drug_row[keep],
        "drug_col": drug_col[keep],
        "cell_line": cell_line[keep],
    }
    labels = None
    if with_label:
        labels = np.where(synergy[keep] > synergy_threshold, "synergistic", "antagonistic")
    return PredictionRecords.from_columns(drugcomb_schema(), columns, labels=labels)


def load_drugcomb_csv(
    path: str | Path,
    *,
    synergy_metric: str = "synergy_loewe",
    max_rows: int | None = None,
    synergy_threshold: float = 0.0,
    with_label: bool = True,
) -> PredictionRecords:
    """Stream a DrugComb summary CSV, reading only the columns needed, into records.

    Args:
        path: Path to a downloaded DrugComb summary CSV.
        synergy_metric: Which synergy column to use as the label source.
        max_rows: Read at most this many data rows (handy for sampling the 1.4 GB file).
        synergy_threshold: Threshold above which a pair is labelled synergistic.
        with_label: Attach the binarized synergy label.
    """
    needed = ["drug_row", "drug_col", "cell_line_name", synergy_metric]
    collected: dict[str, list[str]] = {name: [] for name in needed}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise SchemaError(f"{path} is empty")
        try:
            indices = {name: header.index(name) for name in needed}
        except ValueError as exc:
            raise SchemaError(f"DrugComb CSV missing expected column: {exc}") from exc
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            for name, index in indices.items():
                collected[name].append(row[index])

    return records_from_drugcomb(
        collected,
        synergy_metric=synergy_metric,
        synergy_threshold=synergy_threshold,
        with_label=with_label,
    )


def download_drugcomb_summary(
    dest: str | Path,
    *,
    url: str = DRUGCOMB_SUMMARY_URL,
    chunk_size: int = 1 << 20,
) -> Path:
    """Download the DrugComb summary table to ``dest`` (streamed).

    The file is ~1.4 GB. This performs a network request and is intentionally never
    exercised by the test suite. Data is CC-BY-4.0 — cite the DrugComb paper.
    """
    import urllib.request  # local import: keep the module import-time network-free

    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
    return destination
