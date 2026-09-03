"""End-to-end DrugComb drug--drug--cell-line synergy cold-start split.

With real data (downloaded once, ~1.4 GB, CC-BY-4.0)::

    # from heterosplit.datasets.drugcomb import download_drugcomb_summary
    # download_drugcomb_summary("drugcomb.csv")
    HETEROSPLIT_DRUGCOMB_CSV=drugcomb.csv uv run python examples/drugcomb.py

Offline (a small DrugComb-shaped sample so the example always runs)::

    uv run python examples/drugcomb.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from heterosplit import SplitSpec, split_records
from heterosplit.datasets.drugcomb import load_drugcomb_csv, records_from_drugcomb
from heterosplit.records import PredictionRecords
from heterosplit.report import to_markdown


def _sample_drugcomb_table(n_rows: int = 3000, seed: int = 0) -> dict[str, np.ndarray]:
    """A small table with DrugComb's columns, including rows the loader should drop."""
    rng = np.random.default_rng(seed)
    drugs = np.array([f"Drug{i}" for i in range(80)])
    cells = np.array([f"CELL{i}" for i in range(25)])
    drug_col = rng.choice(drugs, n_rows).astype(object)
    drug_col[rng.random(n_rows) < 0.1] = "NULL"  # mono-therapy rows -> dropped
    cell = rng.choice(cells, n_rows).astype(object)
    cell[rng.random(n_rows) < 0.05] = ""  # missing cell line -> dropped
    return {
        "drug_row": rng.choice(drugs, n_rows).astype(object),
        "drug_col": drug_col,
        "cell_line_name": cell,
        "synergy_loewe": rng.normal(0.0, 10.0, n_rows),
    }


def _load_records(csv_path: str | None, max_rows: int) -> PredictionRecords:
    if csv_path and Path(csv_path).exists():
        print(f"Loading real DrugComb data from {csv_path} (max_rows={max_rows}) ...")
        return load_drugcomb_csv(csv_path, max_rows=max_rows)
    print("No DrugComb CSV found; using a small DrugComb-shaped sample.")
    print("Download real data once via heterosplit.datasets.drugcomb.download_drugcomb_summary().")
    return records_from_drugcomb(_sample_drugcomb_table())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=os.environ.get("HETEROSPLIT_DRUGCOMB_CSV"))
    parser.add_argument("--max-rows", type=int, default=200_000)
    args = parser.parse_args()

    records = _load_records(args.csv, args.max_rows)
    print(
        f"Records: {records.n_records} | drugs: {records.n_entities('drug')} | "
        f"cell lines: {records.n_entities('cell_line')}"
    )

    # Joint cold-start over unordered drug pairs: a test triple needs an unseen drug pair
    # endpoint AND an unseen cell line.
    spec = SplitSpec(
        supervision_edge=("drug", "synergy", "drug"),
        roles=dict(records.schema.roles),
        regime="joint_cold_start",
        holdout={"drug": "either", "cell_line": "all"},
        ratios=(0.8, 0.1, 0.1),
        undirected_pairs=True,
        seed=42,
    )
    result = split_records(records, spec)
    result.audit.raise_for_leakage()
    print(to_markdown(result))


if __name__ == "__main__":
    main()
