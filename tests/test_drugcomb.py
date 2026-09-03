"""Tests for the DrugComb adapter (no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from heterosplit import SplitSpec, split_records
from heterosplit.datasets.drugcomb import (
    DRUGCOMB_SUMMARY_URL,
    drugcomb_schema,
    load_drugcomb_csv,
    records_from_drugcomb,
)


class TestRecordsFromDrugcomb:
    def test_filters_and_labels(self) -> None:
        table = {
            "drug_row": ["A", "A", "B", "C", "D"],
            "drug_col": ["B", "NULL", "C", "D", "A"],  # row 1 is mono-therapy -> dropped
            "cell_line_name": ["x", "x", "", "y", "z"],  # row 2 has no cell line -> dropped
            "synergy_loewe": ["5.0", "1.0", "-3.0", "nan", "2.0"],  # row 3 nan synergy -> dropped
        }
        records = records_from_drugcomb(table)
        # kept: row 0 (A,B,x,+5) and row 4 (D,A,z,+2)
        assert records.n_records == 2
        assert records.has_labels
        labels = records.label_codebook.values.tolist()  # type: ignore[union-attr]
        assert "synergistic" in labels
        assert records.schema == drugcomb_schema()
        assert set(records.raw_ids("cell_line").tolist()) == {"x", "z"}

    def test_synergy_metric_and_threshold(self) -> None:
        table = {
            "drug_row": ["A", "B"],
            "drug_col": ["B", "A"],
            "cell_line_name": ["x", "y"],
            "synergy_zip": ["1.0", "9.0"],
        }
        records = records_from_drugcomb(table, synergy_metric="synergy_zip", synergy_threshold=5.0)
        # only the second pair exceeds the threshold -> synergistic
        decoded = records.label_codebook.decode(records.labels).tolist()  # type: ignore[union-attr]
        assert decoded == ["antagonistic", "synergistic"]

    def test_without_label(self) -> None:
        table = {
            "drug_row": ["A"],
            "drug_col": ["B"],
            "cell_line_name": ["x"],
            "synergy_loewe": ["nan"],  # kept because with_label=False skips the finite check
        }
        records = records_from_drugcomb(table, with_label=False)
        assert records.n_records == 1
        assert not records.has_labels


class TestSplitDrugcomb:
    def test_cold_start_split_is_clean(self) -> None:
        rng = np.random.default_rng(0)
        n = 4000
        drugs = np.array([f"D{i}" for i in range(60)])
        table = {
            "drug_row": rng.choice(drugs, n),
            "drug_col": rng.choice(drugs, n),
            "cell_line_name": rng.choice([f"C{i}" for i in range(20)], n),
            "synergy_loewe": rng.normal(0, 10, n),
        }
        records = records_from_drugcomb(table)
        spec = SplitSpec(
            supervision_edge=("drug", "synergy", "drug"),
            roles=dict(records.schema.roles),
            regime="joint_cold_start",
            holdout={"drug": "either", "cell_line": "all"},
            undirected_pairs=True,
            seed=1,
        )
        result = split_records(records, spec)
        assert result.covers_all_records()
        result.audit.raise_for_leakage()  # no leakage


class TestLoadDrugcombCsv:
    def test_reads_named_columns_only(self, tmp_path: Path) -> None:
        # extra columns present; loader must select by name, not position
        csv_path = tmp_path / "drugcomb.csv"
        csv_path.write_text(
            "block_id,drug_row,drug_col,cell_line_name,synergy_zip,synergy_loewe,study\n"
            "1,A,B,x,3.0,5.0,S1\n"
            "2,A,NULL,x,0.1,1.0,S1\n"  # mono-therapy -> dropped
            "3,C,D,y,-2.0,-4.0,S2\n",
            encoding="utf-8",
        )
        records = load_drugcomb_csv(csv_path, synergy_metric="synergy_loewe")
        assert records.n_records == 2
        assert set(records.raw_ids("drug_row").tolist()) == {"A", "C"}

    def test_max_rows(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "d.csv"
        rows = "\n".join(f"{i},A{i},B{i},x,1.0" for i in range(100))
        csv_path.write_text(
            "block_id,drug_row,drug_col,cell_line_name,synergy_loewe\n" + rows, "utf-8"
        )
        records = load_drugcomb_csv(csv_path, max_rows=10)
        assert records.n_records == 10


def test_summary_url_is_zenodo() -> None:
    assert DRUGCOMB_SUMMARY_URL.startswith("https://zenodo.org/")
