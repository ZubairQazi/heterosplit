"""Tests for the tabular adapter."""

from __future__ import annotations

import numpy as np

from heterosplit import EntityRole, TaskSchema
from heterosplit.adapters.tabular import records_from_table


def _schema() -> TaskSchema:
    return TaskSchema(
        ("drug", "synergy", "drug"),
        {
            "left": EntityRole.source("drug"),
            "right": EntityRole.destination("drug"),
            "cell": EntityRole.context("cell_line"),
        },
    )


class TestRecordsFromTable:
    def test_from_dict(self) -> None:
        table = {
            "left": ["A", "B", "C"],
            "right": ["B", "C", "A"],
            "cell": ["x", "x", "y"],
            "score": [1, 0, 1],
        }
        records = records_from_table(_schema(), table, label="score")
        assert records.n_records == 3
        assert records.n_entities("drug") == 3
        assert records.has_labels
        np.testing.assert_array_equal(records.raw_ids("left"), ["A", "B", "C"])

    def test_label_and_timestamp_as_arrays(self) -> None:
        table = {"left": ["A"], "right": ["B"], "cell": ["x"]}
        records = records_from_table(
            _schema(), table, label=np.array([1]), timestamp=np.array([2020])
        )
        assert records.has_labels
        assert records.timestamps is not None

    def test_dataframe_like(self) -> None:
        class Frame:
            def __init__(self, data: dict[str, list[object]]) -> None:
                self._data = data

            def __getitem__(self, key: str) -> list[object]:
                return self._data[key]

        frame = Frame({"left": ["A", "B"], "right": ["B", "A"], "cell": ["x", "y"], "y": [0, 1]})
        records = records_from_table(_schema(), frame, label="y")
        assert records.n_records == 2
