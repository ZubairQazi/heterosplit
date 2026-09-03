"""Tests for the normalized prediction-records table and codebooks."""

from __future__ import annotations

import numpy as np
import pytest

from heterosplit import EntityRole, SchemaError, TaskSchema
from heterosplit.records import Codebook, PredictionRecords


def synergy_schema() -> TaskSchema:
    return TaskSchema(
        ("drug", "synergy", "drug"),
        {
            "left_drug": EntityRole.source("drug"),
            "right_drug": EntityRole.destination("drug"),
            "cell_line": EntityRole.context("cell_line"),
        },
    )


class TestCodebook:
    def test_build_and_roundtrip(self) -> None:
        book, [codes] = Codebook.build(["b", "a", "c", "a"])
        # values are sorted unique
        np.testing.assert_array_equal(book.values, ["a", "b", "c"])
        np.testing.assert_array_equal(codes, [1, 0, 2, 0])
        np.testing.assert_array_equal(book.decode(codes), ["b", "a", "c", "a"])

    def test_shared_space_across_arrays(self) -> None:
        book, (left, right) = Codebook.build(["a", "b"], ["b", "c"])
        assert len(book) == 3
        # 'b' gets the same code in both columns
        assert left[1] == right[0]

    def test_encode_roundtrips_and_rejects_unknown(self) -> None:
        book, _ = Codebook.build([10, 20, 30])
        np.testing.assert_array_equal(book.encode([30, 10]), [2, 0])
        with pytest.raises(SchemaError, match="not present"):
            book.encode([99])


class TestPredictionRecords:
    def test_shared_codebook_for_same_type(self) -> None:
        # A drug appearing as left in one record and right in another gets one code.
        records = PredictionRecords.from_columns(
            synergy_schema(),
            {
                "left_drug": ["A", "B", "C"],
                "right_drug": ["B", "C", "A"],
                "cell_line": ["MCF7", "MCF7", "A549"],
            },
        )
        assert records.n_records == 3
        assert records.n_entities("drug") == 3
        assert records.n_entities("cell_line") == 2
        # 'A' is code for left[0]; 'A' also appears as right[2] with the same code.
        code_a_left = records.source_codes[0]
        code_a_right = records.destination_codes[2]
        assert code_a_left == code_a_right
        np.testing.assert_array_equal(records.raw_ids("left_drug"), ["A", "B", "C"])

    def test_labels_and_timestamps(self) -> None:
        records = PredictionRecords.from_columns(
            synergy_schema(),
            {"left_drug": ["A", "B"], "right_drug": ["B", "A"], "cell_line": ["x", "y"]},
            labels=["syn", "ant"],
            timestamps=[2020, 2021],
        )
        assert records.has_labels
        assert records.n_labels == 2
        assert records.timestamps is not None
        np.testing.assert_array_equal(records.timestamps, [2020, 2021])

    def test_context_codes(self) -> None:
        records = PredictionRecords.from_columns(
            synergy_schema(),
            {"left_drug": ["A"], "right_drug": ["B"], "cell_line": ["MCF7"]},
        )
        ctx = records.context_codes()
        assert set(ctx) == {"cell_line"}
        assert ctx["cell_line"].shape == (1,)

    def test_missing_column_raises(self) -> None:
        with pytest.raises(SchemaError, match="missing columns"):
            PredictionRecords.from_columns(
                synergy_schema(), {"left_drug": ["A"], "right_drug": ["B"]}
            )

    def test_inconsistent_length_raises(self) -> None:
        with pytest.raises(SchemaError, match="inconsistent lengths"):
            PredictionRecords.from_columns(
                synergy_schema(),
                {"left_drug": ["A", "B"], "right_drug": ["B"], "cell_line": ["x", "y"]},
            )

    def test_label_length_mismatch_raises(self) -> None:
        with pytest.raises(SchemaError, match="labels length"):
            PredictionRecords.from_columns(
                synergy_schema(),
                {"left_drug": ["A"], "right_drug": ["B"], "cell_line": ["x"]},
                labels=["a", "b"],
            )

    def test_integer_ids_supported(self) -> None:
        records = PredictionRecords.from_columns(
            synergy_schema(),
            {"left_drug": [1, 2], "right_drug": [2, 3], "cell_line": [0, 0]},
        )
        assert records.n_entities("drug") == 3
