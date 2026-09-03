"""Regression tests for bugs found by the adversarial core review.

Each test would fail against the pre-fix code and pins the corrected behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from heterosplit import EntityRole, SchemaError, SplitSpec, TaskSchema
from heterosplit.manifest import _values_bytes, fingerprint_records
from heterosplit.records import Codebook, PredictionRecords
from heterosplit.result import SplitResult
from heterosplit.synthetic import make_synthetic_dataset


def _drug_self_schema() -> TaskSchema:
    return TaskSchema(
        ("drug", "synergy", "drug"),
        {"source": EntityRole.source("drug"), "destination": EntityRole.destination("drug")},
    )


class TestAuditValVsTest:
    def test_entity_overlap_between_val_and_test_is_detected(self) -> None:
        # 'A' appears only in val and test (never train) -> train-vs-held misses it.
        schema = _drug_self_schema()
        spec = SplitSpec(
            supervision_edge=("drug", "synergy", "drug"),
            roles=dict(schema.roles),
            regime="source_cold_start",
            ratios=(0.34, 0.33, 0.33),
        )
        records = PredictionRecords.from_columns(
            schema,
            {
                "source": ["A", "A", "B", "C", "D", "E"],
                "destination": ["p", "q", "r", "s", "t", "u"],
            },
        )
        record_split = np.array([1, 2, 0, 0, 0, 0], dtype=np.int64)  # A -> val and test
        result = SplitResult(spec=spec, records=records, record_split=record_split)
        finding = result.audit.get("source_entity_overlap")
        assert finding is not None and finding.count == 1
        assert result.audit.has_leakage


class TestBipartiteMessagePassing:
    def test_does_not_drop_legitimate_edges(self) -> None:
        # user1->item3 (train) and user3->item1 (test) are different edges across
        # independent codebooks; the train edge must survive reconstruction.
        schema = TaskSchema(
            ("user", "buys", "item"),
            {"s": EntityRole.source("user"), "d": EntityRole.destination("item")},
        )
        spec = SplitSpec(
            supervision_edge=("user", "buys", "item"),
            roles=dict(schema.roles),
            regime="random",
            ratios=(0.5, 0.5),
        )
        records = PredictionRecords.from_columns(schema, {"s": [1, 3], "d": [3, 1]})
        result = SplitResult(spec=spec, records=records, record_split=np.array([0, 1], np.int64))
        mp = result.message_passing_edge_index()
        assert mp.shape[1] == 1  # the single train edge is kept, not conflated away
        assert (int(mp[0][0]), int(mp[1][0])) == (
            int(records.source_codes[0]),
            int(records.destination_codes[0]),
        )

    def test_reconstruction_audit_clean_for_bipartite(self) -> None:
        ds = make_synthetic_dataset(
            n_records=800, self_relation=False, source_type="user", destination_type="item", seed=0
        )
        from heterosplit import split_records

        result = split_records(ds.records, ds.spec("both_cold_start"))
        assert result.audit.get("message_passing_reconstruction_leak").count == 0  # type: ignore[union-attr]


class TestSelfLoopDedup:
    def test_undirected_reverse_dedups_self_loops(self) -> None:
        schema = _drug_self_schema()
        spec = SplitSpec(
            supervision_edge=("drug", "synergy", "drug"),
            roles=dict(schema.roles),
            regime="pair_cold_start",
            ratios=(0.5, 0.5),
            undirected_pairs=True,
        )
        records = PredictionRecords.from_columns(schema, {"source": ["A"], "destination": ["A"]})
        result = SplitResult(spec=spec, records=records, record_split=np.array([0], np.int64))
        mp = result.message_passing_edge_index()
        assert mp.shape[1] == 1  # self-loop appears once, not duplicated by add_reverse


class TestFingerprintCollisions:
    def test_embedded_nul_does_not_collide(self) -> None:
        assert _values_bytes(np.array(["a\x00b"])) != _values_bytes(np.array(["a", "b"]))

    def test_int_and_string_do_not_collide(self) -> None:
        assert _values_bytes(np.array([1, 2])) != _values_bytes(np.array(["1", "2"]))

    def test_records_fingerprint_distinguishes_types(self) -> None:
        schema = _drug_self_schema()
        int_records = PredictionRecords.from_columns(
            schema, {"source": [1, 2], "destination": [2, 1]}
        )
        str_records = PredictionRecords.from_columns(
            schema, {"source": ["1", "2"], "destination": ["2", "1"]}
        )
        assert fingerprint_records(int_records) != fingerprint_records(str_records)


class TestCodebookRobustness:
    def test_mixed_types_raise_schema_error(self) -> None:
        with pytest.raises(SchemaError, match="orderable"):
            Codebook.build(np.array([1, "a"], dtype=object))

    def test_nan_rejected(self) -> None:
        with pytest.raises(SchemaError, match="NaN"):
            Codebook.build(np.array([1.0, np.nan]))

    def test_encode_type_mismatch_raises_schema_error(self) -> None:
        book, _ = Codebook.build([1, 2, 3])
        with pytest.raises(SchemaError):
            book.encode(np.array(["a"], dtype=object))


class TestNumpyIntSeed:
    def test_numpy_int_seed_accepted_and_coerced(self) -> None:
        schema = _drug_self_schema()
        spec = SplitSpec(
            supervision_edge=("drug", "synergy", "drug"),
            roles=dict(schema.roles),
            seed=np.int64(7),
        )
        assert spec.seed == 7
        assert type(spec.seed) is int
