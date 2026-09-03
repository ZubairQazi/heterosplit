"""Tests for the leakage audit suite."""

from __future__ import annotations

import numpy as np
import pytest

from heterosplit import (
    EntityRole,
    LeakageError,
    Regime,
    SplitSpec,
    TaskSchema,
    audit_split,
    split_records,
)
from heterosplit.audit import (
    Severity,
    audit_feature_provenance,
    audit_negative_samples,
    contract_for,
)
from heterosplit.records import PredictionRecords
from heterosplit.result import SplitResult
from heterosplit.synthetic import make_synthetic_dataset

DRUG_ROLES = {
    "source": EntityRole.source("drug"),
    "destination": EntityRole.destination("drug"),
}
CTX_ROLES = {**DRUG_ROLES, "cell": EntityRole.context("cell_line")}


def _spec(regime: str, roles: dict[str, EntityRole] = DRUG_ROLES, **kwargs: object) -> SplitSpec:
    return SplitSpec(
        supervision_edge=("drug", "synergy", "drug"),
        roles=roles,
        regime=regime,
        ratios=(0.5, 0.5),
        **kwargs,  # type: ignore[arg-type]
    )


def _manual(
    columns: dict[str, list[object]], record_split: list[int], spec: SplitSpec, **kw: object
) -> SplitResult:
    records = PredictionRecords.from_columns(spec.schema, columns, **kw)  # type: ignore[arg-type]
    return SplitResult(spec=spec, records=records, record_split=np.array(record_split, np.int64))


class TestCleanSplitsPass:
    @pytest.mark.parametrize(
        "regime",
        ["random", "pair_cold_start", "source_cold_start", "either_cold_start", "both_cold_start"],
    )
    def test_no_leakage_on_generated_splits(self, regime: str) -> None:
        ds = make_synthetic_dataset(n_records=1500, n_source_entities=30, seed=0)
        result = split_records(ds.records, ds.spec(regime))
        report = result.audit
        assert not report.has_leakage
        result.audit.raise_for_leakage()  # does not raise
        # our reconstructed MP graph is always clean
        assert result.audit.get("message_passing_reconstruction_leak").count == 0  # type: ignore[union-attr]

    def test_context_clean(self) -> None:
        ds = make_synthetic_dataset(n_records=1500, n_context_entities=15, seed=0)
        result = split_records(ds.records, ds.spec("context_cold_start"))
        assert not result.audit.has_leakage


class TestInjectedLeakageDetected:
    def test_source_overlap(self) -> None:
        result = _manual(
            {"source": ["A", "A", "B", "C"], "destination": ["X", "Y", "Z", "W"]},
            [0, 1, 0, 1],
            _spec("source_cold_start"),
        )
        assert result.audit.has_leakage
        with pytest.raises(LeakageError, match="source_entity_overlap"):
            result.audit.raise_for_leakage()
        finding = result.audit.get("source_entity_overlap")
        assert finding is not None and finding.count == 1
        assert "A" in finding.examples

    def test_pair_overlap_directed(self) -> None:
        result = _manual(
            {"source": ["A", "A", "C", "D"], "destination": ["B", "B", "E", "F"]},
            [0, 1, 0, 1],
            _spec("pair_cold_start"),
        )
        finding = result.audit.get("pair_overlap")
        assert finding is not None and finding.count == 1 and finding.is_violation

    def test_reversed_pair_is_warning_not_violation(self) -> None:
        # (A,B) in train and (B,A) in test: ordered-disjoint, but reversed pair leaks.
        result = _manual(
            {"source": ["A", "B", "C", "D"], "destination": ["B", "A", "E", "F"]},
            [0, 1, 0, 1],
            _spec("pair_cold_start"),
        )
        assert result.audit.get("pair_overlap").count == 0  # type: ignore[union-attr]
        reversed_finding = result.audit.get("reversed_pair_overlap")
        assert reversed_finding is not None and reversed_finding.count == 1
        assert reversed_finding.severity is Severity.WARNING
        assert not result.audit.has_leakage  # warning only

    def test_both_entity_overlap(self) -> None:
        result = _manual(
            {"source": ["A", "C", "E", "A"], "destination": ["B", "D", "F", "G"]},
            [0, 0, 1, 1],
            _spec("both_cold_start"),
        )
        # 'A' appears in a train record and a test record
        assert result.audit.has_leakage

    def test_either_violation(self) -> None:
        # test edge (A,B) with both A and B seen in a train edge -> not cold.
        result = _manual(
            {"source": ["A", "A"], "destination": ["B", "B"]},
            [0, 1],
            _spec("either_cold_start"),
        )
        finding = result.audit.get("either_endpoint_unseen")
        assert finding is not None and finding.count == 1 and finding.is_violation

    def test_context_overlap(self) -> None:
        result = _manual(
            {
                "source": ["A", "C", "E", "G"],
                "destination": ["B", "D", "F", "H"],
                "cell": ["m", "m", "n", "o"],
            },
            [0, 1, 0, 1],
            _spec("context_cold_start", roles=CTX_ROLES),
        )
        finding = result.audit.get("context_overlap:cell")
        assert finding is not None and finding.count == 1 and finding.is_violation

    def test_duplicate_records_across_splits(self) -> None:
        result = _manual(
            {"source": ["A", "A", "C", "D"], "destination": ["B", "B", "E", "F"]},
            [0, 1, 0, 1],
            _spec("random"),
        )
        finding = result.audit.get("duplicate_records_across_splits")
        assert finding is not None and finding.count == 1
        assert finding.severity is Severity.WARNING


class TestOptionalAuditors:
    def test_negative_sample_collision(self) -> None:
        ds = make_synthetic_dataset(n_records=200, n_source_entities=10, seed=0)
        result = split_records(ds.records, ds.spec("pair_cold_start"))
        # use a real positive edge as a "negative" -> collision
        pos_src = ds.records.source_codes[:3]
        pos_dst = ds.records.destination_codes[:3]
        finding = audit_negative_samples(result, pos_src, pos_dst)
        assert finding.count == 3 and finding.is_violation

    def test_feature_provenance_flags_held_out_entities(self) -> None:
        ds = make_synthetic_dataset(n_records=1000, n_source_entities=20, seed=0)
        result = split_records(ds.records, ds.spec("source_cold_start"))
        # "fit" features using every drug id (global fit) -> held-out drugs flagged
        all_ids = ds.records.codebooks["drug"].values.tolist()
        finding = audit_feature_provenance(result, all_ids, role_name="source")
        assert finding.count > 0
        assert finding.severity is Severity.WARNING


class TestContractAndWiring:
    def test_contract_for_joint(self) -> None:
        spec = SplitSpec(
            supervision_edge=("drug", "synergy", "drug"),
            roles=CTX_ROLES,
            regime="joint_cold_start",
            holdout={"drug": "either", "cell_line": "all"},
        )
        contract = contract_for(spec)
        assert contract.either_endpoint_unseen
        assert contract.context_disjoint
        assert not contract.source_disjoint

    def test_audit_is_cached(self) -> None:
        ds = make_synthetic_dataset(n_records=100, seed=0)
        result = split_records(ds.records, ds.spec("pair_cold_start"))
        assert result.audit is result.audit

    def test_audit_split_matches_property(self) -> None:
        ds = make_synthetic_dataset(n_records=100, seed=0)
        result = split_records(ds.records, ds.spec("random"))
        assert audit_split(result).has_leakage == result.audit.has_leakage

    def test_manifest_includes_audit(self) -> None:
        ds = make_synthetic_dataset(n_records=200, seed=0)
        result = split_records(ds.records, ds.spec("both_cold_start"))
        manifest = result.manifest
        assert manifest.audit is not None
        assert manifest.audit["has_leakage"] is False
        assert Regime.coerce(result.spec.regime) is Regime.BOTH
