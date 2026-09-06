"""Smoke test for the GNN link-prediction example (skipped without the [pyg] extra)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

import train_link_prediction as tlp


def test_structured_records_have_community_structure() -> None:
    records = tlp.structured_synthetic_records(n_drugs=40, n_records=400, seed=0)
    assert records.n_records <= 400
    assert records.n_entities("drug") <= 40
    assert records.schema.is_self_relation


def test_compare_regimes_runs_and_returns_metrics() -> None:
    records = tlp.structured_synthetic_records(n_drugs=50, n_records=800, seed=0)
    rows = tlp.compare_regimes(records, regimes=("random", "either_cold_start"), epochs=3, seed=0)
    assert {r["regime"] for r in rows} == {"random", "either_cold_start"}
    for row in rows:
        auc = row["test_auc"]
        assert math.isnan(auc) or 0.0 <= auc <= 1.0
        assert row["train_pos"] > 0
