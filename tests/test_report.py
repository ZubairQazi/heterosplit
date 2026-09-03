"""Tests for the distribution report."""

from __future__ import annotations

import json

from heterosplit import split_records
from heterosplit.report import build_summary, to_json, to_markdown
from heterosplit.synthetic import make_synthetic_dataset


class TestBuildSummary:
    def test_structure(self) -> None:
        ds = make_synthetic_dataset(n_records=1000, n_labels=3, n_context_entities=8, seed=0)
        result = split_records(ds.records, ds.spec("both_cold_start"))
        summary = build_summary(result)
        assert summary["regime"] == "both_cold_start"
        assert set(summary["sizes"]) >= {
            "counts",
            "achieved_ratios",
            "requested_ratios",
            "excluded",
        }
        assert set(summary["entity_counts"]) == set(result.spec.roles)
        assert summary["labels"] is not None
        assert len(summary["labels"]["distribution"]) == 3  # train/val/test
        assert "train" in summary["degree"]
        assert "has_leakage" in summary["audit"]

    def test_no_labels(self) -> None:
        ds = make_synthetic_dataset(n_records=200, n_labels=0, seed=0)
        result = split_records(ds.records, ds.spec("random"))
        assert build_summary(result)["labels"] is None


class TestRendering:
    def test_markdown_has_sections(self) -> None:
        ds = make_synthetic_dataset(n_records=500, n_labels=2, seed=0)
        result = split_records(ds.records, ds.spec("pair_cold_start"))
        md = to_markdown(result)
        assert "# HeteroSplit report" in md
        assert "## Split sizes" in md
        assert "## Leakage audit" in md
        assert "## Label distribution" in md

    def test_json_parseable(self) -> None:
        ds = make_synthetic_dataset(n_records=200, seed=0)
        result = split_records(ds.records, ds.spec("random"))
        parsed = json.loads(to_json(result))
        assert parsed["regime"] == "random"
