"""Tests for the command-line interface and spec round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from heterosplit import EntityRole, Manifest, SplitSpec
from heterosplit.cli import main


class TestSpecRoundTrip:
    def test_from_dict_roundtrips_normalize(self) -> None:
        spec = SplitSpec(
            supervision_edge=("drug", "synergy", "drug"),
            roles={
                "source": EntityRole.source("drug"),
                "destination": EntityRole.destination("drug"),
                "cell": EntityRole.context("cell_line"),
            },
            regime="joint_cold_start",
            holdout={"drug": "either", "cell_line": "all"},
            ratios=(0.7, 0.15, 0.15),
            seed=3,
        )
        assert SplitSpec.from_dict(spec.normalize()).normalize() == spec.normalize()


class TestDemoCommand:
    def test_demo_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["demo", "--regime", "random", "--records", "300", "--seed", "1"])
        assert code == 0
        out = capsys.readouterr().out
        assert "HeteroSplit report" in out

    def test_demo_writes_outputs(self, tmp_path: Path) -> None:
        code = main(
            ["demo", "--regime", "both_cold_start", "--records", "400", "--out-dir", str(tmp_path)]
        )
        assert code == 0
        assert (tmp_path / "manifest.json").exists()
        manifest = Manifest.load(tmp_path / "manifest.json")
        assert manifest.measurements is not None
        assert "runtime_seconds" in manifest.measurements


class TestSplitCommand:
    def _write_inputs(self, tmp_path: Path) -> tuple[Path, Path]:
        rows = ["source,destination,cell,label"]
        rows += [f"d{i % 12},d{(i + 3) % 12},c{i % 5},{i % 2}" for i in range(40)]
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("\n".join(rows), encoding="utf-8")

        spec = {
            "supervision_edge": ["drug", "synergy", "drug"],
            "roles": {
                "source": {"kind": "source", "entity_type": "drug"},
                "destination": {"kind": "destination", "entity_type": "drug"},
                "cell": {"kind": "context", "entity_type": "cell_line"},
            },
            "regime": "source_cold_start",
            "ratios": [0.6, 0.4],
            "seed": 1,
            "label": "label",
        }
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        return csv_path, spec_path

    def test_split_writes_manifest_and_reports(self, tmp_path: Path) -> None:
        csv_path, spec_path = self._write_inputs(tmp_path)
        out_dir = tmp_path / "out"
        code = main(
            ["split", "--input", str(csv_path), "--spec", str(spec_path), "--out-dir", str(out_dir)]
        )
        assert code == 0
        for name in ("manifest.json", "report.md", "report.json", "splits.json"):
            assert (out_dir / name).exists()
        splits = json.loads((out_dir / "splits.json").read_text())
        assert set(splits) == {"train", "test", "excluded"}
        # every record accounted for exactly once
        total = sum(len(v) for v in splits.values())
        assert total == 40
