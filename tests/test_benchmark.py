"""Smoke tests for the benchmark harness."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

import benchmark


class TestBenchmark:
    def test_run_benchmark_smoke(self) -> None:
        rows = benchmark.run_benchmark([500], seed=0)
        assert rows
        assert all("runtime_s" in r and "peak_mb" in r for r in rows)
        assert all(r["reproducible"] for r in rows if r["method"] == "heterosplit")

    def test_heterosplit_balances_at_least_as_well_as_group_shuffle(self) -> None:
        rows = benchmark.run_benchmark([5000], seed=0)
        hs = next(
            r for r in rows if r["method"] == "heterosplit" and r["regime"] == "pair_cold_start"
        )
        gs = next(r for r in rows if r["method"] == "group-shuffle")
        assert hs["ratio_dev"] <= gs["ratio_dev"] + 1e-9

    def test_markdown_table(self) -> None:
        table = benchmark.to_markdown_table(benchmark.run_benchmark([500]))
        assert "| records |" in table
        assert "heterosplit" in table
