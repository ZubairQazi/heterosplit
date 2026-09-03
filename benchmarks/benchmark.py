"""Benchmark HeteroSplit: runtime, peak memory, balance, and reproducibility.

Run::

    uv run python benchmarks/benchmark.py --sizes 5000,50000,200000

Measures, per dataset size and regime: split-construction time, peak memory, deviation
from the requested ratios, label-distribution divergence, and whether re-running yields
an identical manifest. For the pair regime it also runs a naive *group-shuffle* baseline
(assign whole groups to splits by group fraction, no size balancing) to show HeteroSplit's
balance advantage.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from heterosplit import SplitSpec, make_synthetic_dataset, split_records
from heterosplit.canonical import pair_group_ids
from heterosplit.objective import distribution_divergence, size_deviation, value_counts_by_split
from heterosplit.records import PredictionRecords
from heterosplit.result import SplitResult

REGIMES = ["random", "pair_cold_start", "source_cold_start", "both_cold_start"]


def _measure(
    fn: Callable[[PredictionRecords, SplitSpec], SplitResult],
    records: PredictionRecords,
    spec: SplitSpec,
) -> tuple[SplitResult, float, int]:
    tracemalloc.start()
    start = time.perf_counter()
    result = fn(records, spec)
    runtime = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, runtime, int(peak)


def _balance(result: SplitResult) -> tuple[float, float]:
    counts = np.array([result.counts[s] for s in result.split_names], dtype=np.float64)
    ratio_dev = size_deviation(counts, result.spec.ratios)
    labels = result.records.labels
    if labels is None:
        return ratio_dev, 0.0
    vc = value_counts_by_split(
        labels, result.record_split, result.records.n_labels, len(result.split_names)
    )
    return ratio_dev, distribution_divergence(vc, result.spec.ratios)


def group_shuffle_split(records: PredictionRecords, spec: SplitSpec) -> SplitResult:
    """Naive baseline: assign whole canonical-pair groups to splits by group fraction."""
    groups, unique = pair_group_ids(
        records.source_codes, records.destination_codes, undirected=spec.undirected_pairs
    )
    n_groups = int(unique.shape[0])
    rng = np.random.default_rng(spec.seed)
    order = rng.permutation(n_groups)
    position = np.empty(n_groups, dtype=np.float64)
    position[order] = np.arange(n_groups) / max(1, n_groups)
    cumulative = np.cumsum(np.asarray(spec.ratios, dtype=np.float64))
    group_split = np.clip(
        np.searchsorted(cumulative, position, side="right"), 0, len(spec.ratios) - 1
    )
    record_split = group_split[groups].astype(np.int64)
    return SplitResult(spec=spec, records=records, record_split=record_split)


def run_benchmark(sizes: list[int], seed: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_records in sizes:
        n_entities = max(10, n_records // 20)
        dataset = make_synthetic_dataset(
            n_records=n_records, n_source_entities=n_entities, n_labels=3, seed=seed
        )
        for regime in REGIMES:
            spec = dataset.spec(regime, seed=seed)
            result, runtime, peak = _measure(split_records, dataset.records, spec)
            ratio_dev, label_div = _balance(result)
            reproducible = (
                split_records(dataset.records, spec).manifest.digest() == result.manifest.digest()
            )
            rows.append(
                {
                    "n_records": n_records,
                    "n_entities": dataset.records.n_entities("drug"),
                    "method": "heterosplit",
                    "regime": regime,
                    "runtime_s": runtime,
                    "peak_mb": peak / 1e6,
                    "ratio_dev": ratio_dev,
                    "label_div": label_div,
                    "reproducible": reproducible,
                }
            )
        # group-shuffle baseline on the pair regime
        spec = dataset.spec("pair_cold_start", seed=seed)
        result, runtime, peak = _measure(group_shuffle_split, dataset.records, spec)
        ratio_dev, label_div = _balance(result)
        rows.append(
            {
                "n_records": n_records,
                "n_entities": dataset.records.n_entities("drug"),
                "method": "group-shuffle",
                "regime": "pair_cold_start",
                "runtime_s": runtime,
                "peak_mb": peak / 1e6,
                "ratio_dev": ratio_dev,
                "label_div": label_div,
                "reproducible": True,
            }
        )
    return rows


def to_markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| records | entities | method | regime | time (s) | peak (MB) | "
        "ratio dev | label div | reproducible |"
    )
    sep = "|---:|---:|---|---|---:|---:|---:|---:|:--:|"
    lines = [header, sep]
    lines.extend(
        f"| {r['n_records']} | {r['n_entities']} | {r['method']} | {r['regime']} | "
        f"{r['runtime_s']:.4f} | {r['peak_mb']:.1f} | {r['ratio_dev']:.4f} | "
        f"{r['label_div']:.4f} | {'yes' if r['reproducible'] else 'NO'} |"
        for r in rows
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="5000,50000", help="comma-separated record counts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="write the Markdown table here")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    rows = run_benchmark(sizes, seed=args.seed)
    table = to_markdown_table(rows)
    print(table)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
