"""Command-line interface: split a table by a spec, or run a synthetic demo.

Usage::

    heterosplit demo --regime both_cold_start
    heterosplit split --input data.csv --spec spec.json --out-dir out/

The ``split`` command reads a CSV whose columns match the spec's role names, writes a
manifest, JSON + Markdown reports, and the split indices, and exits non-zero if the
audit detects leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

from . import __version__
from .adapters.tabular import records_from_table
from .api import split_records
from .report import to_json, to_markdown
from .result import SplitResult
from .spec import SplitSpec
from .synthetic import make_synthetic_dataset

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="heterosplit", description=__doc__)
    parser.add_argument("--version", action="version", version=f"heterosplit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run a synthetic end-to-end example")
    demo.add_argument("--regime", default="both_cold_start")
    demo.add_argument("--records", type=int, default=2000)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--out-dir", type=Path, default=None)

    split = sub.add_parser("split", help="split a CSV using a JSON spec")
    split.add_argument("--input", required=True, type=Path)
    split.add_argument("--spec", required=True, type=Path)
    split.add_argument("--out-dir", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "demo":
        return _run_demo(args)
    return _run_split(args)


def _run_demo(args: argparse.Namespace) -> int:
    ds = make_synthetic_dataset(
        n_records=args.records, n_context_entities=20, n_labels=3, seed=args.seed
    )
    holdout = {"drug": "either", "cell_line": "all"} if args.regime == "joint_cold_start" else None
    spec = ds.spec(args.regime, seed=args.seed, holdout=holdout)
    result, measurements = _timed_split(ds.records, spec)
    print(to_markdown(result))
    if args.out_dir is not None:
        _write_outputs(result, args.out_dir, measurements)
    return 0 if not result.audit.has_leakage else 1


def _run_split(args: argparse.Namespace) -> int:
    spec_data = json.loads(args.spec.read_text(encoding="utf-8"))
    spec = SplitSpec.from_dict(spec_data)
    columns = _read_csv(args.input)
    records = records_from_table(
        spec.schema, columns, label=spec_data.get("label"), timestamp=spec_data.get("timestamp")
    )
    result, measurements = _timed_split(records, spec)
    _write_outputs(result, args.out_dir, measurements)

    audit = result.audit
    print(f"Wrote outputs to {args.out_dir}")
    print(result.audit.summary())
    if audit.has_leakage:
        print("ERROR: leakage detected — see report for details", flush=True)
        return 1
    return 0


def _timed_split(records: Any, spec: SplitSpec) -> tuple[SplitResult, dict[str, Any]]:
    tracemalloc.start()
    start = time.perf_counter()
    result = split_records(records, spec)
    runtime = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {"runtime_seconds": runtime, "peak_memory_bytes": int(peak)}


def _read_csv(path: Path) -> dict[str, list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        rows = list(reader)
    return {name: [row[name] for row in rows] for name in reader.fieldnames}


def _write_outputs(result: SplitResult, out_dir: Path, measurements: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = result.build_manifest(measurements=measurements)
    manifest.save(out_dir / "manifest.json")
    (out_dir / "report.md").write_text(to_markdown(result), encoding="utf-8")
    (out_dir / "report.json").write_text(to_json(result), encoding="utf-8")
    splits = {name: result.indices(name).tolist() for name in result.split_names}
    splits["excluded"] = result.excluded_indices.tolist()
    (out_dir / "splits.json").write_text(json.dumps(splits), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
