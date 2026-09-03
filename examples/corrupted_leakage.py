"""Demonstrate that the auditor catches injected leakage.

Run with::

    uv run python examples/corrupted_leakage.py

It builds a clean ``source_cold_start`` split, confirms the audit is clean, then moves a
single record so one source drug appears in both training and test — and shows the audit
turning that into a hard leakage error.
"""

from __future__ import annotations

import numpy as np

from heterosplit import LeakageError, split_records
from heterosplit.result import SplitResult
from heterosplit.synthetic import make_synthetic_dataset


def main() -> None:
    ds = make_synthetic_dataset(n_records=1500, n_source_entities=15, seed=0)
    clean = split_records(ds.records, ds.spec("source_cold_start"))

    print("=== Clean source cold-start split ===")
    print(clean.audit.summary())
    print(f"has_leakage = {clean.audit.has_leakage}\n")

    corrupted = _leak_one_source(clean)
    print("=== After moving one record to create a source overlap ===")
    print(corrupted.audit.summary())
    try:
        corrupted.audit.raise_for_leakage()
    except LeakageError as exc:
        print(f"\nraise_for_leakage() raised as expected:\n  {exc}")


def _leak_one_source(result: SplitResult) -> SplitResult:
    """Move one record of an all-training source into test, leaking that source entity."""
    source = result.records.source_codes
    record_split = result.record_split.copy()
    test_index = result.split_names.index("test")
    for code in np.unique(source):
        members = np.flatnonzero(source == code)
        if members.size >= 2 and np.all(record_split[members] == 0):
            record_split[members[0]] = test_index
            break
    return SplitResult(spec=result.spec, records=result.records, record_split=record_split)


if __name__ == "__main__":
    main()
