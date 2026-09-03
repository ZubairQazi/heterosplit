"""Regression tests: intentionally corrupt a clean split; the auditor must catch it.

For every record-partition cold-start regime we take a clean, leakage-free split, move a
single record of an all-training group into the test split (re-introducing the group into
both splits), and assert the auditor now reports leakage. This guards against auditor
false negatives across seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from heterosplit import split_records
from heterosplit.canonical import pair_group_ids
from heterosplit.result import SplitResult
from heterosplit.synthetic import make_synthetic_dataset


def _group_ids(result: SplitResult, regime: str) -> np.ndarray:
    records = result.records
    if regime == "source_cold_start":
        return records.source_codes
    if regime == "destination_cold_start":
        return records.destination_codes
    if regime == "context_cold_start":
        return records.codes("context")
    if regime == "pair_cold_start":
        groups, _ = pair_group_ids(
            records.source_codes, records.destination_codes, undirected=False
        )
        return groups
    raise AssertionError(regime)


def _inject_overlap(result: SplitResult, group_ids: np.ndarray) -> SplitResult | None:
    """Move one record of an all-training group into test so the group spans both splits."""
    record_split = result.record_split.copy()
    train_i = 0
    test_i = result.split_names.index("test")
    for g in np.unique(group_ids):
        members = np.flatnonzero(group_ids == g)
        if members.size >= 2 and np.all(record_split[members] == train_i):
            record_split[members[0]] = test_i
            return SplitResult(spec=result.spec, records=result.records, record_split=record_split)
    return None


@pytest.mark.parametrize(
    "regime",
    ["source_cold_start", "destination_cold_start", "context_cold_start", "pair_cold_start"],
)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_injected_overlap_is_detected(regime: str, seed: int) -> None:
    ds = make_synthetic_dataset(
        n_records=1500, n_source_entities=15, n_context_entities=8, seed=seed
    )
    result = split_records(ds.records, ds.spec(regime))
    assert not result.audit.has_leakage, "baseline split should be clean"

    corrupted = _inject_overlap(result, _group_ids(result, regime))
    assert corrupted is not None, "expected an all-training group to corrupt"
    assert corrupted.audit.has_leakage, "auditor missed injected leakage"
