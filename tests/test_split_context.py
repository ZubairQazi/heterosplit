"""Tests for the context cold-start splitter."""

from __future__ import annotations

import numpy as np
import pytest

from heterosplit import EntityRole, SpecError, SplitSpec, split_records
from heterosplit.synthetic import make_synthetic_dataset


def _context_dataset(**kwargs: object):  # type: ignore[no-untyped-def]
    return make_synthetic_dataset(n_context_entities=12, **kwargs)  # type: ignore[arg-type]


class TestContextColdStart:
    def test_context_entities_disjoint(self) -> None:
        ds = _context_dataset(n_records=2000, seed=0)
        result = split_records(ds.records, ds.spec("context_cold_start"))
        assert result.covers_all_records()
        ctx = ds.records.codes("context")
        train = set(ctx[result.train_indices].tolist())
        test = set(ctx[result.test_indices].tolist())
        assert train.isdisjoint(test)

    def test_endpoints_may_recur(self) -> None:
        ds = _context_dataset(n_records=2000, n_source_entities=15, seed=1)
        result = split_records(ds.records, ds.spec("context_cold_start"))
        src = ds.records.source_codes
        assert set(src[result.train_indices].tolist()) & set(src[result.test_indices].tolist())

    def test_deterministic(self) -> None:
        ds = _context_dataset(n_records=800, seed=2)
        a = split_records(ds.records, ds.spec("context_cold_start", seed=4))
        b = split_records(ds.records, ds.spec("context_cold_start", seed=4))
        np.testing.assert_array_equal(a.record_split, b.record_split)


class TestContextSpecGuard:
    def test_requires_a_context_role(self) -> None:
        with pytest.raises(SpecError, match="exactly one context role"):
            SplitSpec(
                supervision_edge=("drug", "synergy", "drug"),
                roles={
                    "source": EntityRole.source("drug"),
                    "destination": EntityRole.destination("drug"),
                },
                regime="context_cold_start",
            )

    def test_rejects_multiple_context_roles(self) -> None:
        with pytest.raises(SpecError, match="exactly one context role"):
            SplitSpec(
                supervision_edge=("drug", "synergy", "drug"),
                roles={
                    "source": EntityRole.source("drug"),
                    "destination": EntityRole.destination("drug"),
                    "cell_line": EntityRole.context("cell_line"),
                    "tissue": EntityRole.context("tissue"),
                },
                regime="context_cold_start",
            )
