"""Tests for the synthetic dataset generator."""

from __future__ import annotations

import numpy as np
import pytest

from heterosplit import Regime
from heterosplit.synthetic import make_synthetic_dataset


class TestMakeSyntheticDataset:
    def test_defaults(self) -> None:
        ds = make_synthetic_dataset(n_records=100, seed=0)
        r = ds.records
        assert r.n_records == 100
        assert r.schema.is_self_relation
        assert r.has_labels
        assert r.timestamps is None
        assert r.schema.context_names == []

    def test_deterministic_for_fixed_seed(self) -> None:
        a = make_synthetic_dataset(n_records=50, seed=7).records
        b = make_synthetic_dataset(n_records=50, seed=7).records
        np.testing.assert_array_equal(a.source_codes, b.source_codes)
        np.testing.assert_array_equal(a.destination_codes, b.destination_codes)

    def test_different_seeds_differ(self) -> None:
        a = make_synthetic_dataset(n_records=50, seed=1).records
        b = make_synthetic_dataset(n_records=50, seed=2).records
        assert not np.array_equal(a.source_codes, b.source_codes)

    def test_no_self_loops_by_default(self) -> None:
        r = make_synthetic_dataset(n_records=500, n_source_entities=5, seed=3).records
        assert not np.any(r.source_codes == r.destination_codes)

    def test_self_loops_allowed(self) -> None:
        r = make_synthetic_dataset(
            n_records=500, n_source_entities=3, allow_self_loops=True, seed=3
        ).records
        # with a tiny pool and self-loops allowed, some are very likely to occur
        assert np.any(r.source_codes == r.destination_codes)

    def test_bipartite(self) -> None:
        ds = make_synthetic_dataset(
            n_records=80,
            self_relation=False,
            source_type="user",
            destination_type="item",
            n_destination_entities=20,
            relation="rates",
            seed=0,
        )
        r = ds.records
        assert not r.schema.is_self_relation
        assert r.schema.source_type == "user"
        assert r.schema.destination_type == "item"

    def test_context_and_timestamps(self) -> None:
        r = make_synthetic_dataset(
            n_records=60, n_context_entities=4, with_timestamps=True, seed=0
        ).records
        assert r.schema.context_names == ["context"]
        assert r.timestamps is not None
        assert r.n_entities("cell_line") <= 4

    def test_disable_labels(self) -> None:
        r = make_synthetic_dataset(n_records=10, n_labels=0, seed=0).records
        assert not r.has_labels

    def test_spec_builder(self) -> None:
        ds = make_synthetic_dataset(n_records=10, seed=0)
        spec = ds.spec(Regime.PAIR, seed=42, undirected_pairs=True)
        assert spec.regime is Regime.PAIR
        assert spec.seed == 42
        assert spec.schema.supervision_edge == ds.records.schema.supervision_edge

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_source_entities": 0},
            {"n_source_entities": 1},  # self-relation w/o self-loops needs >= 2
            {"self_relation": False, "n_destination_entities": 0},
        ],
    )
    def test_invalid_configs(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            make_synthetic_dataset(n_records=10, seed=0, **kwargs)  # type: ignore[arg-type]
