"""Tests for supervision and message-passing edge reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from heterosplit import EntityRole, SplitSpec, TaskSchema, split_records
from heterosplit.canonical import canonicalize_pairs
from heterosplit.records import PredictionRecords
from heterosplit.synthetic import make_synthetic_dataset

_REGIMES = ["random", "pair_cold_start", "source_cold_start", "both_cold_start"]


def _canonical_pairs(edge_index: np.ndarray) -> set[tuple[int, int]]:
    lo, hi = canonicalize_pairs(edge_index[0], edge_index[1], undirected=True)
    return set(zip(lo.tolist(), hi.tolist(), strict=True))


class TestSupervisionEdgeIndex:
    def test_shape_matches_split(self) -> None:
        ds = make_synthetic_dataset(n_records=200, seed=0)
        result = split_records(ds.records, ds.spec("random"))
        ei = result.supervision_edge_index("train")
        assert ei.shape == (2, result.counts["train"])


class TestMessagePassingInvariant:
    @pytest.mark.parametrize("regime", _REGIMES)
    def test_training_mp_excludes_heldout_edges_and_reverses(self, regime: str) -> None:
        ds = make_synthetic_dataset(n_records=1500, n_source_entities=30, seed=0)
        result = split_records(ds.records, ds.spec(regime))
        mp = result.message_passing_edge_index()
        mp_pairs = _canonical_pairs(mp)

        held = np.concatenate([result.val_indices, result.test_indices])
        h_lo, h_hi = canonicalize_pairs(
            ds.records.source_codes[held], ds.records.destination_codes[held], undirected=True
        )
        held_pairs = set(zip(h_lo.tolist(), h_hi.tolist(), strict=True))
        assert mp_pairs.isdisjoint(held_pairs)

    def test_directed_random_overlap_is_removed(self) -> None:
        # Same undirected edge {A,B} appears as (A,B) and (B,A); if they split across
        # train/test, the training MP graph must not contain that canonical pair.
        schema = TaskSchema(
            ("drug", "rel", "drug"),
            {"source": EntityRole.source("drug"), "destination": EntityRole.destination("drug")},
        )
        # 8 records; enough that train and test both receive some.
        records = PredictionRecords.from_columns(
            schema,
            {
                "source": ["A", "B", "C", "D", "E", "F", "G", "H"],
                "destination": ["B", "A", "D", "C", "F", "E", "H", "G"],
            },
        )
        spec = SplitSpec(
            supervision_edge=("drug", "rel", "drug"),
            roles=dict(schema.roles),
            regime="random",
            ratios=(0.5, 0.5),
            seed=0,
        )
        result = split_records(records, spec)
        mp = result.message_passing_edge_index(remove_heldout=True)
        mp_pairs = _canonical_pairs(mp)
        test = result.test_indices
        t_lo, t_hi = canonicalize_pairs(
            records.source_codes[test], records.destination_codes[test], undirected=True
        )
        test_pairs = set(zip(t_lo.tolist(), t_hi.tolist(), strict=True))
        assert mp_pairs.isdisjoint(test_pairs)

    def test_undirected_adds_both_directions(self) -> None:
        ds = make_synthetic_dataset(n_records=500, n_source_entities=20, seed=1)
        result = split_records(ds.records, ds.spec("pair_cold_start", undirected_pairs=True))
        mp = result.message_passing_edge_index()
        # every edge has its reverse present
        forward = set(zip(mp[0].tolist(), mp[1].tolist(), strict=True))
        reverse = {(b, a) for a, b in forward}
        assert forward == reverse

    def test_remove_heldout_false_may_keep_overlap(self) -> None:
        ds = make_synthetic_dataset(n_records=800, n_source_entities=12, seed=2)
        result = split_records(ds.records, ds.spec("random"))
        safe = result.message_passing_edge_index(remove_heldout=True)
        unsafe = result.message_passing_edge_index(remove_heldout=False)
        assert unsafe.shape[1] >= safe.shape[1]
