"""Tests for the PyTorch Geometric adapter (skipped without the [pyg] extra)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pyg_data = pytest.importorskip("torch_geometric.data")

from heterosplit import EntityRole, SplitSpec  # noqa: E402
from heterosplit.adapters.pyg import (  # noqa: E402
    records_from_heterodata,
    split_heterodata,
    to_link_split,
)
from heterosplit.canonical import canonicalize_pairs  # noqa: E402

EDGE = ("drug", "synergy", "drug")


def _hetero_data(n_edges: int = 300, n_drugs: int = 25, seed: int = 0):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(seed)
    data = pyg_data.HeteroData()
    data["drug"].num_nodes = n_drugs
    src = torch.randint(0, n_drugs, (n_edges,), generator=g)
    dst = torch.randint(0, n_drugs, (n_edges,), generator=g)
    data[EDGE].edge_index = torch.stack([src, dst])
    data[EDGE]["cell_line"] = torch.randint(0, 6, (n_edges,), generator=g)
    data[EDGE].edge_label = torch.randint(0, 2, (n_edges,), generator=g)
    return data


def _spec(regime: str = "pair_cold_start", **kwargs: object) -> SplitSpec:
    return SplitSpec(
        supervision_edge=EDGE,
        roles={
            "source": EntityRole.source("drug"),
            "destination": EntityRole.destination("drug"),
            "cell_line": EntityRole.context("cell_line"),
        },
        regime=regime,
        **kwargs,  # type: ignore[arg-type]
    )


class TestRecordsFromHeterodata:
    def test_extracts_edges_context_labels(self) -> None:
        data = _hetero_data()
        records = records_from_heterodata(data, _spec(), context="cell_line")
        assert records.n_records == 300
        assert records.schema.context_names == ["cell_line"]
        assert records.has_labels  # from edge_label
        expected = data[EDGE].edge_index[0].numpy()
        np.testing.assert_array_equal(records.raw_ids("source"), expected)

    def test_missing_edge_type_raises(self) -> None:
        data = pyg_data.HeteroData()
        data["drug"].num_nodes = 3
        with pytest.raises(KeyError, match="not found"):
            records_from_heterodata(data, _spec(), context=np.zeros(0))

    def test_missing_context_raises(self) -> None:
        data = _hetero_data()
        with pytest.raises(ValueError, match="context"):
            records_from_heterodata(data, _spec())


class TestSplitHeterodata:
    def test_returns_split_result(self) -> None:
        data = _hetero_data()
        result = split_heterodata(data, _spec("pair_cold_start", seed=1), context="cell_line")
        assert result.covers_all_records()
        assert result.manifest.digest()  # manifest works end-to-end


class TestToLinkSplit:
    def test_builds_pyg_link_split(self) -> None:
        data = _hetero_data()
        result = split_heterodata(data, _spec("source_cold_start"), context="cell_line")
        splits = to_link_split(result, data)
        assert set(splits) == {"train", "val", "test"}
        for name, d in splits.items():
            store = d[EDGE]
            assert store.edge_label_index.shape[1] == result.counts[name]
            assert store.edge_label.shape[0] == result.counts[name]
            # node stores preserved
            assert d["drug"].num_nodes == 25

    def test_training_message_passing_is_leakage_safe(self) -> None:
        data = _hetero_data()
        result = split_heterodata(data, _spec("random"), context="cell_line")
        splits = to_link_split(result, data)
        mp = splits["train"][EDGE].edge_index.numpy()
        m_lo, m_hi = canonicalize_pairs(mp[0], mp[1], undirected=True)
        mp_pairs = set(zip(m_lo.tolist(), m_hi.tolist(), strict=True))

        test_labels = splits["test"][EDGE].edge_label_index.numpy()
        t_lo, t_hi = canonicalize_pairs(test_labels[0], test_labels[1], undirected=True)
        test_pairs = set(zip(t_lo.tolist(), t_hi.tolist(), strict=True))
        assert mp_pairs.isdisjoint(test_pairs)
