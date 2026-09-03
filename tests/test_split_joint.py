"""Tests for the joint cold-start splitter."""

from __future__ import annotations

import numpy as np

from heterosplit import split_records
from heterosplit.synthetic import make_synthetic_dataset


def _joint_ds(**kwargs: object):  # type: ignore[no-untyped-def]
    return make_synthetic_dataset(n_context_entities=20, **kwargs)  # type: ignore[arg-type]


class TestJointColdStart:
    def test_test_records_cold_on_both_axes(self) -> None:
        # holdout: either drug new AND cell line new
        ds = _joint_ds(n_records=4000, n_source_entities=40, seed=0)
        result = split_records(
            ds.records,
            ds.spec("joint_cold_start", holdout={"drug": "either", "cell_line": "all"}),
        )
        assert result.covers_all_records()
        src, dst = ds.records.source_codes, ds.records.destination_codes
        ctx = ds.records.codes("context")

        train = result.train_indices
        test = result.test_indices
        train_drugs = np.array(sorted(set(src[train].tolist()) | set(dst[train].tolist())))
        train_cells = np.array(sorted(set(ctx[train].tolist())))

        if test.size:
            # drug axis (either): at least one endpoint is not a training drug
            drug_cold = ~np.isin(src[test], train_drugs) | ~np.isin(dst[test], train_drugs)
            # context axis (all): the cell line is not a training cell line
            cell_cold = ~np.isin(ctx[test], train_cells)
            assert np.all(drug_cold)
            assert np.all(cell_cold)

    def test_context_only_joint_matches_context_disjointness(self) -> None:
        ds = _joint_ds(n_records=2000, seed=1)
        result = split_records(
            ds.records, ds.spec("joint_cold_start", holdout={"cell_line": "all"})
        )
        ctx = ds.records.codes("context")
        train = set(ctx[result.train_indices].tolist())
        test = set(ctx[result.test_indices].tolist())
        assert train.isdisjoint(test)

    def test_both_drug_joint_disjoint_entities(self) -> None:
        ds = _joint_ds(n_records=4000, n_source_entities=40, seed=2)
        result = split_records(
            ds.records,
            ds.spec("joint_cold_start", holdout={"drug": "both", "cell_line": "all"}),
        )
        src, dst = ds.records.source_codes, ds.records.destination_codes
        train_drugs = set(src[result.train_indices].tolist()) | set(
            dst[result.train_indices].tolist()
        )
        test_drugs = set(src[result.test_indices].tolist()) | set(dst[result.test_indices].tolist())
        assert train_drugs.isdisjoint(test_drugs)

    def test_deterministic(self) -> None:
        ds = _joint_ds(n_records=1500, seed=3)
        spec_kwargs = {"holdout": {"drug": "either", "cell_line": "all"}, "seed": 5}
        a = split_records(ds.records, ds.spec("joint_cold_start", **spec_kwargs))
        b = split_records(ds.records, ds.spec("joint_cold_start", **spec_kwargs))
        np.testing.assert_array_equal(a.record_split, b.record_split)

    def test_records_excluded_reported(self) -> None:
        ds = _joint_ds(n_records=4000, n_source_entities=40, seed=4)
        result = split_records(
            ds.records,
            ds.spec("joint_cold_start", holdout={"drug": "either", "cell_line": "all"}),
        )
        # joint splits are strict: some records fall outside every pure tier
        assert result.n_excluded > 0
        assert any("excluded" in w for w in result.warnings)
