"""Property-based invariant tests across regimes.

Hypothesis generates random multipartite prediction-record datasets and we assert the
disjointness / coverage / determinism contracts hold for every regime and shape.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from heterosplit import split_records
from heterosplit.result import SplitResult
from heterosplit.synthetic import SyntheticDataset, make_synthetic_dataset

_RECORD_PARTITION_REGIMES = [
    "random",
    "pair_cold_start",
    "source_cold_start",
    "destination_cold_start",
]
_ALL_REGIMES = [*_RECORD_PARTITION_REGIMES, "either_cold_start", "both_cold_start"]

settings.register_profile("hs", max_examples=60, deadline=None)
settings.load_profile("hs")


@st.composite
def datasets(draw: st.DrawFn) -> SyntheticDataset:
    self_relation = draw(st.booleans())
    kwargs: dict[str, object] = {
        "n_records": draw(st.integers(min_value=20, max_value=250)),
        "n_source_entities": draw(st.integers(min_value=2, max_value=40)),
        "n_labels": draw(st.integers(min_value=0, max_value=4)),
        "self_relation": self_relation,
        "with_timestamps": draw(st.booleans()),
        "seed": draw(st.integers(min_value=0, max_value=9999)),
    }
    if not self_relation:
        kwargs["source_type"] = "user"
        kwargs["destination_type"] = "item"
        kwargs["n_destination_entities"] = draw(st.integers(min_value=2, max_value=40))
    if draw(st.booleans()):
        kwargs["n_context_entities"] = draw(st.integers(min_value=2, max_value=10))
    return make_synthetic_dataset(**kwargs)  # type: ignore[arg-type]


def _endpoint_sets_by_type(result: SplitResult, indices: np.ndarray) -> dict[str, set[int]]:
    """Codes of each *endpoint* entity type appearing in the given records.

    Keyed by entity type so a self-relation unions both roles into one set while a
    bipartite relation keeps its two codebooks (user vs item) separate.
    """
    schema = result.records.schema
    out: dict[str, set[int]] = {}
    for role_name in (schema.source_name, schema.destination_name):
        entity_type = schema.roles[role_name].entity_type
        codes = result.records.codes(role_name)[indices]
        out.setdefault(entity_type, set()).update(codes.tolist())
    return out


@given(datasets(), st.sampled_from(_ALL_REGIMES))
@settings(suppress_health_check=[HealthCheck.data_too_large])
def test_no_record_lost_or_duplicated(ds: SyntheticDataset, regime: str) -> None:
    result = split_records(ds.records, ds.spec(regime))
    assert result.record_split.shape == (ds.records.n_records,)
    assert result.covers_all_records()
    # every index appears in exactly one split or is excluded
    seen = np.concatenate(
        [result.indices(name) for name in result.split_names] + [result.excluded_indices]
    )
    assert seen.size == ds.records.n_records
    np.testing.assert_array_equal(np.sort(seen), np.arange(ds.records.n_records))


@given(datasets(), st.sampled_from(_ALL_REGIMES))
def test_determinism(ds: SyntheticDataset, regime: str) -> None:
    a = split_records(ds.records, ds.spec(regime, seed=123))
    b = split_records(ds.records, ds.spec(regime, seed=123))
    np.testing.assert_array_equal(a.record_split, b.record_split)


@given(datasets())
def test_source_cold_start_disjoint(ds: SyntheticDataset) -> None:
    result = split_records(ds.records, ds.spec("source_cold_start"))
    src = ds.records.source_codes
    train = set(src[result.train_indices].tolist())
    test = set(src[result.test_indices].tolist())
    assert train.isdisjoint(test)


@given(datasets())
def test_both_entity_disjoint(ds: SyntheticDataset) -> None:
    result = split_records(ds.records, ds.spec("both_cold_start"))
    train = _endpoint_sets_by_type(result, result.train_indices)
    test = _endpoint_sets_by_type(result, result.test_indices)
    for entity_type, train_codes in train.items():
        assert train_codes.isdisjoint(test.get(entity_type, set()))


@given(datasets())
def test_either_at_least_one_endpoint_unseen(ds: SyntheticDataset) -> None:
    result = split_records(ds.records, ds.spec("either_cold_start"))
    schema = ds.records.schema
    src, dst = ds.records.source_codes, ds.records.destination_codes
    test = result.test_indices
    if test.size == 0:
        return
    seen = _endpoint_sets_by_type(result, result.train_indices)
    seen_src = np.array(sorted(seen.get(schema.source_type, set())), dtype=np.int64)
    seen_dst = np.array(sorted(seen.get(schema.destination_type, set())), dtype=np.int64)
    src_unseen = ~np.isin(src[test], seen_src)
    dst_unseen = ~np.isin(dst[test], seen_dst)
    assert np.all(src_unseen | dst_unseen)


@given(datasets())
def test_symmetric_pairs_do_not_cross_splits(ds: SyntheticDataset) -> None:
    # Only meaningful for a self-relation with undirected pairs.
    if not ds.records.schema.is_self_relation:
        return
    from heterosplit.canonical import pair_group_ids

    for regime in ("random", "pair_cold_start"):
        result = split_records(ds.records, ds.spec(regime, undirected_pairs=True))
        groups, _ = pair_group_ids(
            ds.records.source_codes, ds.records.destination_codes, undirected=True
        )
        for g in np.unique(groups):
            assert np.unique(result.record_split[groups == g]).size == 1
