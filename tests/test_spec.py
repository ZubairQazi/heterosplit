"""Tests for SplitSpec validation and normalization."""

from __future__ import annotations

import pytest

from heterosplit import EntityRole, Regime, SpecError, SplitSpec


def make_spec(**overrides: object) -> SplitSpec:
    kwargs: dict[str, object] = {
        "supervision_edge": ("drug", "synergy", "drug"),
        "roles": {
            "left_drug": EntityRole.source("drug"),
            "right_drug": EntityRole.destination("drug"),
            "cell_line": EntityRole.context("cell_line"),
        },
    }
    kwargs.update(overrides)
    return SplitSpec(**kwargs)  # type: ignore[arg-type]


class TestRegime:
    def test_coerce_enum_and_string(self) -> None:
        assert Regime.coerce("pair_cold_start") is Regime.PAIR
        assert Regime.coerce(Regime.BOTH) is Regime.BOTH

    def test_aliases(self) -> None:
        assert Regime.coerce("transductive") is Regime.RANDOM
        assert Regime.coerce("cold_start") is Regime.EITHER

    def test_unknown_regime(self) -> None:
        with pytest.raises(SpecError, match="unknown regime"):
            Regime.coerce("banana")


class TestSplitSpecDefaults:
    def test_defaults_and_derived(self) -> None:
        spec = make_spec()
        assert spec.regime is Regime.RANDOM
        assert spec.split_names == ("train", "val", "test")
        assert spec.has_validation
        assert spec.schema.source_name == "left_drug"

    def test_two_way_ratios(self) -> None:
        spec = make_spec(ratios=(0.9, 0.1))
        assert spec.split_names == ("train", "test")
        assert not spec.has_validation


class TestSplitSpecValidation:
    @pytest.mark.parametrize("ratios", [(0.6, 0.4), (0.8, 0.1, 0.1)])
    def test_valid_ratios(self, ratios: tuple[float, ...]) -> None:
        assert make_spec(ratios=ratios).ratios == pytest.approx(ratios)

    @pytest.mark.parametrize(
        "ratios",
        [(0.8, 0.1), (0.5, 0.5, 0.5), (1.0,), (0.8, 0.1, 0.05, 0.05), (0.9, -0.1, 0.2)],
    )
    def test_invalid_ratios(self, ratios: tuple[float, ...]) -> None:
        with pytest.raises(SpecError):
            make_spec(ratios=ratios)

    def test_seed_must_be_non_negative_int(self) -> None:
        with pytest.raises(SpecError, match="seed"):
            make_spec(seed=-1)
        with pytest.raises(SpecError, match="seed"):
            make_spec(seed=True)

    def test_undirected_requires_self_relation(self) -> None:
        make_spec(undirected_pairs=True)  # drug-drug is fine
        with pytest.raises(SpecError, match="self-relation"):
            SplitSpec(
                supervision_edge=("user", "rates", "item"),
                roles={"u": EntityRole.source("user"), "i": EntityRole.destination("item")},
                undirected_pairs=True,
            )

    def test_stratify_by_validation(self) -> None:
        make_spec(stratify_by="label")
        make_spec(stratify_by="cell_line")
        with pytest.raises(SpecError, match="stratify_by"):
            make_spec(stratify_by="nonexistent")


class TestHoldout:
    def test_holdout_rejected_for_non_joint(self) -> None:
        with pytest.raises(SpecError, match="only meaningful for joint"):
            make_spec(regime="either_cold_start", holdout={"drug": "either"})

    def test_joint_requires_holdout(self) -> None:
        with pytest.raises(SpecError, match="requires a non-empty holdout"):
            make_spec(regime="joint_cold_start")

    def test_valid_joint_holdout(self) -> None:
        spec = make_spec(regime="joint_cold_start", holdout={"drug": "either", "cell_line": "all"})
        assert spec.holdout == {"drug": "either", "cell_line": "all"}

    def test_unknown_entity_type_in_holdout(self) -> None:
        with pytest.raises(SpecError, match="unknown entity type"):
            make_spec(regime="joint_cold_start", holdout={"protein": "either"})

    def test_context_mode_not_valid_for_endpoint(self) -> None:
        with pytest.raises(SpecError, match="not valid for entity type"):
            make_spec(regime="joint_cold_start", holdout={"drug": "all"})

    def test_endpoint_mode_not_valid_for_context(self) -> None:
        with pytest.raises(SpecError, match="not valid for entity type"):
            make_spec(regime="joint_cold_start", holdout={"cell_line": "either"})

    def test_all_none_holdout_rejected(self) -> None:
        with pytest.raises(SpecError, match="at least one entity type"):
            make_spec(regime="joint_cold_start", holdout={"drug": "none", "cell_line": "none"})


class TestNormalize:
    def test_normalize_is_canonical_and_serializable(self) -> None:
        import json

        spec = make_spec(regime="pair_cold_start", seed=42, undirected_pairs=True)
        norm = spec.normalize()
        # regime coerced to canonical string; roles sorted by name
        assert norm["regime"] == "pair_cold_start"
        assert list(norm["roles"]) == ["cell_line", "left_drug", "right_drug"]
        assert norm["supervision_edge"] == ["drug", "synergy", "drug"]
        assert norm["undirected_pairs"] is True
        # round-trips through JSON unchanged
        assert json.loads(json.dumps(norm)) == norm

    def test_normalize_stable_across_role_insertion_order(self) -> None:
        a = make_spec()
        b = SplitSpec(
            supervision_edge=("drug", "synergy", "drug"),
            roles={
                "cell_line": EntityRole.context("cell_line"),
                "right_drug": EntityRole.destination("drug"),
                "left_drug": EntityRole.source("drug"),
            },
        )
        assert a.normalize() == b.normalize()
