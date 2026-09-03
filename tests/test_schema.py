"""Tests for the entity-role vocabulary and task schema."""

from __future__ import annotations

import pytest

from heterosplit import EntityRole, RelationMeta, RoleKind, SchemaError, TaskSchema


def synergy_roles() -> dict[str, EntityRole]:
    return {
        "left_drug": EntityRole.source("drug"),
        "right_drug": EntityRole.destination("drug"),
        "cell_line": EntityRole.context("cell_line"),
    }


class TestEntityRole:
    def test_factories_set_kind_and_type(self) -> None:
        s = EntityRole.source("drug")
        d = EntityRole.destination("drug")
        c = EntityRole.context("cell_line")
        assert (s.kind, s.entity_type) == (RoleKind.SOURCE, "drug")
        assert (d.kind, d.entity_type) == (RoleKind.DESTINATION, "drug")
        assert (c.kind, c.entity_type) == (RoleKind.CONTEXT, "cell_line")

    def test_is_predicates(self) -> None:
        assert EntityRole.source("drug").is_source
        assert EntityRole.destination("drug").is_destination
        assert EntityRole.context("cell_line").is_context
        assert not EntityRole.source("drug").is_context

    def test_is_frozen_and_hashable(self) -> None:
        role = EntityRole.source("drug")
        with pytest.raises(AttributeError):
            role.entity_type = "gene"  # type: ignore[misc]
        assert role in {EntityRole.source("drug")}

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_rejects_empty_entity_type(self, bad: str) -> None:
        with pytest.raises(SchemaError):
            EntityRole.source(bad)


class TestRelationMeta:
    def test_symmetric_requires_matching_endpoints(self) -> None:
        RelationMeta(("drug", "synergy", "drug"), symmetric=True)  # ok
        with pytest.raises(SchemaError):
            RelationMeta(("drug", "targets", "gene"), symmetric=True)

    def test_reverse_of_validated(self) -> None:
        rel = RelationMeta(
            ("gene", "rev_targets", "drug"),
            reverse_of=("drug", "targets", "gene"),
        )
        assert rel.reverse_of == ("drug", "targets", "gene")

    @pytest.mark.parametrize("bad", [("a", "b"), ("a", "", "c"), "not-a-tuple"])
    def test_rejects_malformed_edge_type(self, bad: object) -> None:
        with pytest.raises(SchemaError):
            RelationMeta(bad)  # type: ignore[arg-type]


class TestTaskSchema:
    def test_accessors(self) -> None:
        schema = TaskSchema(("drug", "synergy", "drug"), synergy_roles())
        assert schema.source_name == "left_drug"
        assert schema.destination_name == "right_drug"
        assert schema.context_names == ["cell_line"]
        assert schema.source_type == "drug"
        assert schema.relation == "synergy"
        assert schema.destination_type == "drug"
        assert schema.entity_types == {"drug", "cell_line"}
        assert schema.is_self_relation

    def test_bipartite_schema_is_not_self_relation(self) -> None:
        schema = TaskSchema(
            ("user", "rates", "item"),
            {"u": EntityRole.source("user"), "i": EntityRole.destination("item")},
        )
        assert not schema.is_self_relation
        assert schema.context_names == []

    def test_requires_exactly_one_source(self) -> None:
        with pytest.raises(SchemaError, match="source role"):
            TaskSchema(
                ("drug", "synergy", "drug"),
                {
                    "a": EntityRole.source("drug"),
                    "b": EntityRole.source("drug"),
                    "c": EntityRole.destination("drug"),
                },
            )

    def test_requires_exactly_one_destination(self) -> None:
        with pytest.raises(SchemaError, match="destination role"):
            TaskSchema(
                ("drug", "synergy", "drug"),
                {"a": EntityRole.source("drug")},
            )

    def test_source_type_must_match_supervision_edge(self) -> None:
        with pytest.raises(SchemaError, match="source role entity type"):
            TaskSchema(
                ("drug", "synergy", "drug"),
                {"a": EntityRole.source("gene"), "b": EntityRole.destination("drug")},
            )

    def test_empty_roles_rejected(self) -> None:
        with pytest.raises(SchemaError, match="roles must not be empty"):
            TaskSchema(("drug", "synergy", "drug"), {})
