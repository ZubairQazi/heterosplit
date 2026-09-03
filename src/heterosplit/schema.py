"""Task schema and entity-role vocabulary.

A HeteroSplit task is described *structurally* by:

* an :class:`EntityRole` for each prediction column, tagging it as the ``source``,
  ``destination``, or ``context`` of some entity type;
* the supervision edge type ``(source_type, relation, destination_type)`` that is
  being predicted;
* optional :class:`RelationMeta` describing other relations in the graph (used later
  for message-passing and reverse-edge handling).

This module is deliberately independent of any particular split *policy* — a
:class:`~heterosplit.spec.SplitSpec` composes a :class:`TaskSchema` with a regime,
ratios, and a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .errors import SchemaError

__all__ = [
    "EdgeType",
    "EntityRole",
    "RelationMeta",
    "RoleKind",
    "TaskSchema",
]

# A heterogeneous edge type, mirroring PyG's ``(src_type, relation, dst_type)`` triple.
EdgeType = tuple[str, str, str]


class RoleKind(str, Enum):
    """The role a prediction column plays in the supervision edge."""

    SOURCE = "source"
    DESTINATION = "destination"
    CONTEXT = "context"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


def _check_entity_type(entity_type: str) -> str:
    if not isinstance(entity_type, str) or not entity_type.strip():
        raise SchemaError(f"entity_type must be a non-empty string, got {entity_type!r}")
    return entity_type


@dataclass(frozen=True, slots=True)
class EntityRole:
    """An entity type bound to the role it plays in the supervision edge.

    Construct via the factory methods rather than the initializer directly::

        EntityRole.source("drug")
        EntityRole.destination("drug")
        EntityRole.context("cell_line")
    """

    kind: RoleKind
    entity_type: str

    def __post_init__(self) -> None:
        _check_entity_type(self.entity_type)
        if not isinstance(self.kind, RoleKind):
            raise SchemaError(f"kind must be a RoleKind, got {self.kind!r}")

    @classmethod
    def source(cls, entity_type: str) -> EntityRole:
        """A role for the *source* endpoint ``s`` of the supervision edge."""
        return cls(RoleKind.SOURCE, entity_type)

    @classmethod
    def destination(cls, entity_type: str) -> EntityRole:
        """A role for the *destination* endpoint ``d`` of the supervision edge."""
        return cls(RoleKind.DESTINATION, entity_type)

    @classmethod
    def context(cls, entity_type: str) -> EntityRole:
        """A role for a *context* entity ``c`` (e.g. a cell line) attached to the edge."""
        return cls(RoleKind.CONTEXT, entity_type)

    @property
    def is_source(self) -> bool:
        return self.kind is RoleKind.SOURCE

    @property
    def is_destination(self) -> bool:
        return self.kind is RoleKind.DESTINATION

    @property
    def is_context(self) -> bool:
        return self.kind is RoleKind.CONTEXT

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.kind.value}({self.entity_type})"


@dataclass(frozen=True, slots=True)
class RelationMeta:
    """Metadata for a heterogeneous relation (edge type) in the graph.

    Attributes:
        edge_type: The ``(src_type, relation, dst_type)`` triple.
        symmetric: Whether the relation is unordered, i.e. ``(A, B)`` and ``(B, A)``
            denote the same edge. Only meaningful when ``src_type == dst_type``.
        reverse_of: If this relation is a materialized reverse of another edge type,
            the triple of that other edge type. Used by the auditors to detect
            reverse-edge leakage.
    """

    edge_type: EdgeType
    symmetric: bool = False
    reverse_of: EdgeType | None = None

    def __post_init__(self) -> None:
        _validate_edge_type(self.edge_type, "edge_type")
        if self.symmetric and self.edge_type[0] != self.edge_type[2]:
            raise SchemaError(
                "symmetric=True is only valid when the source and destination entity "
                f"types match; got {self.edge_type!r}"
            )
        if self.reverse_of is not None:
            _validate_edge_type(self.reverse_of, "reverse_of")


def _validate_edge_type(edge_type: object, label: str) -> None:
    if not isinstance(edge_type, tuple) or len(edge_type) != 3:
        raise SchemaError(
            f"{label} must be a (src_type, relation, dst_type) triple, got {edge_type!r}"
        )
    for part in edge_type:
        if not isinstance(part, str) or not part.strip():
            raise SchemaError(f"{label} components must be non-empty strings, got {edge_type!r}")


@dataclass(frozen=True, slots=True)
class TaskSchema:
    """The structural description of a link-prediction task.

    A schema pins down *what* is being predicted and *which* entity each column
    refers to. It has exactly one source role and one destination role, plus any
    number of context roles.

    Attributes:
        supervision_edge: The ``(src_type, relation, dst_type)`` triple being predicted.
        roles: Mapping of prediction-column name to :class:`EntityRole`.
        relations: Optional metadata for additional relations in the graph.
    """

    supervision_edge: EdgeType
    roles: dict[str, EntityRole]
    relations: tuple[RelationMeta, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_edge_type(self.supervision_edge, "supervision_edge")
        if not self.roles:
            raise SchemaError("roles must not be empty")

        sources = self.role_names(RoleKind.SOURCE)
        destinations = self.role_names(RoleKind.DESTINATION)
        if len(sources) != 1:
            raise SchemaError(
                f"exactly one source role is required, found {len(sources)}: {sources}"
            )
        if len(destinations) != 1:
            raise SchemaError(
                f"exactly one destination role is required, found "
                f"{len(destinations)}: {destinations}"
            )

        src_type, _relation, dst_type = self.supervision_edge
        got_src = self.roles[sources[0]].entity_type
        got_dst = self.roles[destinations[0]].entity_type
        if got_src != src_type:
            raise SchemaError(
                f"source role entity type {got_src!r} does not match supervision_edge "
                f"source type {src_type!r}"
            )
        if got_dst != dst_type:
            raise SchemaError(
                f"destination role entity type {got_dst!r} does not match supervision_edge "
                f"destination type {dst_type!r}"
            )

    def role_names(self, kind: RoleKind) -> list[str]:
        """Column names whose role has the given :class:`RoleKind`, in insertion order."""
        return [name for name, role in self.roles.items() if role.kind is kind]

    @property
    def source_name(self) -> str:
        """The single source column name."""
        return self.role_names(RoleKind.SOURCE)[0]

    @property
    def destination_name(self) -> str:
        """The single destination column name."""
        return self.role_names(RoleKind.DESTINATION)[0]

    @property
    def context_names(self) -> list[str]:
        """Context column names, in insertion order (possibly empty)."""
        return self.role_names(RoleKind.CONTEXT)

    @property
    def source_type(self) -> str:
        return self.supervision_edge[0]

    @property
    def relation(self) -> str:
        return self.supervision_edge[1]

    @property
    def destination_type(self) -> str:
        return self.supervision_edge[2]

    @property
    def entity_types(self) -> set[str]:
        """Every distinct entity type referenced by a role."""
        return {role.entity_type for role in self.roles.values()}

    @property
    def is_self_relation(self) -> bool:
        """Whether the supervision edge connects two entities of the same type."""
        return self.source_type == self.destination_type
