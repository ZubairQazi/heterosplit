"""The split specification: a validated, serializable description of a split policy.

A :class:`SplitSpec` composes a :class:`~heterosplit.schema.TaskSchema` (what is being
predicted and which entity each column refers to) with a *regime*, target *ratios*,
a *seed*, and regime-specific options. It is the single source of truth a splitter
consumes and the manifest records verbatim, so a fixed spec + input + seed reproduces
the same split.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import SpecError
from .schema import EdgeType, EntityRole, RelationMeta, TaskSchema

__all__ = ["Regime", "SplitSpec"]

_RATIO_TOLERANCE = 1e-6

# Holdout modes allowed for an entity type depending on how it is used by a role.
_ENDPOINT_MODES = frozenset({"source", "destination", "either", "both", "none"})
_CONTEXT_MODES = frozenset({"all", "none"})


class Regime(str, Enum):
    """A split regime, defining the test-set disjointness contract."""

    RANDOM = "random"
    PAIR = "pair_cold_start"
    SOURCE = "source_cold_start"
    DESTINATION = "destination_cold_start"
    EITHER = "either_cold_start"
    BOTH = "both_cold_start"
    CONTEXT = "context_cold_start"
    JOINT = "joint_cold_start"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @classmethod
    def coerce(cls, value: Regime | str) -> Regime:
        """Coerce a string (with a few friendly aliases) or enum into a :class:`Regime`."""
        if isinstance(value, Regime):
            return value
        aliases = {"transductive": cls.RANDOM, "cold_start": cls.EITHER}
        key = str(value).strip().lower()
        if key in aliases:
            return aliases[key]
        try:
            return cls(key)
        except ValueError:
            options = ", ".join(r.value for r in cls)
            raise SpecError(f"unknown regime {value!r}; choose one of: {options}") from None


@dataclass
class SplitSpec:
    """A validated split specification.

    Args:
        supervision_edge: The ``(src_type, relation, dst_type)`` triple to predict.
        roles: Mapping of prediction-column name to :class:`EntityRole` (exactly one
            source and one destination role, plus any number of context roles).
        regime: The split regime (a :class:`Regime` or its string value).
        ratios: Target fractions per split; length 2 (train, test) or 3
            (train, val, test). Must be positive and sum to 1.
        seed: Deterministic seed for assignment.
        holdout: Required for ``joint_cold_start`` only — maps entity type to a holdout
            mode (``source``/``destination``/``either``/``both``/``none`` for endpoint
            types; ``all``/``none`` for context types).
        stratify_by: ``None`` or ``"label"`` (or a role/column name) to balance the
            distribution of that field across splits when feasible.
        undirected_pairs: Treat the supervision relation as unordered; only valid for a
            self-relation (source and destination entity types match).
        relations: Optional metadata for additional graph relations (reverse edges).
    """

    supervision_edge: EdgeType
    roles: dict[str, EntityRole]
    regime: Regime | str = Regime.RANDOM
    ratios: Sequence[float] = (0.8, 0.1, 0.1)
    seed: int = 0
    holdout: Mapping[str, str] | None = None
    stratify_by: str | None = None
    undirected_pairs: bool = False
    relations: tuple[RelationMeta, ...] = ()

    schema: TaskSchema = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Building the schema validates the edge/role structure.
        self.schema = TaskSchema(self.supervision_edge, self.roles, tuple(self.relations))
        self.regime = Regime.coerce(self.regime)
        self.ratios = self._validate_ratios(self.ratios)
        self._validate_seed()
        self._validate_undirected()
        self._validate_stratify()
        self._validate_regime_roles()
        self._validate_holdout()

    # -- validation ----------------------------------------------------------

    @staticmethod
    def _validate_ratios(ratios: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(r) for r in ratios)
        if len(values) not in (2, 3):
            raise SpecError(f"ratios must have length 2 or 3, got {len(values)}")
        if any(r <= 0.0 for r in values):
            raise SpecError(f"all ratios must be positive, got {values}")
        if not math.isclose(sum(values), 1.0, abs_tol=_RATIO_TOLERANCE):
            raise SpecError(f"ratios must sum to 1, got {values} (sum={sum(values)})")
        return values

    def _validate_seed(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise SpecError(f"seed must be a non-negative int, got {self.seed!r}")

    def _validate_undirected(self) -> None:
        if not self.undirected_pairs:
            return
        if not self.schema.is_self_relation:
            raise SpecError(
                "undirected_pairs=True is only valid for a self-relation "
                f"(source type == destination type); got {self.supervision_edge!r}"
            )
        regime = Regime.coerce(self.regime)
        if regime in (Regime.SOURCE, Regime.DESTINATION):
            raise SpecError(
                f"undirected_pairs=True is incompatible with regime {regime.value!r}: "
                "source/destination roles have no stable orientation for unordered pairs"
            )

    def _validate_stratify(self) -> None:
        if self.stratify_by is None or self.stratify_by == "label":
            return
        if self.stratify_by not in self.roles:
            raise SpecError(
                f"stratify_by must be None, 'label', or a role name; got {self.stratify_by!r}"
            )

    def _validate_regime_roles(self) -> None:
        if Regime.coerce(self.regime) is Regime.CONTEXT and len(self.schema.context_names) != 1:
            raise SpecError(
                "context_cold_start requires exactly one context role, got "
                f"{len(self.schema.context_names)}; use joint_cold_start for multiple contexts"
            )

    def _validate_holdout(self) -> None:
        if self.regime is not Regime.JOINT:
            if self.holdout is not None:
                raise SpecError(
                    f"holdout is only meaningful for joint_cold_start; got regime={self.regime}"
                )
            return
        if not self.holdout:
            raise SpecError("joint_cold_start requires a non-empty holdout mapping")

        entity_types = self.schema.entity_types
        endpoint_types = {self.schema.source_type, self.schema.destination_type}
        context_types = {self.roles[name].entity_type for name in self.schema.context_names}
        for entity_type, mode in self.holdout.items():
            if entity_type not in entity_types:
                raise SpecError(
                    f"holdout references unknown entity type {entity_type!r}; "
                    f"known: {sorted(entity_types)}"
                )
            allowed: frozenset[str] = frozenset()
            if entity_type in endpoint_types:
                allowed |= _ENDPOINT_MODES
            if entity_type in context_types:
                allowed |= _CONTEXT_MODES
            if mode not in allowed:
                raise SpecError(
                    f"holdout mode {mode!r} is not valid for entity type {entity_type!r}; "
                    f"allowed: {sorted(allowed)}"
                )
        active = {t: m for t, m in self.holdout.items() if m != "none"}
        if not active:
            raise SpecError("joint_cold_start holdout must hold out at least one entity type")

    # -- derived properties --------------------------------------------------

    @property
    def split_names(self) -> tuple[str, ...]:
        """Ordered split names implied by ``ratios`` length."""
        return ("train", "test") if len(self.ratios) == 2 else ("train", "val", "test")

    @property
    def has_validation(self) -> bool:
        return len(self.ratios) == 3

    # -- serialization -------------------------------------------------------

    def normalize(self) -> dict[str, Any]:
        """Canonical, JSON-serializable form used for manifest fingerprinting."""
        roles = {
            name: {"kind": role.kind.value, "entity_type": role.entity_type}
            for name, role in sorted(self.roles.items())
        }
        relations = [
            {
                "edge_type": list(rel.edge_type),
                "symmetric": rel.symmetric,
                "reverse_of": None if rel.reverse_of is None else list(rel.reverse_of),
            }
            for rel in self.relations
        ]
        holdout = None if self.holdout is None else dict(sorted(self.holdout.items()))
        return {
            "supervision_edge": list(self.supervision_edge),
            "roles": roles,
            "regime": Regime.coerce(self.regime).value,
            "ratios": list(self.ratios),
            "seed": self.seed,
            "holdout": holdout,
            "stratify_by": self.stratify_by,
            "undirected_pairs": self.undirected_pairs,
            "relations": relations,
        }
