"""Deterministic synthetic prediction-record generator.

Used by the test suite, property-based checks, examples, and benchmarks. Given a seed
and a handful of size knobs it produces a :class:`~heterosplit.records.PredictionRecords`
table for a self-relation (e.g. ``drug--drug``) or a bipartite relation (e.g.
``user--item``), optionally with context entities, relation/label values, and
timestamps.

The generator is intentionally simple (uniform sampling) — its job is to exercise the
splitters and auditors across shapes, not to model a realistic degree distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .records import PredictionRecords
from .schema import EntityRole, TaskSchema
from .spec import Regime, SplitSpec

__all__ = ["SyntheticDataset", "make_synthetic_dataset"]


@dataclass(frozen=True)
class SyntheticDataset:
    """A generated dataset plus a convenience builder for matching specs."""

    records: PredictionRecords

    def spec(self, regime: Regime | str = Regime.RANDOM, **kwargs: object) -> SplitSpec:
        """Build a :class:`SplitSpec` over this dataset's schema for the given regime."""
        return SplitSpec(
            supervision_edge=self.records.schema.supervision_edge,
            roles=dict(self.records.schema.roles),
            regime=regime,
            **kwargs,  # type: ignore[arg-type]
        )


def make_synthetic_dataset(
    *,
    n_records: int = 200,
    n_source_entities: int = 40,
    n_destination_entities: int = 40,
    n_context_entities: int = 0,
    n_labels: int = 2,
    with_timestamps: bool = False,
    self_relation: bool = True,
    allow_self_loops: bool = False,
    source_type: str = "drug",
    destination_type: str | None = None,
    context_type: str = "cell_line",
    relation: str = "rel",
    seed: int = 0,
) -> SyntheticDataset:
    """Generate a synthetic :class:`SyntheticDataset`.

    Args:
        n_records: Number of prediction records to generate.
        n_source_entities: Size of the source entity pool. For a self-relation this is
            the shared pool for both endpoints.
        n_destination_entities: Size of the destination pool (ignored for self-relations).
        n_context_entities: Size of the context pool; ``0`` disables the context column.
        n_labels: Number of distinct relation/label values; ``<= 0`` disables labels.
        with_timestamps: Attach an integer timestamp column.
        self_relation: Draw both endpoints from a single shared entity pool.
        allow_self_loops: Permit records whose two endpoints are the same entity.
        source_type: Entity type name for the source (and, for self-relations, both
            endpoints).
        destination_type: Entity type name for the destination; defaults to
            ``source_type`` for self-relations and ``"target"`` otherwise.
        context_type: Entity type name for the context column.
        relation: Relation name of the supervision edge.
        seed: Seed for the underlying ``numpy`` generator.

    Returns:
        A :class:`SyntheticDataset` wrapping the generated records.
    """
    if n_records < 0:
        raise ValueError("n_records must be non-negative")
    if n_source_entities < 1:
        raise ValueError("n_source_entities must be >= 1")
    if self_relation and not allow_self_loops and n_source_entities < 2:
        raise ValueError("a self-relation without self-loops needs n_source_entities >= 2")

    rng = np.random.default_rng(seed)
    dst_type = source_type if self_relation else (destination_type or "target")

    source = rng.integers(0, n_source_entities, size=n_records)
    if self_relation:
        destination = rng.integers(0, n_source_entities, size=n_records)
        if not allow_self_loops:
            _break_self_loops(source, destination, n_source_entities, rng)
    else:
        if n_destination_entities < 1:
            raise ValueError("n_destination_entities must be >= 1 for a bipartite relation")
        destination = rng.integers(0, n_destination_entities, size=n_records)

    roles = {
        "source": EntityRole.source(source_type),
        "destination": EntityRole.destination(dst_type),
    }
    columns: dict[str, np.ndarray] = {"source": source, "destination": destination}

    if n_context_entities > 0:
        roles["context"] = EntityRole.context(context_type)
        columns["context"] = rng.integers(0, n_context_entities, size=n_records)

    schema = TaskSchema((source_type, relation, dst_type), roles)

    labels = rng.integers(0, n_labels, size=n_records) if n_labels >= 1 else None
    timestamps = rng.integers(0, 1000, size=n_records) if with_timestamps else None

    records = PredictionRecords.from_columns(schema, columns, labels=labels, timestamps=timestamps)
    return SyntheticDataset(records=records)


def _break_self_loops(
    source: np.ndarray, destination: np.ndarray, pool: int, rng: np.random.Generator
) -> None:
    """Resample destination entries in place until no record is a self-loop."""
    mask = source == destination
    while bool(mask.any()):
        destination[mask] = rng.integers(0, pool, size=int(mask.sum()))
        mask = source == destination
