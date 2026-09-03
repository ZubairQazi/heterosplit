"""Normalized internal representation of prediction records.

HeteroSplit's *first design decision* is to treat a heterogeneous observation (e.g.
a drug--drug--cell-line synergy measurement) as a **prediction record** with a
source entity ``s``, destination entity ``d``, optional context entities ``c``, an
optional relation/label ``r``, and an optional timestamp ``t`` — rather than as a
first-class hyperedge. This maps cleanly onto ordinary link prediction plus
contextual columns and integrates naturally with PyG.

The key normalization is **per-entity-type codebooks**: every column referring to
the same entity type is factorized against a single shared codebook, so an entity
that appears as a source in one record and a destination in another receives the
*same* integer code. Disjointness is then a statement about integer code sets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .errors import SchemaError
from .schema import TaskSchema

__all__ = ["Codebook", "PredictionRecords"]

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Codebook:
    """A bijection between raw entity/label values and dense integer codes.

    ``values`` holds the sorted unique raw values; the code of a value is its index.
    """

    values: npt.NDArray[np.generic]

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def decode(self, codes: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Map integer codes back to their raw values."""
        return self.values[np.asarray(codes, dtype=np.int64)]

    def encode(self, raw: npt.ArrayLike) -> IntArray:
        """Map raw values to codes, raising :class:`SchemaError` on unknown values."""
        raw_arr = np.asarray(raw)
        codes = np.searchsorted(self.values, raw_arr).astype(np.int64)
        # searchsorted can return len(values) for out-of-range values; clip before
        # the membership check to keep the fancy-index in bounds.
        in_bounds = codes < len(self)
        present = np.zeros(raw_arr.shape, dtype=bool)
        present[in_bounds] = self.values[codes[in_bounds]] == raw_arr[in_bounds]
        if not bool(np.all(present)):
            missing = np.asarray(raw_arr)[~present]
            raise SchemaError(
                f"{missing.size} value(s) not present in codebook, e.g. {missing[:5]!r}"
            )
        return codes

    @classmethod
    def build(cls, *arrays: npt.ArrayLike) -> tuple[Codebook, list[IntArray]]:
        """Build a codebook from one or more raw arrays and return per-array codes.

        All arrays are factorized against a single shared value space, which is what
        gives two columns of the same entity type a common code space.
        """
        materialized = [np.asarray(a) for a in arrays]
        if not materialized:
            raise ValueError("Codebook.build requires at least one array")
        lengths = [a.shape[0] for a in materialized]
        concat = np.concatenate(materialized) if len(materialized) > 1 else materialized[0]
        values, inverse = np.unique(concat, return_inverse=True)
        inverse = np.asarray(inverse, dtype=np.int64).reshape(-1)
        codes: list[IntArray] = []
        offset = 0
        for length in lengths:
            codes.append(inverse[offset : offset + length])
            offset += length
        return cls(values=values), codes


@dataclass(frozen=True)
class PredictionRecords:
    """A validated, code-normalized table of prediction records.

    Attributes:
        schema: The :class:`TaskSchema` describing roles and the supervision edge.
        columns: Mapping of role/column name to its integer code array (length ``n``).
        codebooks: Mapping of entity type to its shared :class:`Codebook`.
        labels: Optional per-record relation/label codes (length ``n``).
        label_codebook: Codebook for ``labels`` if present.
        timestamps: Optional per-record timestamps (length ``n``), kept as-is.
    """

    schema: TaskSchema
    columns: dict[str, IntArray]
    codebooks: dict[str, Codebook]
    labels: IntArray | None = None
    label_codebook: Codebook | None = None
    timestamps: npt.NDArray[np.generic] | None = None

    # -- construction --------------------------------------------------------

    @classmethod
    def from_columns(
        cls,
        schema: TaskSchema,
        columns: Mapping[str, Sequence[object] | npt.ArrayLike],
        *,
        labels: Sequence[object] | npt.ArrayLike | None = None,
        timestamps: Sequence[object] | npt.ArrayLike | None = None,
    ) -> PredictionRecords:
        """Build records from raw per-column values, factorizing per entity type."""
        missing = set(schema.roles) - set(columns)
        if missing:
            raise SchemaError(f"missing columns for roles: {sorted(missing)}")

        raw: dict[str, npt.NDArray[np.generic]] = {
            name: np.asarray(columns[name]) for name in schema.roles
        }
        n = _consistent_length(raw)

        codebooks: dict[str, Codebook] = {}
        code_columns: dict[str, IntArray] = {}
        for entity_type in sorted(schema.entity_types):
            names = [name for name, role in schema.roles.items() if role.entity_type == entity_type]
            codebook, codes = Codebook.build(*(raw[name] for name in names))
            codebooks[entity_type] = codebook
            code_columns.update(zip(names, codes, strict=True))

        label_codes: IntArray | None = None
        label_codebook: Codebook | None = None
        if labels is not None:
            label_arr = np.asarray(labels)
            if label_arr.shape[0] != n:
                raise SchemaError(f"labels length {label_arr.shape[0]} != n_records {n}")
            label_codebook, [label_codes] = Codebook.build(label_arr)

        ts_arr: npt.NDArray[np.generic] | None = None
        if timestamps is not None:
            ts_arr = np.asarray(timestamps)
            if ts_arr.shape[0] != n:
                raise SchemaError(f"timestamps length {ts_arr.shape[0]} != n_records {n}")

        return cls(
            schema=schema,
            columns=code_columns,
            codebooks=codebooks,
            labels=label_codes,
            label_codebook=label_codebook,
            timestamps=ts_arr,
        )

    # -- accessors -----------------------------------------------------------

    @property
    def n_records(self) -> int:
        return int(self.columns[self.schema.source_name].shape[0])

    def codes(self, role_name: str) -> IntArray:
        """Integer code array for a role/column."""
        return self.columns[role_name]

    def codebook_for(self, role_name: str) -> Codebook:
        """The shared codebook of the entity type behind a role."""
        return self.codebooks[self.schema.roles[role_name].entity_type]

    def raw_ids(self, role_name: str) -> npt.NDArray[np.generic]:
        """Decode a role's codes back to raw entity ids."""
        return self.codebook_for(role_name).decode(self.columns[role_name])

    @property
    def source_codes(self) -> IntArray:
        return self.columns[self.schema.source_name]

    @property
    def destination_codes(self) -> IntArray:
        return self.columns[self.schema.destination_name]

    def context_codes(self) -> dict[str, IntArray]:
        """Code arrays for each context column, keyed by column name."""
        return {name: self.columns[name] for name in self.schema.context_names}

    def n_entities(self, entity_type: str) -> int:
        return len(self.codebooks[entity_type])

    @property
    def has_labels(self) -> bool:
        return self.labels is not None

    @property
    def n_labels(self) -> int:
        return 0 if self.label_codebook is None else len(self.label_codebook)


def _consistent_length(raw: Mapping[str, npt.NDArray[np.generic]]) -> int:
    lengths = {name: arr.shape[0] for name, arr in raw.items()}
    for name, arr in raw.items():
        if arr.ndim != 1:
            raise SchemaError(f"column {name!r} must be 1-D, got ndim={arr.ndim}")
    distinct = set(lengths.values())
    if len(distinct) != 1:
        raise SchemaError(f"columns have inconsistent lengths: {lengths}")
    return int(distinct.pop())
