"""Deterministic, serializable split manifests.

A manifest is the auditable record of a split: enough to *verify* that a split was
reproduced, without storing the (potentially huge) index arrays themselves. The
manifest separates a **deterministic core** — versions, an input fingerprint, the
normalized spec, per-split counts, and hashes of the split indices — from
non-deterministic **measurements** (runtime, peak memory). The invariant "a fixed
input, spec, and seed produce the same manifest" applies to the core, captured by
:meth:`Manifest.digest`.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from . import __version__
from .records import PredictionRecords

if TYPE_CHECKING:
    from .result import SplitResult

__all__ = ["MANIFEST_VERSION", "Manifest", "fingerprint_records"]

MANIFEST_VERSION = "1"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _values_bytes(arr: npt.NDArray[np.generic]) -> bytes:
    """Endianness-stable, collision-resistant byte encoding of a value array.

    The dtype string and element count are included so int/str/datetime arrays with the
    same textual form do not collide, and string elements are length-prefixed so an
    embedded NUL can never make two different arrays hash the same.
    """
    a = np.asarray(arr)
    out = bytearray(str(a.dtype.str).encode("ascii") + b"|" + struct.pack("<Q", int(a.shape[0])))
    kind = a.dtype.kind
    if kind in "iu":
        out += a.astype("<i8").tobytes()
    elif kind == "f":
        out += a.astype("<f8").tobytes()
    elif kind == "b":
        out += a.astype(np.int8).tobytes()
    else:
        for value in a.tolist():
            encoded = str(value).encode("utf-8")
            out += struct.pack("<Q", len(encoded)) + encoded
    return bytes(out)


def fingerprint_records(records: PredictionRecords) -> str:
    """A stable content hash of the records (columns, codebooks, labels, timestamps)."""
    h = hashlib.sha256()
    h.update(b"heterosplit-records-v1")
    h.update(repr(tuple(records.schema.supervision_edge)).encode("utf-8"))
    for name in sorted(records.columns):
        h.update(b"\x00col\x00")
        h.update(name.encode("utf-8"))
        h.update(records.columns[name].astype("<i8").tobytes())
    for entity_type in sorted(records.codebooks):
        h.update(b"\x00cb\x00")
        h.update(entity_type.encode("utf-8"))
        h.update(_values_bytes(records.codebooks[entity_type].values))
    if records.labels is not None:
        h.update(b"\x00labels\x00")
        h.update(records.labels.astype("<i8").tobytes())
    if records.label_codebook is not None:
        h.update(_values_bytes(records.label_codebook.values))
    if records.timestamps is not None:
        h.update(b"\x00ts\x00")
        h.update(_values_bytes(records.timestamps))
    return h.hexdigest()


def _hash_indices(indices: npt.NDArray[np.int64]) -> str:
    return _sha256_hex(np.sort(indices).astype("<i8").tobytes())


@dataclass
class Manifest:
    """The auditable, JSON-serializable record of a split.

    Construct with :meth:`from_result`; persist with :meth:`save`; reload with
    :meth:`load` (which does *not* re-run the split).
    """

    manifest_version: str
    heterosplit_version: str
    input_fingerprint: str
    spec: dict[str, Any]
    record_counts: dict[str, int]
    entity_counts: dict[str, dict[str, int]]
    achieved_ratios: dict[str, float]
    index_hashes: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    audit: dict[str, Any] | None = None
    distributions: dict[str, Any] | None = None
    measurements: dict[str, Any] | None = None

    # -- construction --------------------------------------------------------

    @classmethod
    def from_result(
        cls,
        result: SplitResult,
        *,
        audit: dict[str, Any] | None = None,
        distributions: dict[str, Any] | None = None,
        measurements: dict[str, Any] | None = None,
    ) -> Manifest:
        """Build a manifest from a :class:`SplitResult`."""
        records = result.records
        spec = result.spec

        record_counts = dict(result.counts)
        record_counts["excluded"] = result.n_excluded

        entity_counts: dict[str, dict[str, int]] = {}
        for role_name in spec.roles:
            per_split = {
                split: int(np.unique(result.split_codes(role_name, split)).size)
                for split in spec.split_names
            }
            entity_counts[role_name] = per_split

        index_hashes = {split: _hash_indices(result.indices(split)) for split in spec.split_names}
        index_hashes["excluded"] = _hash_indices(result.excluded_indices)

        if audit is None:
            audit = result.audit.to_dict()

        return cls(
            manifest_version=MANIFEST_VERSION,
            heterosplit_version=__version__,
            input_fingerprint=fingerprint_records(records),
            spec=spec.normalize(),
            record_counts=record_counts,
            entity_counts=entity_counts,
            achieved_ratios=result.achieved_ratios(),
            index_hashes=index_hashes,
            warnings=list(result.warnings),
            audit=audit,
            distributions=distributions,
            measurements=measurements,
        )

    # -- serialization -------------------------------------------------------

    def to_dict(self, *, include_measurements: bool = True) -> dict[str, Any]:
        """Serialize to a plain dict. The deterministic core omits ``measurements``."""
        data: dict[str, Any] = {
            "manifest_version": self.manifest_version,
            "heterosplit_version": self.heterosplit_version,
            "input_fingerprint": self.input_fingerprint,
            "spec": self.spec,
            "record_counts": self.record_counts,
            "entity_counts": self.entity_counts,
            "achieved_ratios": self.achieved_ratios,
            "index_hashes": self.index_hashes,
            "warnings": self.warnings,
            "audit": self.audit,
            "distributions": self.distributions,
        }
        if include_measurements:
            data["measurements"] = self.measurements
        return data

    def digest(self) -> str:
        """SHA-256 over the canonical JSON of the deterministic core.

        Two manifests share a digest iff their input, spec, seed, counts, and index
        hashes match — the machine-checkable form of the reproducibility invariant.
        """
        core = self.to_dict(include_measurements=False)
        canonical = json.dumps(core, sort_keys=True, separators=(",", ":"))
        return _sha256_hex(canonical.encode("utf-8"))

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        return cls(
            manifest_version=data["manifest_version"],
            heterosplit_version=data["heterosplit_version"],
            input_fingerprint=data["input_fingerprint"],
            spec=data["spec"],
            record_counts=data["record_counts"],
            entity_counts=data["entity_counts"],
            achieved_ratios=data["achieved_ratios"],
            index_hashes=data["index_hashes"],
            warnings=list(data.get("warnings", [])),
            audit=data.get("audit"),
            distributions=data.get("distributions"),
            measurements=data.get("measurements"),
        )

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        """Load a manifest from disk without re-running the split."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
