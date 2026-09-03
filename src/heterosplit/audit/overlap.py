"""Overlap-based auditors: entity, pair (reverse-aware), context, either, duplicates."""

from __future__ import annotations

import numpy as np

from ..canonical import canonicalize_pairs
from ..records import PredictionRecords
from ..result import SplitResult
from ._common import (
    MAX_EXAMPLES,
    decode_role_examples,
    held_out_indices,
    native,
    pair_set,
)
from .contract import Contract
from .report import AuditFinding, Severity

__all__ = [
    "audit_context_overlap",
    "audit_duplicate_records",
    "audit_either_unseen",
    "audit_endpoint_overlap",
    "audit_pair_overlap",
]


def _role_overlap_codes(result: SplitResult, role_name: str) -> set[int]:
    codes = result.records.codes(role_name)
    train = set(codes[result.indices("train")].tolist())
    held = set(codes[held_out_indices(result)].tolist())
    return train & held


def audit_endpoint_overlap(result: SplitResult, contract: Contract) -> list[AuditFinding]:
    """Source / destination / both-endpoint entity overlap between train and held-out."""
    findings: list[AuditFinding] = []
    schema = result.records.schema
    if contract.source_disjoint:
        findings.append(_role_finding(result, schema.source_name, "source_entity_overlap"))
    if contract.destination_disjoint:
        findings.append(
            _role_finding(result, schema.destination_name, "destination_entity_overlap")
        )
    if contract.both_endpoints_disjoint:
        findings.extend(_endpoint_type_findings(result))
    return findings


def _role_finding(result: SplitResult, role_name: str, check: str) -> AuditFinding:
    overlap = sorted(_role_overlap_codes(result, role_name))
    examples = decode_role_examples(result.records, role_name, overlap)
    return AuditFinding(
        check=check,
        severity=Severity.ERROR,
        count=len(overlap),
        message=(
            f"{len(overlap)} {role_name!r} entities appear in both training and a held-out split"
        ),
        examples=examples,
    )


def _endpoint_type_findings(result: SplitResult) -> list[AuditFinding]:
    schema = result.records.schema
    roles_by_type: dict[str, list[str]] = {}
    for role_name in (schema.source_name, schema.destination_name):
        roles_by_type.setdefault(schema.roles[role_name].entity_type, []).append(role_name)

    findings: list[AuditFinding] = []
    for entity_type, role_names in sorted(roles_by_type.items()):
        train, held = set(), set()
        for role_name in role_names:
            codes = result.records.codes(role_name)
            train |= set(codes[result.indices("train")].tolist())
            held |= set(codes[held_out_indices(result)].tolist())
        overlap = sorted(train & held)
        book = result.records.codebooks[entity_type]
        examples = (
            [native(v) for v in book.decode(np.array(overlap[:MAX_EXAMPLES]))] if overlap else []
        )
        findings.append(
            AuditFinding(
                check=f"both_endpoint_overlap:{entity_type}",
                severity=Severity.ERROR,
                count=len(overlap),
                message=(
                    f"{len(overlap)} {entity_type!r} entities appear in both training and a "
                    "held-out split"
                ),
                examples=examples,
            )
        )
    return findings


def audit_pair_overlap(result: SplitResult, contract: Contract) -> list[AuditFinding]:
    """Pair overlap between train and held-out, including reversed unordered pairs."""
    records = result.records
    src, dst = records.source_codes, records.destination_codes
    train_idx = result.indices("train")
    held_idx = held_out_indices(result)

    ordered_train = pair_set(src, dst, train_idx, undirected=False)
    ordered_held = pair_set(src, dst, held_idx, undirected=False)
    ordered_overlap = ordered_train & ordered_held

    canonical_train = pair_set(src, dst, train_idx, undirected=True)
    canonical_held = pair_set(src, dst, held_idx, undirected=True)
    canonical_overlap = canonical_train & canonical_held

    findings: list[AuditFinding] = []
    if contract.pair_disjoint and result.spec.undirected_pairs:
        findings.append(
            _pair_finding(
                records, "pair_overlap", Severity.ERROR, canonical_overlap, undirected=True
            )
        )
    elif contract.pair_disjoint:
        findings.append(
            _pair_finding(
                records, "pair_overlap", Severity.ERROR, ordered_overlap, undirected=False
            )
        )
        # Canonical overlaps not explained by an exact ordered match => reversed-pair leak.
        reversed_leak = canonical_overlap - {tuple(sorted(p)) for p in ordered_overlap}
        findings.append(
            _pair_finding(
                records,
                "reversed_pair_overlap",
                Severity.WARNING,
                reversed_leak,
                undirected=True,
            )
        )
    return findings


def _pair_finding(
    records: PredictionRecords,
    check: str,
    severity: Severity,
    pairs: set[tuple[int, int]],
    *,
    undirected: bool,
) -> AuditFinding:
    schema = records.schema
    src_book = records.codebooks[schema.source_type]
    dst_book = src_book if undirected else records.codebooks[schema.destination_type]
    examples = [
        (native(src_book.decode(np.array([a]))[0]), native(dst_book.decode(np.array([b]))[0]))
        for a, b in sorted(pairs)[:MAX_EXAMPLES]
    ]
    kind = "canonical" if undirected else "ordered"
    return AuditFinding(
        check=check,
        severity=severity,
        count=len(pairs),
        message=f"{len(pairs)} {kind} pairs appear in both training and a held-out split",
        examples=examples,
    )


def audit_context_overlap(result: SplitResult) -> list[AuditFinding]:
    """Context entity overlap between train and held-out for every context role."""
    return [
        _role_finding(result, role_name, f"context_overlap:{role_name}")
        for role_name in result.records.schema.context_names
    ]


def audit_either_unseen(result: SplitResult) -> AuditFinding:
    """Every held-out edge must have at least one endpoint unseen in training."""
    records = result.records
    schema = records.schema
    src, dst = records.source_codes, records.destination_codes
    train_idx = result.indices("train")

    seen_src = _seen_codes(records, schema.source_type, train_idx)
    seen_dst = _seen_codes(records, schema.destination_type, train_idx)

    held_idx = held_out_indices(result)
    src_seen = np.isin(src[held_idx], seen_src)
    dst_seen = np.isin(dst[held_idx], seen_dst)
    violating = held_idx[src_seen & dst_seen]

    examples = [
        (
            native(records.raw_ids(schema.source_name)[i]),
            native(records.raw_ids(schema.destination_name)[i]),
        )
        for i in violating[:MAX_EXAMPLES]
    ]
    return AuditFinding(
        check="either_endpoint_unseen",
        severity=Severity.ERROR,
        count=int(violating.size),
        message=(
            f"{int(violating.size)} held-out edges have both endpoints seen in training "
            "(no cold endpoint)"
        ),
        examples=examples,
    )


def _seen_codes(records: PredictionRecords, entity_type: str, train_idx: np.ndarray) -> np.ndarray:
    """Sorted codes of ``entity_type`` appearing in training records (any endpoint role)."""
    schema = records.schema
    seen: set[int] = set()
    for role_name in (schema.source_name, schema.destination_name):
        if schema.roles[role_name].entity_type == entity_type:
            seen |= set(records.codes(role_name)[train_idx].tolist())
    return np.array(sorted(seen), dtype=np.int64)


def audit_duplicate_records(result: SplitResult) -> AuditFinding:
    """Identical observations (endpoints + context) landing in more than one split."""
    records = result.records
    src, dst = records.source_codes, records.destination_codes
    if result.spec.undirected_pairs:
        lo, hi = canonicalize_pairs(src, dst, undirected=True)
    else:
        lo, hi = src, dst
    columns = [lo, hi] + [records.codes(name) for name in records.schema.context_names]
    rows = np.stack(columns, axis=1)

    assigned = result.record_split >= 0
    rows_a, splits_a = rows[assigned], result.record_split[assigned]
    if rows_a.shape[0] == 0:
        return AuditFinding("duplicate_records_across_splits", Severity.WARNING, 0, "no records")

    _, inverse = np.unique(rows_a, axis=0, return_inverse=True)
    inverse = np.asarray(inverse, dtype=np.int64).reshape(-1)
    sig_split = np.unique(np.stack([inverse, splits_a], axis=1), axis=0)
    sig_ids, counts = np.unique(sig_split[:, 0], return_counts=True)
    crossing = sig_ids[counts > 1]

    return AuditFinding(
        check="duplicate_records_across_splits",
        severity=Severity.WARNING,
        count=int(crossing.size),
        message=(
            f"{int(crossing.size)} identical observations (endpoints+context) span multiple splits"
        ),
        examples=[tuple(int(v) for v in rows_a[inverse == s][0]) for s in crossing[:MAX_EXAMPLES]],
    )
