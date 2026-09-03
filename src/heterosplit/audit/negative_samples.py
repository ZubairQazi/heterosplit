"""Negative-sampling regime auditor.

Checks a set of proposed negative ``(source, destination)`` samples for a split against
the *positive* edges of the dataset: a "negative" that is actually a positive edge
somewhere is a false negative and a leakage signal. Callers supply the negatives (as
entity codes or raw ids matching the records' encoding) and receive an
:class:`AuditFinding` they can append to a report.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..canonical import canonicalize_pairs
from ..result import SplitResult
from ._common import MAX_EXAMPLES, native
from .report import AuditFinding, Severity

__all__ = ["audit_negative_samples"]


def audit_negative_samples(
    result: SplitResult,
    negative_source: npt.ArrayLike,
    negative_destination: npt.ArrayLike,
    *,
    split: str = "test",
) -> AuditFinding:
    """Flag negative samples that coincide with a real positive edge.

    Args:
        result: The split (used for its positive edges and canonicalization policy).
        negative_source: Source entity codes of the proposed negatives.
        negative_destination: Destination entity codes of the proposed negatives.
        split: The split the negatives are intended for (reported in the message).

    Returns:
        An error-severity finding counting negatives that are actually positive edges.
    """
    records = result.records
    undirected = result.spec.undirected_pairs
    pos_lo, pos_hi = canonicalize_pairs(
        records.source_codes, records.destination_codes, undirected=undirected
    )
    positive = set(zip(pos_lo.tolist(), pos_hi.tolist(), strict=True))

    neg_lo, neg_hi = canonicalize_pairs(
        np.asarray(negative_source, dtype=np.int64),
        np.asarray(negative_destination, dtype=np.int64),
        undirected=undirected,
    )
    colliding = [
        (int(a), int(b))
        for a, b in zip(neg_lo.tolist(), neg_hi.tolist(), strict=True)
        if (a, b) in positive
    ]

    book = records.codebooks[records.schema.source_type]
    dst_book = book if undirected else records.codebooks[records.schema.destination_type]
    examples = [
        (native(book.decode(np.array([a]))[0]), native(dst_book.decode(np.array([b]))[0]))
        for a, b in colliding[:MAX_EXAMPLES]
    ]
    return AuditFinding(
        check="negative_sample_collision",
        severity=Severity.ERROR,
        count=len(colliding),
        message=(
            f"{len(colliding)} negative samples for split {split!r} coincide with real "
            "positive edges (false negatives)"
        ),
        examples=examples,
    )
