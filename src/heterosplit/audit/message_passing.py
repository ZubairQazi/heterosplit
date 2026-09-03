"""Message-passing leakage auditor.

Two checks:

* ``message_passing_shared_pairs`` (warning): how many held-out supervision edges share a
  canonical pair with a *training* supervision edge. A naive message-passing graph that
  simply reuses training edges would leak these; the count quantifies the risk.
* ``message_passing_reconstruction_leak`` (error): verifies that HeteroSplit's own
  reconstructed training message-passing graph contains none of them — the machine-checkable
  form of the "held-out edges and their reverses are absent from the training graph" invariant.
"""

from __future__ import annotations

import numpy as np

from ..canonical import canonicalize_pairs
from ..result import SplitResult
from ._common import MAX_EXAMPLES, held_out_indices, native, pair_set
from .report import AuditFinding, Severity

__all__ = ["audit_message_passing"]


def audit_message_passing(result: SplitResult) -> list[AuditFinding]:
    records = result.records
    src, dst = records.source_codes, records.destination_codes
    train_idx = result.indices("train")
    held_idx = held_out_indices(result)

    train_canonical = pair_set(src, dst, train_idx, undirected=True)
    held_canonical = pair_set(src, dst, held_idx, undirected=True)
    shared = train_canonical & held_canonical

    findings = [
        AuditFinding(
            check="message_passing_shared_pairs",
            severity=Severity.WARNING,
            count=len(shared),
            message=(
                f"{len(shared)} held-out edges share a canonical pair with a training edge; "
                "the reconstructed message-passing graph removes them"
            ),
            examples=_decode_pairs(result, sorted(shared)[:MAX_EXAMPLES]),
        )
    ]

    mp = result.message_passing_edge_index()
    mp_lo, mp_hi = canonicalize_pairs(mp[0], mp[1], undirected=True)
    mp_pairs = set(zip(mp_lo.tolist(), mp_hi.tolist(), strict=True))
    leak = mp_pairs & held_canonical
    findings.append(
        AuditFinding(
            check="message_passing_reconstruction_leak",
            severity=Severity.ERROR,
            count=len(leak),
            message=(
                f"{len(leak)} held-out supervision pairs (or their reverses) remain in the "
                "reconstructed training message-passing graph"
            ),
            examples=_decode_pairs(result, sorted(leak)[:MAX_EXAMPLES]),
        )
    )
    return findings


def _decode_pairs(result: SplitResult, pairs: list[tuple[int, int]]) -> list[tuple[object, object]]:
    book = result.records.codebooks[result.records.schema.source_type]
    return [
        (native(book.decode(np.array([a]))[0]), native(book.decode(np.array([b]))[0]))
        for a, b in pairs
    ]
