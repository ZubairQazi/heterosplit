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
    # Reverse forms only collapse for a self-relation; for a bipartite relation match
    # ordered (src, dst) pairs to avoid conflating the two independent codebooks.
    reverse_aware = records.schema.is_self_relation
    train_idx = result.indices("train")
    held_idx = held_out_indices(result)

    train_pairs = pair_set(src, dst, train_idx, undirected=reverse_aware)
    held_pairs = pair_set(src, dst, held_idx, undirected=reverse_aware)
    shared = train_pairs & held_pairs

    findings = [
        AuditFinding(
            check="message_passing_shared_pairs",
            severity=Severity.WARNING,
            count=len(shared),
            message=(
                f"{len(shared)} held-out edges share a pair with a training edge; "
                "the reconstructed message-passing graph removes them"
            ),
            examples=_decode_pairs(result, sorted(shared)[:MAX_EXAMPLES], undirected=reverse_aware),
        )
    ]

    mp = result.message_passing_edge_index()
    mp_lo, mp_hi = canonicalize_pairs(mp[0], mp[1], undirected=reverse_aware)
    mp_pairs = set(zip(mp_lo.tolist(), mp_hi.tolist(), strict=True))
    leak = mp_pairs & held_pairs
    findings.append(
        AuditFinding(
            check="message_passing_reconstruction_leak",
            severity=Severity.ERROR,
            count=len(leak),
            message=(
                f"{len(leak)} held-out supervision pairs (or their reverses) remain in the "
                "reconstructed training message-passing graph"
            ),
            examples=_decode_pairs(result, sorted(leak)[:MAX_EXAMPLES], undirected=reverse_aware),
        )
    )
    return findings


def _decode_pairs(
    result: SplitResult, pairs: list[tuple[int, int]], *, undirected: bool
) -> list[tuple[object, object]]:
    schema = result.records.schema
    src_book = result.records.codebooks[schema.source_type]
    dst_book = src_book if undirected else result.records.codebooks[schema.destination_type]
    return [
        (native(src_book.decode(np.array([a]))[0]), native(dst_book.decode(np.array([b]))[0]))
        for a, b in pairs
    ]
