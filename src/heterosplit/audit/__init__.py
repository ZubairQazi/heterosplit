"""Leakage auditing: turn each regime's contract into machine-checkable findings.

The auditor is HeteroSplit's strongest differentiator. :func:`audit_split` runs every
automatic check appropriate to the split's regime and returns an :class:`AuditReport`
whose :meth:`~AuditReport.raise_for_leakage` fails loudly on any contract violation.
Every finding carries a count and a few concrete offending ids.

Optional auditors that need extra inputs — negative samples and feature provenance — are
exposed for callers to run and append to a report.
"""

from __future__ import annotations

from ..result import SplitResult
from .contract import Contract, contract_for
from .feature_provenance import audit_feature_provenance
from .message_passing import audit_message_passing
from .negative_samples import audit_negative_samples
from .overlap import (
    audit_context_overlap,
    audit_duplicate_records,
    audit_either_unseen,
    audit_endpoint_overlap,
    audit_pair_overlap,
)
from .report import AuditFinding, AuditReport, Severity

__all__ = [
    "AuditFinding",
    "AuditReport",
    "Contract",
    "Severity",
    "audit_context_overlap",
    "audit_duplicate_records",
    "audit_either_unseen",
    "audit_endpoint_overlap",
    "audit_feature_provenance",
    "audit_message_passing",
    "audit_negative_samples",
    "audit_pair_overlap",
    "audit_split",
    "contract_for",
]


def audit_split(result: SplitResult) -> AuditReport:
    """Run every automatic leakage check appropriate to ``result``'s regime."""
    contract = contract_for(result.spec)
    findings: list[AuditFinding] = []
    findings.extend(audit_endpoint_overlap(result, contract))
    findings.extend(audit_pair_overlap(result, contract))
    if contract.either_endpoint_unseen:
        findings.append(audit_either_unseen(result))
    if contract.context_disjoint:
        findings.extend(audit_context_overlap(result))
    findings.append(audit_duplicate_records(result))
    findings.extend(audit_message_passing(result))
    return AuditReport(findings=findings)
