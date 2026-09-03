"""Feature-provenance auditor.

Precomputed features or embeddings fit on the *whole* entity set leak information about
held-out entities into training. Given the set of entity raw ids that were used to fit a
feature (e.g. a node2vec run, a PCA over all drugs), this auditor flags any that belong
to a held-out split, so globally-fit features are caught before they bias results.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..result import SplitResult
from ._common import MAX_EXAMPLES, held_out_indices, native
from .report import AuditFinding, Severity

__all__ = ["audit_feature_provenance"]


def audit_feature_provenance(
    result: SplitResult,
    fit_entity_ids: Iterable[Any],
    *,
    role_name: str,
    severity: Severity = Severity.WARNING,
) -> AuditFinding:
    """Flag held-out entities that were used to fit a feature/embedding.

    Args:
        result: The split.
        fit_entity_ids: Raw entity ids used when fitting the feature (any iterable).
        role_name: Which role's entity type the feature is over (e.g. ``"source"``).
        severity: ``WARNING`` by default; pass ``ERROR`` to make global fitting a hard fail.

    Returns:
        A finding counting held-out entities present in ``fit_entity_ids``.
    """
    records = result.records
    entity_type = records.schema.roles[role_name].entity_type

    held_idx = held_out_indices(result)
    held_codes: set[int] = set()
    for name, role in records.schema.roles.items():
        if role.entity_type == entity_type:
            held_codes |= set(records.codes(name)[held_idx].tolist())
    held_raw = (
        {native(v) for v in records.codebooks[entity_type].decode(np.array(sorted(held_codes)))}
        if held_codes
        else set()
    )

    leaked = sorted({native(x) for x in fit_entity_ids} & held_raw, key=repr)
    return AuditFinding(
        check="feature_provenance",
        severity=severity,
        count=len(leaked),
        message=(
            f"{len(leaked)} held-out {entity_type!r} entities were used to fit a feature "
            "(features should be fit on training entities only)"
        ),
        examples=leaked[:MAX_EXAMPLES],
    )
