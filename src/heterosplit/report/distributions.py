"""Numeric distribution statistics for a split result."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..objective import count_missing_values, distribution_divergence, value_counts_by_split
from ..result import SplitResult

__all__ = ["degree_summary", "entity_counts", "label_distribution"]


def entity_counts(result: SplitResult) -> dict[str, dict[str, int]]:
    """Distinct entities per role, per split."""
    counts: dict[str, dict[str, int]] = {}
    for role_name in result.spec.roles:
        counts[role_name] = {
            split: int(np.unique(result.split_codes(role_name, split)).size)
            for split in result.split_names
        }
    return counts


def label_distribution(result: SplitResult) -> dict[str, Any] | None:
    """Per-split label distribution, divergence, and missing-label count (or ``None``)."""
    records = result.records
    if records.labels is None or records.label_codebook is None:
        return None
    n_splits = len(result.split_names)
    counts = value_counts_by_split(records.labels, result.record_split, records.n_labels, n_splits)
    distribution = {}
    for i, split in enumerate(result.split_names):
        total = counts[i].sum()
        distribution[split] = (counts[i] / total).tolist() if total else [0.0] * records.n_labels
    return {
        "values": [_native(v) for v in records.label_codebook.values.tolist()],
        "distribution": distribution,
        "divergence": distribution_divergence(counts, result.spec.ratios),
        "missing": count_missing_values(counts),
    }


def degree_summary(result: SplitResult) -> dict[str, dict[str, float]]:
    """Mean endpoint degree (over the full graph) of the records in each split."""
    records = result.records
    schema = records.schema
    src, dst = records.source_codes, records.destination_codes
    src_degree = np.bincount(src, minlength=records.n_entities(schema.source_type))
    dst_degree = np.bincount(dst, minlength=records.n_entities(schema.destination_type))

    summary: dict[str, dict[str, float]] = {}
    for split in result.split_names:
        idx = result.indices(split)
        summary[split] = {
            "mean_source_degree": float(src_degree[src[idx]].mean()) if idx.size else 0.0,
            "mean_destination_degree": float(dst_degree[dst[idx]].mean()) if idx.size else 0.0,
        }
    return summary


def _native(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
