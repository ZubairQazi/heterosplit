"""Assemble and render a split's balance + audit summary."""

from __future__ import annotations

import json
from typing import Any

from ..result import SplitResult
from ..spec import Regime
from .distributions import degree_summary, entity_counts, label_distribution

__all__ = ["build_summary", "to_json", "to_markdown"]


def build_summary(result: SplitResult) -> dict[str, Any]:
    """A JSON-serializable summary of sizes, balance, distributions, and the audit."""
    spec = result.spec
    return {
        "regime": Regime.coerce(spec.regime).value,
        "seed": spec.seed,
        "undirected_pairs": spec.undirected_pairs,
        "sizes": {
            "total_records": result.records.n_records,
            "counts": result.counts,
            "excluded": result.n_excluded,
            "requested_ratios": dict(zip(spec.split_names, spec.ratios, strict=True)),
            "achieved_ratios": result.achieved_ratios(),
        },
        "entity_counts": entity_counts(result),
        "labels": label_distribution(result),
        "degree": degree_summary(result),
        "warnings": list(result.warnings),
        "audit": result.audit.to_dict(),
    }


def to_json(result: SplitResult, *, indent: int | None = 2) -> str:
    return json.dumps(build_summary(result), indent=indent, sort_keys=True)


def to_markdown(result: SplitResult) -> str:
    """Render a compact Markdown report."""
    summary = build_summary(result)
    lines: list[str] = []
    lines.append(f"# HeteroSplit report — `{summary['regime']}` (seed {summary['seed']})")
    lines.append("")

    sizes = summary["sizes"]
    lines.append("## Split sizes")
    lines.append("")
    lines.append("| split | records | achieved | requested |")
    lines.append("|---|---:|---:|---:|")
    for split, count in sizes["counts"].items():
        achieved = sizes["achieved_ratios"].get(split, 0.0)
        requested = sizes["requested_ratios"].get(split, 0.0)
        lines.append(f"| {split} | {count} | {achieved:.3f} | {requested:.3f} |")
    if sizes["excluded"]:
        lines.append(f"| _excluded_ | {sizes['excluded']} | — | — |")
    lines.append("")
    lines.append(f"Total records: {sizes['total_records']}")
    lines.append("")

    lines.append("## Distinct entities per split")
    lines.append("")
    lines.append("| role | " + " | ".join(result.split_names) + " |")
    lines.append("|---" + "|---:" * len(result.split_names) + "|")
    for role, per_split in summary["entity_counts"].items():
        row = " | ".join(str(per_split[s]) for s in result.split_names)
        lines.append(f"| {role} | {row} |")
    lines.append("")

    labels = summary["labels"]
    if labels is not None:
        lines.append("## Label distribution")
        lines.append("")
        lines.append(
            f"Divergence across splits: **{labels['divergence']:.4f}** "
            f"(0 = identical); missing label cells: {labels['missing']}"
        )
        lines.append("")
        lines.append("| split | " + " | ".join(str(v) for v in labels["values"]) + " |")
        lines.append("|---" + "|---:" * len(labels["values"]) + "|")
        for split in result.split_names:
            row = " | ".join(f"{p:.3f}" for p in labels["distribution"][split])
            lines.append(f"| {split} | {row} |")
        lines.append("")

    audit = summary["audit"]
    lines.append("## Leakage audit")
    lines.append("")
    status = "**LEAKAGE DETECTED**" if audit["has_leakage"] else "clean"
    lines.append(
        f"Status: {status} — {audit['n_violations']} violation(s), {audit['n_warnings']} warning(s)"
    )
    lines.append("")
    lines.append("| check | severity | count |")
    lines.append("|---|---|---:|")
    lines.extend(
        f"| {finding['check']} | {finding['severity']} | {finding['count']} |"
        for finding in audit["findings"]
    )
    lines.append("")

    if summary["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        lines.extend(f"- {w}" for w in summary["warnings"])
        lines.append("")

    return "\n".join(lines)
