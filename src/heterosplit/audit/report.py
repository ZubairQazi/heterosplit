"""Audit findings and the aggregate report."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..errors import LeakageError

__all__ = ["AuditFinding", "AuditReport", "Severity"]


class Severity(str, Enum):
    """How a non-zero finding count is treated."""

    ERROR = "error"  # a contract violation -> leakage
    WARNING = "warning"  # a risk worth surfacing, not a violation
    INFO = "info"  # an observation (allowed by the contract)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass
class AuditFinding:
    """One check's result: a name, severity, offending count, and example ids."""

    check: str
    severity: Severity
    count: int
    message: str
    examples: list[Any] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """A finding passes if nothing offending was found."""
        return self.count == 0

    @property
    def is_violation(self) -> bool:
        return self.severity is Severity.ERROR and self.count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity.value,
            "count": self.count,
            "message": self.message,
            "examples": self.examples,
        }


@dataclass
class AuditReport:
    """Aggregate of all findings for a split."""

    findings: list[AuditFinding] = field(default_factory=list)

    @property
    def violations(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.is_violation]

    @property
    def warnings(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity is Severity.WARNING and f.count > 0]

    @property
    def has_leakage(self) -> bool:
        return bool(self.violations)

    def get(self, check: str) -> AuditFinding | None:
        return next((f for f in self.findings if f.check == check), None)

    def add(self, *findings: AuditFinding) -> None:
        self.findings.extend(findings)

    def raise_for_leakage(self) -> None:
        """Raise :class:`LeakageError` if any error-severity check found violations."""
        if not self.has_leakage:
            return
        details = "; ".join(
            f"{f.check}: {f.count} (e.g. {f.examples[:3]})" for f in self.violations
        )
        raise LeakageError(f"leakage detected in {len(self.violations)} check(s): {details}")

    def summary(self) -> str:
        lines = [f"AuditReport: {'LEAKAGE' if self.has_leakage else 'clean'}"]
        for finding in self.findings:
            mark = (
                "x"
                if finding.is_violation
                else ("!" if finding.count and finding.severity is Severity.WARNING else ".")
            )
            lines.append(f"  [{mark}] {finding.check}: {finding.count} ({finding.severity})")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - thin wrapper
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_leakage": self.has_leakage,
            "n_violations": len(self.violations),
            "n_warnings": len(self.warnings),
            "findings": [f.to_dict() for f in self.findings],
        }
