"""Derive the disjointness contract a regime promises, so auditors can verify it."""

from __future__ import annotations

from dataclasses import dataclass

from ..spec import Regime, SplitSpec

__all__ = ["Contract", "contract_for"]


@dataclass
class Contract:
    """The disjointness properties a split regime guarantees between train and held-out.

    Auditors treat a violated ``True`` property as leakage (error severity).
    """

    source_disjoint: bool = False
    destination_disjoint: bool = False
    both_endpoints_disjoint: bool = False
    either_endpoint_unseen: bool = False
    pair_disjoint: bool = False
    context_disjoint: bool = False


def contract_for(spec: SplitSpec) -> Contract:
    """Map a spec's regime (and joint holdout) to the properties auditors must enforce."""
    regime = Regime.coerce(spec.regime)
    simple = {
        Regime.RANDOM: Contract(),
        Regime.PAIR: Contract(pair_disjoint=True),
        Regime.SOURCE: Contract(source_disjoint=True),
        Regime.DESTINATION: Contract(destination_disjoint=True),
        Regime.EITHER: Contract(either_endpoint_unseen=True),
        Regime.BOTH: Contract(both_endpoints_disjoint=True),
        Regime.CONTEXT: Contract(context_disjoint=True),
    }
    if regime in simple:
        return simple[regime]

    # Joint: union of the per-axis contracts implied by the holdout modes.
    contract = Contract()
    holdout = spec.holdout or {}
    for mode in holdout.values():
        if mode == "either":
            contract.either_endpoint_unseen = True
        elif mode == "both":
            contract.both_endpoints_disjoint = True
        elif mode == "source":
            contract.source_disjoint = True
        elif mode == "destination":
            contract.destination_disjoint = True
        elif mode == "all":
            contract.context_disjoint = True
    return contract
