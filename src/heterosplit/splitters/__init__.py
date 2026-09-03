"""Split-policy implementations and the regime -> splitter registry."""

from __future__ import annotations

from ..errors import HeteroSplitError
from ..spec import Regime
from .base import Splitter
from .pair import PairDisjointSplitter
from .random import RandomSplitter

__all__ = ["Splitter", "get_splitter", "supported_regimes"]

_SPLITTERS: dict[Regime, type[Splitter]] = {
    RandomSplitter.regime: RandomSplitter,
    PairDisjointSplitter.regime: PairDisjointSplitter,
}


def get_splitter(regime: Regime | str) -> Splitter:
    """Instantiate the splitter registered for ``regime``."""
    resolved = Regime.coerce(regime)
    try:
        cls = _SPLITTERS[resolved]
    except KeyError:
        raise HeteroSplitError(
            f"regime {resolved.value!r} is not implemented yet; "
            f"available: {sorted(r.value for r in _SPLITTERS)}"
        ) from None
    return cls()


def supported_regimes() -> list[Regime]:
    """Regimes with a registered splitter, in declaration order."""
    return list(_SPLITTERS)
