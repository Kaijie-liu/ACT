# ===- act/pipeline/moe/accounting.py - Experiment Accounting --------====#

"""Auditable accounting identities shared by MoE experiment runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class GuardBinaryAccounting:
    """Disjoint decomposition of actual expert ReLU binary elimination."""

    binaries_before: int
    binaries_after: int
    binary_eliminated: int
    lp_support_eliminated: int
    milp_support_eliminated: int
    structural_or_propagation_eliminated: int

    def as_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


def guard_binary_accounting(
    binaries_before: int,
    binaries_after: int,
    support: Mapping[str, object],
) -> GuardBinaryAccounting:
    """Close and validate the guard binary-elimination accounting identity.

    ``binaries_before`` and ``binaries_after`` count actual expert ReLU binary
    variables. Support counters cover only neurons directly stabilized by their
    LP/MILP support query. Their difference is recorded, not guessed, as the
    structural-or-propagation residual.
    """
    before, after = int(binaries_before), int(binaries_after)
    lp = int(support.get("lp_eliminated", 0))
    milp = int(support.get("milp_eliminated", 0))
    if min(before, after, lp, milp) < 0:
        raise ValueError("guard accounting counts must be nonnegative")
    eliminated = before - after
    if eliminated < 0:
        raise ValueError("guard propagation increased expert ReLU binary width")
    structural = eliminated - lp - milp
    if structural < 0:
        raise ValueError("direct support eliminations exceed actual binary reduction")
    if eliminated != lp + milp + structural:
        raise AssertionError("guard binary accounting identity did not close")
    return GuardBinaryAccounting(
        binaries_before=before,
        binaries_after=after,
        binary_eliminated=eliminated,
        lp_support_eliminated=lp,
        milp_support_eliminated=milp,
        structural_or_propagation_eliminated=structural,
    )
