# ===- act/back_end/moe/verifier.py - Route-A Verification Scheduler ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

from __future__ import annotations

from typing import Callable

from act.back_end.moe.hz_routing import CandidateReport, RouteBranch
from act.back_end.moe.schema import OutputLevelMoESpec
from act.util.stats import VerifyResult, VerifyStatus


ExpertVerifier = Callable[[int, RouteBranch], VerifyResult]
CounterexampleValidator = Callable[[int, VerifyResult], bool]


def verify_output_gate_elimination(
    spec: OutputLevelMoESpec,
    candidates: CandidateReport,
    verify_expert: ExpertVerifier,
    *,
    validate_counterexample: CounterexampleValidator | None = None,
    property_is_convex: bool = False,
    property_is_convex_cone: bool = False,
) -> VerifyResult:
    """Run the output-level gate-elimination sufficient certificate.

    Every feasible candidate expert is checked only on its top-k membership
    region.  A weighted-MoE expert violation yields UNKNOWN, not FALSIFIED,
    because gate elimination is incomplete there.  For hard top-1, a concrete
    witness may be reported only after full-model forward validation.
    """
    applicable = (spec.gate_elimination_applicable and property_is_convex) or (
        spec.affine_decoder and spec.nonnegative and property_is_convex_cone
    )
    if not applicable and not spec.hard_routing:
        return VerifyResult(
            VerifyStatus.UNKNOWN,
            metadata={"source": "moe_route_a", "reason": "gate_elimination_not_applicable"},
        )

    by_expert = {branch.expert: branch for branch in candidates.branches}
    results: list[tuple[int, VerifyResult]] = []
    for expert in candidates.candidates:
        result = verify_expert(expert, by_expert[expert])
        results.append((expert, result))
        if result.status == VerifyStatus.FALSIFIED and spec.hard_routing:
            validated = (
                validate_counterexample(expert, result)
                if validate_counterexample is not None
                else False
            )
            if validated:
                result.metadata.update(
                    {
                        "source": "moe_route_a",
                        "expert": expert,
                        "route_validated": True,
                    }
                )
                return result

    statuses = [result.status for _, result in results]
    metadata = {
        "source": "moe_route_a",
        "candidates": candidates.candidates,
        "infeasible": candidates.infeasible,
        "candidate_set_minimal": candidates.minimal,
        "expert_statuses": {
            expert: result.status.value for expert, result in results
        },
    }
    if all(status == VerifyStatus.CERTIFIED for status in statuses):
        return VerifyResult(VerifyStatus.CERTIFIED, metadata=metadata)
    if any(status == VerifyStatus.TIMEOUT for status in statuses):
        metadata["reason"] = "candidate_expert_timeout"
        return VerifyResult(VerifyStatus.TIMEOUT, metadata=metadata)
    metadata["reason"] = (
        "weighted_gate_elimination_incomplete"
        if not spec.hard_routing
        else "candidate_expert_undecided_or_unvalidated"
    )
    return VerifyResult(VerifyStatus.UNKNOWN, metadata=metadata)
