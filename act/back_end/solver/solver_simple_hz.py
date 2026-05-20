"""Demo HZ solver: 50-line template using only ACT algorithms.

This file is the worked example for "how to author a new HZ solver
under ACT". It composes only modules that live in
``act.back_end.hybridz_tf.algorithms.*`` and ``act.back_end.solver.*``.
No HyZor-repo imports.

The demo verifies a property by:
  1. Running interval / hybridz TF propagation (already in main)
  2. Compute output HZono bounds (via solver_hz.compute_bounds)
  3. Calling ``algorithms.lp_verify.check_unsafe_for_act`` on the
     output HZono + ASSERT layer
  4. Mapping the verdict to a ``BatchLPSolution`` (N=1).

This is a minimal, deliberately not-tuned baseline. Future HZ solvers
should follow the same shape but plug in their cascade / bound-tightening
strategies (e.g. eq_lagr_v8 + binary_probe in the cascade) by composing
the algorithms package.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch

from act.back_end.solver.solver_base import (
    Solver, SolverCaps, SolveStatus,
    BatchLPProblem, BatchLPSolution,
)


class SimpleHZSolver(Solver):
    """Template solver: HZ propagation + unsafe-feasibility LP.

    Implements the standard solver interface (capabilities + solve_batch)
    but rejects ``solve_batch`` since this solver consumes the HZ chain
    produced by ``analyze`` rather than a pre-built ``BatchLPProblem``.
    The intended entry is via the verifier's HZ pipeline; the
    ``solve_batch`` raise mirrors ``HZSolver``.
    """

    def __init__(self):
        self._last_bounds = None

    def capabilities(self) -> SolverCaps:
        return SolverCaps(
            supports_gpu=True, supports_csp=False, supports_hz=True,
        )

    def solve_batch(
        self,
        problem: BatchLPProblem,
        timelimit: Optional[float] = None,
    ) -> BatchLPSolution:
        raise NotImplementedError(
            "SimpleHZSolver consumes an HZ chain from analyze, not a "
            "BatchLPProblem. See module docstring."
        )

    def verify_hz(self, out_hz, assert_layer, *, timeout_s: float = 30.0
                  ) -> Tuple[str, Optional[torch.Tensor]]:
        """Decide spec on an output HZono using the unsafe-feasibility LP.

        Returns ``(status, witness)`` where status is one of
        ``"verified" / "unknown" / "feasible"``.
        """
        from act.back_end.hybridz_tf.algorithms.lp_verify import (
            check_unsafe_for_act,
        )
        st, xi = check_unsafe_for_act(
            out_hz, assert_layer, timeout_s=timeout_s,
        )
        if st == "infeasible":
            return "verified", None
        if st == "feasible":
            return "feasible", (None if xi is None
                                else torch.as_tensor(xi))
        return "unknown", None
