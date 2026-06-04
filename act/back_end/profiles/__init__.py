"""Structural gates for HZ-verifier profiles.

Per advisor 2026-06-03: profile gates should be structural triggers
(net + vnnlib + snapshot shape), not per-benchmark name checks. This
module hosts the helper predicates and their tests.
"""

from act.back_end.profiles.generic_mlp_endcap_gate import (
    supports_generic_mlp_endcap,
    GateDiagnostic,
)

__all__ = ["supports_generic_mlp_endcap", "GateDiagnostic"]
