"""ReLU encoding cascade controller for HZ-based forward verification.

Background
----------
Tight ReLU encodings (``eq_lagr_v8``: 4 continuous + 1 binary +
3 equality rows per unstable neuron) give the precision needed at the
verification boundary but are expensive on every layer of a deep
network. The cascade controller picks a CHEAPER encoding on early
layers (interval / triangle) and switches to the tighter encoding only
on the final ``K`` layers near the output — analogous to
``α/β-CROWN``'s last-layer LP refinement.

This module exposes a pure-policy ``pick_relu_method(layer_idx,
total_relus, *, last_k=None)`` that returns the method name to use at
each ReLU. The TF dispatcher consumes the choice. The policy is
deliberately stateless and config-driven so future HZ solvers can plug
in their own cascade strategies.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class CascadePolicy:
    """Per-network cascade configuration.

    Attributes:
        early_method: ReLU encoding for layers OUTSIDE the last_k tail.
            "triangle" (DeepZ-style single-chord; no binary slot) is the
            default; "bigM" or "interval" are alternatives.
        late_method: ReLU encoding for the last ``last_k`` layers.
            "eq_lagr_v8" is the HyZor tight encoding (linking equality +
            binary z + 4 xi); "exact_lp" / "bigM" are alternatives.
        last_k: how many trailing ReLUs use late_method. 0 disables the
            tail (cascade off, use early_method everywhere). None means
            "use the value of HYZOR_LARGE_CLS_EQ_LAYERS env var, default 3".
    """

    early_method: str = "triangle"
    late_method: str = "eq_lagr_v8"
    last_k: Optional[int] = None


def _resolve_last_k(policy_last_k: Optional[int]) -> int:
    if policy_last_k is not None:
        return int(policy_last_k)
    return int(os.environ.get("HYZOR_LARGE_CLS_EQ_LAYERS", "3"))


def pick_relu_method(relu_idx: int, total_relus: int,
                     policy: Optional[CascadePolicy] = None) -> str:
    """Return the ReLU encoding name to apply at the ``relu_idx``-th
    ReLU (0-indexed) of a network with ``total_relus`` ReLUs total.

    Args:
        relu_idx: 0-based position in the ReLU sequence of the network.
        total_relus: number of ReLU layers in the network.
        policy: cascade configuration. None ⇒ default policy.

    Returns:
        Method name string (e.g. ``"triangle"`` or ``"eq_lagr_v8"``).
    """
    pol = policy if policy is not None else CascadePolicy()
    k = _resolve_last_k(pol.last_k)
    if k <= 0:
        return pol.early_method
    if k >= total_relus:
        return pol.late_method
    # last_k tail: ReLUs at indices [total_relus - k, total_relus)
    cutoff = total_relus - k
    return pol.late_method if relu_idx >= cutoff else pol.early_method


def cascade_schedule(total_relus: int,
                     policy: Optional[CascadePolicy] = None) -> List[str]:
    """Return the full per-ReLU schedule of method names.

    Equivalent to ``[pick_relu_method(i, total_relus, policy) for i in
    range(total_relus)]``. Use this when you want to plan the cascade
    once at network-build time and pass it to the TF dispatcher.
    """
    return [pick_relu_method(i, total_relus, policy)
            for i in range(total_relus)]


# --- Self-tests (run with: python -m act.back_end.hybridz_tf.algorithms.cascade) ---


def _test_last_k_zero_returns_early_everywhere():
    sched = cascade_schedule(10, CascadePolicy(last_k=0))
    assert sched == ["triangle"] * 10


def _test_last_k_three_of_ten():
    sched = cascade_schedule(10, CascadePolicy(last_k=3))
    assert sched == ["triangle"] * 7 + ["eq_lagr_v8"] * 3, sched


def _test_last_k_exceeds_total():
    sched = cascade_schedule(2, CascadePolicy(last_k=5))
    assert sched == ["eq_lagr_v8", "eq_lagr_v8"]


def _test_env_default_used_when_none():
    saved = os.environ.get("HYZOR_LARGE_CLS_EQ_LAYERS")
    os.environ["HYZOR_LARGE_CLS_EQ_LAYERS"] = "2"
    try:
        sched = cascade_schedule(5)  # default policy ⇒ env wins
        assert sched == ["triangle"] * 3 + ["eq_lagr_v8"] * 2
    finally:
        if saved is None:
            del os.environ["HYZOR_LARGE_CLS_EQ_LAYERS"]
        else:
            os.environ["HYZOR_LARGE_CLS_EQ_LAYERS"] = saved


if __name__ == "__main__":
    _test_last_k_zero_returns_early_everywhere()
    _test_last_k_three_of_ten()
    _test_last_k_exceeds_total()
    _test_env_default_used_when_none()
    print("OK: cascade tests pass")
