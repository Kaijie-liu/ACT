"""Default-off scoped replacement of the lift, not a second runtime path."""

from contextlib import contextmanager
from experiments.neural_hz_20260831 import c9_live_runtime_v1 as base
from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift as original_lift
from experiments.neural_hz_20260831.c10_fused_emission_v1 import lift as fused_lift

SelectedRejected = base.SelectedRejected
numeric_roots = base.numeric_roots


@contextmanager
def installed(*, enabled=False, **callbacks):
    if not enabled:
        yield
        return
    if base.lift is not original_lift:
        raise ValueError('fused runtime requires the unmodified registered lift hook')
    base.lift = fused_lift
    try:
        with base.installed(enabled=True, **callbacks):
            yield
    finally:
        base.lift = original_lift
