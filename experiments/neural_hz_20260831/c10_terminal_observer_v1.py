"""Exact-delegating scalar diagnostics; never use status to change decisions."""

from contextlib import contextmanager
import math
import time
from act.back_end.solver import solver_hz as backend


def scalar(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@contextmanager
def observed(emit):
    original_milp, original_validator = backend.milp, backend._valid_milp_point
    def solve(*args, **kwargs):
        started = time.monotonic()
        try:
            result = original_milp(*args, **kwargs)
        except Exception as exc:
            emit({'event': 'ordinary_milp_exception', 'exception': type(exc).__name__,
                'message': str(exc)[:2048], 'wall_s': time.monotonic() - started})
            raise
        emit({'event': 'ordinary_milp_return', 'status': scalar(getattr(result, 'status', None)),
            'message': str(getattr(result, 'message', ''))[:2048],
            'success': bool(getattr(result, 'success', False)), 'incumbent_present': getattr(result, 'x', None) is not None,
            'nodes': scalar(getattr(result, 'mip_node_count', None)), 'gap': scalar(getattr(result, 'mip_gap', None)),
            'wall_s': time.monotonic() - started,
            'configured_call_limit_s': scalar(kwargs.get('options', {}).get('time_limit'))})
        return result
    def validate(*args, **kwargs):
        result = original_validator(*args, **kwargs)
        emit({'event': 'ordinary_point_validation', 'accepted': bool(result),
            'tolerance': scalar(kwargs.get('tol', args[-1] if args else None))})
        return result
    backend.milp, backend._valid_milp_point = solve, validate
    try:
        yield
    finally:
        backend.milp, backend._valid_milp_point = original_milp, original_validator
