"""Default-off single-thread research binding of the unchanged C24 graph."""

import threading

from experiments.neural_hz_20260831 import c24_dense_graph_v1 as original
from experiments.neural_hz_20260831.c24_dense_ownership_v1 import DenseOwnerEngine
from experiments.neural_hz_20260831.c128_channel_support_v1 import FactorizedOwnerEngine


_BINDING = threading.Lock()


def graph(expr, keep, max_work, *, uid_start, enabled=False):
    """Keep every graph/source/UID/ownership operation except Conv propagation.

    No report key, source map, UID or auxiliary quantity is fabricated or
    rewritten. Work counters come from the unchanged graph's actual engine
    charges; branch costs retain the complete original max-parent recurrence.
    """
    if type(enabled) is not bool:
        raise ValueError('enabled must be an explicit Boolean')
    if not enabled:
        return original.graph(expr, keep, max_work, uid_start=uid_start)
    if threading.active_count() != 1 or threading.current_thread() is not threading.main_thread():
        raise ValueError('channel graph binding is single-thread research only')
    if not _BINDING.acquire(blocking=False):
        raise ValueError('nested or concurrent channel graph binding')
    previous = original.DenseOwnerEngine
    try:
        if previous is not DenseOwnerEngine:
            raise ValueError('conflicting C24 graph engine binding')

        def enabled_engine(max_visits):
            return FactorizedOwnerEngine(max_visits, enabled=True)

        original.DenseOwnerEngine = enabled_engine
        return original.graph(expr, keep, max_work, uid_start=uid_start)
    finally:
        original.DenseOwnerEngine = previous
        _BINDING.release()
