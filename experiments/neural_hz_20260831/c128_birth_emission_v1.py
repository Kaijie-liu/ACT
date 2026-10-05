"""Default-off C97 source construction with only exact graph support replaced.

Research-only, single-threaded binding. All original encoding, sparse quotient,
UID allocation, binary/frame handling, guards and inverse construction remain.
"""
import threading

from experiments.neural_hz_20260831 import c97_birth_emission_v1 as original
from experiments.neural_hz_20260831.c24_dense_graph_v1 import graph as old_graph
from experiments.neural_hz_20260831.c128_channel_graph_v1 import graph as channel_graph


def lift(expr, keep_rows, *, enabled=False, **kwargs):
    if type(enabled) is not bool:
        raise ValueError('enabled must be an explicit Boolean')
    if not enabled:
        return None
    if threading.active_count() != 1 or threading.current_thread() is not threading.main_thread():
        raise ValueError('single-thread research source binding required')
    if original.graph is not old_graph:
        raise ValueError('nested or conflicting original graph binding')

    def bound_graph(*args, **options):
        return channel_graph(*args, enabled=True, **options)

    original.graph = bound_graph
    try:
        return original.lift(expr, keep_rows, enabled=True, **kwargs)
    finally:
        original.graph = old_graph
