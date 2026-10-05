# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measured, complete C78 traversal at the four original runtime boundaries."""
from typing import Callable

from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c78_complete_roots_v1 import collect


class RuntimeRoots:
    """One shared diagnostic token pool; no HZ-generation budget is reset."""

    def __init__(self, emit: Callable[[dict], None], *, enabled: bool = False):
        self.enabled = enabled
        self.emit = emit
        self.pool = WorkPool(256_000_000)
        self.calls = 0
        self.receipts = []

    def collect(self, tf: object, extra: dict, *, boundary: str):
        """Keep all numeric hashes and identity tokens; measure both 1-GiB gates."""
        if not self.enabled:
            return None
        if self.calls >= 4:
            raise ValueError('only the four preregistered full traversals are allowed')
        self.calls += 1
        start_work = self.pool.used
        self.pool.charge('c101_diagnostic_boundary_metadata', 512)
        completed = {}

        def observe(event: dict) -> None:
            if event['event'] == 'complete_root_traversal_finished':
                completed.update(event)
            self.emit(dict(event='c101_complete_traversal_progress', boundary=boundary,
                           traversal=self.calls, details=event))

        roots, stats = measured(
            lambda: collect(tf, extra, pool=self.pool, enabled=True, observe=observe),
            observe=lambda stats: self.emit(dict(event='c101_complete_traversal_transient',
                boundary=boundary, traversal=self.calls, measurement=stats)))
        receipt = dict(event='c101_complete_traversal_returned', boundary=boundary,
            traversal=self.calls, measurement=stats, token_work=self.pool.used-start_work,
            shared_token_work=self.pool.used, shared_token_parts=dict(self.pool.parts),
            complete_traversal=completed, numeric_roots=len(roots.numeric),
            python_shallow_bytes=roots.python_shallow_bytes, unique_objects=roots.unique_objects,
            unchanged_full_numeric_hashes_executed=True,
            numeric_hash_traffic_in_token_pool=False, all_CPU_work_in_generation_cap=False,
            authentication_scope='separate C32 authentication/diagnostic boundary', formal_gain=0)
        self.receipts.append(receipt)
        self.emit(receipt)
        return roots
