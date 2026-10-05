# SPDX-License-Identifier: AGPL-3.0-or-later
"""One exact-delegating BASE observation, with no solver/native replacement."""
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

from act.back_end.solver import solver_hz as backend

TOOL = 3
RESERVATION = 262144
_USED = False
_MILP = importlib.import_module('scipy.optimize._milp')
_WRAPPER = importlib.import_module('scipy.optimize._highspy._highs_wrapper')
SOURCE_HASHES = {
    _MILP.__file__: 'a840bc816dce9e9ca995e2c4cd8d85328c5bbe9656555e74951f798d1eb90482',
    _WRAPPER.__file__: 'c15a943a0985bf716d84b93719d0b2f572e698373655ab0258614b4b2ab67303',
}
BOUNDARIES = {
    _MILP.milp.__code__: {
        371: 'scipy_input_start', 372: 'scipy_input_return',
        374: 'highs_wrapper_start', 377: 'highs_wrapper_return',
        394: 'scipy_result_ready',
    },
    _WRAPPER._highs_wrapper.__code__: {
        117: 'highs_lp_setup_start', 135: 'highs_lp_setup_return',
        183: 'passOptions_start', 184: 'passOptions_return',
        193: 'passModel_start', 194: 'passModel_return',
        206: 'run_start', 207: 'run_return', 217: 'extraction_start',
    },
}
EXPECTED = [
    'scipy_input_start', 'scipy_input_return', 'highs_wrapper_start',
    'highs_lp_setup_start', 'highs_lp_setup_return', 'passOptions_start',
    'passOptions_return', 'passModel_start', 'passModel_return',
    'run_start', 'run_return', 'extraction_start', 'highs_wrapper_return',
    'scipy_result_ready',
]


@contextmanager
def installed(*, pool, emit, enabled=False):
    """Observe first existing backend call only; pass original objects through."""
    global _USED
    if not enabled:
        yield None
        return
    if _USED:
        raise ValueError('C108 is one shot in a fresh process')
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('installed SciPy boundary source differs')
    monitoring = sys.monitoring
    if monitoring.get_tool(TOOL) is not None or monitoring.get_events(TOOL):
        raise ValueError('C108 monitoring slot is not unused')
    if any(monitoring.get_local_events(TOOL, code) for code in BOUNDARIES):
        raise ValueError('C108 local event slot is not unused')
    pool.charge('c108_single_base_phase_observation_reserve', RESERVATION)
    _USED = True
    original = backend.milp
    report = dict(callbacks=0, events=0, serialized_bytes=0, calls=0,
                  reservation=RESERVATION, restored=False)

    def solve(*args, **kwargs):
        report['calls'] += 1
        if report['calls'] != 1:
            return original(*args, **kwargs)
        started = time.monotonic()

        def observe(code, line):
            report['callbacks'] += 1
            if report['callbacks'] > 512:
                raise MemoryError('C108 callback bound exceeded')
            phase = BOUNDARIES[code].get(line)
            if phase is not None:
                report['events'] += 1
                event = dict(event='c108_ordinary_phase', phase=phase,
                    call_wall_s=time.monotonic()-started,
                    monotonic_s=time.monotonic(), line=line,
                    callbacks=report['callbacks'], call=1)
                size = len(json.dumps(event).encode('utf8')) + 1
                report['serialized_bytes'] += size
                if report['events'] > 32 or report['serialized_bytes'] > 65536:
                    raise MemoryError('C108 scalar event bound exceeded')
                emit(event)
            return monitoring.DISABLE

        monitoring.use_tool_id(TOOL, 'C108-single-ordinary-BASE')
        old_callback = None
        try:
            old_callback = monitoring.register_callback(TOOL, monitoring.events.LINE, observe)
            if old_callback is not None:
                raise ValueError('unused C108 slot retained a callback')
            for code in BOUNDARIES:
                monitoring.set_local_events(TOOL, code, monitoring.events.LINE)
            # Do not inspect, copy, mutate or replace any solver arguments/result.
            return original(*args, **kwargs)
        finally:
            for code in BOUNDARIES:
                monitoring.set_local_events(TOOL, code, 0)
            monitoring.register_callback(TOOL, monitoring.events.LINE, old_callback)
            monitoring.free_tool_id(TOOL)
            report['restored'] = True
            emit(dict(event='c108_ordinary_observation_closed', **report,
                      shared_diagnostic_work=pool.used))

    backend.milp = solve
    try:
        yield report
    finally:
        backend.milp = original
