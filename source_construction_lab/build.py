"""Function-local parser substitution; original producer algebra is unchanged.

No module or global is patched. Independent checking uses the old, uncached
checker. All source validation, affine/ReLU rules, ties and LP rows still run.
"""
import time
from types import FunctionType
from upstream_source import checker
from source_enclosure import format as fmt, produce
from full_source import lift, obligations
from scoped_source import build as original
from scoped_source.graph import clock
from source_construction_lab.cache import MatrixParser

MODES = ('reference', 'uncached', 'cached')


def bind(function, **overrides):
    if function.__closure__ or not set(overrides) <= function.__globals__.keys():
        raise ValueError('only reviewed closure-free function dependencies')
    fn = FunctionType(function.__code__, {**function.__globals__, **overrides},
        function.__name__, function.__defaults__)
    fn.__kwdefaults__ = function.__kwdefaults__
    return fn


class Trace:
    def __init__(self, emit=None):
        self._emit = emit or (lambda event: None)
        self._stack = []
        self._timers = {}

    def wrap(self, name, fn, *, details=None):
        def measured(*args, **kwargs):
            meta = {} if details is None else details(args, kwargs)
            start = time.monotonic()
            self._emit({'event': 'ENTER', 'operation': name, **meta})
            frame = [0.]
            self._stack.append(frame)
            failed = False
            try:
                return fn(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                seconds = time.monotonic() - start
                self._stack.pop()
                if self._stack:
                    self._stack[-1][0] += seconds
                row = self._timers.setdefault(name, dict(calls=0, inclusive_seconds=0., exclusive_seconds=0.))
                row['calls'] += 1
                row['inclusive_seconds'] += seconds
                row['exclusive_seconds'] += max(0., seconds-frame[0])
                self._emit({'event': 'EXIT_ERROR' if failed else 'EXIT', 'operation': name,
                    'seconds': seconds, **meta})
        return measured

    def timers(self):
        return {k: dict(v) for k, v in self._timers.items()}


def construct(doc, *, expected_source_sha256, deadline, mode='reference', emit=None):
    if mode not in MODES:
        raise ValueError('unregistered construction mode')
    tick = clock(deadline)
    trace = Trace(emit)
    if mode == 'reference':
        bundle = trace.wrap('construction', original.build)(doc,
            expected_source_sha256=expected_source_sha256, deadline=deadline)
        tick()
        return bundle, {'mode': mode, 'timings': trace.timers(), 'parser': None,
            'algebra_changed': False, 'checks_elided': False, 'source_validation_cached': False,
            'bound_or_verdict_cached': False, 'not_a_proof_verdict': True}
    parser = MatrixParser(expected_source_sha256, enabled=mode == 'cached', tick=tick) if mode != 'reference' else None
    parse = checker.csr if parser is None else parser.csr
    decoded = bind(checker.hz, csr=trace.wrap('csr_parse', parse))
    unpack = trace.wrap('state_unpack', bind(fmt.unpack, hz=decoded))
    tag = lambda args, kwargs: {'tag': args[-1]}
    pair = lambda args, kwargs: {'pair': list(args[1])}
    functions = {
        'validate': trace.wrap('source_validate', original.validate),
        'box': trace.wrap('input_enclosure', original.box),
        'operator': trace.wrap('decode_operator', original.operator),
        'identity': trace.wrap('state_hash', original.identity),
        'affine': trace.wrap('affine_lift', bind(lift.affine, unpack=unpack), details=tag),
        'relu': trace.wrap('relu_graph', bind(produce.relu, unpack=unpack), details=tag),
        'join': trace.wrap('shared_join', bind(original.join, unpack=unpack)),
        'guards': trace.wrap('pair_guard', bind(original.guards, unpack=unpack), details=pair),
        'project': trace.wrap('property_projection', bind(original.project, unpack=unpack)),
        'lp_build': trace.wrap('output_lp', bind(obligations.build, unpack=unpack), details=pair),
    }
    try:
        bundle = trace.wrap('construction', bind(original.build, **functions))(
            doc, expected_source_sha256=expected_source_sha256, deadline=deadline)
        tick()
    finally:
        if parser is not None:
            parser.clear()
    return bundle, {'mode': mode, 'timings': trace.timers(),
        'parser': None if parser is None else parser.stats(),
        'algebra_changed': False, 'checks_elided': False, 'source_validation_cached': False,
        'bound_or_verdict_cached': False, 'not_a_proof_verdict': True}
