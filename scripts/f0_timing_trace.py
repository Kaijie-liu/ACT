"""Diagnostic-only, durable call spans. Never changes ACT arguments or results.

Only registered function references in ACT modules are wrapped. No global
Python profiler, tensor serialization, solver callback or solver-option edits.
Logging is charged to the request and can perturb deadline-sensitive outcomes.
"""
import functools
import hashlib
import importlib
import inspect
import json
import math
import os
import sys
import time


class TraceFailure(BaseException):
    """Cannot be swallowed by an algorithm's ordinary solver-error fallback."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def scalar(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if hasattr(value, 'item'):
        return scalar(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else {'nonfinite': str(value)}
    return str(value)


def structure(obj):
    out = {}
    for key in ('n_cont', 'n_bin', 'n_out', 'n_var'):
        if hasattr(obj, key):
            out[key] = scalar(getattr(obj, key))
    matrix = getattr(obj, 'A', None)
    if matrix is not None:
        out['matrix_shape'] = list(matrix.shape)
        if hasattr(matrix, 'nnz'):
            out['matrix_nnz'] = int(matrix.nnz)
    return out


def arguments(fn, args, kwargs, caller):
    bound = inspect.signature(fn).bind(*args, **kwargs).arguments
    out = {}
    for key in ('time_limit', 'difference_time_limit', 'relax_binaries', 'stage',
                'cap', 'obligations', 'until', 'tolerance', 'expert_relation'):
        if key in bound:
            out[key] = scalar(bound[key])
    for key in ('pair', 'route_set'):
        value = bound.get(key, caller.get(key))
        if isinstance(value, (list, tuple)) and all(isinstance(v, int) for v in value):
            out[key] = list(value)
    for key in ('index', 'property_index'):
        if isinstance(caller.get(key), int):
            out['property_index'] = caller[key]
    layer = caller.get('L')
    if layer is not None and hasattr(layer, 'id'):
        out['layer_id'] = scalar(layer.id)
    if 'rows' in bound and hasattr(bound['rows'], '__len__'):
        out['selected_rows'] = len(bound['rows'])
    for key in ('hz', 'entry_hz'):
        if bound.get(key) is not None:
            out[key] = structure(bound[key])
    if 'encodings' in bound:
        out['encoding_count'] = len(bound['encodings'])
    # Native SciPy call: record dimensions/options, never the arrays themselves.
    if fn.__module__.startswith('scipy.') and fn.__name__ == 'milp':
        import numpy as np
        out['objective_size'] = int(np.size(bound['c']))
        integrality = bound.get('integrality')
        out['integral_entries'] = int(np.count_nonzero(integrality)) if integrality is not None else 0
        out['options'] = {k: scalar(v) for k, v in (bound.get('options') or {}).items()}
        constraints = bound.get('constraints')
        if hasattr(constraints, 'A'):
            out.update(structure(constraints))
    return out


def outcome(result):
    out = structure(result)
    for key in ('status', 'solver_status', 'reason', 'elapsed_seconds', 'elapsed', 'fun',
                'mip_dual_bound', 'mip_gap', 'mip_node_count', 'exact', 'solves',
                'minimum', 'solver_gap', 'solver_primal_objective', 'solver_dual_objective'):
        value = result.get(key) if isinstance(result, dict) else getattr(result, key, None)
        if value is not None and not isinstance(value, (list, tuple, dict)):
            out[key] = scalar(value)
    for key in ('lower_status', 'upper_status'):
        values = getattr(result, key, None)
        if isinstance(values, tuple):
            from collections import Counter
            out[key + '_counts'] = dict(Counter(values))
    if isinstance(result, (int, float)):
        out['value'] = scalar(result)
    return out


class Recorder:
    def __init__(self, path, started, identity):
        self.handle = path.open('xb', buffering=0)
        self.started = started
        self.stack = []
        self.seq = 0
        self.previous = '0' * 64
        self.logging_seconds = 0.0
        self.emit('IDENTITY', identity=identity)

    def emit(self, kind, **fields):
        before = time.monotonic()
        record = {'seq': self.seq, 'previous': self.previous, 'kind': kind,
                  'elapsed': before - self.started,
                  'logging_seconds_before': self.logging_seconds, **fields}
        try:
            digest = hashlib.sha256(canonical(record)).hexdigest()
            data = canonical({**record, 'sha256': digest}) + b'\n'
            view = memoryview(data)
            while view:
                count = self.handle.write(view)
                if not count:
                    raise OSError('short trace write')
                view = view[count:]
            os.fsync(self.handle.fileno())
        except BaseException as exc:
            raise TraceFailure(str(exc)) from exc
        self.logging_seconds += time.monotonic() - before
        self.previous, self.seq = digest, self.seq + 1
        return record['seq']

    def wrap(self, fn, name):
        @functools.wraps(fn)
        def observed(*args, **kwargs):
            try:
                meta = arguments(fn, args, kwargs, sys._getframe(1).f_locals)
                span = self.emit('BEGIN', name=name, parent=self.stack[-1] if self.stack else None,
                                 arguments=meta)
            except TraceFailure:
                raise
            except BaseException as exc:
                raise TraceFailure(f'trace metadata: {exc}') from exc
            self.stack.append(span)
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                self.emit('RAISE', span=span, exception=type(exc).__name__)
                raise
            else:
                try:
                    self.emit('END', span=span, result=outcome(result))
                except TraceFailure:
                    raise
                except BaseException as exc:
                    raise TraceFailure(f'trace outcome: {exc}') from exc
                return result
            finally:
                self.stack.pop()
        return observed

    def close(self):
        self.handle.close()


TARGETS = {
    'act.pipeline.moe.experiment1': ('_propagate_component', 'shared_input_pair_propagation', '_solve_output'),
    'act.pipeline.moe.route_complexity_schedule': ('prepare',),
    'act.pipeline.moe.paired_monolithic': ('run_monolithic',),
    'act.pipeline.moe.staged_verifier': ('_run_f0',),
    'act.back_end.moe.weighted_top2': ('compute_weighted_top2_gate_range',
        'compute_weighted_top2_difference_range', 'build_weighted_top2_f0', 'solve_weighted_top2_f0'),
    'act.back_end.moe.monolithic_f0': ('_build_disjunction', 'solve_monolithic_weighted_top2_f0'),
    'act.back_end.solver.solver_hz': ('_lower_hz_milp', 'hz_support_bounds', 'hz_minimize_output'),
    'act.back_end.hybridz_tf.tf_mlp': ('_guarded_support_query',),
    'scipy.optimize': ('milp',),
}


def install(recorder):
    replacements = {}
    for module_name, names in TARGETS.items():
        module = importlib.import_module(module_name)
        for name in names:
            original = getattr(module, name)
            replacements[id(original)] = (original, recorder.wrap(original, module_name + '.' + name))
    patches = []
    # All registered ACT modules imported above before alias replacement. Dynamic
    # imports from their module globals will subsequently see the same wrappers.
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith('act.') or module is None:
            continue
        for name, original in list(vars(module).items()):
            match = replacements.get(id(original))
            if match is not None and original is match[0]:
                patches.append((module, name, original))
                setattr(module, name, match[1])
    from act.pipeline.moe.request_budget import RequestBudget
    original = RequestBudget.limit
    patches.append((RequestBudget, 'limit', original))
    RequestBudget.limit = recorder.wrap(original, 'RequestBudget.limit')
    recorder.emit('INSTALLED', aliases=[f'{getattr(o, "__name__", "?")}.{n}' for o, n, _ in patches])
    return patches


def restore(patches):
    for owner, name, original in reversed(patches):
        setattr(owner, name, original)
