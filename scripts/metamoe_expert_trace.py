"""Observation-only wrappers for the frozen class-separated expert path.

Original arguments, return objects, exceptions and solver options are retained.
No additional solver, feasibility check, bound computation or witness search.
The synchronous hash-chain writes are charged and may perturb timing.
"""
import functools
import importlib
import inspect
import sys
import time

from f0_timing_trace import Recorder, TraceFailure, arguments, outcome, scalar, structure, restore


def small(value):
    if isinstance(value, dict):
        return {str(k): small(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [small(v) for v in value]
    return scalar(value)


def metadata(fn, args, kwargs, frame):
    out = arguments(fn, args, kwargs, frame.f_locals)
    bound = inspect.signature(fn).bind(*args, **kwargs).arguments
    for key in ('timelimit', 'feasibility_tol', 'batch_size', 'n_out'):
        if key in bound:
            out[key] = scalar(bound[key])
    if 'deadline' in bound:
        out['remaining_at_entry_seconds'] = bound['deadline'] - time.monotonic()
    for key in ('model', 'output_hz', 'input_hz'):
        obj = bound.get(key)
        if obj is not None:
            out[key] = structure(obj)
            if hasattr(obj, 'exact'):
                out[key]['exact'] = bool(obj.exact)
    if 'extra_A' in bound or fn.__name__ == '_solve_hz_feasibility':
        extra = bound.get('extra_A')
        out['extra_rows'] = int(extra.shape[0]) if extra is not None else 0
    layer = bound.get('L', bound.get('layer'))
    if layer is not None:
        out['layer_id'] = scalar(getattr(layer, 'id', None))
        out['layer_kind'] = scalar(getattr(layer, 'kind', None))
    # Recover identifiers only, never arrays or arbitrary caller objects. The
    # source-bound callsite distinguishes expanded and contracted feasibility.
    for _ in range(12):
        if frame is None:
            break
        if frame.f_code.co_name == 'evaluate_spec':
            local = frame.f_locals
            out['evaluate_callsite_line'] = frame.f_lineno
            for key in ('lane', 'row', 'exact_witness', 'M'):
                if key in local:
                    out[key] = scalar(local[key])
            break
        frame = frame.f_back
    return out


def observed_result(value):
    out = outcome(value)
    for key in ('message', 'success', 'nodes'):
        item = value.get(key) if isinstance(value, dict) else getattr(value, key, None)
        if item is not None:
            out[key] = scalar(item)
    if hasattr(value, 'x'):
        x = value.x
        out['has_incumbent'] = x is not None
        out['incumbent_size'] = int(x.size) if x is not None else None
    if isinstance(value, list) and all(hasattr(v, 'metadata') for v in value):
        out['verify_results'] = [
            {'status': v.status.value, 'metadata': small(v.metadata),
             'has_counterexample': v.counterexample is not None} for v in value]
    return out


class ExpertRecorder(Recorder):
    def wrap(self, fn, name):
        @functools.wraps(fn)
        def observed(*args, **kwargs):
            try:
                meta = metadata(fn, args, kwargs, sys._getframe(1))
                span = self.emit('BEGIN', name=name,
                    parent=self.stack[-1] if self.stack else None, arguments=meta)
            except TraceFailure:
                raise
            except BaseException as exc:
                raise TraceFailure(f'metadata: {exc}') from exc
            self.stack.append(span)
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                self.emit('RAISE', span=span, exception=type(exc).__name__,
                          message=str(exc)[:4096])
                raise
            else:
                try:
                    self.emit('END', span=span, result=observed_result(result))
                except TraceFailure:
                    raise
                except BaseException as exc:
                    raise TraceFailure(f'outcome: {exc}') from exc
                return result
            finally:
                self.stack.pop()
        return observed


TARGETS = {
    'act.back_end.moe.class_separated_top1': ('verify_class_separated_box',),
    'act.back_end.moe.route_a': ('_analyze_router', 'verify_once'),
    'act.back_end.moe.hz_routing': ('analyze_candidates',),
    'act.back_end.solver.solver_hz': ('_lower_hz_milp', '_solve_hz_feasibility',
        '_valid_milp_point', 'hz_support_bounds', 'milp'),
}


def install(recorder):
    replacements = {}
    for module_name, names in TARGETS.items():
        module = importlib.import_module(module_name)
        for name in names:
            original = getattr(module, name)
            replacements[id(original)] = (original, recorder.wrap(original, module_name+'.'+name))
    patches = []
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith('act.') or module is None:
            continue
        for name, original in list(vars(module).items()):
            match = replacements.get(id(original))
            if match is not None and original is match[0]:
                patches.append((module, name, original))
                setattr(module, name, match[1])
    from act.back_end.solver.solver_hz import HZSolver
    from act.back_end.hybridz_tf import HybridzTF
    for owner, name in ((HZSolver, 'evaluate_spec'), (HybridzTF, 'apply')):
        original = getattr(owner, name)
        patches.append((owner, name, original))
        setattr(owner, name, recorder.wrap(original, owner.__name__+'.'+name))
    recorder.emit('INSTALLED', aliases=[f'{o.__name__}.{n}' for o, n, _ in patches])
    return patches
