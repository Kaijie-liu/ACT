"""Default-off shared C9 reducer materialization inside one native apply scope."""

from contextlib import contextmanager

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build


class SelectedRejected(BaseException):
    """Do not let a native speculative MemoryError handler try another path."""


def supported(expr, limit):
    if limit != 64_000_000 or expr.frame_id is None or not expr.terms:
        return False
    if any(not term.operators for term in expr.terms):
        return False
    reducer = expr.terms[0].operators[-1]
    return (sp.isspmatrix_csr(reducer) and 0 < reducer.shape[0] < reducer.shape[1]
        and reducer.shape[0] == expr.n_out
        and all(term.operators[-1] is reducer for term in expr.terms)
        and any(type(op) is ImplicitConv2DOp for term in expr.terms for op in term.operators[:-1]))


def value_view(lifted, keep):
    lifted.validate()
    keep = np.asarray(keep)
    if keep.dtype != np.dtype(bool) or keep.shape != (lifted.hz.n_out,):
        raise ValueError('C9 invalid native row mask')
    hz = lifted.hz
    if np.all(keep):
        return hz
    gc, gb = hz.Gc.multiply(keep[:, None]).tocsr(), hz.Gb.multiply(keep[:, None]).tocsr()
    gc.eliminate_zeros()
    gb.eliminate_zeros()
    return SparseHZono(hz.c.copy(), gc, gb, hz.Ac, hz.Ab, hz.b, hz.Auc, hz.Aub, hz.ub,
        frame_id=hz.frame_id, exact=hz.exact)


def numeric_roots(state):
    expected = {'tf', 'layer', 'expression', 'lifted', 'views', 'construction',
        'entry_widths', 'entry_slots', 'apply_args', 'apply_kwargs', 'consumer_construction'}
    if set(state) != expected:
        raise ValueError('unregistered C9 runtime state')
    result = {key: state[key] for key in ('layer', 'expression', 'views',
        'entry_widths', 'entry_slots', 'apply_args', 'apply_kwargs')}
    if state['lifted'] is not None:
        result.update(state['lifted'].numeric_roots())
    return result


@contextmanager
def installed(*, enabled=False, before=None, ready=None, consumed=None, emit=None):
    if not enabled:
        yield
        return
    original_apply, original_materialize = HybridzTF.apply, cnn._lazy_materialize
    original_relu = cnn._sparse_apply_relu
    stack = []

    def report(name, **data):
        if emit is not None:
            emit({'event': name, **data})

    def applied(tf, layer, *args, **kwargs):
        state = {'tf': tf, 'layer': layer, 'expression': None, 'lifted': None, 'views': [],
            'construction': None, 'entry_widths': None, 'entry_slots': None,
            'apply_args': args, 'apply_kwargs': kwargs, 'consumer_construction': None}
        stack.append(state)
        try:
            result = original_apply(tf, layer, *args, **kwargs)
            if state['lifted'] is not None and consumed is not None:
                consumed(state, result)
            return result
        except SelectedRejected:
            raise
        except Exception as exc:
            if state['expression'] is not None:
                raise SelectedRejected(str(exc)) from exc
            raise
        finally:
            if stack.pop() is not state:
                raise RuntimeError('C9 native apply stack mismatch')

    def materialize(expr, keep_rows, limit, *, allow_transient_sum=False):
        if not stack:
            return original_materialize(expr, keep_rows, limit, allow_transient_sum=allow_transient_sum)
        state = stack[-1]
        if state['expression'] is None and not supported(expr, limit):
            return original_materialize(expr, keep_rows, limit, allow_transient_sum=allow_transient_sum)
        try:
            tf = state['tf']
            if state['expression'] is None:
                state['expression'] = expr
                state['entry_widths'] = dict(tf._sparse_frame_widths)
                state['entry_slots'] = dict(tf._sparse_relu_slots)
                widths = state['entry_widths'].get(expr.frame_id)
                if widths is None:
                    raise ValueError('C9 missing live global frame')
                if before is not None:
                    before(state)
                state['lifted'], state['construction'] = measured_build(lambda: lift(expr,
                    np.ones(expr.n_out, dtype=bool), enabled=True, frame_widths=widths,
                    observe=lambda name, data: report(name, **data)))
                if tf._sparse_frame_widths != state['entry_widths'] or tf._sparse_relu_slots != state['entry_slots']:
                    raise ValueError('C9 construction mutated shared frame')
                if ready is not None:
                    ready(state)
                report('c9_live_constructed', construction=state['construction'],
                    n_cont=state['lifted'].hz.n_cont, n_bin=state['lifted'].hz.n_bin)
            elif expr is not state['expression'] or not supported(expr, limit):
                raise ValueError('C9 selected expression or structure changed')
            if tf._sparse_frame_widths != state['entry_widths'] or tf._sparse_relu_slots != state['entry_slots']:
                raise ValueError('C9 frame advanced between materialization requests')
            result = value_view(state['lifted'], keep_rows)
            state['views'].append(result)
            report('c9_live_value_view', selected_rows=int(np.count_nonzero(keep_rows)),
                request=len(state['views']), n_cont=result.n_cont, n_bin=result.n_bin)
            return result
        except Exception as exc:
            report('c9_selected_rejected', reason=str(exc), exception=type(exc).__name__)
            raise SelectedRejected(str(exc)) from exc

    def relu(*args, **kwargs):
        if not stack or stack[-1]['lifted'] is None:
            return original_relu(*args, **kwargs)
        state = stack[-1]
        try:
            if state['consumer_construction'] is not None:
                raise ValueError('C9 selected apply requested a second ReLU construction')
            result, state['consumer_construction'] = measured_build(lambda: original_relu(*args, **kwargs))
            report('c9_native_relu_constructed', construction=state['consumer_construction'])
            return result
        except Exception as exc:
            raise SelectedRejected(str(exc)) from exc

    HybridzTF.apply, cnn._lazy_materialize, cnn._sparse_apply_relu = applied, materialize, relu
    try:
        yield
    finally:
        HybridzTF.apply, cnn._lazy_materialize, cnn._sparse_apply_relu = original_apply, original_materialize, original_relu
