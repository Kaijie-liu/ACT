"""Opt-in process-local structural C5 dispatch; never imported by production."""

from contextlib import contextmanager
import time

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c5_native_budgeted_materialization_v1 import BudgetedMaterializer


def supported(expr, limit):
    if int(limit) != 64_000_000 or not expr.terms:
        return False
    for term in expr.terms:
        if len(term.operators) != 4:
            return False
        a, d, b, e = term.operators
        if type(a) is not ImplicitConv2DOp or type(b) is not ImplicitConv2DOp:
            return False
        if a._groups != 1 or b._groups != 1:
            return False
        if not sp.isspmatrix_csr(d) or not sp.isspmatrix_csr(e):
            return False
        for diagonal in (d, e):
            if diagonal.shape[0] != diagonal.shape[1]:
                return False
            other = (diagonal - sp.diags(diagonal.diagonal(), format="csr")).tocsr()
            other.eliminate_zeros()
            if other.nnz:
                return False
    return True


@contextmanager
def installed(*, enabled=False, emit=None):
    if not enabled:
        yield
        return
    original_materialize, original_deferred = cnn._lazy_materialize, cnn._try_deferred_expr_conv_relu
    stack = []

    def report(event):
        if emit is not None:
            emit(event)

    def deferred(layer, expr, result, tf):
        state = {"expr": None, "budget": None, "failed": False, "stage": 0}
        stack.append(state)
        try:
            return original_deferred(layer, expr, result, tf)
        finally:
            popped = stack.pop()
            if popped is not state:
                raise RuntimeError("C5 island stack mismatch")
            if state["budget"] is not None:
                report({"event": "c5_island_end", "layer": layer.id, "failed": state["failed"],
                        "stages": state["stage"], "sequence_products": 256_000_000 - state["budget"].remaining_whole})

    def materialize(expr, keep_rows, limit, *, allow_transient_sum=False):
        if not stack:
            return original_materialize(expr, keep_rows, limit, allow_transient_sum=allow_transient_sum)
        state = stack[-1]
        if state["failed"]:
            raise MemoryError("C5 selected island already rejected")
        if not supported(expr, limit):
            if state["budget"] is not None:
                state["failed"] = True
                raise MemoryError("C5 selected island changed its registered structure")
            return original_materialize(expr, keep_rows, limit, allow_transient_sum=allow_transient_sum)
        if state["expr"] is None:
            state["expr"] = expr
            state["budget"] = BudgetedMaterializer(len(expr.terms))
        elif state["expr"] is not expr:
            state["failed"] = True
            raise MemoryError("C5 selected island expression identity changed")
        mask = np.asarray(keep_rows, dtype=bool).reshape(-1)
        if mask.size != expr.n_out:
            state["failed"] = True
            raise ValueError("C5 keep-row shape mismatch")
        rows = np.flatnonzero(mask)
        tick = time.monotonic()
        try:
            out, stats = state["budget"].run(expr, rows)
            report({"event": "c5_materialized", "stage": state["stage"], "selected_rows": rows.size,
                    "elapsed_s": time.monotonic() - tick, "term_stats": stats,
                    "n_cont": out.n_cont, "n_bin": out.n_bin, "frame_id": out.frame_id})
            state["stage"] += 1
            return out
        except BaseException:
            state["failed"] = True
            report({"event": "c5_selected_failure", "stage": state["stage"], "elapsed_s": time.monotonic() - tick})
            raise

    cnn._lazy_materialize, cnn._try_deferred_expr_conv_relu = materialize, deferred
    try:
        yield
    finally:
        cnn._lazy_materialize, cnn._try_deferred_expr_conv_relu = original_materialize, original_deferred
