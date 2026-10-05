"""Test-only oracle: original scalar composer, no coefficient-demand reuse."""

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import _left_compose_rows
from act.back_end.hybridz_tf.tf_cnn import (
    SparseHZAffineExpr, SparseHZAffineTerm, _lazy_left_compose, _lazy_materialize,
)
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import live_value_rows


class SourceRestrictedInner:
    def __init__(self, inner, source, observe=None):
        self.inner = inner
        self.shape = inner.shape
        self.live = live_value_rows(source)
        self.row_cache = {}
        self.observe = observe

    def row(self, index):
        if index not in self.row_cache:
            columns, values = self.inner._row(index)
            keep = self.live[columns]
            self.row_cache[index] = (columns[keep], values[keep])
        return self.row_cache[index]

    def left_compose(self, left, max_nnz):
        result = _left_compose_rows(left, operator_shape=self.shape, row=self.row, max_nnz=max_nnz)
        if self.observe is not None:
            self.observe(result)
        return result


def reference_matrix(source, inner, middle, outer, rows, output, *, restricted=True):
    rows = np.asarray(rows, dtype=np.int64)
    q = sp.csr_matrix((np.ones(rows.size), (np.arange(rows.size), rows)),
                      shape=(rows.size, outer.shape[0]))
    for op in (sp.diags(output, format="csr"), outer, sp.diags(middle, format="csr"),
               SourceRestrictedInner(inner, source) if restricted else inner):
        q = _lazy_left_compose(q, op, 64_000_000)
    return q


def reference_materialize(expr, rows, *, restricted=True, observe=None):
    proxies, terms = {}, []
    for index, term in enumerate(expr.terms):
        inner, middle, outer, output = term.operators
        key = (id(inner), id(term.source))
        if key not in proxies:
            callback = None if observe is None else lambda matrix, i=index: observe(i, matrix)
            proxies[key] = SourceRestrictedInner(inner, term.source, callback) if restricted else inner
        terms.append(SparseHZAffineTerm(term.source, (proxies[key], middle, outer, output)))
    reference_expr = SparseHZAffineExpr(tuple(terms), expr.bias, expr.n_out, expr.frame_id)
    keep = np.zeros(expr.n_out, dtype=bool)
    keep[rows] = True
    return _lazy_materialize(reference_expr, keep, 64_000_000)


def equal_payload(left, right):
    if sp.issparse(left) or sp.issparse(right):
        return (sp.issparse(left) and sp.issparse(right) and left.shape == right.shape
                and all(equal_payload(getattr(left, name), getattr(right, name))
                        for name in ("data", "indices", "indptr")))
    left, right = np.asarray(left), np.asarray(right)
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()


def compare_hz(left, right):
    checks = {name: equal_payload(getattr(left, name), getattr(right, name))
              for name in ("c", "Gc", "Gb", "Ac", "Ab", "b", "Auc", "Aub", "ub")}
    checks.update({name: getattr(left, name) == getattr(right, name)
                   for name in ("frame_id", "exact", "n_cont", "n_bin", "n_out")})
    return checks
