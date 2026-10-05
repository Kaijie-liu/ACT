"""Exact zero transfer with full latent predicates; no operator expansion."""

import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import (
    sparse_hz_linear, sparse_hz_add_same_frame, sparse_hz_add_const, sparse_hz_pad_frame,
)
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import compare_hz


def zero_transfer_reference(expr):
    # Group by SOURCE IDENTITY, not source equality or frame number. Repeated
    # source terms produce one zero operator before the predicate join.
    sources = {}
    for term in expr.terms:
        if not term.source.exact or term.source.frame_id != expr.frame_id:
            raise ValueError('zero transfer has unproved source/frame')
        sources.setdefault(id(term.source), term.source)
    if not sources:
        raise ValueError('zero transfer has no sources')
    result = None
    for source in sources.values():
        zero = sp.csr_matrix((expr.n_out, source.n_out), dtype=np.float64)
        part = sparse_hz_linear(source, zero)
        result = part if result is None else sparse_hz_add_same_frame(result, part)
    return sparse_hz_add_const(result, expr.bias)


def verify_zero_transfer(expr, actual):
    expected = zero_transfer_reference(expr)
    checks = compare_hz(actual, expected)
    if not all(checks.values()):
        raise ValueError('zero transfer lost exact bias/predicates/widths')
    return checks


def verify_negative_relu(preactivation, actual, bounds, widths):
    if not np.all(bounds.ub.detach().cpu().numpy() <= 0):
        raise ValueError('zero-output ReLU lacks authoritative all-negative bounds')
    nc, nb = widths.get(preactivation.frame_id, (preactivation.n_cont, preactivation.n_bin))
    padded = sparse_hz_pad_frame(preactivation, max(nc, preactivation.n_cont), max(nb, preactivation.n_bin))
    expected = sparse_hz_linear(padded, sp.csr_matrix((actual.n_out, preactivation.n_out), dtype=np.float64))
    checks = compare_hz(actual, expected)
    if not all(checks.values()):
        raise ValueError('all-negative ReLU lost exact latent predicates')
    return checks
