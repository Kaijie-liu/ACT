"""Solver-free current-problem feasibility proposals for canonical ReLU HZ.

Only the independent full-matrix checker has acceptance authority. The proposal
is NOT a robustness certificate, exact rational witness, or network replay.
No history/cache, search, matrix change, bound tightening, or solver call.
This experimental primitive is not installed into the frozen production path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time

import numpy as np
import scipy.sparse as sp

from act.back_end.solver import solver_hz as sh


class ProposalUnavailable(ValueError):
    """Not an infeasibility result: caller must retain UNKNOWN/fall back."""


@dataclass(frozen=True)
class AssignmentScope:
    request_sha256: str
    input_sha256: str
    evaluation_nonce: str

    def validate(self):
        for value in (self.request_sha256, self.input_sha256):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ProposalUnavailable('invalid request/input identity')
        if not self.evaluation_nonce:
            raise ProposalUnavailable('current evaluation nonce required')


@dataclass(frozen=True)
class AssignmentProposal:
    scope: AssignmentScope
    model_sha256: str
    point: np.ndarray
    free_prefix: int
    relu_rows: int
    construction_seconds: float


def _time_left(deadline):
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise ProposalUnavailable('assignment deadline')


def _check_model(model):
    n, nc, nb = model.n_var, model.n_cont, model.n_bin
    if (nc < 0 or nb < 0 or not sp.isspmatrix_csr(model.A) or
            not sp.isspmatrix_csr(model.value_matrix) or
            not model.A.has_canonical_format or not model.value_matrix.has_canonical_format or
            model.A.shape[1] != n or model.value_matrix.shape != (model.value_center.size, n) or
            model.row_lb.shape != (model.A.shape[0],) or model.row_ub.shape != model.row_lb.shape or
            model.var_lb.shape != (n,) or model.var_ub.shape != (n,)):
        raise ProposalUnavailable('unsupported model shape/storage')
    if (not all(np.isfinite(a).all() for a in (model.A.data, model.value_matrix.data,
            model.value_center, model.var_lb, model.var_ub)) or
            np.isnan(model.row_lb).any() or np.isnan(model.row_ub).any() or
            np.isposinf(model.row_lb).any() or np.isneginf(model.row_ub).any() or
            np.any(model.row_lb > model.row_ub) or
            not np.array_equal(model.integrality, np.r_[np.zeros(nc), np.ones(nb)]) or
            not np.array_equal(model.var_lb, np.r_[-np.ones(nc), np.zeros(nb)]) or
            not np.array_equal(model.var_ub, np.ones(n))):
        raise ProposalUnavailable('nonfinite/changed model or factor domains')


def model_fingerprint(model):
    """Binds ALL constraints/domains and output factor map, without densifying."""
    _check_model(model)
    h = hashlib.sha256()
    h.update(json.dumps([model.n_cont, model.n_bin], separators=(',', ':')).encode())
    def array(name, value):
        a = np.ascontiguousarray(value)
        h.update(json.dumps([name, a.dtype.str, list(a.shape)], separators=(',', ':')).encode())
        h.update(memoryview(a).cast('B'))
    for name in ('A', 'value_matrix'):
        m = getattr(model, name)
        array(name+'.shape', np.asarray(m.shape, dtype=np.int64))
        for field in ('data', 'indices', 'indptr'):
            array(name+'.'+field, getattr(m, field))
    for name in ('row_lb', 'row_ub', 'var_lb', 'var_ub', 'integrality', 'value_center'):
        array(name, getattr(model, name))
    return h.hexdigest()


def propose_current_assignment(model, scope, deadline, *, free_values=None):
    """One deterministic proposal, no retries/clamping/optimizer.

    For equality j, the two new continuous columns are p+2j,p+2j+1,
    the new binary is nc+j; coefficients are a<0,b<0,2a. Earlier columns
    have already been assigned. If R is rhs minus earlier contributions,
    t=R-a-b is the represented preactivation. t>=0 uses (1,1+t/b,0),
    t<0 uses (t/a-1,1,1). ALL rows are still checked afterwards.
    A zero free prefix is a factor-space seed, not an asserted concrete
    network-center replay or assumption that a guard contains that center.
    """
    started = time.monotonic()
    _time_left(deadline)
    scope.validate()
    identity = model_fingerprint(model)
    nc, nb = model.n_cont, model.n_bin
    prefix = nc-2*nb
    equalities = np.flatnonzero(np.isfinite(model.row_lb) & (model.row_lb == model.row_ub))
    if prefix < 0 or len(equalities) != nb:
        raise ProposalUnavailable('not the supported appended ReLU equality structure')
    seed = np.zeros(prefix) if free_values is None else np.asarray(free_values, dtype=np.float64)
    if seed.shape != (prefix,) or not np.isfinite(seed).all() or np.any(np.abs(seed) > 1):
        raise ProposalUnavailable('invalid free-factor seed')
    point = np.zeros(model.n_var)
    point[:prefix] = seed
    assigned = np.zeros(model.n_var, dtype=bool)
    assigned[:prefix] = True
    A = model.A
    for j, row in enumerate(equalities):
        _time_left(deadline)
        start, end = A.indptr[row:row+2]
        cols, values = A.indices[start:end], A.data[start:end]
        new = np.asarray([prefix+2*j, prefix+2*j+1, nc+j])
        found = np.searchsorted(cols, new)
        if np.any(found >= cols.size) or not np.array_equal(cols[found], new):
            raise ProposalUnavailable('missing appended factor')
        a, b, binary = values[found]
        if not (a < 0 and b < 0 and binary == 2*a):
            raise ProposalUnavailable('noncanonical ReLU equality')
        old = np.ones(cols.size, dtype=bool)
        old[found] = False
        if not np.all(assigned[cols[old]]):
            raise ProposalUnavailable('future/unmapped factor dependency')
        residual = model.row_lb[row]-float(values[old] @ point[cols[old]])
        pre = residual-a-b
        if pre >= 0:
            point[new] = (1., (residual-a)/b, 0.)
        else:
            point[new] = ((residual-b-2*a)/a, 1., 1.)
        assigned[new] = True
    _time_left(deadline)
    if not assigned.all() or not np.isfinite(point).all():
        raise ProposalUnavailable('partial/nonfinite assignment')
    point.setflags(write=False)
    return AssignmentProposal(scope, identity, point, prefix, nb, time.monotonic()-started)


def check_current_assignment(model, proposal, expected_scope, deadline):
    """Independent of ReLU construction: original full feasibility gate only.

    Returns a checked BASE point or None, never infeasible/SAFE/UNSAFE. A failed
    check supplies no negative evidence. Hashing and checking share deadline.
    """
    started = time.monotonic()
    report = {'accepted': False, 'reason': 'not_checked', 'source_complete': False,
              'evidence_grade': 'HZ_FLOAT_FEASIBILITY_POLICY', 'solver_calls': 0}
    point = None
    try:
        _time_left(deadline)
        expected_scope.validate()
        if proposal.scope != expected_scope or proposal.model_sha256 != model_fingerprint(model):
            raise ProposalUnavailable('current request/model identity mismatch')
        x = np.asarray(proposal.point, dtype=np.float64)
        if x.shape != (model.n_var,) or not np.isfinite(x).all():
            raise ProposalUnavailable('incomplete/nonfinite candidate')
        tol = sh.HZ_NUMERICAL_POLICY.feasibility_tolerance
        y = np.asarray(model.A @ x).reshape(-1)
        if not np.isfinite(y).all():
            raise ProposalUnavailable('nonfinite row evaluation')
        low, high = np.isfinite(model.row_lb), np.isfinite(model.row_ub)
        eq = low & high & (model.row_lb == model.row_ub)
        report.update(max_equality_residual=float(np.max(np.abs(y[eq]-model.row_lb[eq]), initial=0.)),
            max_row_violation=float(max(0., np.max(model.row_lb[low]-y[low], initial=0.),
                                        np.max(y[high]-model.row_ub[high], initial=0.))),
            max_box_violation=float(max(0., np.max(model.var_lb-x, initial=0.), np.max(x-model.var_ub, initial=0.))),
            max_integrality_residual=float(np.max(np.abs(x[model.n_cont:]-np.rint(x[model.n_cont:])), initial=0.)),
            constraints_checked=model.A.shape[0], variables_checked=model.n_var, tolerance=tol,
            model_sha256=proposal.model_sha256)
        valid = sh._valid_milp_point(model, x, model.A, model.row_lb, model.row_ub, tol)
        _time_left(deadline)
        if valid:
            point = x.copy()
            report.update(accepted=True, reason='all_current_base_constraints_checked')
        else:
            report['reason'] = 'full_feasibility_check_failed'
    except (ProposalUnavailable, ValueError, OverflowError) as exc:
        report['reason'] = str(exc)
    report['checking_seconds'] = time.monotonic()-started
    if time.monotonic() >= deadline:
        point = None
        report.update(accepted=False, reason='assignment deadline')
    return point, report
