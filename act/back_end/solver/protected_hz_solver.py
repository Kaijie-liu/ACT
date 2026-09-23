"""Opt-in protected-property schedule over UNCHANGED HZ/MILP constraints.

V1 scope: one lane, nonempty LINEAR_LE obligations. Base gets at most 10% of
the SAME expert allocation; remaining rows share its remainder in fixed order.
Unknown base never licenses CERTIFIED. No default ACT source is replaced.
"""
from __future__ import annotations
from contextlib import contextmanager
import math
from pathlib import Path
import threading
import time

import numpy as np
import scipy.sparse as sp
import torch
from act.back_end.solver import solver_hz as sh
from act.back_end.solver.isolated_feasibility import NativeSession, save_npz
from act.back_end.solver.native_feasibility_worker import digest, publish
from act.front_end.specs import OutKind
from act.util.stats import VerifyResult, VerifyStatus

BASE_FRACTION = .1
ORIGINAL_SOLVER = sh.HZSolver


class ProtectedHZSolver(ORIGINAL_SOLVER):
    def __init__(self, *, directory, time_limit=30., tolerance=1e-7, session_factory=NativeSession):
        if tolerance != sh.HZ_NUMERICAL_POLICY.feasibility_tolerance:
            raise ValueError('protected v1 retains the frozen feasibility tolerance')
        super().__init__(time_limit=time_limit, tolerance=tolerance)
        self.directory = Path(directory)
        self.session_factory = session_factory

    def evaluate_spec(self, output_hz, out_spec, *, batch_size, n_out,
                      input_hz=None, input_shape=None, timelimit=None):
        started = time.monotonic()
        budget = float(self.time_limit if timelimit is None else timelimit)
        if not math.isfinite(budget) or not 0 < budget <= 30:
            raise ValueError('expert allocation must be in (0, 30] seconds')
        deadline = started+budget
        self.directory.mkdir(parents=True, exist_ok=False)
        plan = {'schema': 'protected-hz-v1', 'started_monotonic': started,
            'deadline_monotonic': deadline, 'total_seconds': budget,
            'base_fraction': BASE_FRACTION, 'base_deadline_monotonic': started+budget*BASE_FRACTION,
            'batch_size': batch_size, 'n_out': n_out, 'tolerance': self.tolerance,
            'scope': 'LINEAR_LE, one lane; same original point/infeasibility acceptance',
            'numerical_policy': sh.hz_numerical_policy_manifest()}
        publish(self.directory/'plan.json', plan)
        rows, session, base, M = [], None, sh._MILPResult('unknown', None), 0

        def finish(status, reason, witness=None):
            cleanup = time.monotonic()
            if session is not None:
                session.close()
            cleanup_seconds = time.monotonic()-cleanup
            # Every unvisited property stays visible; never treat a short list
            # of excluded rows as complete coverage.
            while len(rows) < M:
                rows.append({'row': len(rows), 'status': 'NOT_STARTED', 'reason': 'deadline_or_early_terminal'})
            if time.monotonic() >= deadline:
                status, reason, witness = VerifyStatus.UNKNOWN, 'expert_terminal_deadline', None
            meta = {'source': 'protected_hz_milp_v1', 'reason': reason, 'lane': 0,
                'base_status': base.status, 'properties': rows, 'required_properties': M,
                'native_queries_started': sum(q['native_started'] for q in session.queries) if session else 0,
                'base_fraction': BASE_FRACTION, 'allocation_seconds': budget,
                'elapsed_before_publication': time.monotonic()-started, 'cleanup_seconds': cleanup_seconds,
                'queries': session.queries if session else [],
                'matrix_sha256': session.model_hash if session else None,
                'source_complete': False}
            publish(self.directory/'result.json', {'status': status.value, 'metadata': meta,
                'has_counterexample': witness is not None, 'finished_monotonic': time.monotonic()})
            if status != VerifyStatus.UNKNOWN and time.monotonic() >= deadline:
                status, reason, witness = VerifyStatus.UNKNOWN, 'expert_publication_deadline', None
                meta['reason'] = reason
                publish(self.directory/'late_result_rejected.json', {'status': status.value, 'reason': reason})
            self.last_stats = meta
            return [VerifyResult(status, counterexample=witness, metadata=meta)]

        try:
            if batch_size != 1 or output_hz is None or out_spec.kind != OutKind.LINEAR_LE:
                return finish(VerifyStatus.UNKNOWN, 'unsupported_protected_scope')
            model = sh._lower_hz_milp(output_hz)
            encoded = out_spec.encode_linear(B=1, n_out=n_out, device=torch.device('cpu'), dtype=torch.float64)
            C = encoded['C'].detach().cpu().double().numpy()
            t = encoded['thresholds'].detach().cpu().double().numpy()[0]
            M = int(encoded['M'])
            if M < 1 or model.value_center.size != n_out:
                return finish(VerifyStatus.UNKNOWN, 'shape_or_empty_property')
            coeff = (sp.csr_matrix(C) @ model.value_matrix).tocsr()
            const = C @ model.value_center
            save_npz(self.directory/'properties.npz', C=C, thresholds=t, constants=const)
            publish(self.directory/'property_identity.json', {'properties_sha256': digest(self.directory/'properties.npz'),
                'count': M, 'n_cont': model.n_cont, 'n_bin': model.n_bin,
                'constraint_rows': model.A.shape[0], 'constraint_nnz': model.A.nnz})
            session = self.session_factory(model, self.directory)
            base = session.query(plan['base_deadline_monotonic'], scope={'phase': 'base'}, tolerance=self.tolerance)
            if base.status == 'infeasible':
                return finish(VerifyStatus.UNKNOWN, 'empty_hz')  # Keep original nonvacuity gate.
            exact_witness = (isinstance(output_hz, sh.SparseHZono) and output_hz.exact
                and input_hz is not None and output_hz.frame_id is not None
                and output_hz.frame_id == input_hz.frame_id and input_shape is not None)
            for row in range(M):
                now = time.monotonic()
                if now >= deadline:
                    break
                # Fixed ordered rows; no outcome-driven sample/row selection.
                until = now+(deadline-now)/(M-row)
                expanded = session.query(until, scope={'phase': 'expanded', 'row': row}, extra_A=coeff[row],
                    extra_lb=np.array([t[row]-self.tolerance-const[row]]),
                    extra_ub=np.array([np.inf]), tolerance=self.tolerance)
                item = {'row': row, 'status': expanded.status, 'expanded_query': len(session.queries)-1}
                rows.append(item)
                if expanded.status == 'infeasible':
                    continue
                if expanded.status == 'feasible' and exact_witness:
                    value = const[row]+float((coeff[row] @ expanded.x).item())
                    witness = expanded.x if value >= t[row]+self.tolerance else None
                    if witness is None:
                        contracted = session.query(until, scope={'phase': 'contracted', 'row': row}, extra_A=coeff[row],
                            extra_lb=np.array([t[row]+self.tolerance-const[row]]),
                            extra_ub=np.array([np.inf]), tolerance=self.tolerance)
                        item['contracted_query'] = len(session.queries)-1
                        if contracted.status == 'feasible':
                            value = const[row]+float((coeff[row] @ contracted.x).item())
                            if value >= t[row]+self.tolerance:
                                witness = contracted.x
                    if witness is not None:
                        point = self._recover_input(model, witness, input_hz, input_shape, 0)
                        if point is not None:
                            # A validated violating point also satisfies the base
                            # constraints. Full MoE replay remains upstream.
                            return finish(VerifyStatus.FALSIFIED, 'exact_violation_witness', point)
            if len(rows) == M and all(r['status'] == 'infeasible' for r in rows):
                if base.status == 'feasible':
                    return finish(VerifyStatus.CERTIFIED, 'expanded_violations_infeasible')
                return finish(VerifyStatus.UNKNOWN, 'properties_excluded_but_base_unproved')
            return finish(VerifyStatus.UNKNOWN, 'protected_properties_unresolved')
        except Exception as exc:
            return finish(VerifyStatus.UNKNOWN, 'protected_exception:'+type(exc).__name__+':'+str(exc))


_ACTIVE = False


@contextmanager
def protected_experts(directory):
    """Single-process opt-in factory; verifier's local import sees this class.

Does not touch candidate analysis, score support, propagation or frozen files.
"""
    global _ACTIVE
    if _ACTIVE or threading.current_thread() is not threading.main_thread():
        raise RuntimeError('one main-thread protected expert context required')
    _ACTIVE = True
    old, count = sh.HZSolver, 0
    class BoundSolver(ProtectedHZSolver):
        def __init__(self, time_limit=30., tolerance=1e-7):
            nonlocal count
            folder = Path(directory)/f'evaluation_{count:03d}'
            count += 1
            super().__init__(directory=folder, time_limit=time_limit, tolerance=tolerance)
    try:
        sh.HZSolver = BoundSolver
        yield BoundSolver
    finally:
        sh.HZSolver = old
        _ACTIVE = False
