"""Opt-in checked current assignment for BASE only; no property shortcuts.

One attempt, same base deadline, same full-matrix float feasibility policy.
Native fallback only receives the remainder. Native/property acceptance and
ProtectedHZSolver are unchanged. No incumbent sharing across requests.
"""
from contextlib import contextmanager
from dataclasses import asdict
import math
from pathlib import Path
import time
import uuid

from act.back_end.solver import solver_hz as sh
from act.back_end.solver.current_assignment import (
    AssignmentScope, propose_current_assignment, check_current_assignment,
)
from act.back_end.solver.isolated_feasibility import NativeSession, save_npz, csr_arrays
from act.back_end.solver.native_feasibility_worker import digest, publish
from act.back_end.solver.protected_hz_solver import protected_experts


class CheckedBaseSession(NativeSession):
    def __init__(self, model, directory, *, assignment_scope, **kwargs):
        super().__init__(model, directory, **kwargs)
        assignment_scope.validate()
        self.assignment_scope = assignment_scope
        self.attempted = False

    def query(self, deadline, *, scope, extra_A=None, extra_lb=None, extra_ub=None, tolerance=1e-7):
        if not math.isfinite(deadline):
            raise ValueError('finite query deadline required')
        if tolerance != sh.HZ_NUMERICAL_POLICY.feasibility_tolerance:
            raise ValueError('checked base retains original tolerance')
        kwargs = dict(scope=scope, extra_A=extra_A, extra_lb=extra_lb, extra_ub=extra_ub, tolerance=tolerance)
        if (self.attempted or self.queries or scope != {'phase': 'base'} or
                any(x is not None for x in (extra_A, extra_lb, extra_ub))):
            return super().query(deadline, **kwargs)
        self.attempted = True
        started = time.monotonic()
        folder = self.root/'checked_base'
        folder.mkdir(exist_ok=False)
        record = {'schema': 'checked-base-v1', 'scope': scope,
            'assignment_scope': asdict(self.assignment_scope), 'started_monotonic': started,
            'deadline_monotonic': deadline, 'native_started': False, 'native_result': None,
            'returned_status': 'unknown', 'evidence_kind': 'CURRENT_FULL_MATRIX_POINT',
            'source_complete': False, 'terminal': 'NOT_STARTED'}
        point = None
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError('base deadline before construction')
            publish(folder/'begin.json', record)
            # Charge exactly the same base serialization as the native path.
            model = self.model
            save_npz(self.model_file, **csr_arrays(model.A), row_lb=model.row_lb, row_ub=model.row_ub,
                var_lb=model.var_lb, var_ub=model.var_ub, integrality=model.integrality,
                value_center=model.value_center, **{'value_'+k:v for k,v in csr_arrays(model.value_matrix).items()})
            self.model_hash = digest(self.model_file)
            proposal = propose_current_assignment(model, self.assignment_scope, deadline)
            point, check = check_current_assignment(model, proposal, self.assignment_scope, deadline)
            record.update(check=check, model_fingerprint=proposal.model_sha256,
                model_sha256=self.model_hash, construction_seconds=proposal.construction_seconds,
                free_prefix=proposal.free_prefix, relu_rows=proposal.relu_rows,
                proposal_scope=asdict(proposal.scope))
            save_npz(folder/'candidate.npz', point=proposal.point)
            record['candidate_sha256'] = digest(folder/'candidate.npz')
            record['terminal'] = 'CHECKED' if point is not None else 'REJECTED'
        except Exception as exc:
            point = None
            record.update(terminal='ERROR_OR_DEADLINE', error=type(exc).__name__+':'+str(exc))
        if time.monotonic() >= deadline:
            point = None
            record['terminal'] = 'BASE_DEADLINE'
        record.update(returned_status='feasible' if point is not None else 'unknown',
            return_elapsed_seconds=time.monotonic()-started, finished_monotonic=time.monotonic())
        publish(folder/'return.json', record)
        if time.monotonic() >= deadline:
            point = None
            record.update(returned_status='unknown', terminal='PUBLICATION_DEADLINE')
            publish(folder/'late_return_rejected.json', {'reason':'same_base_deadline'})
        if point is not None:
            # Explicit non-native base evidence in the ledger, not a fake
            # optimal/infeasible solver result. Every property still runs.
            record['return_elapsed_seconds'] = time.monotonic()-started
            self.queries.append(record)
            return sh._MILPResult('feasible', point)
        # Failure never establishes infeasibility and never resets the clock.
        # This also publishes a normal UNKNOWN when there is no time left.
        return super().query(deadline, **kwargs)


@contextmanager
def checked_base_experts(directory, *, request_sha256, input_sha256):
    """Reuse the protected context's single-writer/main-thread guard."""
    AssignmentScope(request_sha256, input_sha256, 'validate-before-install').validate()
    with protected_experts(directory) as cls:
        original_init = cls.__init__
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Fresh scope per evaluation, never imported from another request.
            scope = AssignmentScope(request_sha256, input_sha256, uuid.uuid4().hex)
            self.session_factory = lambda model, folder: CheckedBaseSession(
                model, folder, assignment_scope=scope)
        cls.__init__ = init
        yield cls
