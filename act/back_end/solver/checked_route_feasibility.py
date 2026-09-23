"""Opt-in current-query feasibility for guarded routing HZs, not support bounds.

The incoming HZ already contains the route guard. Every retained constraint,
factor bound and binary is checked. Failure is not exclusion: the original
native query gets the remaining original deadline. No cross-query cache.
"""
from contextlib import contextmanager
from dataclasses import asdict
import math
from pathlib import Path
import threading
import time
import uuid

from act.back_end.solver import solver_hz as sh
from act.back_end.solver.current_assignment import AssignmentScope, propose_current_assignment, check_current_assignment
from act.back_end.solver.isolated_feasibility import save_npz, csr_arrays
from act.back_end.solver.native_feasibility_worker import digest, publish


class CheckedRouteFeasibility:
    def __init__(self, directory, *, request_sha256, input_sha256, enabled):
        AssignmentScope(request_sha256,input_sha256,'initial-identity-check').validate()
        if type(enabled) is not bool:raise ValueError('explicit boolean mode required')
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=False)
        self.request_hash,self.input_hash=request_sha256,input_sha256
        self.enabled=enabled;self.records=[]

    def __call__(self,hz,*,time_limit=30.,tolerance=1e-7):
        started=time.monotonic();limit=float(time_limit)
        if not math.isfinite(limit) or not 0<=limit<=30:
            raise ValueError('routing control requires original <=30s query allocation')
        if tolerance!=sh.HZ_NUMERICAL_POLICY.feasibility_tolerance:
            raise ValueError('routing control retains original feasibility policy')
        deadline=started+limit;proposal_deadline=min(deadline,started+3.)
        index=len(self.records);folder=self.root/f'query_{index:03d}';folder.mkdir()
        scope=AssignmentScope(self.request_hash,self.input_hash,uuid.uuid4().hex)
        record={'schema':'checked-route-feasibility-v1','index':index,'enabled':self.enabled,
            'scope':asdict(scope),'started_monotonic':started,'deadline_monotonic':deadline,
            'proposal_deadline_monotonic':proposal_deadline,'tolerance':tolerance,
            'numerical_policy':sh.hz_numerical_policy_manifest(),
            'incoming_hz_exact':bool(getattr(hz,'exact',False)),
            'status':'unknown','evidence':'NONE','native_invoked':False,'source_complete':False}
        publish(folder/'begin.json',record)
        result=sh._MILPResult('unknown',None)
        point=None;model=None
        try:
            if time.monotonic()>=deadline:raise TimeoutError('query expired before lowering')
            model=sh._lower_hz_milp(hz)
            # This is the FULL incoming guarded model, not the unguarded router.
            save_npz(folder/'model.npz',**csr_arrays(model.A),row_lb=model.row_lb,row_ub=model.row_ub,
                var_lb=model.var_lb,var_ub=model.var_ub,integrality=model.integrality,
                value_center=model.value_center,**{'value_'+k:v for k,v in csr_arrays(model.value_matrix).items()})
            record.update(model_sha256=digest(folder/'model.npz'),n_cont=model.n_cont,n_bin=model.n_bin,
                          constraint_rows=model.A.shape[0],constraint_nnz=model.A.nnz)
            if self.enabled:
                try:
                    proposal=propose_current_assignment(model,scope,proposal_deadline)
                    point,check=check_current_assignment(model,proposal,scope,proposal_deadline)
                    save_npz(folder/'proposal.npz',point=proposal.point)
                    record['proposal']={'scope':asdict(proposal.scope),'model_fingerprint':proposal.model_sha256,
                        'point_sha256':digest(folder/'proposal.npz'),'check':check,
                        'construction_seconds':proposal.construction_seconds}
                except Exception as exc:
                    record['proposal_error']=type(exc).__name__+':'+str(exc)
                    point=None
                # Publication is part of the proposal's budget as well.
                publish(folder/'attempt.json',record)
                if time.monotonic()>=proposal_deadline:
                    point=None;record['proposal_deadline_rejected']=True
            if point is not None:
                result=sh._MILPResult('feasible',point)
                record['evidence']='CURRENT_FULL_GUARDED_MATRIX_POINT'
            elif time.monotonic()<deadline:
                record.update(native_invoked=True,native_started_monotonic=time.monotonic())
                publish(folder/'native_entry.json',record)
                # Preserve the frozen native fallback and its numerical gate.
                # Its limit is soft; a late return is recorded, NOT concealed.
                result=sh._solve_hz_feasibility(model,deadline,feasibility_tol=tolerance)
                native_finished=time.monotonic()
                record.update(native_finished_monotonic=native_finished,native_status=result.status,
                              native_returned_after_local_deadline=native_finished>=deadline)
                if result.x is not None:
                    save_npz(folder/'native_point.npz',point=result.x)
                    record['native_point_sha256']=digest(folder/'native_point.npz')
                record['evidence']='ORIGINAL_NATIVE_FEASIBILITY_POLICY'
        except Exception as exc:
            result=sh._MILPResult('unknown',None);record.update(error=type(exc).__name__+':'+str(exc),evidence='NONE')
        record.update(status=result.status,nodes=result.nodes,finished_monotonic=time.monotonic(),
                      elapsed_before_publication=time.monotonic()-started)
        publish(folder/'result.json',record)
        if record['evidence']=='CURRENT_FULL_GUARDED_MATRIX_POINT' and time.monotonic()>=proposal_deadline:
            # No late point can make a branch feasible. Native fallback after a
            # terminal publication is intentionally not attempted a second time.
            result=sh._MILPResult('unknown',None)
            record.update(status='unknown',evidence='NONE',late_publication_rejected=True)
            publish(folder/'late_result_rejected.json',{'status':'unknown','reason':'proposal_publication_deadline'})
        record['elapsed_through_publication']=time.monotonic()-started
        self.records.append(record)
        return sh.HZFeasibilityResult(result.status,result.nodes,time.monotonic()-started)


_ACTIVE=False


@contextmanager
def checked_route_feasibility(directory,*,request_sha256,input_sha256,enabled=True):
    """Only route-module feasibility binding changes; support/experts untouched."""
    global _ACTIVE
    from act.back_end.moe import hz_routing
    if _ACTIVE or threading.current_thread() is not threading.main_thread():
        raise RuntimeError('one main-thread checked-routing context required')
    runner=CheckedRouteFeasibility(directory,request_sha256=request_sha256,input_sha256=input_sha256,enabled=enabled)
    original=hz_routing.hz_check_feasibility;_ACTIVE=True
    try:
        hz_routing.hz_check_feasibility=runner
        yield runner
    finally:
        hz_routing.hz_check_feasibility=original;_ACTIVE=False
