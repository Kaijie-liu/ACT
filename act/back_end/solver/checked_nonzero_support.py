"""Opt-in nonzero precheck on the CURRENT selected-score HZ only.

Reuse the existing generator enclosure, not a feasible point. A separate exact
sign check of stored generator coefficients may only veto that shortcut. It
does not tighten ranges, change native acceptance, or prove source lowering.
"""
from contextlib import contextmanager
from dataclasses import asdict
from fractions import Fraction
import math
from pathlib import Path
import threading
import time
import uuid

import torch
from act.back_end.core import Bounds
from act.back_end.solver import solver_hz as sh
from act.back_end.solver.current_assignment import AssignmentScope
from act.back_end.solver.isolated_feasibility import csr_arrays,save_npz
from act.back_end.solver.native_feasibility_worker import digest,publish


def nonzero(lo,hi):
    return math.isfinite(lo) and math.isfinite(hi) and lo<=hi and (lo>0 or hi<0)


def exact_generator_sign(hz,row,deadline):
    """Check only the unconditioned generator box, a superset of every guard.

    Use the original continuous and +/-1 binary generators, NOT the shifted
    0/1 MILP matrix. Separate occurrences of duplicate columns can only widen
    this box. No coefficient construction or support solve is trusted here.
    """
    center=Fraction.from_float(float(hz.c[row]));radius=Fraction(0);terms=0
    for matrix in (hz.Gc,hz.Gb):
        for value in matrix.data[matrix.indptr[row]:matrix.indptr[row+1]]:
            if terms%128==0 and time.monotonic()>=deadline:raise TimeoutError('sign check deadline')
            radius+=abs(Fraction.from_float(float(value)));terms+=1
    if time.monotonic()>=deadline:raise TimeoutError('sign check deadline')
    lo,hi=center-radius,center+radius
    return {'lower':str(lo),'upper':str(hi),'sign':1 if lo>0 else (-1 if hi<0 else 0),'terms':terms}


def snapshot(hz,path):
    fields={'c':hz.c,'b':hz.b,'ub':hz.ub}
    for name in ('Gc','Gb','Ac','Ab','Auc','Aub'):
        fields.update({name+'_'+k:v for k,v in csr_arrays(getattr(hz,name)).items()})
    save_npz(path,**fields)


def unknown(row,started):
    return sh.HZSupportBoundsResult((row,),Bounds(torch.tensor([[-math.inf]],dtype=torch.float64),
        torch.tensor([[math.inf]],dtype=torch.float64)),('nonzero_unresolved',),('nonzero_unresolved',),
        (None,),time.monotonic()-started,0,False)


def describe(result):
    lo,hi=float(result.bounds.lb.item()),float(result.bounds.ub.item())
    return {'row':result.rows[0],'lower':lo if math.isfinite(lo) else None,
        'upper':hi if math.isfinite(hi) else None,'accepted':nonzero(lo,hi),
        'lower_status':list(result.lower_status),'upper_status':list(result.upper_status),
        'solves':result.solves,'exact':result.exact}


class CheckedNonzeroSupport:
    def __init__(self,directory,*,request_sha256,input_sha256,enabled,fallback):
        AssignmentScope(request_sha256,input_sha256,'identity').validate()
        if type(enabled) is not bool:raise ValueError('explicit boolean required')
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=False)
        self.request_hash,self.input_hash=request_sha256,input_sha256
        self.enabled,self.fallback=enabled,fallback;self.records=[]

    def __call__(self,hz,row,*,time_limit):
        started=time.monotonic();limit=float(time_limit)
        if type(row) is not int or row<0:raise ValueError('one explicit score row required')
        if not math.isfinite(limit) or not 0<=limit<=30:raise ValueError('unchanged <=30s allocation required')
        deadline=started+limit;precheck_deadline=min(deadline,started+3.)
        folder=self.root/f'query_{len(self.records):03d}';folder.mkdir()
        scope=AssignmentScope(self.request_hash,self.input_hash,uuid.uuid4().hex)
        record={'schema':'selected-score-nonzero-v1','scope':asdict(scope),'row':row,'enabled':self.enabled,
            'started_monotonic':started,'deadline_monotonic':deadline,'precheck_deadline_monotonic':precheck_deadline,
            'numerical_policy':sh.hz_numerical_policy_manifest(),'source_complete':False,
            'native_invoked':False,'evidence':'NONE'}
        publish(folder/'begin.json',record)
        out=unknown(row,started);fast=None;accepted=False
        try:
            if time.monotonic()>=deadline:raise TimeoutError('entry deadline')
            if isinstance(hz,sh.SparseHZono):
                if row>=hz.n_out:raise IndexError('score row out of range')
                snapshot(hz,folder/'source_hz.npz')
                record.update(source_hz_sha256=digest(folder/'source_hz.npz'),frame_id=hz.frame_id,
                    hz_exact=bool(hz.exact),n_out=hz.n_out,n_cont=hz.n_cont,n_bin=hz.n_bin,
                    n_eq=hz.n_eq,n_ineq=hz.n_ineq)
                if self.enabled:
                    try:
                        if time.monotonic()>=precheck_deadline:raise TimeoutError('snapshot deadline')
                        # EXACTLY the old support routine's zero-budget fallback.
                        fast=sh.hz_support_bounds(hz,[row],time_limit=0.,relax_binaries=False)
                        if fast.rows!=(row,) or fast.solves!=0 or fast.bounds.lb.numel()!=1 or fast.bounds.ub.numel()!=1:
                            raise ValueError('wrong fast support identity')
                        record['fast']=describe(fast)
                        if record['fast']['accepted']:
                            exact=exact_generator_sign(hz,row,precheck_deadline);record['exact_sign']=exact
                            expected=1 if record['fast']['lower']>0 else -1
                            accepted=exact['sign']==expected
                        record['precheck_accepted']=accepted
                    except Exception as exc:
                        record['precheck_error']=type(exc).__name__+':'+str(exc);accepted=False
                    publish(folder/'attempt.json',record)
                    if time.monotonic()>=precheck_deadline:accepted=False;record['precheck_deadline_rejected']=True
            else:record['unsupported_precheck']='sparse HZ only; native fallback retained'
            if accepted:
                out=sh.HZSupportBoundsResult((row,),fast.bounds,('fast_nonzero_checked',),('fast_nonzero_checked',),
                    (None,),time.monotonic()-started,0,False)
                record['evidence']='CURRENT_GENERATOR_BOX_NONZERO'
            elif time.monotonic()<deadline:
                record.update(native_invoked=True,native_started_monotonic=time.monotonic())
                publish(folder/'native_entry.json',record)
                remaining=max(0.,deadline-time.monotonic())
                record['native_budget_seconds']=remaining
                if remaining>0:
                    out=self.fallback(hz,row,time_limit=remaining)
                    finished=time.monotonic()
                    record.update(native_finished_monotonic=finished,native_late=finished>=deadline,
                                  native_result=describe(out),evidence='ORIGINAL_SUPPORT_POLICY')
        except Exception as exc:
            out=unknown(row,started);record.update(error=type(exc).__name__+':'+str(exc),evidence='NONE')
        record.update(result=describe(out),finished_monotonic=time.monotonic(),
                      elapsed_before_publication=time.monotonic()-started)
        publish(folder/'result.json',record)
        if record['evidence']=='CURRENT_GENERATOR_BOX_NONZERO' and time.monotonic()>=precheck_deadline:
            publish(folder/'late_result_rejected.json',{'accepted':False,'reason':'precheck publication deadline'})
            out=unknown(row,started)
        self.records.append(record)
        return out


_ACTIVE=False


@contextmanager
def checked_nonzero_support(directory,*,request_sha256,input_sha256,enabled=True):
    """Only class-separated top1's definedness callsite; no global patch."""
    global _ACTIVE
    from act.back_end.moe import class_separated_top1 as entry
    if _ACTIVE or threading.current_thread() is not threading.main_thread():
        raise RuntimeError('one main-thread nonzero context required')
    original=entry.selected_score_support
    runner=CheckedNonzeroSupport(directory,request_sha256=request_sha256,input_sha256=input_sha256,
        enabled=enabled,fallback=original)
    _ACTIVE=True
    try:
        entry.selected_score_support=runner
        yield runner
    finally:entry.selected_score_support=original;_ACTIVE=False
