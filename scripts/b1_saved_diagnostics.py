"""Numerical attribution on captured B1 matrices, never online acceptance.

Each solver call is issued by the separate frozen diagnostic queue. No loading
of a checkpoint, propagation, hidden range query, or exact-proof prerequisite.
The caller must enforce its registered wall-clock deadline and count the call.
"""
import base64
import hashlib
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp
from scoped_proof.io import load, tick


def number(x):
    return None if x is None or not math.isfinite(float(x)) else float(x)


def minimum_observation(result):
    """Keep already returned telemetry; raw duals never become accepted bounds."""
    value={key:number(getattr(result,key,None)) for key in
        ('minimum','candidate_objective','solver_primal_objective',
         'solver_dual_objective','solver_gap','elapsed')}
    value.update(status=result.status,solver_status=result.solver_status,
        reason=result.reason,solver_bound_kind=getattr(result,'solver_bound_kind',None),
        objective_scope='PRIMAL_DUAL_AND_CANDIDATE_INCLUDE_OUTPUT_CONSTANT',
        exact_feasibility_checked=False,raw_dual_independently_checked=False)
    return value


class StoredProblem:
    def remaining(self, limit):
        tick(self.deadline)
        remaining = min(float(limit),self.deadline-time.monotonic()-1.)
        if remaining <= 0:
            raise TimeoutError('no diagnostic solve after construction cutoff')
        return remaining

    def __init__(self, path, digest, deadline, *, work=None):
        self.deadline, self.work = deadline, work
        tick(deadline)
        self.value = load(path, digest=digest, limit=128*2**20)
        if self.value.get('schema') != 'NATIVE_F0_DIAGNOSTIC_CAPTURE_V1':
            raise ValueError('captured native schema')
        if self.value.get('native') is None:
            raise ValueError('native problem unavailable')
        self.digest, self.cache = digest, {}
        self.native = self.value['native']
        self.offset = float.fromhex(self.native['objective_offset'])
        self.objective = self.array('native.objective')
        self.A = self.matrix('native.A')
        self.row_lower, self.row_upper = self.array('native.row_lb'), self.array('native.row_ub')
        self.lower, self.upper = self.array('native.var_lb'), self.array('native.var_ub')
        self.integrality = self.array('native.integrality')
        if (self.objective.size != self.native['variables']
                or self.A.shape != (len(self.row_lower), len(self.objective))
                or len(self.row_upper) != len(self.row_lower)
                or any(len(v) != len(self.objective) for v in (self.lower,self.upper,self.integrality))):
            raise ValueError('same-matrix dimensions')
        tick(deadline)

    def array(self, name):
        tick(self.deadline)
        if name not in self.cache:
            item = self.value['arrays'][name]
            if item['dtype'] not in ('<f8', '<i8'):
                raise ValueError('stored numeric dtype')
            raw = base64.b64decode(item['base64'], validate=True)
            if len(raw) != item['count']*8:
                raise ValueError('stored numeric length')
            arr = np.frombuffer(raw, dtype=item['dtype'])
            arr.setflags(write=False)
            self.cache[name] = arr
            if self.work is not None: self.work.visit('coefficient_visits',int(arr.size))
        return self.cache[name]

    def matrix(self, name):
        spec = self.value['matrices'][name]
        if self.work is not None: self.work.visit('row_visits',int(spec['shape'][0]))
        return sp.csr_matrix((self.array(spec['data']), self.array(spec['indices']),
                              self.array(spec['indptr'])), shape=tuple(spec['shape']))

    def hz(self, name):
        from act.back_end.solver.solver_hz import SparseHZono
        meta = self.value['sources'][name]
        return SparseHZono(**{k:self.array(name+'.'+k) for k in ('c','b','ub')},
            **{k:self.matrix(name+'.'+k) for k in ('Gc','Gb','Ac','Ab','Auc','Aub')},
            frame_id=meta['frame_id'], exact=meta['exact_flag'])

    def residuals(self, point, *, relaxed):
        if point is None:
            return None
        x = np.asarray(point)
        if x.shape != self.objective.shape or not np.isfinite(x).all():
            return {'candidate_valid_shape_finite':False}
        value = self.A@x
        maximum = lambda *a: max([0.]+[float(np.max(v)) for v in a if len(v)])
        row = maximum(self.row_lower-value, value-self.row_upper)
        box = maximum(self.lower-x, x-self.upper)
        binary = self.integrality != 0
        integer = maximum(np.abs(x[binary]-np.rint(x[binary])))
        return dict(candidate_valid_shape_finite=True, row_violation=row,
            box_violation=box, integrality_violation=integer,
            numerical_feasible_1e7=row<=1e-7 and box<=1e-7 and (relaxed or integer<=1e-7),
            exact_feasibility_checked=False,
            objective=number(self.offset+self.objective@x))

    def solve_same_matrix(self, *, relaxed, time_limit):
        limit = self.remaining(time_limit)
        started = time.monotonic()
        result = milp(c=self.objective.copy(),
            integrality=np.zeros_like(self.integrality) if relaxed else self.integrality.copy(),
            bounds=Bounds(self.lower.copy(),self.upper.copy()),
            constraints=LinearConstraint(self.A.copy(),self.row_lower.copy(),self.row_upper.copy()),
            options={'time_limit':limit,'mip_rel_gap':0.})
        tick(self.deadline)
        primal, dual = number(getattr(result,'fun',None)), number(getattr(result,'mip_dual_bound',None))
        # Keep raw timeout dual telemetry separate from the unchanged policy's
        # status-0, finite, outward-corrected lower bound.
        from act.back_end.solver.solver_hz import _certified_solver_lower_bound, hz_outward_slack
        flags = np.zeros_like(self.integrality) if relaxed else self.integrality
        certified = _certified_solver_lower_bound(result, flags)
        policy_lower = None
        if certified is not None:
            bound, kind = certified
            policy_lower = float(np.nextafter(self.offset+bound-hz_outward_slack(self.offset,bound),-np.inf))
        raw_lower = primal if relaxed and result.status==0 else dual
        return dict(source_sha256=self.digest, scope=self.value['scope'],
            mode='CONTINUOUS_RELAXATION' if relaxed else 'ORIGINAL_INTEGRAL_MATRIX',
            status=int(result.status), message=str(result.message),
            primal=None if primal is None else primal+self.offset,
            dual=None if dual is None else dual+self.offset,
            raw_solver_lower=None if raw_lower is None else raw_lower+self.offset,
            numerical_lower=policy_lower,
            bound_kind=None if certified is None else certified[1],
            mip_gap=number(getattr(result,'mip_gap',None)),
            candidate=self.residuals(getattr(result,'x',None),relaxed=relaxed),
            seconds=time.monotonic()-started, native_calls=1,
            grade='OFFLINE_NUMERICAL_DIAGNOSTIC_NOT_EXACT_OPTIMALITY_OR_NETWORK_PROOF',
            repairs_online=False)

    def joint_property(self, gate=None, difference=False):
        from act.back_end.solver.solver_hz import sparse_hz_linear
        hz = self.hz('upstream_expert_joint')
        q = np.asarray(self.value['scope']['q'])
        if hz.n_out != 2*len(q): raise ValueError('all classes retained')
        weights = np.r_[q,-q] if difference else np.r_[float(gate)*q,(1-float(gate))*q]
        constant = 0. if difference else float(self.value['scope']['constant'])
        return sparse_hz_linear(hz, weights.reshape(1,-1), np.array([constant]))

    def difference_extreme(self, *, upper, time_limit):
        from act.back_end.solver.solver_hz import sparse_hz_linear, hz_minimize_output
        hz = self.joint_property(difference=True)
        if upper: hz = sparse_hz_linear(hz, np.array([[-1.]]))
        result = hz_minimize_output(hz, 0, input_hz=None, input_shape=None,
            time_limit=self.remaining(time_limit))
        tick(self.deadline)
        bound = result.minimum
        return dict(source_sha256=self.digest, scope=self.value['scope'],
            mode='INTEGER_DIFFERENCE_UPPER' if upper else 'INTEGER_DIFFERENCE_LOWER',
            numerical_bound=None if bound is None else (-bound if upper else bound),
            result=minimum_observation(result),result_objective='-d' if upper else 'd',
            status=result.status, solver_status=result.solver_status, native_calls=1,
            grade='OFFLINE_NUMERICAL_POLICY_NOT_CHECKED_RANGE', repairs_online=False)

    def replace_difference_range(self, *, lower_record, upper_record, time_limit):
        """One weighted solve; reuse the two separately charged range results."""
        from types import SimpleNamespace
        from act.back_end.moe.weighted_top2 import (SharedInputPairHZ,
            WeightedTop2GateRange, WeightedTop2DifferenceRange, build_weighted_top2_f0)
        from act.back_end.solver.solver_hz import hz_minimize_output
        for row, mode in ((lower_record,'INTEGER_DIFFERENCE_LOWER'),
                          (upper_record,'INTEGER_DIFFERENCE_UPPER')):
            if row['source_sha256'] != self.digest or row['mode'] != mode or row['scope'] != self.value['scope']:
                raise ValueError('same-source difference result binding')
        old = self.value['source_metadata']
        newlo, newhi = lower_record['numerical_bound'], upper_record['numerical_bound']
        if newlo is None or newhi is None:
            return dict(source_sha256=self.digest, mode='REPLACE_DIFFERENCE_RANGE',
                        status='UNEXECUTED_MISSING_RANGE', native_calls=0, repairs_online=False)
        dlo = max(float(newlo),old['ranges']['difference'][0])
        dhi = min(float(newhi),old['ranges']['difference'][1])
        if dlo>dhi: raise ValueError('conflicting numerical range; no inward repair')
        joint,router,entry = [self.hz(n) for n in
            ('upstream_expert_joint','upstream_conditioned_router','upstream_guarded_entry')]
        pair = tuple(old['selected_pair']); q = tuple(self.value['scope']['q'])
        shared = SharedInputPairHZ(joint,entry,tuple(old['a_rows']),tuple(old['b_rows']),
            old['shared'][0],old['shared'][1],old['private_a'][0],old['private_b'][0],
            old['private_a'][1],old['private_b'][1])
        # These metadata objects record already captured/performed range work;
        # they are not independent certificates and cause no support query.
        gate_support = SimpleNamespace(**old['margin_support'])
        difference_support = SimpleNamespace(lower_status=(lower_record['status'],),
            upper_status=(upper_record['status'],), exact=False, solves=0, elapsed=0., solver_gap=(None,))
        gate = WeightedTop2GateRange(pair,router,router.frame_id,router.n_out,
            tuple(old['ranges']['margin']),tuple(old['ranges']['gate']),gate_support)
        difference = WeightedTop2DifferenceRange(pair,joint,joint.frame_id,joint.n_out,q,
            (dlo,dhi),difference_support)
        encoding = build_weighted_top2_f0(shared,router,pair,q,self.value['scope']['constant'],
            difference_time_limit=0.,gate_range=gate,difference_range=difference)
        result = hz_minimize_output(encoding.output_hz,0,input_hz=None,input_shape=None,
            time_limit=self.remaining(time_limit))
        tick(self.deadline)
        return dict(source_sha256=self.digest, scope=self.value['scope'],
            mode='REPLACE_DIFFERENCE_RANGE', difference=[dlo,dhi],
            status=result.status, lower=number(result.minimum), solver_status=result.solver_status,
            result=minimum_observation(result),
            native_calls=1, seconds=result.elapsed,
            grade='NEW_F0_INTEGER_DOMAIN_NUMERICAL_DIAGNOSTIC_NOT_SAME_NATIVE_LP', repairs_online=False)

    def slice_endpoint(self, *, slice_index, endpoint, time_limit):
        """One of the frozen 16 closed slices x two endpoints, same integer P."""
        from act.back_end.solver.solver_hz import SparseHZono, hz_minimize_output
        if type(slice_index) is not int or not 0<=slice_index<16 or endpoint not in (0,1):
            raise ValueError('registered sixteen-slice query')
        metadata = self.value['source_metadata']
        lo,hi = metadata['ranges']['margin']
        cuts = np.linspace(lo,hi,17)
        a,b = float(cuts[slice_index]),float(cuts[slice_index+1])
        # Closed shared endpoints avoid dropping a tie; expand internal cuts.
        if slice_index: a = float(np.nextafter(a,-np.inf))
        if slice_index<15: b = float(np.nextafter(b,np.inf))
        sigmoid = lambda x: (1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x)))
        gate = float(np.nextafter(sigmoid(a if endpoint==0 else b),
                                  -np.inf if endpoint==0 else np.inf))
        gate = min(1.,max(0.,gate))
        hz = self.joint_property(gate=gate)
        router = self.hz('upstream_conditioned_router')
        pair = metadata['selected_pair']
        if router.frame_id != hz.frame_id: raise ValueError('same stored factor frame')
        m0 = router.c[pair[0]]-router.c[pair[1]]
        def padded(matrix, width):
            row = matrix.getrow(pair[0])-matrix.getrow(pair[1])
            if row.shape[1]>width: raise ValueError('router factor prefix')
            return sp.hstack((row,sp.csr_matrix((1,width-row.shape[1]))),format='csr')
        mc,mb = padded(router.Gc,hz.n_cont),padded(router.Gb,hz.n_bin)
        sliced = SparseHZono(hz.c,hz.Gc,hz.Gb,hz.Ac,hz.Ab,hz.b,
            sp.vstack((hz.Auc,mc,-mc),format='csr'),
            sp.vstack((hz.Aub,mb,-mb),format='csr'),
            np.r_[hz.ub,b-m0,m0-a],frame_id=hz.frame_id,exact=hz.exact)
        result = hz_minimize_output(sliced,0,input_hz=None,input_shape=None,
            time_limit=self.remaining(time_limit))
        tick(self.deadline)
        return dict(source_sha256=self.digest,scope=self.value['scope'],
            mode='CLOSED_MARGIN_SLICE_ENDPOINT', slice_index=slice_index,endpoint=endpoint,
            interval=[a,b],gate=gate,lower=number(result.minimum),status=result.status,
            result=minimum_observation(result),
            solver_status=result.solver_status,seconds=result.elapsed,native_calls=1,
            domain='same shared joint HZ with integer factors, margin slice only',
            comparison_scope='SLICE_PLUS_ENDPOINT_SUPPORT_NOT_ONLY_A_NARROWER_F0_GATE_RANGE',
            grade='NUMERICAL_SIGMOID_NEXTAFTER_NOT_RATIONAL_ENCLOSURE',repairs_online=False)
