"""Source-explicit ordered two-elimination algebra fixture, not a benchmark.

Per block: input x; old unit u=s*x/2+offset; half z=unit_sign*(u-offset);
retained output r; legacy a=t*u/2. The C32 unit consumer is exactly the later
C40 half definition.
All original input coordinates precede the internal variables. A separate
binary factor per block remains in an inequality on BOTH representations.
No SplicedState/private source permit/native receipt is manufactured.
"""
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c32_fresh_lineage_v1 import ReversibleLineage,encode_splice,REDIRECT
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import incidence_oracle
from experiments.neural_hz_20260831.c38_all_consumer_coefficients_v1 import discover
from experiments.neural_hz_20260831.c39_half_alias_census_v1 import select_halves,audit_rows
from experiments.neural_hz_20260831.c40_compact_half_gauge_v1 import materialize
from experiments.neural_hz_20260831.c40_half_gauge_transaction_v1 import pack_descriptors


def build(count,sign,alias_sign,*,pool,unit_sign=1,pivot=1.,offset=0.):
    if (type(count) is not int or not 1<=count<=128 or sign not in (-1,1)
            or alias_sign not in (-1,1) or unit_sign not in (-1,1) or pivot not in (.5,1.,2.)
            or offset not in (-.125,0.,.125)):
        raise ValueError('ordered synthetic fixture outside frozen domain')
    pool.charge('c46_complete_ordered_fixture_construction',16384*count+1024)
    nc=5*count;pre_rows=[];pre_rhs=[];post_rows=[];le_rows=[];out_rows=[]
    roots=np.empty(4*count,np.int64);scales=np.zeros(4*count,np.int64)
    columns=[];retired=[];tails=[];definitions=[]
    for i in range(count):
        u,z,r,a=(count+4*i+j for j in range(4))
        pre_rows.extend([[(i,-sign*pivot/2),(u,pivot)],[(u,-unit_sign*pivot),(z,pivot)],[(z,1.),(r,.25)]])
        pre_rhs.extend([pivot*offset,-unit_sign*pivot*offset,0.])
        post_rows.extend([[(i,-unit_sign*sign*pivot/2),(z,pivot)],[(z,1.),(r,.25)]])
        le_rows.append([(z,1.),(r,-.25)]);out_rows.append([(r,1.)])
        roots[4*i:4*i+4]=[encode_splice(3*i,3*i+1,False,pivot,unit_sign,0),
            REDIRECT|((3*i+1)<<20)|(3*i),3*i+2,-u-1]
        scales.view(np.float64)[4*i]=pivot*offset
        scales.view(np.float64)[4*i+3]=alias_sign/2
        columns.append(u);retired.append(((3*i+1)<<20)|(3*i))
        tails.append(((z-count)<<40)|((3*i+1)<<20)|(3*i))
        definitions.extend([-1,2*i,2*i+1,-1])
    def matrix(rows):
        ptr=[0];cols=[];values=[]
        for row in rows:
            for col,value in row:cols.append(col);values.append(value)
            ptr.append(len(cols))
        return sp.csr_matrix((np.array(values,np.float64),np.array(cols,np.int32),np.array(ptr,np.int32)),shape=(len(rows),nc))
    gc=matrix(out_rows);auc=matrix(le_rows);aub=sp.eye(count,format='csr',dtype=np.float64)*.125
    def hz(rows,rhs):
        return SparseHZono(np.zeros(count),gc,sp.csr_matrix((count,count)),matrix(rows),
            sp.csr_matrix((len(rows),count)),np.asarray(rhs,np.float64),auc,aub,np.ones(count),frame_id=46321,exact=True)
    pre=hz(pre_rows,pre_rhs);old=hz(post_rows,np.zeros(2*count))
    lineage=ReversibleLineage(roots,scales,np.array(columns,np.int32),np.array(retired,np.uint64),
        np.array(tails,np.uint64),count,0)
    lineage.seal=lineage.fingerprint();lineage.validate()
    # UID of each surviving consumer receives the deleted producer's UID.
    eq=np.array([v for i in range(count) for v in (3*i,3*i+2)],np.int64)
    tables=dict(definitions=np.array(definitions,np.int64),eq=eq,le=np.arange(3*count,4*count,dtype=np.int64))
    state=SimpleNamespace(hz=old,original_fields=dict(old_n_cont=count,logical_n_cont=nc,
        old_n_eq=0,eq_roots=roots))
    claimed=incidence_oracle(old,tables['eq'],tables['le'],count,nc,pool=pool)
    cohort,_=discover(state,old,tables,claimed,pool=pool);half,_=select_halves(cohort,pool=pool)
    proof,rows=audit_rows(old,tables,claimed,half,old_nc=count,logical_nc=nc,pool=pool,branch=pool)
    if not proof['all_joint_row_arithmetic_proved'] or len(half)!=count:raise ValueError('complete ordered half cohort not proved')
    new,_,desc,runs,writer=materialize(old,old,half,rows,pool=pool,enabled=True)
    words=pack_descriptors(desc,pool=pool)
    bycol={int(v['column']):int(v['degree']) for v in half}
    degrees=np.array([bycol[int(t['column'])] for t in desc],np.int32)
    return pre,old,new,lineage,words,runs,degrees,writer
