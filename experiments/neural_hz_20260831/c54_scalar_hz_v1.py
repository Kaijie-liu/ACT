"""Exact general-scalar nonconvex HZ source writer, default-off and no native adapter."""
import hashlib
import json
from types import SimpleNamespace
import numpy as np
from experiments.neural_hz_20260831.c52_signed_first_write_v1 import validate_program,source_hash
from experiments.neural_hz_20260831.c54_exact_dyadic_v1 import source,canonical,multiply,add,unit_bounded,quotient_by_power,Pool,ONE,ZERO
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect

FIELDS={'schema','n_cont','n_bin','n_out','frame_id','original_nc','global_ids','inverse_roots','inverse_values',
    'removed','eq_uids','ineq_uids','Ac','Ab','Auc','Aub','Gc','Gb','b','ub','c','scalars','source_sha256','report','seal'}


def state_hash(state):
    if type(state) is not dict or set(state)!=FIELDS:raise ValueError('unregistered exact HZ payload fields')
    digest=hashlib.sha256()
    def visit(value):
        if type(value) is np.ndarray:
            if value.dtype.hasobject or not value.flags.c_contiguous:raise ValueError('canonical numeric storage required')
            digest.update(str((value.shape,value.dtype.str)).encode())
            if value.size:digest.update(memoryview(value).cast('B'))
        elif type(value) is dict:
            for k in sorted(value):digest.update(k.encode());visit(value[k])
        elif type(value) in (tuple,list):
            digest.update(str(len(value)).encode())
            for v in value:visit(v)
        elif value is None or type(value) in (str,int,float,bool):digest.update(json.dumps(value).encode()+b'\0')
        else:raise ValueError('unregistered exact HZ storage')
    visit({k:v for k,v in state.items() if k!='seal'})
    return digest.hexdigest()


def _translated(row,roots,weights,work):
    result={}
    work.charge('c54_complete_root_row_visit_and_order',16*len(row['cc'])+len(row['cc'])*max(1,len(row['cc']).bit_length()))
    for c,v in zip(row['cc'],row['cv']):
        root=int(roots[int(c)]);value=multiply(source(v),weights[int(c)],work)
        result[root]=add(result.get(root,ZERO),value,work)
    continuous=sorted((c,v) for c,v in result.items() if v!=ZERO)
    binary=[(int(c),source(v)) for c,v in zip(row['bc'],row['bv'])]
    return continuous,binary,source(row['rhs'])


def _matrix(rows,binary,width,positions,pool):
    starts=[0];columns=[];values=[]
    for row in rows:
        for c,v in row[1 if binary else 0]:
            columns.append(c if binary else int(positions[c]));values.append(pool.intern(v))
        starts.append(len(columns))
    return dict(indptr=np.array(starts,np.int32),indices=np.array(columns,np.int32),
        coefficients=np.array(values,np.int32),shape=(len(rows),width))


def build(program,*,enabled=False,pool=None):
    if not enabled:return None
    whole=WorkPool(256_000_000) if pool is None else pool;work=BranchPool(whole)
    protected,total=validate_program(program,work);before=source_hash(program)
    nc=program['nc'];roots=np.arange(nc,dtype=np.int64);weights=[ONE]*nc
    equations=[];inequalities=[];eq_uids=[];ineq_uids=[];removed=[];derived=0;original_nnz=0
    for row in program['commands']:
        original_nnz+=len(row['cv'])+len(row['bv'])
        current=_translated(row,roots,weights,work);continuous,binary,rhs=current;column=row['column'];chosen=False
        if row['kind']=='def' and not protected[column] and len(continuous)==2 and not binary and rhs==ZERO:
            (parent,value),(fresh,pivot)=continuous
            if fresh!=column or not parent<column:raise ValueError('original defining pivot/order lost')
            if pivot[0]==1:
                ratio=quotient_by_power(value,pivot)
                if unit_bounded(ratio):
                    roots[column]=parent;weights[column]=ratio;removed.append((row['uid'],column));chosen=True
                    derived+=int(len(row['cc'])>2)
        if not chosen:
            target,labels=(inequalities,ineq_uids) if row['kind']=='ineq' else (equations,eq_uids)
            target.append(current);labels.append(row['uid'])
    retained=np.flatnonzero(roots==np.arange(nc,dtype=np.int64)).astype(np.int64)
    positions=np.full(nc,-1,np.int32);positions[retained]=np.arange(len(retained),dtype=np.int32)
    outputs=[_translated(row,roots,weights,work) for row in program['outputs']]
    scalar_pool=Pool(work)
    inverse_roots=positions[roots].astype(np.int32) if removed else np.empty(0,np.int32)
    inverse_values=np.array([scalar_pool.intern(v) for v in weights],np.int32) if removed else np.empty(0,np.int32)
    result=dict(schema='c54_exact_scalar_nonconvex_HZ_v1',n_cont=len(retained),n_bin=program['nb'],n_out=len(outputs),
        frame_id=program['frame_id'],original_nc=nc,global_ids=retained,inverse_roots=inverse_roots,inverse_values=inverse_values,
        removed=np.asarray(removed,np.int64).reshape(-1,2),eq_uids=np.array(eq_uids,np.int64),ineq_uids=np.array(ineq_uids,np.int64),
        source_sha256=before,report={},seal='')
    for ckey,bkey,rows in [('Ac','Ab',equations),('Auc','Aub',inequalities),('Gc','Gb',outputs)]:
        result[ckey]=_matrix(rows,False,len(retained),positions,scalar_pool)
        result[bkey]=_matrix(rows,True,program['nb'],positions,scalar_pool)
    result['b']=np.array([scalar_pool.intern(r[2]) for r in equations],np.int32)
    result['ub']=np.array([scalar_pool.intern(r[2]) for r in inequalities],np.int32)
    result['c']=np.array([scalar_pool.intern(source(v)) for v in program['bias']],np.int32)
    result['scalars']=scalar_pool.pack()
    new_nnz=sum(len(result[k]['indices']) for k in ('Ac','Ab','Auc','Aub'))
    if removed and not new_nnz<original_nnz:raise ValueError('no strict total predicate nnz reduction')
    work.charge('c54_complete_compact_maps_and_source_hash',24*(total+nc+len(program['commands']))+256)
    result['report']=dict(eliminated=len(removed),coalescing_exposed_relations=derived,original_predicate_nnz=original_nnz,
        new_predicate_nnz=new_nnz,first_write_no_removed_predicate_stored=True,whole_work=whole.used,branch_work=work.used,
        native_lowering_or_solver_admission_proved=False,formal_gain=0)
    entries=sum(a.size for value in result.values() for a in (value.values() if type(value) is dict else [value]) if type(a) is np.ndarray)
    if entries>64_000_000:raise MemoryError('unchanged complete numeric entry cap')
    if source_hash(program)!=before:raise ValueError('source program changed')
    result['seal']=state_hash(result)
    return result


def storage(program,state):
    if state_hash(state)!=state['seal']:raise ValueError('exact state changed')
    roots=collect(SimpleNamespace(),dict(complete_common_source=program,complete_state=state));measured=roots.measure()
    return dict(numeric_bytes=measured.resident_bytes,numeric_entries=measured.resident_entries,
        python_shallow_bytes=roots.python_shallow_bytes,numeric_roots=len(roots.numeric),
        numeric_plus_reported_shallow_bytes=measured.resident_bytes+roots.python_shallow_bytes,
        complete_source_scalar_pool_inverse_UIDs_and_maps_included=True)
