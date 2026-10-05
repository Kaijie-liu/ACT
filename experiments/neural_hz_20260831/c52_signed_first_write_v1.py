"""Default-off source-first signed congruence, not a native/frame adapter."""
from fractions import Fraction
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c10_alias_quotient_v1 import exact_sum
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest

PROGRAM={'old_nc','nc','nb','frame_id','commands','outputs','bias'}
COMMAND={'kind','uid','column','cc','cv','bc','bv','rhs'}
FIELDS={'hz','global_ids','inverse','eq_uids','ineq_uids','removed','source_sha256','original_nc','report','seal'}
HZ_FIELDS={'c','Gc','Gb','Ac','Ab','b','Auc','Aub','ub','frame_id','exact'}


def _arrays(row,nc,nb):
    for cols,vals,width in ((row['cc'],row['cv'],nc),(row['bc'],row['bv'],nb)):
        if (type(cols) is not np.ndarray or cols.dtype!=np.dtype(np.int64) or cols.ndim!=1
                or type(vals) is not np.ndarray or vals.dtype!=np.dtype(np.float64)
                or vals.shape!=cols.shape or not cols.flags.c_contiguous or not vals.flags.c_contiguous
                or np.any(cols<0) or np.any(cols>=width) or np.any(np.diff(cols)<=0)
                or not np.isfinite(vals).all() or np.any(np.abs(vals)<2.**-20) or np.any(np.abs(vals)>2.**40)):
            raise ValueError('canonical finite original row in the unchanged coefficient window required')
    if type(row['rhs']) is not float or not np.isfinite(row['rhs']):raise ValueError('finite original RHS required')


def validate_program(program,pool):
    if type(program) is not dict or set(program)!=PROGRAM:raise ValueError('complete typed source program required')
    old,nc,nb=(program[n] for n in ('old_nc','nc','nb'))
    if any(type(n) is not int for n in (old,nc,nb,program['frame_id'])) or not 0<=old<=nc<=64_000_000 or not 0<=nb<=64_000_000:
        raise ValueError('bounded original global frame required')
    if type(program['commands']) is not list or type(program['outputs']) is not list:raise ValueError('complete ordered source streams required')
    pool.charge('c52_complete_source_shape_and_frame',32*nc+64*(len(program['commands'])+len(program['outputs']))+256)
    frontier=old;uids=set();nnz=0
    for row in program['commands']:
        if type(row) is not dict or set(row)!=COMMAND:raise ValueError('unknown source row fields')
        if row['kind'] not in ('def','eq','ineq') or type(row['uid']) is not int or not 0<=row['uid']<2**63 or row['uid'] in uids:
            raise ValueError('every original predicate UID must occur once')
        uids.add(row['uid'])
        defining=row['kind']=='def'
        if type(row['column']) is not int or row['column']!=(frontier if defining else -1):raise ValueError('fresh topological definition required')
        _arrays(row,frontier+int(defining),nb)
        if defining:
            if not len(row['cc']) or row['cc'][-1]!=frontier or row['cv'][-1]<=0:raise ValueError('fresh positive defining pivot required')
            frontier+=1
        nnz+=len(row['cv'])+len(row['bv'])
    if frontier!=nc:raise ValueError('a global continuous definition is missing')
    protected=np.zeros(nc,bool)
    for row in program['outputs']:
        if type(row) is not dict or set(row)!={'cc','cv','bc','bv','rhs'} or row['rhs']!=0.:raise ValueError('complete linear output map required')
        _arrays(row,nc,nb);protected[row['cc']]=True
        nnz+=len(row['cv'])+len(row['bv'])
    bias=program['bias']
    if type(bias) is not np.ndarray or bias.dtype!=np.dtype(np.float64) or bias.shape!=(len(program['outputs']),) or not np.isfinite(bias).all():raise ValueError('complete finite output bias required')
    if nnz+nc+nb+len(uids)>64_000_000:raise MemoryError('source entry cap exceeded')
    pool.charge('c52_complete_source_coefficient_checks',16*nnz+len(bias))
    return protected,nnz


def source_hash(program):
    h=hashlib.sha256(json.dumps([program[n] for n in ('old_nc','nc','nb','frame_id')]).encode())
    for row in [*program['commands'],*program['outputs']]:
        h.update(json.dumps([row.get('kind','output'),row.get('uid'),row.get('column'),row['rhs']]).encode())
        for name in ('cc','cv','bc','bv'):
            value=row[name];h.update(str((name,value.shape,value.dtype.str)).encode());h.update(memoryview(value).cast('B'))
    h.update(memoryview(program['bias']).cast('B'));return h.hexdigest()


def _canonical(row,links,pool):
    count=len(row['cc']);pool.charge('c52_signed_root_gather_and_canonicalization',16*count+count*max(1,(count-1).bit_length()))
    codes=links[row['cc']];columns=np.abs(codes)-1;values=row['cv']*np.where(codes<0,-1.,1.)
    if count:
        order=np.argsort(columns,kind='stable');columns,values=columns[order],values[order]
        starts=np.r_[0,np.flatnonzero(np.diff(columns))+1,count]
        out_c=[];out_v=[]
        for a,b in zip(starts[:-1],starts[1:]):
            if b-a>1:
                pool.charge('c52_exact_shared_root_collision',64*int(b-a));value=exact_sum(values[a:b])
            else:value=float(values[a])
            if value!=0.:out_c.append(int(columns[a]));out_v.append(value)
        columns=np.asarray(out_c,np.int64);values=np.asarray(out_v,np.float64)
    return columns,values,row['bc'].copy(),row['bv'].copy(),row['rhs']


def _matrix(rows,binary,width):
    position=2 if binary else 0
    sizes=np.asarray([len(r[position]) for r in rows],np.int32)
    ptr=np.empty(len(rows)+1,np.int32);ptr[0]=0;np.cumsum(sizes,out=ptr[1:])
    cols=np.concatenate([r[position] for r in rows]).astype(np.int32) if rows else np.zeros(0,np.int32)
    values=np.concatenate([r[position+1] for r in rows]) if rows else np.zeros(0,np.float64)
    return sp.csr_matrix((values,cols,ptr),shape=(len(rows),width))


def _emit(program,*,signed,pool):
    protected,total=validate_program(program,pool);before=source_hash(program)
    nc,nb=program['nc'],program['nb'];links=np.arange(1,nc+1,dtype=np.int64)
    equations=[];inequalities=[];eq_uid=[];le_uid=[];removed=[];writes=0;original_nnz=0
    for row in program['commands']:
        original_nnz+=len(row['cv'])+len(row['bv'])
        current=_canonical(row,links,pool) if signed else tuple(row[k].copy() if isinstance(row[k],np.ndarray) else row[k] for k in ('cc','cv','bc','bv','rhs'))
        cc,cv,bc,bv,rhs=current;column=row['column'];chosen=False
        if signed and row['kind']=='def' and not protected[column] and len(cc)==2 and not len(bc) and rhs==0.:
            if cc[-1]!=column:raise ValueError('signed rewrite lost its fresh defining pivot')
            if cv[0]==-cv[1] or cv[0]==cv[1]:
                sign=1 if cv[0]==-cv[1] else -1
                links[column]=sign*int(links[int(cc[0])]);removed.append((row['uid'],column));chosen=True
        if not chosen:
            rows,labels=(inequalities,le_uid) if row['kind']=='ineq' else (equations,eq_uid)
            rows.append(current);labels.append(row['uid']);writes+=len(cv)+len(bv)
    retained=np.flatnonzero(links==np.arange(1,nc+1,dtype=np.int64)).astype(np.int64)
    positions=np.full(nc,-1,np.int64);positions[retained]=np.arange(len(retained),dtype=np.int64)
    inverse=np.sign(links)*(positions[np.abs(links)-1]+1) if removed else np.zeros(0,np.int64)
    if removed and np.any(inverse==0):raise ValueError('an inverse root is not retained')
    output=[_canonical(r,links,pool) if signed else tuple(r[k].copy() if isinstance(r[k],np.ndarray) else r[k] for k in ('cc','cv','bc','bv','rhs')) for r in program['outputs']]
    for rows in (equations,inequalities,output):
        for index,(cc,cv,bc,bv,rhs) in enumerate(rows):
            mapped=positions[cc]
            if np.any(mapped<0):raise ValueError('a published row still uses an eliminated factor')
            rows[index]=(mapped,cv,bc,bv,rhs)
    pool.charge('c52_complete_compact_CSR_maps_UIDs_hashes',24*(total+nc+len(program['commands']))+256)
    hz=SparseHZono(program['bias'].copy(),_matrix(output,False,len(retained)),_matrix(output,True,nb),
        _matrix(equations,False,len(retained)),_matrix(equations,True,nb),np.asarray([r[4] for r in equations],np.float64),
        _matrix(inequalities,False,len(retained)),_matrix(inequalities,True,nb),np.asarray([r[4] for r in inequalities],np.float64),
        frame_id=program['frame_id'],exact=True)
    if source_hash(program)!=before:raise ValueError('original source program changed')
    result=dict(hz=hz,global_ids=retained,inverse=inverse.astype(np.int64),eq_uids=np.asarray(eq_uid,np.int64),
        ineq_uids=np.asarray(le_uid,np.int64),removed=np.asarray(removed,np.int64).reshape(-1,2),source_sha256=before,
        original_nc=nc,report=dict(signed_first_write=signed,eliminated=len(removed),original_predicate_nnz=original_nnz,
            emitted_predicate_coefficients=writes,new_predicate_nnz=hz.Ac.nnz+hz.Ab.nnz+hz.Auc.nnz+hz.Aub.nnz,
            whole_work=pool.whole.used if hasattr(pool,'whole') else pool.used,branch_work=pool.used,
            first_write_no_removed_predicate_stored=True,native_or_solver_executed=False,formal_gain=0),seal='')
    result['seal']=state_hash(result);return result


def state_hash(state):
    if type(state) is not dict or set(state)!=FIELDS:raise ValueError('unregistered compact source state fields')
    if type(state['hz']) is not SparseHZono or set(vars(state['hz']))!=HZ_FIELDS:raise ValueError('unregistered HZ payload fields')
    h=hashlib.sha256(source_digest(state['hz']).encode())
    h.update(json.dumps([state['source_sha256'],state['original_nc'],state['report']],sort_keys=True).encode())
    for n in ('global_ids','inverse','eq_uids','ineq_uids','removed'):
        a=state[n]
        if type(a) is not np.ndarray or a.dtype!=np.dtype(np.int64) or not a.flags.c_contiguous:raise ValueError('owned canonical int64 frame/UID map required')
        h.update(str((n,a.shape,a.dtype.str)).encode())
        if a.size:h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def build(program,*,pool=None,enabled=False):
    if not enabled:return None
    whole=WorkPool(256_000_000) if pool is None else pool
    return _emit(program,signed=True,pool=BranchPool(whole))


def reference(program,*,pool=None):
    return _emit(program,signed=False,pool=WorkPool(256_000_000) if pool is None else pool)


def reconstruct(state,continuous):
    if state_hash(state)!=state['seal']:raise ValueError('compact representation changed')
    values=[Fraction(x) for x in continuous]
    if len(values)!=state['hz'].n_cont or any(abs(x)>1 for x in values):raise ValueError('complete compact latent box point required')
    if not len(state['inverse']):return values
    return [(1 if code>0 else -1)*values[abs(int(code))-1] for code in state['inverse']]


def storage(program,state):
    if state_hash(state)!=state['seal']:raise ValueError('compact representation changed')
    roots=collect(SimpleNamespace(),{'complete_common_source':program,'complete_state':state})
    measured=roots.measure()
    return dict(numeric_bytes=measured.resident_bytes,numeric_entries=measured.resident_entries,
        python_shallow_bytes=roots.python_shallow_bytes,numeric_roots=len(roots.numeric),
        numeric_plus_reported_shallow_bytes=measured.resident_bytes+roots.python_shallow_bytes,
        common_source_included=True,all_inverse_frame_UID_and_report_fields_included=True)
