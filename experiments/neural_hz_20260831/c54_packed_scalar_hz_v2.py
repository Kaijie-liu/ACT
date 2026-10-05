"""Packed exact nonconvex HZ layout; audit view is NOT binary64/native lowering."""
import hashlib
import json
from types import SimpleNamespace
import weakref
import numpy as np
from experiments.neural_hz_20260831.c54_scalar_hz_v1 import build as source_build,state_hash as split_hash
from experiments.neural_hz_20260831.c54_scalar_hz_audit_v1 import audit as split_audit,layout as split_layout,reconstruct as split_reconstruct,feasible as split_feasible,value as split_value
from experiments.neural_hz_20260831.c52_signed_first_write_v1 import storage as reference_storage
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect

FIELDS={'schema','n_cont','n_bin','n_out','n_eq','n_ineq','frame_id','original_nc','global_ids','inverse',
    'removed','eq_uids','ineq_uids','csr','rhs','scalars','source_sha256','report','seal'}


def state_hash(state):
    if type(state) is not dict or set(state)!=FIELDS:raise ValueError('unregistered packed HZ fields')
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
        else:raise ValueError('unregistered packed storage')
    visit({k:v for k,v in state.items() if k!='seal'});return digest.hexdigest()


def _pack(raw,pool):
    ptr=[0];columns=[];ids=[];nc=raw['n_cont']
    for ckey,bkey in [('Ac','Ab'),('Auc','Aub'),('Gc','Gb')]:
        cm,bm=raw[ckey],raw[bkey]
        for row in range(cm['shape'][0]):
            for matrix,binary in [(cm,False),(bm,True)]:
                a,b=map(int,matrix['indptr'][row:row+2]);columns.extend(map(int,matrix['indices'][a:b]+(nc if binary else 0)))
                ids.extend(map(int,matrix['coefficients'][a:b]))
            ptr.append(len(columns))
    pool.charge('c54v2_complete_joint_CSR_and_inverse_packing',16*(len(ids)+len(ptr)+len(raw['inverse_roots']))+256)
    result={k:raw[k] for k in ('n_cont','n_bin','n_out','frame_id','original_nc','global_ids','removed','eq_uids','ineq_uids','scalars','source_sha256')}
    result.update(schema='c54_joint_latent_exact_HZ_v2',n_eq=len(raw['eq_uids']),n_ineq=len(raw['ineq_uids']),
        csr=dict(indptr=np.array(ptr,np.int32),indices=np.array(columns,np.int32),coefficients=np.array(ids,np.int32),
            shape=(len(ptr)-1,raw['n_cont']+raw['n_bin'])),rhs=np.concatenate([raw['b'],raw['ub'],raw['c']]),
        inverse=(raw['inverse_values'].astype(np.uint64)<<np.uint64(32))|raw['inverse_roots'].astype(np.uint64),
        report=dict(raw['report']),seal='')
    return result


def build(program,*,enabled=False,pool=None):
    if not enabled:return None
    whole=WorkPool(256_000_000) if pool is None else pool;raw=source_build(program,enabled=True,pool=whole)
    prior=raw['report']['branch_work'];work=BranchPool(whole,cap=200_000_000-prior)
    retired=[weakref.ref(a) for key in ('Ac','Ab','Auc','Aub','Gc','Gb') for a in raw[key].values() if type(a) is np.ndarray]
    retired.extend(weakref.ref(raw[k]) for k in ('b','ub','c','inverse_roots','inverse_values'))
    state=_pack(raw,work);del raw
    work.charge('c54v2_complete_split_buffer_retirement',128*len(retired)+256)
    if any(ref() is not None for ref in retired):raise ValueError('unshared split-layout numeric buffers remain live')
    state['report'].update(whole_work=whole.used,branch_work=prior+work.used,
        packed_layout_work=work.used,unshared_split_numeric_buffers_retired=len(retired),
        raw_native_binary64_admission_proved=False)
    state['seal']=state_hash(state);return state


def split_view(state,*,pool=None):
    """Checked exact-integer coefficient IDs only; no native floating matrices."""
    if state_hash(state)!=state['seal'] or state['schema']!='c54_joint_latent_exact_HZ_v2':raise ValueError('packed exact state changed')
    ne,nu,no,nc,nb=(state[k] for k in ('n_eq','n_ineq','n_out','n_cont','n_bin'));matrix=state['csr']
    if (type(matrix) is not dict or set(matrix)!={'shape','indptr','indices','coefficients'}
            or matrix['shape']!=(ne+nu+no,nc+nb) or len(state['eq_uids'])!=ne or len(state['ineq_uids'])!=nu):raise ValueError('complete row/domain partitions required')
    for a in (matrix['indptr'],matrix['indices'],matrix['coefficients'],state['rhs']):
        if type(a) is not np.ndarray or a.dtype!=np.dtype(np.int32) or a.ndim!=1:raise ValueError('canonical packed CSR arrays required')
    ptr=matrix['indptr'];cols=matrix['indices'];ids=matrix['coefficients']
    if (len(ptr)!=ne+nu+no+1 or ptr[0]!=0 or ptr[-1]!=len(cols) or len(ids)!=len(cols)
            or np.any(np.diff(ptr)<0) or np.any(cols<0) or np.any(cols>=nc+nb) or len(state['rhs'])!=ne+nu+no):raise ValueError('complete joint CSR spans differ')
    for i in range(ne+nu+no):
        a,b=map(int,ptr[i:i+2])
        if np.any(np.diff(cols[a:b])<=0):raise ValueError('joint row is not canonical')
    inverse=state['inverse']
    if type(inverse) is not np.ndarray or inverse.dtype!=np.dtype(np.uint64) or inverse.ndim!=1:raise ValueError('canonical64-bit inverse records required')
    low=inverse & np.uint64((1<<32)-1);high=inverse>>np.uint64(32)
    if (len(inverse)!=(state['original_nc'] if len(state['removed']) else 0) or np.any(low>=nc)
            or np.any(high>=len(state['scalars']['sign']))):raise ValueError('inverse record fields outside complete frames')
    if pool is not None:pool.charge('c54v2_complete_exact_audit_view',16*(len(cols)+len(ptr)+len(inverse))+256)
    raw={k:state[k] for k in ('n_cont','n_bin','n_out','frame_id','original_nc','global_ids','removed','eq_uids','ineq_uids','scalars','source_sha256','report')}
    raw.update(schema='c54_exact_scalar_nonconvex_HZ_v1',inverse_roots=low.astype(np.int32),inverse_values=high.astype(np.int32),seal='')
    offset=0
    for ckey,bkey,rhs,nrow in [('Ac','Ab','b',ne),('Auc','Aub','ub',nu),('Gc','Gb','c',no)]:
        for key,binary,width in [(ckey,False,nc),(bkey,True,nb)]:
            starts=[0];out_c=[];out_i=[]
            for i in range(offset,offset+nrow):
                a,b=map(int,ptr[i:i+2])
                for col,idx in zip(cols[a:b],ids[a:b]):
                    if bool(col>=nc)==binary:out_c.append(int(col)-(nc if binary else 0));out_i.append(int(idx))
                starts.append(len(out_c))
            raw[key]=dict(indptr=np.array(starts,np.int32),indices=np.array(out_c,np.int32),coefficients=np.array(out_i,np.int32),shape=(nrow,width))
        raw[rhs]=state['rhs'][offset:offset+nrow].copy();offset+=nrow
    raw['seal']=split_hash(raw);split_layout(raw)
    return raw


def audit(program,state,*,pool=None):
    work=WorkPool(256_000_000) if pool is None else pool
    raw=split_view(state,pool=work);proof=split_audit(program,raw,pool=work)
    return dict(proof,schema='c54_joint_exact_Fraction_source_audit_v2',packed_state_sha256=state['seal'],
        complete_joint_row_and_discrete_domain_partitions_checked=True,native_lowering_proved=False)


def reconstruct(state,compact):return split_reconstruct(split_view(state),compact)
def feasible(state,continuous,binary):return split_feasible(split_view(state),continuous,binary)
def value(state,continuous,binary):return split_value(split_view(state),continuous,binary)


def check_points(kind,program,reference,state):
    from experiments.neural_hz_20260831.c54_scalar_fixtures_v1 import check_points as original_point_checks
    return original_point_checks(kind,program,reference,split_view(state))


def storage(program,state):
    if state_hash(state)!=state['seal']:raise ValueError('packed HZ changed')
    roots=collect(SimpleNamespace(),dict(complete_common_source=program,complete_state=state));measured=roots.measure()
    return dict(numeric_bytes=measured.resident_bytes,numeric_entries=measured.resident_entries,
        python_shallow_bytes=roots.python_shallow_bytes,numeric_roots=len(roots.numeric),
        numeric_plus_reported_shallow_bytes=measured.resident_bytes+roots.python_shallow_bytes,
        complete_source_scalar_pool_inverse_UIDs_and_maps_included=True)


def comparison(program,reference,state):
    before=reference_storage(program,reference);after=storage(program,state)
    return dict(before=before,after=after,
        strict_predicate_nnz_decrease=state['report']['new_predicate_nnz']<state['report']['original_predicate_nnz'],
        strict_numeric_bytes_decrease=after['numeric_bytes']<before['numeric_bytes'],
        strict_numeric_entries_decrease=after['numeric_entries']<before['numeric_entries'],
        strict_combined_reported_accounting_decrease=after['numeric_plus_reported_shallow_bytes']<before['numeric_plus_reported_shallow_bytes'],
        comparator='same_source_unchanged_C52_binary64_reference_not_an_enlarged_exact_format_reference',
        native_or_C31_whole_request_reduction_proved=False,formal_gain=0)
