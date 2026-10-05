"""Exact radix16 continuous lift; mathematical coefficient format, not solver API."""
import hashlib
import json
import math
import numpy as np
from experiments.neural_hz_20260831.c54_exact_dyadic_v1 import unpack
from experiments.neural_hz_20260831.c54_packed_scalar_hz_v2 import split_view, state_hash as exact_hash
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool

SHARED=('frame_id','original_nc','global_ids','inverse','removed','scalars','source_sha256')
FIELDS=set(SHARED)|{'schema','n_cont','n_bin','n_out','n_eq','n_ineq','base_nc','base_ne',
    'eq_uids','ineq_uids','csr','rhs','lifts','exact_semantic_sha256','report','seal'}


def state_hash(state):
    if type(state) is not dict or set(state)!=FIELDS:raise ValueError('unregistered lifted HZ fields')
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
        else:raise ValueError('unregistered lifted storage')
    visit({k:v for k,v in state.items() if k!='seal'});return digest.hexdigest()


def native(value):
    m,e=value
    if abs(m).bit_length()>53:raise ValueError('long exact scalar cannot be cast to binary64')
    return math.ldexp(float(m),e)


def exact_binding(exact):
    if exact_hash(exact)!=exact['seal']:raise ValueError('exact source changed')
    # A shared execution pool's cumulative counter is not the mathematical state.
    normalized=dict(exact,report=dict(exact['report'],whole_work=0))
    return exact_hash(normalized)


def digit_rows(value):
    """(digit, width), low-to-high; no approximate scalar arithmetic."""
    m,e=value;number=abs(m);bits=number.bit_length();result=[]
    while bits:
        width=min(bits,16);result.append((number&((1<<width)-1),width))
        number>>=width;bits-=width
    return result


def build(exact,*,enabled=False,pool=None):
    if not enabled:return None
    whole=WorkPool(256_000_000) if pool is None else pool;work=BranchPool(whole,cap=16_000_000)
    split_view(exact,pool=work)  # Full existing checked frame/domain/scalar layout.
    table=unpack(exact['scalars']);matrix=exact['csr'];base=exact['n_cont'];ne=exact['n_eq']
    keys=set();work.charge('c55_complete_source_scalar_scan',32*(len(matrix['indices'])+len(table)+len(exact['rhs'])))
    for col,idx in zip(matrix['indices'],matrix['coefficients']):
        if abs(table[int(idx)][0]).bit_length()>53:
            if col>=base:raise ValueError('C54 binary source coefficients must remain binary64')
            keys.add((int(col),int(idx)))
    for idx in exact['rhs']:native(table[int(idx)])
    keys=sorted(keys);count=sum(len(digit_rows(table[idx])) for col,idx in keys)
    # Upper bounds include all additional native index/data/pointer/RHS entries.
    if count>16384 or 8*count>131072:raise MemoryError('unchanged radix auxiliary/entry cap')
    if base+count+exact['n_bin']>=2**31:raise MemoryError('native compact coordinate cap')
    work.charge('c55_digit_plan_and_preallocation',256*(count+len(keys))+128)
    nc=base+count;aux_rows=[];records=[];replacement={};next_col=base
    for col,idx in keys:
        digits=digit_rows(table[idx]);start=next_col
        for digit,width in digits:
            terms={next_col:1.}
            if digit:terms[col]=-math.ldexp(float(digit),-width)
            if next_col>start:terms[next_col-1]=-math.ldexp(1.,-width)
            aux_rows.append((sorted(terms.items()),0.));next_col+=1
        m,e=table[idx];scale=math.ldexp(float(1 if m>0 else -1),abs(m).bit_length()+e)
        if not 2**-20<=abs(scale)<=2**40:raise ValueError('lift scale outside unchanged window')
        replacement[col,idx]=(next_col-1,scale);records.append((col,idx,start,len(digits)))
    original=[];ptr=matrix['indptr']
    for row in range(matrix['shape'][0]):
        terms=[]
        for i in range(int(ptr[row]),int(ptr[row+1])):
            col=int(matrix['indices'][i]);idx=int(matrix['coefficients'][i])
            if (col,idx) in replacement:terms.append(replacement[col,idx])
            else:terms.append((col+count if col>=base else col,native(table[idx])))
        terms.sort()
        if any(terms[i][0]==terms[i-1][0] for i in range(1,len(terms))):raise ValueError('unexpected duplicate lifted column')
        original.append((terms,native(table[int(exact['rhs'][row])])) )
    rows=original[:ne]+aux_rows+original[ne:];starts=[0];columns=[];values=[];rhs=[]
    for terms,b in rows:
        columns.extend(c for c,v in terms);values.extend(v for c,v in terms);starts.append(len(columns));rhs.append(b)
    if 2*len(columns)+2*len(rows)>64_000_000:raise MemoryError('unchanged complete entry bound')
    uid_max=max([0,*map(int,exact['eq_uids']),*map(int,exact['ineq_uids']),*map(int,exact['removed'][:,0])])
    if uid_max+count>=2**63:raise ValueError('no fresh equality UID range')
    result={k:exact[k] for k in SHARED}
    result.update(schema='c55_binary64_coefficient_lift_HZ_v1',n_cont=nc,n_bin=exact['n_bin'],n_out=exact['n_out'],
        n_eq=ne+count,n_ineq=exact['n_ineq'],base_nc=base,base_ne=ne,
        eq_uids=np.concatenate([exact['eq_uids'],np.arange(uid_max+1,uid_max+count+1,dtype=np.int64)]),
        ineq_uids=exact['ineq_uids'],lifts=np.asarray(records,np.int32).reshape(-1,4),
        csr=dict(indptr=np.asarray(starts,np.int32),indices=np.asarray(columns,np.int32),data=np.asarray(values,np.float64),shape=(len(rows),nc+exact['n_bin'])),
        rhs=np.asarray(rhs,np.float64),exact_semantic_sha256=exact_binding(exact),report={},seal='')
    work.charge('c55_complete_binary64_emission_and_seal',32*(len(values)+len(rows)+count)+256)
    new_nnz=starts[ne+count+exact['n_ineq']]
    result['report']=dict(auxiliary_continuous=count,scalar_coordinate_pairs=len(keys),
        predicate_nnz=new_nnz,original_predicate_nnz=exact['report']['original_predicate_nnz'],
        lift_work=work.used,formal_gain=0,native_solver_admission_proved=False)
    if exact_hash(exact)!=exact['seal']:raise ValueError('exact source changed')
    result['seal']=state_hash(result);return result


def layout(state):
    if state_hash(state)!=state['seal'] or state['schema']!='c55_binary64_coefficient_lift_HZ_v1':raise ValueError('lifted state changed')
    nc,nb,ne,nu,no=(state[k] for k in ('n_cont','n_bin','n_eq','n_ineq','n_out'))
    base,old_ne=state['base_nc'],state['base_ne'];m=state['csr']
    if min(base,old_ne,nb,nu,no)<0 or not base<=nc or ne-old_ne!=nc-base:raise ValueError('lifted domain/row dimensions differ')
    if set(m)!={'indptr','indices','data','shape'} or m['shape']!=(ne+nu+no,nc+nb):raise ValueError('lifted CSR shape differs')
    for a,dtype in [(m['indptr'],np.int32),(m['indices'],np.int32),(m['data'],np.float64),(state['rhs'],np.float64)]:
        if type(a) is not np.ndarray or a.dtype!=np.dtype(dtype) or a.ndim!=1 or not a.flags.c_contiguous:raise ValueError('canonical native numeric array required')
    ptr=m['indptr'];cols=m['indices'];data=m['data']
    if (len(ptr)!=ne+nu+no+1 or ptr[0]!=0 or ptr[-1]!=len(cols) or len(cols)!=len(data) or np.any(np.diff(ptr)<0)
        or np.any(cols<0) or np.any(cols>=nc+nb) or len(state['rhs'])!=ne+nu+no):raise ValueError('complete native CSR spans required')
    for a in (data,state['rhs']):
        nonzero=np.abs(a[a!=0])
        if np.any(~np.isfinite(a)) or np.any(nonzero<2**-20) or np.any(nonzero>2**40):raise ValueError('native scalar window differs')
    for i in range(ne+nu+no):
        a,b=map(int,ptr[i:i+2])
        if np.any(np.diff(cols[a:b])<=0) or np.any(data[a:b]==0):raise ValueError('native row is not canonical')
    if len(state['eq_uids'])!=ne or len(state['ineq_uids'])!=nu:raise ValueError('complete equality/inequality UIDs required')
    lifts=state['lifts']
    if type(lifts) is not np.ndarray or lifts.dtype!=np.dtype(np.int32) or lifts.ndim!=2 or lifts.shape[1]!=4:raise ValueError('complete lift map required')
    table=unpack(state['scalars']);cursor=base;seen=set()
    for col,idx,start,count in lifts:
        col,idx,start,count=map(int,(col,idx,start,count))
        if not 0<=col<base or not 0<=idx<len(table) or start!=cursor or not 1<=count<=32 or (col,idx) in seen:raise ValueError('lift map does not cover auxiliary frame')
        seen.add((col,idx));cursor+=count
    if cursor!=nc or nc-base>16384 or 8*(nc-base)>131072:raise ValueError('lift frame/cap differs')
    return table
