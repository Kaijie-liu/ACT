"""Exact common-affine-carrier radix lift with dyadic row gauges; default off."""
import hashlib
import json
import math
import numpy as np
from experiments.neural_hz_20260831.c54_exact_dyadic_v1 import unpack
from experiments.neural_hz_20260831.c54_packed_scalar_hz_v2 import split_view,state_hash as exact_hash
from experiments.neural_hz_20260831.c55_binary64_lift_v1 import SHARED,native,exact_binding
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool

FIELDS=set(SHARED)|{'schema','n_cont','n_bin','n_out','n_eq','n_ineq','base_nc','base_ne',
    'eq_uids','ineq_uids','csr','rhs','exact_semantic_sha256','report','seal'}


def state_hash(state):
    if type(state) is not dict or set(state)!=FIELDS:raise ValueError('unregistered carrier HZ fields')
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
        else:raise ValueError('unregistered carrier storage')
    visit({k:v for k,v in state.items() if k!='seal'});return digest.hexdigest()


def digit_rows(value,max_width=53):
    if type(max_width) is not int or not 1<=max_width<=53:raise ValueError('exact native mantissa width required')
    number=abs(value[0]);bits=number.bit_length();result=[]
    while bits:
        width=min(bits,max_width);result.append((number&((1<<width)-1),width))
        number>>=width;bits-=width
    return result


def build(exact,*,enabled=False,pool=None):
    if not enabled:return None
    whole=WorkPool(256_000_000) if pool is None else pool;work=BranchPool(whole,cap=16_000_000)
    split_view(exact,pool=work)
    table=unpack(exact['scalars']);matrix=exact['csr'];base=exact['n_cont'];ne=exact['n_eq'];ptr=matrix['indptr']
    work.charge('c56_complete_exact_consumer_scan',64*(len(matrix['indices'])+len(table)+len(exact['rhs'])))
    plans={};consumers=[];group_occurrences=0
    for row in range(matrix['shape'][0]):
        literal=[];buckets={}
        for i in range(int(ptr[row]),int(ptr[row+1])):
            col=int(matrix['indices'][i]);value=table[int(matrix['coefficients'][i])];m,e=value
            if abs(m).bit_length()<=53:literal.append((col,native(value)))
            else:
                if col>=base:raise ValueError('long binary coefficient outside C54 source grammar')
                buckets.setdefault((abs(m),e),[]).append((col,1 if m>0 else -1))
        keys=[]
        for (m,e),terms in sorted(buckets.items()):
            terms=tuple(terms);key=(m,e,terms);k=(len(terms)-1).bit_length()
            if k>=60:raise MemoryError('carrier too wide for unchanged coefficient window')
            if key not in plans:plans[key]=(k,digit_rows((m,e),min(53,60-k)))
            keys.append(key);group_occurrences+=1
        consumers.append((literal,keys,native(table[int(exact['rhs'][row])])) )
    ordered=sorted(plans);count=0;aux_nnz=0
    for key in ordered:
        k,digits=plans[key];count+=len(digits)
        aux_nnz+=sum(1+(i>0)+(len(key[2]) if d else 0) for i,(d,w) in enumerate(digits))
        scale=math.ldexp(1.,key[0].bit_length()+key[1]+k)
        if not 2**-20<=scale<=2**40:raise ValueError('consumer multiplier outside unchanged window')
    added_entries=2*aux_nnz+3*count  # data/index, row-pointer/RHS AND equality UID.
    if count>16384 or added_entries>131072:raise MemoryError('complete carrier radix auxiliary/entry cap')
    if base+count+exact['n_bin']>=2**31:raise MemoryError('native compact coordinate cap')
    work.charge('c56_complete_carrier_preallocation',256*(count+len(plans))+32*aux_nnz+128)
    nc=base+count;aux_rows=[];replacement={};next_col=base
    for key in ordered:
        m,e,roots=key;k,digits=plans[key];start=next_col
        for digit,width in digits:
            q=max(0,width+k-20);terms={next_col:math.ldexp(1.,q)}
            if digit:
                coefficient=math.ldexp(float(digit),q-width-k)
                for col,sign in roots:terms[col]=-sign*coefficient
            if next_col>start:terms[next_col-1]=-math.ldexp(1.,q-width)
            aux_rows.append((sorted(terms.items()),0.));next_col+=1
        replacement[key]=(next_col-1,math.ldexp(1.,m.bit_length()+e+k))
    original=[]
    for literal,keys,b in consumers:
        terms=[(col+count if col>=base else col,value) for col,value in literal]
        terms.extend(replacement[key] for key in keys);terms.sort()
        if any(terms[i][0]==terms[i-1][0] for i in range(1,len(terms))):raise ValueError('duplicate carrier consumer column')
        original.append((terms,b))
    rows=original[:ne]+aux_rows+original[ne:];starts=[0];columns=[];values=[];rhs=[]
    for terms,b in rows:
        columns.extend(c for c,v in terms);values.extend(v for c,v in terms);starts.append(len(columns));rhs.append(b)
    if 2*len(columns)+2*len(rows)+count>64_000_000:raise MemoryError('complete numeric entry bound')
    uid_max=max([0,*map(int,exact['eq_uids']),*map(int,exact['ineq_uids']),*map(int,exact['removed'][:,0])])
    if uid_max+count>=2**63:raise ValueError('no fresh equality UID range')
    result={k:exact[k] for k in SHARED}
    result.update(schema='c56_gauged_affine_carrier_HZ_v1',n_cont=nc,n_bin=exact['n_bin'],n_out=exact['n_out'],
        n_eq=ne+count,n_ineq=exact['n_ineq'],base_nc=base,base_ne=ne,
        eq_uids=np.concatenate([exact['eq_uids'],np.arange(uid_max+1,uid_max+count+1,dtype=np.int64)]),
        ineq_uids=exact['ineq_uids'],csr=dict(indptr=np.asarray(starts,np.int32),indices=np.asarray(columns,np.int32),
            data=np.asarray(values,np.float64),shape=(len(rows),nc+exact['n_bin'])),
        rhs=np.asarray(rhs,np.float64),exact_semantic_sha256=exact_binding(exact),report={},seal='')
    work.charge('c56_complete_gauged_emission_and_seal',32*(len(values)+len(rows)+count)+256)
    result['report']=dict(auxiliary_continuous=count,scalar_affine_carriers=len(plans),consumer_groups=group_occurrences,
        predicate_nnz=starts[ne+count+exact['n_ineq']],original_predicate_nnz=exact['report']['original_predicate_nnz'],
        lift_work=work.used,added_radix_entries=added_entries,formal_gain=0,native_solver_admission_proved=False)
    if exact_hash(exact)!=exact['seal']:raise ValueError('exact source changed')
    result['seal']=state_hash(result);return result


def layout(state):
    if state_hash(state)!=state['seal'] or state['schema']!='c56_gauged_affine_carrier_HZ_v1':raise ValueError('carrier state changed')
    nc,nb,ne,nu,no=(state[k] for k in ('n_cont','n_bin','n_eq','n_ineq','n_out'))
    base,old_ne=state['base_nc'],state['base_ne'];m=state['csr']
    if min(base,old_ne,nb,nu,no)<0 or not base<=nc or ne-old_ne!=nc-base:raise ValueError('carrier row/coordinate dimensions differ')
    if set(m)!={'indptr','indices','data','shape'} or m['shape']!=(ne+nu+no,nc+nb):raise ValueError('carrier CSR shape differs')
    for a,dtype in [(m['indptr'],np.int32),(m['indices'],np.int32),(m['data'],np.float64),(state['rhs'],np.float64)]:
        if type(a) is not np.ndarray or a.dtype!=np.dtype(dtype) or a.ndim!=1 or not a.flags.c_contiguous:raise ValueError('canonical native numeric array required')
    ptr=m['indptr'];cols=m['indices'];data=m['data']
    if (len(ptr)!=ne+nu+no+1 or ptr[0]!=0 or ptr[-1]!=len(cols) or len(cols)!=len(data) or np.any(np.diff(ptr)<0)
        or np.any(cols<0) or np.any(cols>=nc+nb) or len(state['rhs'])!=ne+nu+no):raise ValueError('complete carrier CSR spans required')
    for a in (data,state['rhs']):
        nonzero=np.abs(a[a!=0])
        if np.any(~np.isfinite(a)) or np.any(nonzero<2**-20) or np.any(nonzero>2**40):raise ValueError('native scalar window differs')
    for i in range(ne+nu+no):
        a,b=map(int,ptr[i:i+2])
        if np.any(np.diff(cols[a:b])<=0) or np.any(data[a:b]==0):raise ValueError('carrier row is not canonical')
    if len(state['eq_uids'])!=ne or len(state['ineq_uids'])!=nu:raise ValueError('complete equality/inequality UIDs required')
    actual_added=2*(int(ptr[ne])-int(ptr[old_ne]))+3*(nc-base)
    if nc-base>16384 or actual_added>131072 or actual_added!=state['report']['added_radix_entries']:raise ValueError('actual carrier entry budget differs')
    return unpack(state['scalars'])
