"""Complete physical HZ with explicit circuit roots and exact source inverse."""
from fractions import Fraction as F
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import closed_uid_tables
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX,UID_LIMIT,validate_words
from experiments.neural_hz_20260831.c62_local_equations_v1 import reconstruct,SCHEMA as LOCAL

SCHEMA='c91_complete_circuit_source_v1'


def widen(matrix,columns):
    if columns<matrix.shape[1]:raise ValueError('original frame cannot shrink')
    return sp.csr_matrix((matrix.data,matrix.indices,matrix.indptr),shape=(matrix.shape[0],columns),copy=False)


def row(matrix,index):
    a,b=map(int,matrix.indptr[index:index+2]);return matrix.indices[a:b],matrix.data[a:b]


def make(fields,circuits,source_proof,circuit_proof,*,pool,enabled=False):
    """Apply a caller-authenticated structural plan; never a source/solver permit."""
    if not enabled:return None
    hz=fields['hz'];nc=hz.n_cont;neq=hz.n_eq;main=int(fields['logical_n_cont'])
    old=int(fields['old_n_cont']);first_uid=int(fields['report']['radix_uid_base'])+len(fields['def_rows'])
    pool.charge('c91_complete_current_source_headers',8*(neq+hz.n_ineq+len(fields['owners']))+1024)
    if (not hz.exact or fields['report']['new_lineage_schema']!=LOCAL
        or len(fields['owners'])!=main-old or not isinstance(source_proof,bytes) or not isinstance(circuit_proof,bytes)):
        raise ValueError('complete explicit current local source required')
    eq_uids,_=closed_uid_tables(SimpleNamespace(**fields))
    total=sum(c['new_factors'] for c in circuits)
    emission=int(fields['report']['actual_radix_work'])+sum(64*len(c['rhs'])+16*(len(c['native'])+len(c['rhs'])) for c in circuits)
    if total<=0 or total+len(fields['def_rows'])>16384 or first_uid+total>UID_LIMIT:
        raise MemoryError('shared whole auxiliary/UID reserve exceeded')
    if emission>16_000_000:raise MemoryError('shared declared circuit emission reserve exceeded')
    nc2=nc+total;new_rows=neq+total
    if nc2>=2**31 or new_rows>=2**31:raise MemoryError('native compact index domain exceeded')
    replacements={};auxiliary=[];routes=[];blocks=[];cursor=0
    for packet in circuits:
        count=int(packet['new_factors']);nrows=len(packet['rhs'])
        pool.charge('c91_complete_changed_literal_headers',64*nrows+16*len(packet['native']))
        if (count<0 or len(packet['pivots'])!=nrows or len(packet['gauges'])!=nrows
            or len(packet['indptr'])!=nrows+1 or packet['indptr'][0]!=0
            or packet['indptr'][-1]!=len(packet['native']) or len(packet['columns'])!=len(packet['native'])
            or np.any(np.diff(packet['indptr'])<=0) or np.any(packet['ab_indptr'])
            or not np.isfinite(packet['native']).all() or np.any((np.abs(packet['native'])<2.**-20)|(np.abs(packet['native'])>2.**40))
            or np.any(packet['rhs']!=0) or not np.array_equal(packet['pivots'][:count],nc+cursor+np.arange(count))):
            raise ValueError('complete proven native circuit prefix required')
        begin=len(routes)
        for i in range(nrows):
            a,b=map(int,packet['indptr'][i:i+2]);cs=packet['columns'][a:b];vs=packet['native'][a:b]
            pivot=int(packet['pivots'][i]);gauge=int(packet['gauges'][i])
            if np.any(cs<0) or np.any(cs>=nc2) or np.any(np.diff(cs)<=0):raise ValueError('canonical full-frame circuit row required')
            p=np.flatnonzero(cs==pivot)
            if len(p)!=1 or vs[int(p[0])]<=0:raise ValueError('positive actual circuit pivot missing')
            if i<count:
                if np.any(cs[cs!=pivot]>=pivot):raise ValueError('new auxiliary inverse not topological')
                auxiliary.append((cs,vs,pivot,gauge,float(vs[int(p[0])])))
            else:
                rank=int(fields['old_n_eq'])+pivot-old
                if not old<=pivot<main or not 0<=rank<len(fields['eq_roots']):raise ValueError('original logical output rank required')
                physical=int(fields['eq_roots'][rank])
                if physical<0 or physical>=neq or physical in replacements:raise ValueError('output removed or replaced twice')
                if hz.Ab.indptr[physical+1]!=hz.Ab.indptr[physical] or hz.b[physical]!=0:
                    raise ValueError('binary/affine-offset original row cannot be replaced')
                replacements[physical]=(cs,vs,pivot,gauge)
                routes.append((physical,pivot,gauge))
        blocks.append((nc,nc+cursor,count,begin,len(routes),neq+cursor,first_uid+cursor,0))
        cursor+=count
    if cursor!=total or len(auxiliary)!=total:raise ValueError('complete circuit row partition differs')
    lengths=np.r_[np.diff(hz.Ac.indptr),[len(a[0]) for a in auxiliary]].astype(np.int64)
    for physical,(cs,_,_,_) in replacements.items():lengths[physical]=len(cs)
    nnz=int(lengths.sum())
    if nnz>=hz.Ac.nnz or nnz>=2**31:raise ValueError('whole Ac nnz does not strictly decrease')
    pool.charge('c91_complete_native_CSR_assembly',4*nnz+8*new_rows)
    ptr=np.r_[0,np.cumsum(lengths)].astype(np.int32)
    cs=np.empty(nnz,np.int32);vs=np.empty(nnz,np.float64)
    previous=0
    for physical in sorted(replacements):
        a,b=int(hz.Ac.indptr[previous]),int(hz.Ac.indptr[physical]);d=int(ptr[previous])
        cs[d:d+b-a]=hz.Ac.indices[a:b];vs[d:d+b-a]=hz.Ac.data[a:b]
        cols,vals,_,_=replacements[physical];d,e=map(int,ptr[physical:physical+2])
        cs[d:e]=cols;vs[d:e]=vals;previous=physical+1
    a,b=int(hz.Ac.indptr[previous]),hz.Ac.nnz;d=int(ptr[previous])
    cs[d:d+b-a]=hz.Ac.indices[a:b];vs[d:d+b-a]=hz.Ac.data[a:b]
    for i,(cols,vals,_,_,_) in enumerate(auxiliary):
        d,e=map(int,ptr[neq+i:neq+i+2]);cs[d:e]=cols;vs[d:e]=vals
    ac=sp.csr_matrix((vs,cs,ptr),shape=(new_rows,nc2),copy=False)
    bp=np.r_[hz.Ab.indptr,np.full(total,hz.Ab.indptr[-1],hz.Ab.indptr.dtype)]
    ab=sp.csr_matrix((hz.Ab.data,hz.Ab.indices,bp),shape=(new_rows,hz.n_bin),copy=False)
    after=SparseHZono(hz.c,widen(hz.Gc,nc2),hz.Gb,ac,ab,np.r_[hz.b,np.zeros(total)],
        widen(hz.Auc,nc2),hz.Aub,hz.ub,frame_id=hz.frame_id,exact=True)
    owners=fields['owners'].copy();new_owners=np.zeros(total,np.int64)
    def change(columns,uid,sign):
        pool.charge('c91_known_owner_incidence_updates',4*len(columns))
        mask=(columns>=old)&(columns<main);np.add.at(owners,columns[mask]-old,sign*(RADIX+uid))
        mask=columns>=nc;np.add.at(new_owners,columns[mask]-nc,sign*(RADIX+uid))
    for physical,(cols,_,_,_) in replacements.items():
        change(row(hz.Ac,physical)[0],int(eq_uids[physical]),-1);change(cols,int(eq_uids[physical]),1)
    for i,(cols,_,_,_,_) in enumerate(auxiliary):change(cols,first_uid+i,1)
    validate_words(owners);validate_words(new_owners)
    pool.charge('c91_complete_auxiliary_inverse_owner_records',64*total+32*len(routes)+64*len(blocks))
    aux=np.empty((total,8),np.int64)
    bound_bits=np.array([-1.,1.],np.float64).view(np.int64)
    for i,(_,_,pivot,gauge,pivot_value) in enumerate(auxiliary):
        aux[i]=(neq+i,first_uid+i,pivot,np.array(pivot_value,np.float64).view(np.int64).item(),gauge,
            int(new_owners[i]),int(bound_bits[0]),int(bound_bits[1]))
    new_fields=dict(fields);new_fields['hz']=after;new_fields['owners']=owners
    eqscales=fields['eq_scales']
    ranks=np.array([int(fields['old_n_eq'])+p-old for _,p,_ in routes],np.int64)
    gauges=np.array([g for _,_,g in routes],np.int64)
    if not np.array_equal(eqscales[ranks],gauges):
        eqscales=eqscales.copy();eqscales[ranks]=gauges
    new_fields['eq_scales']=eqscales
    report=dict(fields['report']);report.update(source_representation_schema=SCHEMA,
        n_cont=nc2,n_eq=new_rows,circuit_auxiliaries=total,circuit_rows_replaced=len(routes),
        total_shared_auxiliaries=total+len(fields['def_rows']),fresh_generation_work_proved=False,
        declared_circuit_emission_reserve_work=emission,
        inherited_report_scope='C69 base generation only; excludes C91 construction')
    new_fields['report']=report
    return dict(schema=SCHEMA,fields=new_fields,old_source_n_cont=nc,old_source_n_eq=neq,
        auxiliary_records=aux,output_routes=np.array(routes,np.int64),block_records=np.array(blocks,np.int64),
        original_source_proof=source_proof,original_circuit_proof=circuit_proof,
        native_or_LIVE_admission=False,formal_gain=0)


def extend(state,point,*,pool):
    """Unique new-coordinate extension from actual HZ equations, not word packets."""
    if state['schema']!=SCHEMA or len(point)!=state['old_source_n_cont']:raise ValueError('explicit original source point required')
    pool.charge('c91_full_original_point',16*len(point));result=list(map(F,point));hz=state['fields']['hz']
    if any(abs(v)>1 for v in result):raise ValueError('original box violated')
    for physical,_,pivot,_,_,_,_,_ in state['auxiliary_records']:
        cols,values=row(hz.Ac,int(physical));pool.charge('c91_exact_new_inverse',64*len(cols))
        pivot=int(pivot)
        if pivot!=len(result):raise ValueError('new inverse order differs')
        coefs={int(c):F(float(v)) for c,v in zip(cols,values,strict=True)};p=coefs.pop(pivot)
        if p<=0 or any(c>=pivot for c in coefs) or sum(map(abs,coefs.values()),F(0))>p:
            raise ValueError('new inverse topology/box differs')
        v=(F(float(hz.b[int(physical)]))-sum((a*result[c] for c,a in coefs.items()),F(0)))/p
        if abs(v)>1:raise ValueError('new inverse outside redundant box')
        result.append(v)
    return result


def recover(state,point,*,pool):
    """Drop circuit variables and restore every original local scalar equation."""
    if state['schema']!=SCHEMA or len(point)!=state['fields']['hz'].n_cont:raise ValueError('explicit complete circuit point required')
    fields=state['fields'];base=state['old_source_n_cont']
    pool.charge('c91_complete_local_inverse_composition',16*base+128*int(np.count_nonzero(fields['eq_roots']<0)))
    return reconstruct(point[:base],fields['eq_roots'],fields['eq_scales'],old_n_cont=fields['old_n_cont'],
        old_n_eq=fields['old_n_eq'],n_cont=base,schema=LOCAL)


def _incidence(matrix,rows,uids,width,*,pool):
    """Independent counts/UID sums; large packed words never enter floats."""
    sizes=matrix.indptr[rows+1]-matrix.indptr[rows]
    total=int(sizes.sum())
    pool.charge('c91_independent_split_incidence',8*total+8*width)
    if len(rows)>UID_LIMIT or np.any(uids<0) or np.any(uids>=UID_LIMIT):
        raise ValueError('exact split UID sum domain exceeded')
    columns=np.concatenate([row(matrix,int(r))[0] for r in rows]) if total else np.empty(0,np.int32)
    weights=np.repeat(uids,sizes)
    counts=np.bincount(columns,minlength=width)
    sums=np.bincount(columns,weights=weights.astype(np.float64),minlength=width)
    integers=sums.astype(np.int64)
    if (len(counts)!=width or counts.max(initial=0)>UID_LIMIT or np.any(sums!=integers)
        or np.any(integers>counts*(UID_LIMIT-1))):raise ValueError('exact split incidence domain differs')
    return counts.astype(np.int64)*RADIX+integers


def audit(before,state,circuits,*,pool,enabled=False):
    """Every old/new row and all owner/map fields, from a qualified old source."""
    if not enabled:return None
    if state['schema']!=SCHEMA or state['native_or_LIVE_admission']:raise ValueError('explicit non-admitted circuit source required')
    fields=state['fields'];old=before['hz'];new=fields['hz'];aux=state['auxiliary_records'];routes=state['output_routes']
    pool.charge('c91_all_original_row_buffer_comparison',4*old.Ac.nnz+8*old.n_eq)
    if (new.frame_id!=old.frame_id or not new.exact or new.n_bin!=old.n_bin
        or new.n_cont!=old.n_cont+len(aux) or new.n_eq!=old.n_eq+len(aux)
        or not np.array_equal(new.c,old.c) or not np.array_equal(new.ub,old.ub)
        or not np.array_equal(new.b[:old.n_eq],old.b) or np.any(new.b[old.n_eq:])):
        raise ValueError('complete original frame/constant/binary partition differs')
    for name in ('Gc','Gb','Auc','Aub'):
        a,b=getattr(old,name),getattr(new,name)
        if any(not np.array_equal(getattr(a,k),getattr(b,k)) for k in ('data','indices','indptr')):
            raise ValueError('original value/inequality coefficients differ')
    if (not np.array_equal(old.Ab.data,new.Ab.data) or not np.array_equal(old.Ab.indices,new.Ab.indices)
        or not np.array_equal(old.Ab.indptr,new.Ab.indptr[:old.n_eq+1])
        or np.any(new.Ab.indptr[old.n_eq+1:]!=old.Ab.indptr[-1])):
        raise ValueError('original binary equations differ')
    pool.charge('c91_full_unchanged_maps_and_value_arrays',4*sum(before[k].size for k in
        ('keep','eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows','uid_slabs','radix_gauges'))
        +4*sum(getattr(old,k).nnz+getattr(old,k).shape[0]+1 for k in ('Gc','Gb','Ab','Auc','Aub')))
    changed=np.sort(routes[:,0]);previous=0
    for physical in [*map(int,changed),old.n_eq]:
        oa,ob=int(old.Ac.indptr[previous]),int(old.Ac.indptr[physical])
        na,nb=int(new.Ac.indptr[previous]),int(new.Ac.indptr[physical])
        if (not np.array_equal(old.Ac.data[oa:ob],new.Ac.data[na:nb])
            or not np.array_equal(old.Ac.indices[oa:ob],new.Ac.indices[na:nb])
            or not np.array_equal(np.diff(old.Ac.indptr[previous:physical+1]),np.diff(new.Ac.indptr[previous:physical+1]))):
            raise ValueError('an unchanged original physical row differs')
        previous=physical+1
    eq_uids,_=closed_uid_tables(SimpleNamespace(**before));first=int(before['report']['radix_uid_base'])+len(before['def_rows'])
    expected_routes=[];expected_blocks=[];offset=0;written_nnz=0
    for packet in circuits:
        begin=len(expected_routes)
        n=packet['new_factors'];pool.charge('c91_every_written_circuit_literal',8*len(packet['native'])+64*len(packet['rhs']))
        for i,pivot in enumerate(packet['pivots']):
            a,b=map(int,packet['indptr'][i:i+2]);pivot=int(pivot)
            if i<n:
                physical=old.n_eq+offset+i
                p=int(np.flatnonzero(packet['columns'][a:b]==pivot)[0]);pv=float(packet['native'][a+p])
                record=aux[offset+i]
                if (record[0]!=physical or record[1]!=first+offset+i or record[2]!=pivot
                    or record[3]!=np.array(pv,np.float64).view(np.int64).item() or record[4]!=packet['gauges'][i]
                    or not np.array_equal(record[6:8].view(np.float64),np.array([-1.,1.]))):
                    raise ValueError('complete new inverse/UID/bound record differs')
            else:
                rank=before['old_n_eq']+pivot-before['old_n_cont'];physical=int(before['eq_roots'][rank])
                expected_routes.append((physical,pivot,int(packet['gauges'][i])))
                if fields['eq_scales'][rank]!=packet['gauges'][i]:raise ValueError('actual output row gauge differs')
            cols,vals=row(new.Ac,physical)
            if (not np.array_equal(cols,packet['columns'][a:b]) or not np.array_equal(vals,packet['native'][a:b])):
                raise ValueError('actual written circuit literal differs')
            written_nnz+=b-a
        expected_blocks.append((old.n_cont,old.n_cont+offset,n,begin,len(expected_routes),old.n_eq+offset,first+offset,0))
        offset+=n
    if not np.array_equal(routes,np.array(expected_routes,np.int64)):raise ValueError('complete circuit root routes differ')
    if not np.array_equal(state['block_records'],np.array(expected_blocks,np.int64)):raise ValueError('complete block routing differs')
    if (offset!=len(aux) or state['old_source_n_cont']!=old.n_cont or state['old_source_n_eq']!=old.n_eq
        or fields['expression'] is not before['expression']):raise ValueError('complete original source identity differs')
    for name in ('keep','eq_roots','ineq_roots','ineq_scales','def_rows','uid_slabs','radix_gauges'):
        if not np.array_equal(before[name],fields[name]):raise ValueError('original source/inverse map changed: '+name)
    unchanged_scales=np.ones(len(before['eq_scales']),bool)
    unchanged_scales[before['old_n_eq']+routes[:,1]-before['old_n_cont']]=False
    if not np.array_equal(before['eq_scales'][unchanged_scales],fields['eq_scales'][unchanged_scales]):
        raise ValueError('old scalar inverse numerator/tag changed')
    old_words=_incidence(old.Ac,changed,eq_uids[changed],new.n_cont,pool=pool)
    added_rows=np.r_[changed,np.arange(old.n_eq,new.n_eq)].astype(np.int64)
    added_uids=np.r_[eq_uids[changed],np.arange(first,first+len(aux))].astype(np.int64)
    new_words=_incidence(new.Ac,added_rows,added_uids,new.n_cont,pool=pool)
    left,right=before['old_n_cont'],before['logical_n_cont']
    if (not np.array_equal(fields['owners'],before['owners']+new_words[left:right]-old_words[left:right])
        or not np.array_equal(aux[:,5],new_words[old.n_cont:])):
        raise ValueError('complete old/new owner vectors differ from independent incidence')
    if new.Ac.nnz>=old.Ac.nnz:raise ValueError('whole physical predicate nnz does not decrease')
    return dict(complete_original_rows_checked=old.n_eq,actual_original_rows_replaced=len(routes),
        actual_auxiliary_rows_checked=len(aux),actual_written_native_coefficients=written_nnz,
        complete_old_MAIN_owners_checked=len(fields['owners']),complete_new_owners_checked=len(aux),
        old_scalar_equations_retained=int(np.count_nonzero(fields['eq_roots']<0)),
        original_binary_factors=old.n_bin,all_original_maps_and_other_predicates_preserved=True,
        independent_complete_owner_delta_proved=True,whole_predicate_nnz_before=int(old.Ac.nnz+old.Ab.nnz+old.Auc.nnz+old.Aub.nnz),
        whole_predicate_nnz_after=int(new.Ac.nnz+new.Ab.nnz+new.Auc.nnz+new.Aub.nnz),
        source_and_box_theorem_requires_authenticated_C90=True,full_LIVE_admission=False,formal_gain=0)
