"""Exact physical matrices and local lineage, not a native-admitted Closed."""
from fractions import Fraction as F
import hashlib
import json
import math
import numpy as np
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c59_full_consumer_census_v2 import Products,gauge
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import native_word,fraction,in_window
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA,decode
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import unpack
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import validate_words,RADIX


def equal(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint8),b.view(np.uint8))


def same_matrix(a,b):
    return a.shape==b.shape and all(equal(getattr(a,n),getattr(b,n)) for n in ('data','indices','indptr'))


def native(value,shift):
    m,e=value
    if abs(m).bit_length()>53 or not in_window((m,e+shift)):raise ValueError('new row coefficient is not native/window exact')
    result=math.ldexp(float(m),e+shift)
    if F(result)!=fraction(value)*F(2)**shift:raise ValueError('native coefficient conversion differs')
    return result


def source_maps(saved,legacy,pool):
    hz=saved['hz'];old=legacy['hz'];main=int(saved['logical_n_cont']-saved['old_n_cont']);oe=int(saved['old_n_eq'])
    pool.charge('c62_complete_original_UID_and_old_row_bijection',16*(hz.n_eq+hz.n_ineq+main)+1024)
    if (old.n_cont,old.n_bin,legacy['old_n_eq'],legacy['logical_n_cont'])!=(hz.n_cont,hz.n_bin,oe,saved['logical_n_cont']):
        raise ValueError('same original global frame and MAIN universe required')
    if any(not equal(saved[n],legacy[n]) for n in ('keep',)) or len(saved['def_rows'])!=len(legacy['def_rows']):
        raise ValueError('same original output/radix cohort required')
    mapped=np.full(hz.n_eq,-1,np.int64);roots=saved['eq_roots'];old_roots=legacy['eq_roots'];active=old_roots>=0
    if old_roots.shape!=roots.shape:raise ValueError('complete old lineage width differs')
    mapped[roots[active]]=old_roots[active];mapped[saved['def_rows']]=legacy['def_rows']
    if not np.array_equal(np.sort(mapped[mapped>=0]),np.arange(old.n_eq)):raise ValueError('old complete physical EQ bijection differs')
    uid=np.full(hz.n_eq,-1,np.int64);uid[roots[:oe]]=np.arange(oe)
    seen=np.zeros(main,bool)
    for raw in legacy['uid_slabs']:
        first,rank,count=unpack(raw)
        if rank+count>main or seen[rank:rank+count].any():raise ValueError('original UID slabs overlap/outside MAIN')
        seen[rank:rank+count]=True;uid[roots[oe+rank:oe+rank+count]]=first+np.arange(count)
    if not seen.all():raise ValueError('original UID slab coverage incomplete')
    base=legacy['report']['radix_uid_base'];uid[saved['def_rows']]=base+np.arange(len(saved['def_rows']))
    le=np.full(hz.n_ineq,-1,np.int64);le[saved['ineq_roots']]=oe+np.arange(hz.n_ineq)
    all_uids=np.r_[uid,le]
    if np.any(all_uids<0) or np.any(all_uids>=2**20) or np.unique(all_uids).size!=len(all_uids):
        raise ValueError('complete unique bounded predicate UIDs required')
    old_le=np.full(hz.n_ineq,-1,np.int64);old_le[saved['ineq_roots']]=legacy['ineq_roots']
    if not np.array_equal(np.sort(old_le),np.arange(old.n_ineq)):raise ValueError('old complete INEQ bijection differs')
    return mapped,old_le,uid,le



def prepare(saved,legacy,plan,*,pool,enabled=False):
    """Prove only changed boundary rows; returned packet contains no source HZ."""
    if not enabled:return None
    hz=saved['hz'];old=legacy['hz'];marked=plan['selected'];roots=plan['roots'];weights=plan['weights']
    if hz.frame_id!=old.frame_id or not hz.exact or not old.exact:raise ValueError('exact original shared frame required')
    if not equal(hz.c,old.c) or not same_matrix(hz.Gc,old.Gc) or not same_matrix(hz.Gb,old.Gb):
        raise ValueError('same complete original output mapping required')
    om,olm,eu,lu=source_maps(saved,legacy,pool);oe=int(saved['old_n_eq']);oc=int(saved['old_n_cont'])
    pool.charge('c62_complete_boundary_exchange_maps',8*hz.n_cont+16*len(saved['eq_roots'])+1024)
    oldroots=np.arange(hz.n_cont,dtype=np.int64);old_alias=legacy['eq_roots'][oe:]<0
    oldcols=oc+np.flatnonzero(old_alias);oldroots[oldcols]=-legacy['eq_roots'][oe:][old_alias]-1
    if np.any(oldroots[oldroots[oldcols]]!=oldroots[oldcols]):raise ValueError('old independent frontier changed')
    different=roots!=oldroots;products=Products(pool);parts={};qcount={};digest=hashlib.sha256()
    for cm,bm,rname,oldmap,uids in [('Ac','Ab','b',om,eu),('Auc','Aub','ub',olm,lu)]:
        matrix=getattr(hz,cm);binary=getattr(hz,bm);rhs=getattr(hz,rname);nr=matrix.shape[0]
        keep=~plan['erased'] if cm=='Ac' else np.ones(nr,bool)
        pool.charge('c62_complete_migration_row_partition',16*nr+1024)
        mapping=np.cumsum(keep,dtype=np.int64)-1;mapping[~keep]=-1
        patches={};qrows=np.zeros(nr,np.int64)
        # Both old and new rules leave all other original source rows unchanged.
        candidates=np.flatnonzero(keep & (plan['raw_hits'][cm] | (oldmap<0)))
        for row0 in candidates:
            row=int(row0);a,b=map(int,matrix.indptr[row:row+2]);cols=matrix.indices[a:b];vals=matrix.data[a:b]
            pool.charge('c62_complete_changed_boundary_dispatch',16+len(cols))
            if oldmap[row]>=0 and not different[cols].any():continue
            ba,bb=map(int,binary.indptr[row:row+2])
            pool.charge('c62_exact_changed_patch_sort',64+16*len(cols)+len(cols)*max(1,len(cols).bit_length()))
            translated={}
            for col,value in zip(cols,vals):
                col=int(col);target=int(roots[col])
                if target in translated:raise ValueError('coalescing outside complete no-collision boundary class')
                translated[target]=products(native_word(value),weights[col]) if marked[col] else native_word(value)
            operands=list(translated.values())+[native_word(x) for x in binary.data[ba:bb]]
            if rhs[row]:operands.append(native_word(rhs[row]))
            unused,unused2,q=gauge(operands)
            if q is None:raise ValueError('complete changed row has no native gauge')
            order=np.array(sorted(translated),dtype=matrix.indices.dtype)
            values=np.array([native(translated[int(c)],q) for c in order])
            bc=binary.indices[ba:bb].copy();bv=np.array([native(native_word(x),q) for x in binary.data[ba:bb]])
            r=native(native_word(rhs[row]),q)
            if len(order)!=len(cols):raise ValueError('unregistered patch nnz change')
            patches[row]=(order,values,bc,bv,r)
            qrows[row]=q;qcount[str(q)]=qcount.get(str(q),0)+1
            digest.update(json.dumps([cm,row,q,order.tolist(),[x.hex() for x in values]]).encode())
        missing=(oldmap<0)&keep
        if any(int(r) not in patches for r in np.flatnonzero(missing)):raise ValueError('missing restored definition patch')
        parts[cm]=dict(oldmap=oldmap,mapping=mapping,keep=keep,patches=patches,uids=uids,qrows=qrows,
                       old_rows=getattr(old,cm).shape[0],new_rows=int(keep.sum()))
    pool.charge('c62_local_lineage_copy_and_encoding',8*len(saved['eq_roots'])+16*int(marked.sum())+1024)
    er=parts['Ac']['mapping'][saved['eq_roots']].copy();es=saved['eq_scales']+parts['Ac']['qrows'][saved['eq_roots']]
    proved={};denoms={}
    for v in plan['parents']:
        if not marked[v]:continue
        i=oe+v-oc;er[i]=plan['tags'][v];es.view(np.float64)[i]=plan['numerators'][v]
        bits=int(er[i])+(1<<64);parent=bits&((1<<32)-1);q=(bits>>32)&63;key=(q,float(es.view(np.float64)[i]))
        if key not in proved:
            pool.charge('c62_distinct_written_inverse_equation_proof',128)
            unused_parent,r=decode(int(er[i]),key[1],column=v,n_cont=hz.n_cont,schema=SCHEMA);proved[key]=r
        if parent!=plan['parents'][v] or proved[key]!=plan['local'][v] or not parent<v or bits&~((1<<63)|((1<<38)-1)):
            raise ValueError('complete written local-edge binding differs')
        denoms[str(q)]=denoms.get(str(q),0)+1
    lr=parts['Auc']['mapping'][saved['ineq_roots']].copy()
    ls=saved['ineq_scales']+parts['Auc']['qrows'][saved['ineq_roots']]
    dr=parts['Ac']['mapping'][saved['def_rows']].copy();dq=parts['Ac']['qrows'][saved['def_rows']].copy()
    added=marked[oc:saved['logical_n_cont']] & ~old_alias;lost=old_alias & ~marked[oc:saved['logical_n_cont']]
    classes={}
    for label,mask in [('new_not_old',added),('old_not_new',lost)]:
        for v0 in oc+np.flatnonzero(mask):
            v=int(v0);k=label+('_power_two' if abs(plan['local'][v][0])==1 else '_general');classes[k]=classes.get(k,0)+1
    report=dict(schema='c62_bound_boundary_migration_plan_v1',complete=True,plan=plan['report'],
        exchange_classes=classes,local_equations_checked=int(marked.sum()),local_denominator_histogram=denoms,
        patch_counts={k:len(p['patches']) for k,p in parts.items()},patch_gauge_histogram=qcount,
        radix_nonzero_gauges=int(np.count_nonzero(dq)),independent_Fraction_pairs=len(products.values),
        exact_product_lookups=products.lookups,patch_identity_sha256=digest.hexdigest(),
        original_predicate_nnz=int(sum(getattr(hz,k).nnz for k in ('Ac','Ab','Auc','Aub'))),formal_gain=0)
    # No C9 expression, source matrix, ancestor dictionaries or scalar cache escapes.
    return dict(parts=parts,eq_roots=er,eq_scales=es,ineq_roots=lr,ineq_scales=ls,def_rows=dr,
                radix_gauges=dq,removed_main=marked[oc:saved['logical_n_cont']].copy(),report=report)


def emit_matrix(old,part,which,pool):
    """Copy/check maximal unchanged CSR runs and write exact changed patches."""
    mapping=part['mapping'];oldmap=part['oldmap'];sources=np.flatnonzero(part['keep']);patches=part['patches']
    widths=np.empty(len(sources),np.int64)
    pool.charge('c62_matrix_row_layout',16*len(mapping)+1024)
    for j,row0 in enumerate(sources):
        row=int(row0)
        widths[j]=len(patches[row][which]) if row in patches else old.indptr[int(oldmap[row])+1]-old.indptr[int(oldmap[row])]
    count=int(widths.sum());pool.charge('c62_copy_write_and_complete_byte_comparison',3*count+8*len(sources)+1024)
    values=np.empty(count,np.float64);indices=np.empty(count,old.indices.dtype)
    indptr=np.empty(len(sources)+1,old.indptr.dtype);indptr[0]=0;np.cumsum(widths,out=indptr[1:])
    j=0
    while j<len(sources):
        row=int(sources[j]);a,b=map(int,indptr[j:j+2])
        if row in patches:
            cc,cv=patches[row][which:which+2]
            indices[a:b]=cc;values[a:b]=cv
            if not equal(indices[a:b],cc) or not equal(values[a:b],cv):raise ValueError('physical patch write differs')
            j+=1;continue
        begin=j;old_start=int(oldmap[row]);j+=1
        while j<len(sources) and int(sources[j]) not in patches and int(oldmap[sources[j]])==old_start+j-begin:j+=1
        end=int(indptr[j]);start=int(old.indptr[old_start]);stop=int(old.indptr[old_start+j-begin])
        if end-a!=stop-start:raise ValueError('unchanged complete CSR run length differs')
        values[a:end]=old.data[start:stop];indices[a:end]=old.indices[start:stop]
        if not equal(values[a:end],old.data[start:stop]) or not equal(indices[a:end],old.indices[start:stop]):
            raise ValueError('unchanged source-bound CSR run differs')
    out=sp.csr_matrix((values,indices,indptr),shape=(len(sources),old.shape[1]))
    if not out.has_canonical_format or np.any(out.data==0):raise ValueError('emitted complete matrix is not canonical nonzero')
    return out


def emit(legacy,prepared,*,pool,enabled=False):
    if not enabled:return None
    old=legacy['hz'];parts=prepared['parts'];matrices={};rhs={};owners=legacy['owners'].copy();expected=legacy['owners'].copy()
    pool.charge('c62_complete_owner_delta_and_rhs_metadata',8*len(owners)+8*(old.n_eq+old.n_ineq)+1024)
    old_nc=legacy['old_n_cont'];logical=legacy['logical_n_cont'];owner_rows=owner_terms=0
    for cm,bm,rname in [('Ac','Ab','b'),('Auc','Aub','ub')]:
        p=parts[cm];matrices[cm]=emit_matrix(getattr(old,cm),p,0,pool);matrices[bm]=emit_matrix(getattr(old,bm),p,2,pool)
        sources=np.flatnonzero(p['keep']);out_rhs=np.empty(len(sources),np.float64)
        for j,row0 in enumerate(sources):
            row=int(row0);out_rhs[j]=p['patches'][row][4] if row in p['patches'] else getattr(old,rname)[p['oldmap'][row]]
        rhs[rname]=out_rhs
        # All other row UIDs are unchanged, with byte-identical copied payload.
        changed=set(p['patches'])|set(map(int,np.flatnonzero((p['oldmap']>=0)&~p['keep'])))
        for row in sorted(changed):
            oi=int(p['oldmap'][row]);ni=int(p['mapping'][row]);uid=int(p['uids'][row]);owner_rows+=1
            for sign,index,matrix in [(-1,oi,getattr(old,cm)),(1,ni,matrices[cm])]:
                if index<0:continue
                cc=matrix.indices[matrix.indptr[index]:matrix.indptr[index+1]]
                ids=cc[(cc>=old_nc)&(cc<logical)]-old_nc
                pool.charge('ownership_known_incidence_updates',8*len(ids));owner_terms+=len(ids)
                count=expected[ids]//RADIX;total=expected[ids]%RADIX
                nc=count+sign;nt=total+sign*uid
                if np.any(nc<0) or np.any(nc>2**20) or np.any(nt<0) or np.any(nt>=RADIX):
                    raise ValueError('intermediate exact owner fields exceed unchanged range')
                np.add.at(owners,ids,sign*(RADIX+uid));expected[ids]=nc*RADIX+nt
    validate_words(owners)
    if not np.array_equal(owners,expected) or np.any(owners[prepared['removed_main']]):
        raise ValueError('complete emitted-row owner delta differs')
    hz=SparseHZono(old.c,old.Gc,old.Gb,matrices['Ac'],matrices['Ab'],rhs['b'],matrices['Auc'],matrices['Aub'],rhs['ub'],frame_id=old.frame_id,exact=True)
    nnz=int(sum(getattr(hz,k).nnz for k in ('Ac','Ab','Auc','Aub')))
    report=dict(prepared['report'],actual_predicate_nnz=nnz,all_MAIN_owners_checked=len(owners),
                complete_owner_delta_rows=owner_rows,owner_delta_occurrences=owner_terms,
                fresh_full_actual_words_scan=False,source_first_or_native_admission=False,concrete_network_witness=False)
    if nnz!=report['original_predicate_nnz']-2*report['local_equations_checked']:raise ValueError('complete native nnz reduction differs')
    fields={k:v for k,v in legacy.items() if k!='report'}
    fields.update(hz=hz,owners=owners,producer_report=legacy['report'])
    fields.update({k:prepared[k] for k in ('eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows','radix_gauges')})
    return dict(schema='c62_checked_physical_boundary_state_v1',lineage_schema=SCHEMA,fields=fields,proof=report),report


def build(saved,legacy,plan,*,pool,enabled=False):
    if not enabled:return None
    return emit(legacy,prepare(saved,legacy,plan,pool=pool,enabled=True),pool=pool,enabled=True)
