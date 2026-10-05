"""Full original-row/inverse proof and complete certified-owner delta partition.

No legacy decoder is used on new local-equation tags. No proof follows merely
from a generator report, exact flag, expected score, hash or solver status.
"""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import math
import numpy as np
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import native_word,word,fraction
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA,decode
from experiments.neural_hz_20260831.c62_precision_plan_v1 import odd_significands
from experiments.neural_hz_20260831.c62_physical_quotient_v1 import equal,same_matrix,source_maps
from experiments.neural_hz_20260831.c63_birth_blocks_v1 import route_rows
from experiments.neural_hz_20260831.c64_gauged_products_v1 import gauge_row
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX,validate_words
from experiments.neural_hz_20260831.c18_sparse_rewrite_v1 import sort_cost
from experiments.neural_hz_20260831.c7_factored_hz_v1 import expression_binding


def audit(saved,legacy,candidate,*,pool,enabled=False):
    if not enabled:return None
    fields=candidate['fields'];construction=candidate['construction'];hz=fields['hz'];original=saved['hz']
    old=int(saved['old_n_cont']);oe=int(saved['old_n_eq']);logical=int(saved['logical_n_cont']);main=logical-old
    if (candidate['lineage_schema']!=SCHEMA or candidate['schema']!='c65_owned_normal_birth_precision_draft_v1'
            or not candidate['independent_qualification_pending'] or not hz.exact or not original.exact
            or hz.frame_id!=original.frame_id or hz.n_cont!=original.n_cont or hz.n_bin!=original.n_bin
            or fields['old_n_cont']!=old or fields['old_n_eq']!=oe or fields['logical_n_cont']!=logical
            or fields['old_n_bin']!=original.n_bin or fields['expression'] is not saved['expression']
            or construction['origin_binding']!=expression_binding(saved['expression'])):
        raise ValueError('complete original expression/frame and new lineage schema required')
    pool.charge('c65_complete_original_header_and_graph',8*(main+hz.n_cont+original.n_eq)+1024)
    for name in ('eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows','radix_gauges','owners'):
        if type(fields[name]) is not np.ndarray or fields[name].ndim!=1 or fields[name].dtype!=np.dtype(np.int64):
            raise ValueError('native complete int64 lineage/owner vector required: '+name)
    if fields['eq_scales'].shape!=fields['eq_roots'].shape:raise ValueError('complete local inverse word pairs required')
    if not equal(fields['keep'],saved['keep']) or fields['eq_roots'].shape!=(oe+main,):raise ValueError('original full logical frame differs')
    old_nodes=saved['definition_graph'];nodes=construction['nodes']
    if len(nodes)!=len(old_nodes) or construction['root']!=saved['root']:raise ValueError('complete source DAG differs')
    pool.charge('c65_all_new_graph_array_bits',4*sum(n[k].size for n in old_nodes for k in ('support','needed','slots','exponents')))
    for a,b in zip(nodes,old_nodes):
        if any(a[k]!=b[k] for k in ('kind','parents','width')):raise ValueError('source DAG shape differs')
        for key in ('support','needed','slots','exponents'):
            if not equal(a[key],b[key]):raise ValueError('source graph array differs: '+key)
        for key in ('source','op'):
            if key in b and a[key] is not b[key]:raise ValueError('original source/operator sharing differs')
    for name in ('Gc','Gb'):
        if not same_matrix(getattr(hz,name),getattr(original,name)):raise ValueError('complete original output differs')
    if not equal(hz.c,original.c):raise ValueError('source output center differs')
    physical=saved['eq_roots'][oe:];start=original.Ac.indptr[physical];stop=original.Ac.indptr[physical+1]
    rawmask=(stop-start==2)&(original.Ab.indptr[physical+1]==original.Ab.indptr[physical])&(original.b[physical]==0)
    protected=np.zeros(hz.n_cont,bool);protected[original.Gc.indices]=True;columns=old+np.arange(main)
    rawmask&=~protected[columns]
    last=np.full(main,-1,np.int64);nonempty=stop>start;last[nonempty]=original.Ac.indices[stop[nonempty]-1]
    rawmask&=last==columns
    raw=np.zeros(hz.n_cont,bool);raw[columns[rawmask]]=True
    selected=np.zeros(hz.n_cont,bool);selected[old:logical]=fields['eq_roots'][oe:]<0
    if np.any(selected&~raw):raise ValueError('new local equation outside original raw cohort')
    parents={};local={};roots=np.arange(hz.n_cont,dtype=np.int64);weights=[(1,0)]*hz.n_cont
    own=np.zeros(original.n_eq,bool);erased=np.zeros(original.n_eq,bool);decoded={}
    pool.charge('c65_all_original_local_equations',32*int(raw.sum())+1024)
    for v0 in np.flatnonzero(raw):
        v=int(v0);i=oe+v-old;r=int(saved['eq_roots'][i]);a=int(original.Ac.indptr[r])
        p=int(original.Ac.indices[a]);left=native_word(original.Ac.data[a]);pivot=native_word(original.Ac.data[a+1])
        if not 0<=p<v or pivot[0]!=1:raise ValueError('original exact topological singleton differs')
        ratio=word(-left[0],left[1]-pivot[1])
        if not 0<abs(fraction(ratio))<=1:raise ValueError('original singleton box is not redundant')
        parents[v]=p;local[v]=ratio;own[r]=True
        if selected[v]:
            tag=int(fields['eq_roots'][i]);num=float(fields['eq_scales'].view(np.float64)[i]);bits=tag+(1<<64)
            key=((bits>>32)&63,num)
            if key not in decoded:
                pool.charge('c65_distinct_local_inverse_decode',128)
                _,decoded[key]=decode(tag,num,column=v,n_cont=hz.n_cont,schema=SCHEMA)
            if bits&~((1<<63)|((1<<38)-1)) or bits&((1<<32)-1)!=p or decoded[key]!=ratio:
                raise ValueError('complete written inverse is not the original local equation')
            erased[r]=True
            if selected[p]:
                pool.charge('c57_exact_dyadic_product',64*max(1,(abs(weights[p][0]).bit_length()+63)//64))
                weights[v]=word(ratio[0]*weights[p][0],ratio[1]+weights[p][1]);roots[v]=roots[p]
            else:weights[v]=ratio;roots[v]=p
    mapping=np.cumsum(~erased,dtype=np.int64)-1;mapping[erased]=-1
    surviving=fields['eq_roots']>=0
    if (not np.array_equal(fields['eq_roots'][surviving],mapping[saved['eq_roots'][surviving]])
            or hz.n_eq!=int((~erased).sum()) or hz.n_ineq!=original.n_ineq
            or not np.array_equal(fields['def_rows'],mapping[saved['def_rows']])):
        raise ValueError('complete new physical predicate partition differs')
    qrows=np.zeros(original.n_eq,np.int64)
    qrows[saved['eq_roots'][surviving]]=fields['eq_scales'][surviving]-saved['eq_scales'][surviving]
    if fields['radix_gauges'].shape!=saved['def_rows'].shape:raise ValueError('complete radix gauge map required')
    qrows[saved['def_rows']]=fields['radix_gauges']
    if np.any((qrows<0)|(qrows>60)):raise ValueError('unregistered whole-row gauge')
    routed,_=route_rows(saved,np.flatnonzero(raw),pool=pool,enabled=True)
    oldmap,old_le,original_uids,le_uids=source_maps(saved,legacy,pool)
    expected_euids=original_uids[~erased]
    if not equal(construction['eq_uids'],expected_euids) or not equal(construction['ineq_uids'],le_uids):
        raise ValueError('all emitted UIDs differ from original source proof')
    if not equal(fields['uid_slabs'],legacy['uid_slabs']):raise ValueError('reserved UID/slab identities differ')
    for name in ('ineq_roots','ineq_scales'):
        if not equal(fields[name],saved[name]):raise ValueError('old complete inequality map differs')
    for name in ('Auc','Aub'):
        if not same_matrix(getattr(hz,name),getattr(original,name)):raise ValueError('old complete inequality differs')
    if not equal(hz.ub,original.ub):raise ValueError('old complete inequality RHS differs')
    pool.charge('c65_complete_original_and_written_predicate_comparison',
        2*int(original.Ac.nnz+original.Ab.nnz+original.Auc.nnz+original.Aub.nnz)+8*(original.n_eq+original.n_ineq)+1024)
    maxima=np.zeros(hz.n_cont,np.uint64);row_gauges=Counter();fractions=changed_rows=0
    owner=legacy['owners'].copy();oldroots=np.arange(hz.n_cont,dtype=np.int64)
    alias=legacy['eq_roots'][oe:]<0;oldcols=old+np.flatnonzero(alias)
    oldroots[oldcols]=-legacy['eq_roots'][oe:][alias]-1
    different=roots!=oldroots;owner_rows=owner_terms=0
    for r in range(original.n_eq):
        a,b=map(int,original.Ac.indptr[r:r+2]);cc=original.Ac.indices[a:b];cv=original.Ac.data[a:b]
        ba,bb=map(int,original.Ab.indptr[r:r+2]);bc=original.Ab.indices[ba:bb];bv=original.Ab.data[ba:bb];rhs=float(original.b[r])
        if routed[r] and not own[r]:
            positions=np.flatnonzero(raw[cc])
            pool.charge('c65_independent_complete_source_maximum',16*len(positions))
            np.maximum.at(maxima,cc[positions],odd_significands(cv[positions]))
        target=int(mapping[r]);oi=int(oldmap[r]);uid=int(original_uids[r])
        if target>=0:
            positions=np.flatnonzero(selected[cc]) if routed[r] else np.empty(0,np.int64)
            outc=cc;outv=cv;outb=bv;outr=rhs;q=0
            if len(positions):
                pool.charge('c65_all_written_products_Fraction',64*len(positions))
                outc=cc.copy();outv=cv.copy()
                for j in positions:
                    v=int(cc[j]);exact=F(float(cv[j]))*fraction(weights[v]);value=float(exact)
                    if F(value)!=exact:raise ValueError('complete exact source product is not native')
                    outc[j]=roots[v];outv[j]=value;fractions+=1
                pool.charge('rewrite_sort',sort_cost(len(cc)))
                order=np.argsort(outc,kind='stable');outc=outc[order];outv=outv[order]
                if np.any(np.diff(outc)<=0):raise ValueError('source root coalescence outside full boundary class')
                outv,outb,outr,q=gauge_row(outv,bv,rhs,pool=pool)
                changed_rows+=1;row_gauges[q]+=1
            if q!=int(qrows[r]):raise ValueError('complete independent row gauge differs')
            ca,cb=map(int,hz.Ac.indptr[target:target+2]);da,db=map(int,hz.Ab.indptr[target:target+2])
            if (not equal(hz.Ac.indices[ca:cb],outc) or not equal(hz.Ac.data[ca:cb],outv)
                    or not equal(hz.Ab.indices[da:db],bc) or not equal(hz.Ab.data[da:db],outb)
                    or hz.b[target]!=outr):raise ValueError('complete written physical row differs from exact original relation')
        # The certified C31 owner vector covers unchanged column sets/UIDs.
        # Every changed column set is computed from ACTUAL old/new CSR rows.
        changed=(target<0 and oi>=0) or (target>=0 and (oi<0 or (routed[r] and different[cc].any())))
        if changed:
            owner_rows+=1
            for sign,index,matrix in [(-1,oi,legacy['hz'].Ac),(1,target,hz.Ac)]:
                if index<0:continue
                ids=matrix.indices[matrix.indptr[index]:matrix.indptr[index+1]]
                ids=ids[(ids>=old)&(ids<logical)]-old
                pool.charge('c65_independent_actual_changed_owner_incidence',8*len(ids));owner_terms+=len(ids)
                np.add.at(owner,ids,sign*(RADIX+uid))
                if np.any(owner[ids]<0):raise ValueError('independent changed ownership underflow')
    validate_words(owner)
    if not equal(owner,fields['owners']) or np.any(owner[selected[old:logical]]):raise ValueError('all MAIN owners differ from complete source partition')
    digest=hashlib.sha256();pool.charge('c65_complete_reconstructed_boundary_identity',16*len(parents))
    for v in sorted(parents):
        digest.update(json.dumps([v,parents[v],local[v],int(maxima[v]),bool(selected[v]),int(roots[v]),weights[v]]).encode())
    identity=digest.hexdigest();report=fields['report']['alias_quotient']
    if (identity!=report['identity_sha256'] or report['optimum']!=int(selected.sum())
            or report['rewritten_rows']!=changed_rows or report['row_gauges']!=dict(row_gauges)):
        raise ValueError('reported complete boundary/gauges not independently reproduced')
    nnz=int(sum(getattr(hz,k).nnz for k in ('Ac','Ab','Auc','Aub')))
    if nnz!=int(sum(getattr(original,k).nnz for k in ('Ac','Ab','Auc','Aub')))-2*int(selected.sum()):
        raise ValueError('complete exact strict predicate reduction differs')
    return dict(schema='c65_complete_original_rows_local_inverse_owner_proof_v1',
        original_EQ_checked=original.n_eq,original_INEQ_checked=original.n_ineq,
        original_raw_factors=len(parents),local_inverse_equations=int(selected.sum()),
        exact_Fraction_products=fractions,all_MAIN_owners_checked=len(owner),
        actual_owner_delta_rows=owner_rows,actual_owner_delta_occurrences=owner_terms,
        certified_unchanged_owner_rows_and_all_changed_actual_rows_proved=True,
        separate_full_actual_words_scan=False,original_nonconvex_binary_factors=hz.n_bin,
        all_original_graph_arrays_checked=4*len(nodes),actual_predicate_nnz=nnz,
        original_boxes_and_all_new_inverse_boxes_proved=True,complete_source_boundary_sha256=identity,
        new_native_or_LIVE_admission=False,concrete_network_witness=False,formal_gain=0)
