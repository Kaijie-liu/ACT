"""Original-DAG-routed exact sufficient statistics; no HZ or source writer."""
from collections import Counter
import hashlib
import json
import numpy as np
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import native_word,word,fraction
from experiments.neural_hz_20260831.c62_local_equations_v1 import encode
from experiments.neural_hz_20260831.c63_birth_blocks_v1 import route_rows

LIMIT=(1<<53)-1
BAD=-10**9


def odd_significands(values):
    a=np.asarray(values)
    if a.dtype!=np.dtype(np.float64) or a.ndim!=1:raise ValueError('native binary64 vector required')
    bits=a.view(np.uint64);exponent=(bits>>np.uint64(52))&np.uint64(2047)
    if np.any((exponent==0)|(exponent==2047)):raise ValueError('nonzero normal native coefficients required')
    m=(bits&np.uint64((1<<52)-1))|np.uint64(1<<52)
    divisor=m&(~m+np.uint64(1))
    return m//divisor


def choose(parents,local,maxima,pool):
    """Leaf-closed C61 recurrence, with exact integer maximum constraints."""
    pool.charge('c62_compact_forest_metadata',16*len(parents)+1024)
    columns=sorted(parents);children={v:[] for v in columns};states={};scores={};choices={}
    for v in columns:
        p=parents[v]
        if not 0<=p<v:raise ValueError('original topological parent required')
        if p in parents:children[p].append(v)
        size=1+len(states.get(p,{}));pool.charge('c62_original_ancestor_segments',32*size)
        states[v]={p:local[v]}
        for a,r in states.get(p,{}).items():
            # Native output is never made from this possibly long exact word.
            pool.charge('c57_exact_dyadic_product',64*max(1,(abs(r[0]).bit_length()+63)//64))
            states[v][a]=word(local[v][0]*r[0],local[v][1]+r[1])
    leaf_states=nonleaf_states=0
    for v in reversed(columns):
        scores[v]={};choices[v]={};c=children[v]
        pool.charge('c62_leaf_or_nonleaf_recurrence',16*len(states[v])+16*len(c)*(len(states[v])+1))
        kept=sum(scores[t][v] for t in c) if all(scores[t][v]>=0 for t in c) else BAD
        bound=LIMIT//int(maxima.get(v,0)) if maxima.get(v,0) else None
        for a,r in states[v].items():
            keep=kept if abs(r[0])<=LIMIT else BAD
            can_remove=bound is None or abs(r[0])<=bound
            if not c:
                remove=1 if can_remove else BAD;leaf_states+=1
            else:
                remove=1+sum(scores[t][a] for t in c) if can_remove and all(scores[t][a]>=0 for t in c) else BAD
                nonleaf_states+=1
            scores[v][a]=max(keep,remove);choices[v][a]=remove>keep
    pool.charge('c62_boundary_selection_record',32*len(columns))
    tops=[v for v in columns if parents[v] not in parents]
    if any(scores[v][parents[v]]<0 for v in tops):raise ValueError('no exact boundary representation')
    optimum=sum(scores[v][parents[v]] for v in tops);selected={};anchors={};weights={}
    stack=[(v,parents[v]) for v in tops]
    while stack:
        v,a=stack.pop();selected[v]=choices[v][a];anchors[v]=a if selected[v] else v
        weights[v]=states[v][a] if selected[v] else (1,0)
        stack.extend((t,anchors[v]) for t in children[v])
    if sum(selected.values())!=optimum:raise ValueError('incomplete optimum reconstruction')
    return selected,anchors,weights,dict(optimum=optimum,ancestor_states=sum(map(len,states.values())),
        leaf_states=leaf_states,nonleaf_states=nonleaf_states)


def plan(saved,*,pool,enabled=False):
    if not enabled:return None
    hz=saved['hz'];old=int(saved['old_n_cont']);logical=int(saved['logical_n_cont']);old_eq=int(saved['old_n_eq'])
    main=logical-old;eq_roots=saved['eq_roots'];defs=saved['def_rows']
    if not 0<=old<=logical<=hz.n_cont<2**32 or eq_roots.shape!=(old_eq+main,):raise ValueError('complete original source frame required')
    pool.charge('c62_complete_MAIN_header_partition',8*(main+hz.n_cont+hz.n_eq)+1024)
    if not np.array_equal(np.sort(np.r_[eq_roots,defs]),np.arange(hz.n_eq)):raise ValueError('original physical EQ partition differs')
    physical=eq_roots[old_eq:];a=hz.Ac.indptr[physical];b=hz.Ac.indptr[physical+1]
    proposed=(b-a==2)&(hz.Ab.indptr[physical+1]==hz.Ab.indptr[physical])&(hz.b[physical]==0)
    slots=old+np.arange(main);protected=np.zeros(hz.n_cont,bool);protected[hz.Gc.indices]=True
    proposed&=~protected[slots]
    # Packed physical roots cannot masquerade as a logical singleton definition.
    nonempty=b>a;last=np.full(main,-1,np.int64);last[nonempty]=hz.Ac.indices[b[nonempty]-1]
    proposed&=last==slots
    chosen_slots=slots[proposed];parents={};local={};own=np.full(hz.n_eq,-1,np.int64)
    tags={};numerators={};equations={};pool.charge('c62_original_local_equation_binding',32*len(chosen_slots))
    for v0 in chosen_slots:
        v=int(v0);row=int(eq_roots[old_eq+v-old]);start=int(hz.Ac.indptr[row])
        p=int(hz.Ac.indices[start]);left=native_word(hz.Ac.data[start]);pivot=native_word(hz.Ac.data[start+1])
        if not 0<=p<v or pivot[0]!=1:raise ValueError('topological positive dyadic original definition required')
        r=word(-left[0],left[1]-pivot[1])
        if r not in equations:
            pool.charge('c62_distinct_local_equation_encoding',128)
            equations[r]=encode(0,r)
        tag,num=equations[r];tag+=p
        parents[v]=p;local[v]=r;own[row]=v;tags[v]=tag;numerators[v]=num
    lookup=np.full(hz.n_cont,-1,np.int64);lookup[chosen_slots]=np.arange(len(chosen_slots));maxima=np.zeros(len(chosen_slots),np.uint64)
    occurrences=0;rows_total=0;raw_hits={};routing=None
    for name in ('Ac','Auc','Gc'):
        matrix=getattr(hz,name);nr=matrix.shape[0]
        if name=='Ac':
            routed,routing=route_rows(saved,chosen_slots,pool=pool,enabled=True)
            raw_hits[name]=own>=0
            routed[own>=0]=False
            rows=np.flatnonzero(routed)
        else:
            pool.charge('c59_full_consumer_incidence_scan',3*matrix.nnz+8*nr+1024)
            hit=lookup[matrix.indices]>=0;prefix=np.empty(matrix.nnz+1,np.int64);prefix[0]=0;np.cumsum(hit,out=prefix[1:])
            affected=prefix[matrix.indptr[1:]]>prefix[matrix.indptr[:-1]]
            raw_hits[name]=affected.copy()
            if name=='Gc' and affected.any():raise ValueError('output-live factor cannot be projected')
            rows=np.flatnonzero(affected);del hit,prefix,affected
        for row in rows:
            start,stop=map(int,matrix.indptr[row:row+2])
            if name=='Ac':pool.charge('c63_routed_incidence_row',16+3*(stop-start))
            ids=lookup[matrix.indices[start:stop]];mask=ids>=0
            if not mask.any():continue
            raw_hits[name][row]=True;rows_total+=1
            values=matrix.data[start:stop][mask];pool.charge('c62_complete_external_maximum',32+8*len(ids)+16*len(values))
            np.maximum.at(maxima,ids[mask],odd_significands(values));occurrences+=len(values)
    routing['routed_nonraw_rows_inspected']=int(np.count_nonzero(routed))
    routing['routed_nonraw_coefficients_inspected']=int(np.diff(hz.Ac.indptr)[routed].sum())
    maximum_map={int(v):int(m) for v,m in zip(chosen_slots,maxima)}
    selected,anchors,weights,stats=choose(parents,local,maximum_map,pool)
    pool.charge('c62_owned_plan_arrays',8*hz.n_cont+16*len(chosen_slots)+1024)
    marked=np.zeros(hz.n_cont,bool);root=np.arange(hz.n_cont,dtype=np.int64);ratio=[(1,0)]*hz.n_cont
    for v in parents:marked[v]=selected[v];root[v]=anchors[v];ratio[v]=weights[v]
    erased=np.zeros(hz.n_eq,bool)
    for row in np.flatnonzero(own>=0):erased[row]=selected[int(own[row])]
    classes=Counter('power_two' if abs(local[v][0])==1 else 'general' for v in parents if selected[v])
    h=hashlib.sha256()
    for v in sorted(parents):h.update(json.dumps([v,parents[v],local[v],maximum_map[v],selected[v],anchors[v],weights[v]]).encode())
    return dict(selected=marked,roots=root,weights=ratio,erased=erased,parents=parents,local=local,tags=tags,numerators=numerators,raw_hits=raw_hits,
        report=dict(schema='c62_complete_extremal_significand_plan_v1',raw_nodes=len(parents),
            selected_classes=dict(classes),external_occurrences=occurrences,external_rows=rows_total,
            identity_sha256=h.hexdigest(),birth_routing=routing,**stats))

