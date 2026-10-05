"""Same exact forest recurrence with direct states and sparse extra ancestors."""
import numpy as np
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import word

LIMIT=(1<<53)-1
BAD=-10**9


def choose(parents,local,maxima,n_cont,*,pool):
    columns=sorted(parents)
    if not 0<=n_cont<=64_000_000:raise ValueError('bounded global frame required')
    edges=sum(p in parents for p in parents.values())
    pool.charge('c64_compact_DP_arrays',4*n_cont+8*len(columns)+16*edges+1024)
    children={};extras={};scores=np.full(n_cont,BAD,np.int64);choices=np.zeros(n_cont,bool)
    extra_scores={};extra_choices={}
    for v in columns:
        p=parents[v]
        if not 0<=p<v<n_cont:raise ValueError('original topological forest required')
        if p in parents:
            children.setdefault(p,[]).append(v)
            inherited=[(parents[p],local[p]),*extras.get(p,{}).items()]
            pool.charge('c64_only_extra_ancestor_states',32*len(inherited))
            here={}
            for a,r in inherited:
                pool.charge('c57_exact_dyadic_product',64*max(1,(abs(r[0]).bit_length()+63)//64))
                here[a]=word(local[v][0]*r[0],local[v][1]+r[1])
            extras[v]=here
    states_total=leaf_states=nonleaf_states=0
    def score(t,a):
        return int(scores[t]) if a==parents[t] else extra_scores[t][a]
    for v in reversed(columns):
        cs=children.get(v,());segments=[(parents[v],local[v]),*extras.get(v,{}).items()]
        pool.charge('c62_leaf_or_nonleaf_recurrence',16*len(segments)+16*len(cs)*(len(segments)+1))
        kept=sum(int(scores[t]) for t in cs) if all(scores[t]>=0 for t in cs) else BAD
        bound=LIMIT//int(maxima[v]) if maxima[v] else None
        if v in extras:extra_scores[v]={};extra_choices[v]={}
        for a,r in segments:
            keep=kept if abs(r[0])<=LIMIT else BAD
            allowed=bound is None or abs(r[0])<=bound
            remove=1+sum(score(t,a) for t in cs) if allowed and all(score(t,a)>=0 for t in cs) else BAD
            best=max(keep,remove);deleted=remove>keep
            if a==parents[v]:scores[v]=best;choices[v]=deleted
            else:extra_scores[v][a]=best;extra_choices[v][a]=deleted
            states_total+=1;leaf_states+=not cs;nonleaf_states+=bool(cs)
    tops=[v for v in columns if parents[v] not in parents]
    if any(scores[v]<0 for v in tops):raise ValueError('no complete precision-feasible boundary')
    optimum=sum(int(scores[v]) for v in tops)
    pool.charge('c64_direct_dense_boundary_output',3*n_cont+8*len(columns)+4*edges+1024)
    selected=np.zeros(n_cont,bool);roots=np.arange(n_cont,dtype=np.int64);weights=[(1,0)]*n_cont
    stack=[(v,parents[v]) for v in tops]
    while stack:
        v,a=stack.pop();direct=a==parents[v]
        deleted=bool(choices[v]) if direct else extra_choices[v][a]
        if deleted:
            selected[v]=True;roots[v]=a;weights[v]=local[v] if direct else extras[v][a]
        stack.extend((t,int(roots[v])) for t in children.get(v,()))
    if int(selected.sum())!=optimum:raise ValueError('complete boundary reconstruction differs')
    return selected,roots,weights,dict(optimum=optimum,raw_nodes=len(columns),ancestor_states=states_total,
        extra_ancestor_states=sum(map(len,extras.values())),leaf_states=leaf_states,nonleaf_states=nonleaf_states)

