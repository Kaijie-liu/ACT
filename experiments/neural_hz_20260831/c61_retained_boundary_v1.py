"""Default-off exact original-factor forest optimization; diagnostic only."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import numpy as np
from experiments.neural_hz_20260831.c59_full_consumer_census_v2 import decode_frame,Products,gauge,interval
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import word,native_word,fraction,in_window
from experiments.neural_hz_20260831.c10_alias_quotient_v1 import frontier

ONE=(1,0)
BAD=-10**9


def precise(value):
    return abs(value[0]).bit_length()<=53


def complete_interval(lo,hi,binary,rhs):
    for value in binary:
        l,h=interval(native_word(value));lo=max(lo,l);hi=min(hi,h)
    if rhs:
        l,h=interval(native_word(rhs));lo=max(lo,l);hi=min(hi,h)
    return lo,hi,None if lo>hi else min(max(0,lo),hi)


def optimize(parents,states,keep_ok,remove_ok,pool):
    """Exact maximum removal in a forest; state keys are proper ancestors."""
    count=len(parents)
    pool.charge('c61_complete_forest_topology',64*count+count*max(1,count.bit_length())+1024)
    columns=sorted(parents);children={v:[] for v in columns};tops=[]
    for v in columns:
        p=parents[v]
        if not 0<=p<v:raise ValueError('original topological parent required')
        if p in parents:children[p].append(v)
        else:tops.append(v)
        expected=[p,*states[p]] if p in parents else [p]
        if list(states[v])!=expected:raise ValueError('incomplete original ancestor state set')
    scores={};choices={}
    for v in reversed(columns):
        scores[v]={};choices[v]={}
        for a in states[v]:
            pool.charge('c61_exact_forest_DP_state',64+16*len(children[v]))
            kept=[scores[c][v] for c in children[v]]
            removed=[scores[c][a] for c in children[v]]
            k=sum(kept) if keep_ok[v][a] and all(x>=0 for x in kept) else BAD
            r=1+sum(removed) if remove_ok[v][a] and all(x>=0 for x in removed) else BAD
            scores[v][a]=max(k,r);choices[v][a]=r>k
    total=sum(scores[v][parents[v]] for v in tops)
    if any(scores[v][parents[v]]<0 for v in tops):raise ValueError('no precision-feasible original boundary representation')
    selected={};anchors={};stack=[(v,parents[v]) for v in tops]
    while stack:
        v,a=stack.pop();pool.charge('c61_complete_boundary_selection',48)
        selected[v]=choices[v][a];anchors[v]=a if selected[v] else v
        stack.extend((c,anchors[v]) for c in children[v])
    if sum(selected.values())!=total:raise ValueError('DP selection and optimum disagree')
    return total,selected,anchors,scores


def assess(saved,packet,*,pool,enabled=False,observe=None):
    if not enabled:return None
    hz=saved['hz'];old=int(saved['old_n_cont']);logical=int(saved['logical_n_cont']);old_eq=int(saved['old_n_eq'])
    if packet['n_cont']!=hz.n_cont or packet['n_bin']!=hz.n_bin or not hz.exact:raise ValueError('original HZ frame differs')
    roots,weights,inverse_proof=decode_frame(packet,pool)
    columns=np.flatnonzero(roots!=np.arange(hz.n_cont));chosen=roots!=np.arange(hz.n_cont)
    if np.any(columns<old) or np.any(columns>=logical):raise ValueError('old/radix coordinate cannot be removed')
    pool.charge('c59_complete_defining_row_binding',160*len(columns)+8*hz.n_cont+hz.n_eq+1024)
    own=np.full(hz.n_eq,-1,np.int64);parents={};local={};pivots={};states={};keep_ok={};remove_ok={};legacy={}
    products=Products(pool);defining_gauges=Counter();defining_bad=0;state_count=0;digest=hashlib.sha256()
    for raw in columns:
        v=int(raw);row=int(saved['eq_roots'][old_eq+v-old]);a,b=map(int,hz.Ac.indptr[row:row+2])
        cc=hz.Ac.indices[a:b];cv=hz.Ac.data[a:b]
        if (len(cc)!=2 or int(cc[-1])!=v or not 0<=int(cc[0])<v or own[row]>=0
                or hz.Ab.indptr[row]!=hz.Ab.indptr[row+1] or hz.b[row]!=0):
            raise ValueError('unproved original homogeneous defining row')
        parent=int(cc[0]);pivot=native_word(cv[-1]);left=native_word(cv[0])
        if pivot[0]!=1 or roots[v]!=roots[parent]:raise ValueError('positive dyadic pivot/shared original root required')
        ratio=word(-left[0],left[1]-pivot[1])
        if not 0<abs(fraction(ratio))<=1 or not precise(ratio):raise ValueError('native local ratio and redundant box required')
        if products(ratio,weights[parent])!=weights[v]:raise ValueError('original inverse and local edge differ')
        pool.charge('c61_complete_ancestor_state_and_envelope',64+32*(1+len(states.get(parent,{}))))
        own[row]=v;parents[v]=parent;local[v]=ratio;pivots[v]=pivot;legacy[v]=abs(fraction(ratio))>=F(1,2**60)
        states[v]={parent:ratio}
        if parent in parents:
            for ancestor,value in states[parent].items():states[v][ancestor]=products(ratio,value)
        state_count+=len(states[v]);keep_ok[v]={a:precise(r) for a,r in states[v].items()}
        remove_ok[v]={a:True for a in states[v]}
        # A retained defining row's coefficient is -pivot*r(v,a); scaling is exact.
        coefficients=[pivot,*[word(r[0],r[1]+pivot[1]) for r in states[v].values() if precise(r)]]
        low,high,q=gauge(coefficients);defining_bad+=q is None
        if q is not None:defining_gauges[str(q)]+=1
        if parent in parents:
            # This is the old all-use test on the parent's local substitution.
            grandparent=parents[parent];r=states[v][grandparent]
            value=word(r[0],r[1]+pivot[1])
            legacy[parent]&=precise(value) and in_window(value)
        digest.update(json.dumps([v,parent,row,ratio,list(states[v].items()),low,high,q]).encode())
    if observe:observe(dict(event='complete_original_forest_bound',nodes=len(columns),ancestor_states=state_count,
        defining_envelope_failures=defining_bad,original_inverse_equations=inverse_proof))
    external_counts=Counter();shifts=Counter();matrix_counts={};external_bad=0;support_conflicts=0
    for name,bname,rhs,predicate in [('Ac','Ab',hz.b,True),('Auc','Aub',hz.ub,True),('Gc','Gb',hz.c,False)]:
        matrix=getattr(hz,name);binary=getattr(hz,bname);nr=matrix.shape[0]
        if rhs.shape!=(nr,) or binary.shape!=(nr,hz.n_bin):raise ValueError('complete RHS/binary layout required')
        pool.charge('c59_full_consumer_incidence_scan',3*matrix.nnz+8*nr+1024)
        hits=chosen[matrix.indices];prefix=np.empty(matrix.nnz+1,np.int64);prefix[0]=0;np.cumsum(hits,out=prefix[1:])
        affected=(prefix[matrix.indptr[1:]]-prefix[matrix.indptr[:-1]])>0
        if name=='Ac':affected[own>=0]=False
        if not predicate and affected.any():raise ValueError('output-live original candidate')
        rows=np.flatnonzero(affected);del hits,prefix,affected
        before=dict(external_counts)
        for row in rows:
            a,b=map(int,matrix.indptr[row:row+2]);ba,bb=map(int,binary.indptr[row:row+2])
            cols=matrix.indices[a:b];vals=matrix.data[a:b]
            pool.charge('c61_complete_external_row_envelope',64+16*len(cols)+8*(bb-ba))
            mapped=roots[cols]
            if len(set(map(int,mapped)))!=len(cols):support_conflicts+=1
            lo,hi=-10**9,10**9
            for col,value in zip(cols,vals):
                col=int(col);base=native_word(value);l,h=interval(base);lo=max(lo,l);hi=min(hi,h)
                if not chosen[col]:continue
                external_counts['original_candidate_occurrences']+=1
                for ancestor,ratio in states[col].items():
                    result=products(base,ratio);ok=precise(result)
                    external_counts['ancestor_product_occurrences']+=1
                    external_counts['inexact_product_occurrences']+=not ok
                    remove_ok[col][ancestor]&=ok
                    if ancestor==parents[col]:legacy[col]&=ok and in_window(result)
                    if ok:
                        l,h=interval(result);lo=max(lo,l);hi=min(hi,h)
            lo,hi,q=complete_interval(lo,hi,binary.data[ba:bb],rhs[row]);external_bad+=q is None
            if q is not None:shifts[str(q)]+=1
            external_counts['rows']+=1
            digest.update(json.dumps([name,int(row),lo,hi,q]).encode())
        matrix_counts[name]={k:v-before.get(k,0) for k,v in external_counts.items()}
        if observe:observe(dict(event='complete_ancestor_consumer_matrix',matrix=name,counts=matrix_counts[name],
            envelope_failures=external_bad,support_conflicts=support_conflicts))
    if support_conflicts:raise ValueError('full-root support coalescence outside this forest-only rule')
    optimum,selected,anchors,scores=optimize(parents,states,keep_ok,remove_ok,pool)
    pool.charge('c61_complete_legacy_comparator_and_summary',128*len(columns)+64*state_count+4096)
    eligible=np.array([legacy[int(v)] for v in columns],bool)
    old_selected=frontier(dict(column=columns,parent=np.array([parents[int(v)] for v in columns]),
        all_products_exact=eligible,products_window_safe=eligible),hz.n_cont)
    old_set=set(map(int,columns[old_selected]));new_set={v for v in selected if selected[v]}
    for v in sorted(parents):
        digest.update(json.dumps([v,list(keep_ok[v].items()),list(remove_ok[v].items()),list(scores[v].items()),selected[v],anchors[v]]).encode())
    bins=Counter(('power_two' if abs(local[v][0])==1 else 'general') for v in new_set)
    added=new_set-old_set;lost=old_set-new_set;exchange=Counter()
    for label,items in [('new_not_old',added),('old_not_new',lost)]:
        for v in items:exchange[label+('_power_two' if abs(local[v][0])==1 else '_general')]+=1
    original=int(hz.Ac.nnz+hz.Ab.nnz+hz.Auc.nnz+hz.Aub.nnz)
    exact=not(external_bad or defining_bad)
    return dict(schema='c61_complete_precision_optimal_original_boundaries_v1',complete=True,
        raw_nodes=len(columns),ancestor_states=state_count,original_coordinates=hz.n_cont,
        local_general=sum(abs(r[0])!=1 for r in local.values()),precision_optimal_removed=optimum,
        native_symbolic_optimality_proved=exact,universal_external_envelope_failures=external_bad,
        universal_defining_envelope_failures=defining_bad,universal_external_shift_histogram=dict(shifts),
        universal_defining_shift_histogram=dict(defining_gauges),full_root_support_conflicts=support_conflicts,
        complete_matrix_counts=matrix_counts,external_counts=dict(external_counts),selected_classes=dict(bins),
        boundary_exchange_vs_old=dict(exchange),additional_removed_vs_old=optimum-len(old_selected),
        legacy_local=len(columns),legacy_eligible=int(eligible.sum()),legacy_selected=len(old_selected),
        legacy_incidence_hits=int(external_counts['original_candidate_occurrences']+sum(p in parents for p in parents.values())),
        original_predicate_nnz=original,symbolic_predicate_nnz=original-2*optimum,
        legacy_predicate_nnz=original-2*len(old_selected),new_auxiliary_variables=0,
        exact_product_lookups=products.lookups,exact_product_unique_pairs=len(products.values),exact_product_reuses=products.hits,
        complete_boundary_profile_sha256=digest.hexdigest(),inverse_equations_checked=inverse_proof,
        complete_original_input_source_binding_inherited=True,original_local_edge_inverse_implementation_proved=False,
        physical_HZ_native_or_source_first_proved=False,formal_gain=0)
