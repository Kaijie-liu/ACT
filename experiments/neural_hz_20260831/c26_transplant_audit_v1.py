"""Complete source-derived plans and independent tagged-UID transplant checks."""

from fractions import Fraction as F
import math
import numpy as np
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import Plan
from experiments.neural_hz_20260831.c22_uid_runs_v1 import row_for_uid
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import unique_other
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import incidence_oracle
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def row(m,r):
    a,b=map(int,m.indptr[r:r+2])
    return m.indices[a:b],m.data[a:b]


def plans_from_checked_incidence(closed,post,columns,words,eq,le,*,pool):
    """words/columns must come from the COMPLETE independent C23 phase proof.

Not an input API for unproved packed words, a sealed column whitelist or a
live production selector. The worker retains and charges the complete proof.
"""
    pool.charge('transplant_diagnostic_UID_lookup',8*(len(eq)+len(le)))
    lookup={int(u):(False,i) for i,u in enumerate(eq)}
    lookup.update({int(u):(True,i) for i,u in enumerate(le)})
    plans=[]
    for raw in columns:
        pool.charge('transplant_source_plan_scalar_checks',96)
        col=int(raw); d=int(closed.eq_roots[closed.old_n_eq+col-closed.old_n_cont])
        pc,pv=row(post.Ac,d); bc,bv=row(post.Ab,d)
        u=int(eq[d]); v=unique_other(int(words[col-closed.old_n_cont]),u)
        if v not in lookup: raise ValueError('checked unique consumer does not resolve')
        kind,c=lookup[v]; tc,tv=row(post.Auc if kind else post.Ac,c)
        if not len(pc) or not len(tc) or int(pc[-1])!=col or int(tc[0])!=col or len(bc):
            raise ValueError('pair is not a binary-free ordered MAIN definition')
        pivot=float(pv[-1]); sign=-1 if tv[0]>0 else 1
        if pivot<=0 or math.frexp(pivot)[0]!=.5 or float(tv[0])!=-sign*pivot:
            raise ValueError('pair lost its exact unit pivot')
        pool.charge('transplant_source_tail_read',4*(len(tc)-1))
        main=row_for_uid(closed.uid_slabs,v,pool=pool)
        main=None if main is None else closed.old_n_cont+main
        plans.append(Plan(col,d,c,kind,u,v,pivot,sign,float(post.b[d]),main,tuple(map(int,tc[1:]))))
    return plans


def verify(closed,post,original_overlay,columns,eq,le,draft,oracle,*,pool,expected_oracle_fingerprint):
    """Oracle is the independently authenticated completed C15 component.

Only its unchanged output map is used to BRIDGE the original whole-HZ hash;
its predicate matrices are the independently proved exact spliced reference.
No completed HZ is input to the metadata compiler or future fresh generator.
"""
    closed.validate(); draft.validate(); oracle.validate()
    if oracle.fingerprint()!=expected_oracle_fingerprint:
        raise ValueError('independently anchored complete oracle content changed, including resealing')
    bridge=SparseHZono(oracle.hz.c,oracle.hz.Gc,oracle.hz.Gb,post.Ac,post.Ab,post.b,
        post.Auc,post.Aub,post.ub,frame_id=post.frame_id,exact=post.exact)
    if source_digest(bridge)!=oracle.original_digest:
        raise ValueError('actual post predicates do not match independently proved C15 source')
    if not np.array_equal(columns,oracle.columns) or not np.array_equal(draft.columns,columns):
        raise ValueError('incomplete source-discovered unit population')
    pool.charge('independent_transplant_full_lineage_validation',8*len(closed.eq_roots))
    pool.charge('independent_transplant_row_UID_tables',8*(len(eq)+len(le)))
    lookup={int(u):(False,i) for i,u in enumerate(eq)}
    lookup.update({int(u):(True,i) for i,u in enumerate(le)})
    definitions=[int(closed.eq_roots[closed.old_n_eq+int(col)-closed.old_n_cont]) for col in columns]
    keep=np.ones(post.n_eq,bool); keep[definitions]=False
    eq_new=eq[keep].copy(); le_new=le.copy(); row_map=np.cumsum(keep,dtype=np.int64)-1
    by_slot={}; redirects={}; retired={}; parent_terms=tail_terms=0; phase_consumers=old_consumers=0
    for (col,newrow,pivot,offset,sign),d in zip(oracle.decoded(),definitions):
        # C15 decoder target has negative encoding for inequalities.
        kind=newrow<0; newrow=-newrow-1 if kind else newrow
        slot=closed.old_n_eq+col-closed.old_n_cont; raw=int(draft.eq_roots[slot])
        allowed=(1<<62)|((1<<48)-1)
        if raw<1<<62 or raw&~allowed or (raw>>28)&((1<<20)-1)!=d:
            raise ValueError('independent producer/tag decode mismatch')
        oldrow=raw&((1<<20)-1)
        if (bool(raw&(1<<20))!=kind or math.ldexp(1.,((raw>>21)&63)-20)!=pivot
                or (-1 if raw&(1<<27) else 1)!=sign
                or float(draft.eq_scales.view(np.float64)[slot])!=offset
                or (oldrow if kind else int(row_map[oldrow]))!=newrow):
            raise ValueError('tagged target/pivot/sign/RHS differs from independent exact certificate')
        pc,pv=row(post.Ac,d); tc,tv=row(post.Auc if kind else post.Ac,oldrow)
        nc,nv=row(oracle.hz.Auc if kind else oracle.hz.Ac,newrow)
        pool.charge('independent_transplant_all_parent_prefix_and_tail',4*(len(pc)+len(tc)))
        if (int(pc[-1])!=col or int(tc[0])!=col or len(nc)!=len(pc)+len(tc)-2
                or not np.array_equal(nc[:len(pc)-1],pc[:-1])
                or not np.array_equal(nv[:len(pc)-1],sign*pv[:-1])
                or not np.array_equal(nc[len(pc)-1:],tc[1:])
                or not np.array_equal(nv[len(pc)-1:],tv[1:])):
            raise ValueError('full surviving row cannot reconstruct the exact producer')
        old_rhs=(post.ub if kind else post.b)[oldrow]; new_rhs=(oracle.hz.ub if kind else oracle.hz.b)[newrow]
        if F(float(new_rhs))!=F(float(old_rhs))+sign*F(offset): raise ValueError('exact RHS mismatch')
        u=int(eq[d]); v=int((le if kind else eq)[oldrow]); retired[v]=u
        (le_new if kind else eq_new)[newrow]=u
        by_slot[slot]=(d,oldrow,kind,pivot,offset,sign)
        main=row_for_uid(closed.uid_slabs,v,pool=pool)
        if main is not None: redirects[closed.old_n_eq+main]=u
        parent_terms+=len(pc)-1; tail_terms+=len(tc)-1
        is_phase=oldrow>=(closed.hz.n_ineq if kind else closed.hz.n_eq)
        phase_consumers+=is_phase; old_consumers+=not is_phase
    for i,(r,s) in enumerate(zip(draft.eq_roots,draft.eq_scales)):
        if i in by_slot: continue
        expected=((1<<61)|redirects[i]) if i in redirects else int(closed.eq_roots[i])
        if int(r)!=expected or s!=closed.eq_scales[i]: raise ValueError('untouched lineage or consumer redirect changed')
    # Check EVERY old physical UID, not selected examples, against the sparse
    # retirement reader; no UID table is accepted as a runtime selector.
    for uid in lookup:
        if draft.retired_to(uid,pool=pool)!=retired.get(uid): raise ValueError('retired UID index is incomplete')
    for oldrow in range(post.n_eq):
        if draft.eq_row(oldrow,pool=pool)!=(int(row_map[oldrow]) if keep[oldrow] else None):
            raise ValueError('all-physical-row rank deletion differs')
    if len(set(map(int,np.r_[eq_new,le_new])))!=len(eq_new)+len(le_new):
        raise ValueError('transferred physical UID reused')
    actual=incidence_oracle(oracle.hz,eq_new,le_new,closed.old_n_cont,closed.logical_n_cont,pool=pool)
    for i,word in enumerate(draft.iter_words(original_overlay,pool=pool)):
        if word!=int(actual[i]): raise ValueError('transferred ownership differs from full new sparse incidence')
    return {'all_original_physical_UIDs_checked':len(lookup),'all_original_EQ_row_queries_checked':post.n_eq,
        'all_transferred_MAIN_words_checked':len(actual),'all_tagged_unit_pairs_checked':len(columns),
        'all_lineage_slots_checked':len(draft.eq_roots),'all_definition_parent_terms_checked':parent_terms,
        'all_consumer_tail_terms_checked':tail_terms,'new_phase_consumers':phase_consumers,
        'preexisting_consumers':old_consumers,'redirected_MAIN_consumers':len(redirects),
        'complete_transferred_incidence_equal':True,'independent_exact_source_and_box_proof_transferred':True,
        'all_binaries_and_global_frame_retained_in_reference':True,'new_HZ_or_native_consumer_constructed':False,
        'formal_gain':0}
