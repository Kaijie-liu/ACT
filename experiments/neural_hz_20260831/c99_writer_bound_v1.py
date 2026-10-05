"""Full unchanged C30 writer charge inventory from complete discovered plans."""


def writer_bound(view,plans,*,pool,enabled=False):
    if not enabled:return None
    n=len(plans);pool.charge('c99_full_writer_bound_headers',128+64*n)
    if not n:raise ValueError('strict native splice population required')
    tails=sum(len(p.tail) for p in plans)
    neg=sum(p.sign==-1 for p in plans)
    prefixes=sum(int(view.pre.Ac.indptr[p.definition+1]-view.pre.Ac.indptr[p.definition])-1
                 for p in plans if p.sign==-1)
    neq=sum(not p.inequality for p in plans);nle=n-neq
    schedule=0
    for edits in (n+neq,n,nle,0,n+neq,nle):
        schedule+=64*(edits+2)+4*edits*max(1,(edits-1).bit_length())
    rows=2*(view.n_eq-n+1)+2*(view.n_ineq+1)
    dispatcher=4*(22+n+neg)
    incremental=384+192*n+4*tails+schedule+4*rows+512+prefixes+dispatcher
    original=sum(int(m.nnz) for k in ('Ac','Ab','Auc','Aub') for m in view.blocks(k))
    native=2*(original-2*n)+view.n_eq-n+view.n_ineq
    return dict(incremental_upper=incremental,native_payload_upper=native,
        dispatcher_upper=dispatcher,all_plans=n,all_tail_terms=tails,
        actual_negative_parent_prefixes=prefixes,strict_predicate_nnz=original-2*n,
        source_predicate_nnz=original,tariffs_and_native_boundary_unchanged=True)
