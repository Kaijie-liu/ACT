"""Full C64 source model with the actually changed C65 operation groups."""
from collections import Counter
import numpy as np
from experiments.neural_hz_20260831.c64_source_budget_v1 import bound as old_bound


def bound(saved,legacy,reference,*,pool,enabled=False):
    if not enabled:return None
    result=old_bound(saved,legacy,reference,pool=pool,enabled=True)
    hz=saved['hz'];old=saved['old_n_cont'];oe=saved['old_n_eq'];parents=reference['parents']
    pool.charge('c65_complete_owned_normal_event_partition',8*(hz.n_eq+len(parents))+1024)
    own=np.zeros(hz.n_eq,bool);own[[int(saved['eq_roots'][oe+v-old]) for v in parents]]=True
    dyadic=np.zeros(hz.n_eq,bool)
    for node in saved['definition_graph']:
        counts=Counter(node['parents'])
        if node['kind']=='sum' and all(n>0 and n&(n-1)==0 for n in counts.values()):
            slots=node['slots'][node['needed']];dyadic[saved['eq_roots'][oe+slots-old]]=True
    # Extra radix rows conservatively use the general reader, as in C64's bound.
    events=int(np.count_nonzero(reference['raw_hits']['Ac']&~own&~dyadic))
    hits=result['external_occurrences']-result['dyadic_source_occurrences']
    removed=6*hits;added=32+8*events
    parts=dict(result['work_parts'])
    original=parts.pop('c62_complete_external_maximum',0)
    if original<removed:raise ValueError('invalid complete original normal-statistic work')
    if original:parts['c65_owned_normal_external_maximum']=original-removed
    parts['c65_fresh_normal_producer_binding']=32
    if events:parts['c65_owned_normal_domain_read']=8*events
    extra=sum(parts.values())
    if extra!=result['coupled_extra_upper']-removed+added:raise ValueError('complete C65 bound does not reconcile')
    whole=result['whole_base_work']+extra;branch=result['branch_base_work']+extra
    result.update(schema='c65_complete_source_construction_bound_v1',work_parts=parts,
        inherited_C64_whole_bound=result['whole_work_upper'],actual_removed_exponent_domain_operations=removed,
        new_bound_producer_operations=added,owned_normal_reader_events=events,owned_normal_occurrences=hits,
        coupled_extra_upper=extra,whole_work_upper=whole,branch_work_upper=branch,
        whole_headroom=256_000_000-whole,branch_headroom=200_000_000-branch,
        work_caps_fit=whole<=256_000_000 and branch<=200_000_000)
    return result
