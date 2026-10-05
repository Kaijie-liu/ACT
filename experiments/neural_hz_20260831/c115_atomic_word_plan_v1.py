# SPDX-License-Identifier: AGPL-3.0-or-later
"""Source-first atomic V/M dyadic plan; no old HZ or native CSR is constructed."""
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from experiments.neural_hz_20260831.c113_prepared_tile_v1 import (
    prepare_tile as prepare_routes, _emit_row as realize_row)
from experiments.neural_hz_20260831.c88_inline_tile_v1 import NativeUnproved


def _plan_row(terms,pivot,power=None,*,pool):
    raw_count = len(terms)+1
    merged = {}
    for col, number, exponent in terms:
        if not number:
            continue
        col, number, exponent = int(col), int(number), int(exponent)
        if col in merged:
            prior, old_power = merged[col]
            common = min(old_power, exponent)
            number = (prior << (old_power-common))+(number << (exponent-common))
            exponent = common
        merged[col] = (number, exponent)
    merged = {c: v for c, v in merged.items() if v[0]}
    if pivot in merged:
        raise ValueError('original or fresh pivot occurs in its own defining form')
    if power is None:
        power = max(0, max((abs(n).bit_length()+e for n, e in merged.values()), default=0)
                    +(max(1, len(merged))-1).bit_length())
    if not 0 <= power <= 1023:
        raise NativeUnproved('normalizing pivot outside original exponent domain')
    merged[int(pivot)] = (1, int(power))
    cols = sorted(merged)
    final_count = len(cols)
    if final_count > raw_count:
        raise ValueError('coalescence increased coefficient population')
    pool.charge('c96_no_credit_for_coalesced_away_guards', 8*(raw_count-final_count))
    if any(abs(merged[c][0]) > (1 << 62) or not -4096 <= merged[c][1] <= 4096 for c in cols):
        raise NativeUnproved('complete exact integer row outside bounded native-word domain')
    return dict(columns=tuple(cols), words=tuple(merged[c][0] for c in cols),
                powers=tuple(merged[c][1] for c in cols), pivot=int(pivot),
                pivot_power=int(power), rhs=0.)


@dataclass(slots=True)
class PreparedAtomic:
    state: dict | None
    pool: object
    source_n_cont: int
    inventory: dict


def reduce_rows(rows,aux_count,m_slots,source_base,reused_v,*,pool):
    """One prospective transaction; total strictness is checked before commit."""
    pool.charge('c115_complete_group_incidence',8*sum(len(r['columns']) for r in rows))
    pool.charge('c115_complete_group_inventory',64*len(m_slots))
    index={r['pivot']:i for i,r in enumerate(rows[:aux_count])}
    if len(index)!=aux_count or set(m_slots)-set(index):
        raise ValueError('complete unique symbolic defining inventory required')
    incidence=defaultdict(list)
    for i,r in enumerate(rows):
        for col in r['columns']:
            if col in m_slots and col!=r['pivot']:incidence[col].append(i)
    groups=defaultdict(list)
    for slot in sorted(m_slots):
        consumers=tuple(incidence[slot])
        if not consumers or any(i<aux_count for i in consumers):
            continue
        groups[consumers].append(slot)
    proposed=list(rows);removed=set();decisions=[]
    initial_nnz=sum(len(r['columns']) for r in rows)
    for consumers,slots in sorted(groups.items()):
        selected=set(slots);upper=0
        old_count=sum(len(proposed[index[s]]['columns']) for s in slots)
        old_count+=sum(len(proposed[i]['columns']) for i in consumers)
        for i in consumers:
            r=proposed[i]
            count=sum(len(proposed[index[c]]['columns'])-1 if c in selected else 1 for c in r['columns'])
            pool.charge('c115_complete_union_support_bound',8*count)
            union=set()
            for col in r['columns']:
                if col in selected:
                    union.update(c for c in proposed[index[col]]['columns'] if c!=col)
                else:union.add(col)
            upper+=len(union)
        decision=dict(slots=list(slots),consumers=list(consumers),old_local_nnz=old_count,
            complete_new_nnz_upper=upper,prospective_nonincrease=False)
        if upper>old_count:
            decision['reason']='complete_union_upper_not_nonincreasing';decisions.append(decision);continue
        rewritten=[]
        for i in consumers:
            r=proposed[i];terms=[]
            expanded=sum(len(proposed[index[c]]['columns'])-1 if c in selected else 1 for c in r['columns'])-1
            pool.charge('c96_exact_row_prefix_and_numeric',64+8*(expanded+1))
            for col,number,power in zip(r['columns'],r['words'],r['powers'],strict=True):
                if col==r['pivot']:continue
                if col not in selected:
                    terms.append((col,number,power));continue
                child=proposed[index[col]]
                for parent,value,exponent in zip(child['columns'],child['words'],child['powers'],strict=True):
                    if parent!=col:
                        terms.append((parent,-number*value,exponent+power-child['pivot_power']))
            rewritten.append(_plan_row(terms,r['pivot'],r['pivot_power'],pool=pool))
        count=sum(len(r['columns']) for r in rewritten)
        if count>upper:raise ValueError('exact support exceeded complete union upper')
        for i,r in zip(consumers,rewritten,strict=True):proposed[i]=r
        removed.update(slots)
        decision.update(prospective_nonincrease=True,reason='prospective_nonincreasing_exact_substitution',
            actual_new_local_nnz=count,nnz_delta=count-old_count)
        decisions.append(decision)
    final=[r for r in proposed[:aux_count] if r['pivot'] not in removed]+proposed[aux_count:]
    final_nnz=sum(len(r['columns']) for r in final)
    reference_lower=initial_nnz+2*reused_v
    strict=final_nnz<reference_lower
    if not strict:
        final=rows;removed=set();final_nnz=initial_nnz
    remaining={r['pivot'] for r in final[:aux_count-len(removed)]}
    if any(c>=source_base and c not in remaining for r in final for c in r['columns']):
        raise ValueError('unresolved planned auxiliary after atomic decision')
    return dict(rows=final,new_factors=aux_count-len(removed),source_base=source_base,
        removed_m=sorted(removed),decisions=decisions,initial_quotient_nnz=initial_nnz,
        final_nnz=final_nnz,original_C96_nnz_lower=reference_lower,strict_atomic_nnz_proved=strict,
        all_M_changes_rolled_back=not strict,original_rows_not_mutated=True,
        exact_inverse_is_original_source_projection=True,original_HZ_never_constructed=True)


def prepare_tile(transformed,parent_ids,parent_powers,output_ids,output_powers,
                 base_n_cont,*,pool,enabled=False):
    if not enabled:return None
    prepared=prepare_routes(transformed,parent_ids,parent_powers,output_ids,output_powers,
        base_n_cont,pool=pool,enabled=True)
    state=prepared.state;prepared.state=None
    numbers=state['numbers'];kernel_powers=state['kernel_powers'];kcount=state['kcount']
    forms=state['forms'];users=state['users'];channels=state['channels'];keep_m=state['keep_m']
    v_uses=state['v_uses'];keep_v=state['keep_v'];representatives=state['representatives']
    sharing=state['sharing'];reused_v=state['reused_v'];outputs=state['outputs']
    outpowers=state['outpowers'];omap=state['omap']
    rows,vmap,mmap=[],{},{}
    representative_maps = {}
    for c, t in representatives:
        pool.charge('c96_exact_row_prefix_and_numeric', 64+8*(len(forms[c, t])+1))
        slot = base_n_cont+len(rows)
        row = _plan_row([(col, -sign, exp) for col, sign, exp in forms[c, t]], slot, pool=pool)
        representative_maps[c, t] = (slot, row['pivot_power'])
        rows.append(row)
    for key, (representative, sign, shift) in sharing.items():
        slot, unit = representative_maps[representative]
        vmap[key] = (slot, unit+shift, sign)

    def vterms(c, t, number, power):
        if (c, t) in vmap:
            slot, unit, sign = vmap[c, t]
            if (c, t) in reused_v:
                pool.charge('c111_every_alias_consumer_route', 8)
                number = int(number)*sign
            return [(slot, int(number), int(power)+unit)]
        return [(col, int(number)*sign, int(power)+exp) for col, sign, exp in forms[c, t]]

    for k, t in sorted(keep_m):
        size = sum(1 if (c, t) in vmap else len(forms[c, t]) for c in channels[k, t])
        pool.charge('c96_exact_row_prefix_and_numeric', 64+8*(size+1))
        terms = [term for c in channels[k, t]
                 for term in vterms(c, t, -int(numbers[k, c, t]), int(kernel_powers[k, c]))]
        slot = base_n_cont+len(rows)
        row = _plan_row(terms, slot, pool=pool)
        mmap[k, t] = (slot, row['pivot_power'])
        rows.append(row)
    aux_count = len(rows)
    for k in range(kcount):
        for s in range(4):
            if outputs[k, s] < 0:
                continue
            used = [t for t in range(16) if omap[s, t] and (k, t) in channels]
            size = sum(1 if (k, t) in mmap else sum(1 if (c, t) in vmap else len(forms[c, t])
                       for c in channels[k, t]) for t in used)
            pool.charge('c96_exact_row_prefix_and_numeric', 64+8*(size+1))
            terms = []
            for t in used:
                sign = -int(omap[s, t])
                if (k, t) in mmap:
                    slot, unit = mmap[k, t]
                    terms.append((slot, sign, unit))
                else:
                    for c in channels[k, t]:
                        terms.extend(vterms(c, t, sign*int(numbers[k, c, t]), int(kernel_powers[k, c])))
            rows.append(_plan_row(terms, int(outputs[k, s]), int(outpowers[k, s]), pool=pool))

    atomic=reduce_rows(rows,aux_count,{slot for slot,unit in mmap.values()},
        base_n_cont,len(reused_v),pool=pool)
    atomic['source_report']=dict(kept_v=len(representatives),kept_m=len(keep_m)-len(atomic['removed_m']),
        original_retained_v=len(keep_v),reused_v=len(reused_v),original_kept_m=len(keep_m),
        original_used_v=len(v_uses),original_used_m=len(channels),
        inlined_v=len(v_uses)-len(keep_v),inlined_m=len(channels)-len(keep_m)+len(atomic['removed_m']))
    inventory=dict(prepared.inventory,new_factors=atomic['new_factors'],
        removed_m=len(atomic['removed_m']),final_nnz=atomic['final_nnz'],
        strict_atomic_nnz_proved=atomic['strict_atomic_nnz_proved'])
    return PreparedAtomic(atomic,pool,int(base_n_cont),inventory)


def emit_tile(prepared,base_n_cont,*,pool,enabled=False):
    if not enabled:return None
    if (type(prepared) is not PreparedAtomic or prepared.state is None or prepared.pool is not pool
            or base_n_cont<prepared.source_n_cont):
        raise ValueError('same live atomic preparation and pool required')
    state=prepared.state;prepared.state=None
    rows=state['rows'];n=state['new_factors'];source=state['source_base']
    pool.charge('c115_final_compact_name_emission',8*state['final_nnz']+64*len(rows))
    names={r['pivot']:base_n_cont+i for i,r in enumerate(rows[:n])}
    def rename(col):
        return names[col] if col>=source else col
    native=[]
    for r in rows:
        terms=[(rename(c),v,e) for c,v,e in zip(r['columns'],r['words'],r['powers'],strict=True)
               if c!=r['pivot']]
        native.append(realize_row(terms,rename(r['pivot']),r['pivot_power'],pool=pool))
    sizes=np.array([len(r['columns']) for r in native],np.int64)
    ptr=np.r_[0,np.cumsum(sizes)].astype(np.int64)
    packet=dict(indptr=ptr,columns=np.concatenate([r['columns'] for r in native]) if native else np.empty(0,np.int32),
        words=np.concatenate([r['words'] for r in native]) if native else np.empty(0,np.int64),
        powers=np.concatenate([r['powers'] for r in native]) if native else np.empty(0,np.int32),
        native=np.concatenate([r['native'] for r in native]) if native else np.empty(0,np.float64),
        pivots=np.array([r['pivot'] for r in native],np.int32),
        pivot_powers=np.array([r['pivot_power'] for r in native],np.int32),
        gauges=np.array([r['gauge'] for r in native],np.int32),rhs=np.zeros(len(native),np.float64))
    if int(ptr[-1])!=state['final_nnz'] or n!=prepared.inventory['new_factors']:
        raise ValueError('emitted complete native inventory differs from prepared plan')
    report=dict(state['source_report'],native_coefficients_pass=True,new_factors=n,
        removed_m=len(state['removed_m']),initial_quotient_nnz=state['initial_quotient_nnz'],
        original_C96_nnz_lower=state['original_C96_nnz_lower'],
        strict_atomic_nnz_proved=state['strict_atomic_nnz_proved'],
        all_M_changes_rolled_back=state['all_M_changes_rolled_back'],
        complete_group_decisions=state['decisions'],rows=len(native),nnz=int(ptr[-1]),base_n_cont=base_n_cont,
        no_old_native_CSR_or_HZ_created=True,local_names_are_not_combined_HZ=True,
        all_factor_counts_known_before_emission=True,global_auxiliary_gate_proved=False,
        complete_physical_reduction_proved=False,source_runtime_LIVE_admitted=False,formal_gain=0)
    return report,packet


def construct(transformed,parent_ids,parent_powers,output_ids,output_powers,base_n_cont,*,pool,enabled=False):
    if not enabled:return None
    ready=prepare_tile(transformed,parent_ids,parent_powers,output_ids,output_powers,base_n_cont,pool=pool,enabled=True)
    return emit_tile(ready,base_n_cont,pool=pool,enabled=True)
