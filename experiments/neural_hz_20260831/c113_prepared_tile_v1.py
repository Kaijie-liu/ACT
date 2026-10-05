# SPDX-License-Identifier: AGPL-3.0-or-later
"""Owned once-prepared exact V quotient and single-consumption row emission."""
from dataclasses import dataclass
from experiments.neural_hz_20260831.c112_factored_key_v1 import routes
import numpy as np
from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import T, R
from experiments.neural_hz_20260831.c7_factored_hz_v1 import scaled_exact
from experiments.neural_hz_20260831.c88_inline_tile_v1 import Prepared, prepare, NativeUnproved, actual_rows


def _row(terms, pivot, power=None, *, pool):
    """Standalone already-existing terms; construction prepays before forming them."""
    pool.charge('c96_exact_row_prefix_and_numeric', 64+8*(len(terms)+1))
    return _emit_row(terms, pivot, power, pool=pool)


def _emit_row(terms, pivot, power=None, *, pool):
    """Internal row stage after the caller has prepaid term construction."""
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
    words = np.array([merged[c][0] for c in cols], np.int64)
    powers = np.array([merged[c][1] for c in cols], np.int32)
    floating = words.astype(np.float64)
    if not np.array_equal(floating.astype(np.int64), words):
        raise NativeUnproved('coalesced exact row is not binary64')
    # These actual arrays are one-dimensional; exponent type/range and every
    # finite nonzero exactly represented word were established just above.
    # Do not repeat C9.exponent_data's8 per-final-element domain scans.
    mantissas, exponents = np.frexp(np.abs(floating))
    exponents = exponents.astype(np.int64)+powers.astype(np.int64)
    lower = -19-int(exponents.min())
    maximum = int(exponents.max())
    mantissa = float(mantissas[exponents == maximum].max())
    upper = (41 if mantissa == .5 else 40)-maximum
    if lower > upper:
        raise NativeUnproved('complete pivot/coefficient window incompatible')
    shift = min(max(0, lower), upper)
    try:
        native = scaled_exact(floating, powers.astype(np.int64)+shift)
    except (ValueError, FloatingPointError) as exc:
        raise NativeUnproved('complete native word scaling not exact') from exc
    if np.any((np.abs(native) < 2.**-20) | (np.abs(native) > 2.**40)):
        raise NativeUnproved('actual native coefficient window differs')
    return dict(columns=np.array(cols, np.int32), words=words, powers=powers, native=native,
                pivot=int(pivot), pivot_power=int(power), gauge=int(shift), rhs=0.)


@dataclass(slots=True)
class PreparedTile:
    """Local producer state; all numeric roots remain owned by its source scope."""
    state: dict | None
    pool: object
    source_n_cont: int
    inventory: dict


def prepare_tile(transformed, parent_ids, parent_powers, output_ids, output_powers,
              base_n_cont, *, pool, enabled=False):
    """Internal source ownership transfer, not an independent source certificate."""
    if not enabled:
        return None
    ids, exponents = np.asarray(parent_ids), np.asarray(parent_powers)
    outputs, outpowers = np.asarray(output_ids), np.asarray(output_powers)
    if type(transformed) is not Prepared:
        transformed = prepare(transformed, pool=pool)
    dense, raw = transformed.dense, transformed.transformed
    numbers = raw['numerator'].reshape(*raw['numerator'].shape[:2], 16)
    kernel_powers = raw['exponent']
    kcount, ccount = numbers.shape[:2]
    pool.charge('c88_complete_tile_topology_headers', 1024+128*(ccount+kcount))
    if (ids.shape != (ccount, 4, 4) or exponents.shape != ids.shape
            or outputs.shape != (kcount, 2, 2) or outpowers.shape != outputs.shape
            or ids.dtype.kind not in 'iu' or outputs.dtype.kind not in 'iu'
            or exponents.dtype.kind not in 'iu' or outpowers.dtype.kind not in 'iu'
            or np.any(ids < -1) or np.any(outputs < -1)
            or np.any(ids >= base_n_cont) or np.any(outputs >= base_n_cont)
            or set(map(int, ids[ids >= 0])) & set(map(int, outputs[outputs >= 0]))
            or len(set(map(int, outputs[outputs >= 0]))) != int((outputs >= 0).sum())):
        raise ValueError('complete bounded original input/output coordinate maps required')
    imap, omap = np.kron(T, T), np.kron(R, R)
    ids, exponents = ids.reshape(ccount, 16), exponents.reshape(ccount, 16)
    outputs, outpowers = outputs.reshape(kcount, 4), outpowers.reshape(kcount, 4)
    forms = {}
    for c in range(ccount):
        for t in range(16):
            forms[c, t] = [(int(ids[c, p]), int(imap[t, p]), int(exponents[c, p]))
                           for p in range(16) if ids[c, p] >= 0 and imap[t, p]]
    users = {(k, t): [s for s in range(4) if outputs[k, s] >= 0 and omap[s, t]]
             for k in range(kcount) for t in range(16)}
    active = np.array([[bool(forms[c, t]) for t in range(16)] for c in range(ccount)])
    if dense:
        pool.charge('c88_dense_transform_factorized_degrees', 64*(ccount+kcount))
        shared_channels = {t: np.flatnonzero(active[:, t]).tolist() for t in range(16)}
        channels = {(k, t): shared_channels[t] for k, t in users if users[k, t]}
    else:
        pairs = sum(int(active[:, t].sum()) for k, t in users if users[k, t])
        pool.charge('c88_complete_degree_and_demand_pairs', 8*pairs)
        channels = {(k, t): np.flatnonzero(active[:, t] & (numbers[k, :, t] != 0)).tolist()
                    for k, t in users if users[k, t]}
    channels = {key: cs for key, cs in channels.items() if cs}
    keep_m = {key for key, cs in channels.items() if len(cs) > 1 and len(users[key]) > 1}
    v_uses = {}
    if dense:
        totals = {t: sum(1 if (k, t) in keep_m else len(users[k, t])
                        for k in range(kcount) if (k, t) in channels) for t in range(16)}
        v_uses = {(c, t): totals[t] for t in range(16) for c in shared_channels[t] if totals[t]}
    else:
        for (k, t), cs in channels.items():
            for c in cs:
                v_uses[c, t] = v_uses.get((c, t), 0)+(1 if (k, t) in keep_m else len(users[k, t]))
    keep_v = {key for key, uses in v_uses.items() if len(forms[key]) > 1 and uses > 1}
    representatives, sharing = routes(forms, keep_v, ids, exponents, pool=pool)
    reused_v = {key for key, (rep, _, _) in sharing.items() if key != rep}
    pool.charge('c113_once_prepared_inventory',64+8*len(keep_v))
    inventory=dict(new_factors=len(representatives)+len(keep_m),
        reused_v=len(reused_v),removed_definition_nnz=sum(len(forms[key])+1 for key in reused_v),
        alias_consumer_occurrences=sum(v_uses[key] for key in reused_v))
    state=dict(numbers=numbers,kernel_powers=kernel_powers,kcount=kcount,forms=forms,
        users=users,channels=channels,keep_m=keep_m,v_uses=v_uses,keep_v=keep_v,
        representatives=representatives,sharing=sharing,reused_v=reused_v,
        outputs=outputs,outpowers=outpowers,omap=omap)
    return PreparedTile(state,pool,int(base_n_cont),inventory)


def emit_tile(prepared,base_n_cont,*,pool,enabled=False):
    """Consume the same owned preparation once, after global factor allocation."""
    if not enabled:return None
    if (type(prepared) is not PreparedTile or prepared.state is None
        or prepared.pool is not pool or base_n_cont<prepared.source_n_cont):
        raise ValueError('live once-prepared tile and same construction pool required')
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
        row = _emit_row([(col, -sign, exp) for col, sign, exp in forms[c, t]], slot, pool=pool)
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
        row = _emit_row(terms, slot, pool=pool)
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
            rows.append(_emit_row(terms, int(outputs[k, s]), int(outpowers[k, s]), pool=pool))
    sizes = np.array([len(row['columns']) for row in rows], np.int64)
    ptr = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    packet = dict(indptr=ptr, columns=np.concatenate([r['columns'] for r in rows]) if rows else np.empty(0, np.int32),
        words=np.concatenate([r['words'] for r in rows]) if rows else np.empty(0, np.int64),
        powers=np.concatenate([r['powers'] for r in rows]) if rows else np.empty(0, np.int32),
        native=np.concatenate([r['native'] for r in rows]) if rows else np.empty(0, np.float64),
        pivots=np.array([r['pivot'] for r in rows], np.int32),
        pivot_powers=np.array([r['pivot_power'] for r in rows], np.int32),
        gauges=np.array([r['gauge'] for r in rows], np.int32), rhs=np.zeros(len(rows), np.float64))
    report = dict(native_coefficients_pass=True, new_factors=aux_count, kept_v=len(representatives), kept_m=len(keep_m),
        original_retained_v=len(keep_v), reused_v=len(reused_v),
        original_used_v=len(v_uses), original_used_m=len(channels), inlined_v=len(v_uses)-len(keep_v),
        inlined_m=len(channels)-len(keep_m), rows=len(rows), nnz=int(ptr[-1]),
        base_n_cont=base_n_cont, local_names_are_not_combined_HZ=True,
        global_auxiliary_gate_proved=False, complete_physical_reduction_proved=False)
    if aux_count!=prepared.inventory['new_factors']:
        raise ValueError('complete prepared factor count changed during emission')
    return report, packet


def construct(transformed,parent_ids,parent_powers,output_ids,output_powers,
              base_n_cont,*,pool,enabled=False):
    """Convenience path with exactly one preparation and one emission."""
    if not enabled:return None
    prepared=prepare_tile(transformed,parent_ids,parent_powers,output_ids,output_powers,
        base_n_cont,pool=pool,enabled=True)
    return emit_tile(prepared,base_n_cont,pool=pool,enabled=True)
