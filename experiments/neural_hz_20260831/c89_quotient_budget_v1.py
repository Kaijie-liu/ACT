"""Exact quotient composition and one globally budgeted native circuit plan."""
import numpy as np

from experiments.neural_hz_20260831.c88_inline_tile_v1 import _row, NativeUnproved
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import native_word, word


def compose_row(columns, words, powers, pivot, pivot_power, roots, weights, *, pool):
    """Substitute only old coordinates; preserve a surviving original/new pivot."""
    pool.charge('c89_exact_quotient_row', 64+24*len(columns))
    if pivot < len(roots) and (int(roots[pivot]) != pivot or weights[pivot] != (1, 0)):
        raise NativeUnproved('original output pivot is already projected')
    terms = []
    seen_pivot = False
    for col, number, power in zip(columns, words, powers, strict=True):
        col, number, power = int(col), int(number), int(power)
        if col == pivot:
            if seen_pivot or (number, power) != (1, pivot_power):
                raise ValueError('missing unique original dyadic pivot literal')
            seen_pivot = True
            continue
        if col < len(roots):
            anchor, (multiplier, exponent) = int(roots[col]), weights[col]
            if abs(multiplier).bit_length() > 53:
                raise NativeUnproved('old quotient weight exceeds native-word composition domain')
            number, power = word(number*multiplier, power+exponent)
            col = anchor
        terms.append((col, number, power))
    if not seen_pivot:
        raise ValueError('complete original pivot absent')
    return _row(terms, pivot, pivot_power)


def lower(report, packet, roots, weights, *, pool, enabled=False):
    """No source word arrays are retained by this native-only result."""
    if not enabled:
        return None
    if len(roots) != report['base_n_cont'] or len(weights) != len(roots):
        raise ValueError('full source-bound quotient frame required')
    rows = []
    changed = 0
    for r in range(report['rows']):
        a, b = map(int, packet['indptr'][r:r+2])
        cols = packet['columns'][a:b]
        changed += sum(c < len(roots) and int(roots[c]) != int(c) for c in cols)
        rows.append(compose_row(cols, packet['words'][a:b], packet['powers'][a:b],
            int(packet['pivots'][r]), int(packet['pivot_powers'][r]), roots, weights, pool=pool))
    sizes = [len(r['columns']) for r in rows]
    if sum(sizes) >= 2**31 or len(rows) >= 2**31:
        raise MemoryError('native compact pointer domain exceeded')
    result = dict(indptr=np.r_[0, np.cumsum(sizes)].astype(np.int32),
        columns=np.concatenate([r['columns'] for r in rows]) if rows else np.empty(0,np.int32),
        native=np.concatenate([r['native'] for r in rows]) if rows else np.empty(0,np.float64),
        rhs=np.zeros(len(rows), np.float64),
        pivots=np.array([r['pivot'] for r in rows], np.int32),
        gauges=np.array([r['gauge'] for r in rows], np.int32),
        ab_indptr=np.zeros(len(rows)+1, np.int32))
    return result, dict(changed_old_coordinate_occurrences=int(changed),
        rows=len(rows), nnz=sum(sizes), new_factors=report['new_factors'],
        old_scalar_factors_restored=0, diagnostic_words_retained=False)


def direct_count(hz, physical_rows, pivots, roots, weights, *, pool):
    """Count actual exact quotient-image source rows, not raw source incidence."""
    total = 0
    for row, pivot in zip(physical_rows, pivots, strict=True):
        row, pivot = int(row), int(pivot)
        a, b = map(int, hz.Ac.indptr[row:row+2])
        if hz.b[row] != 0 or hz.Ab.indptr[row+1] != hz.Ab.indptr[row]:
            raise NativeUnproved('linear circuit cannot replace binary or affine-offset row')
        cs, vs = hz.Ac.indices[a:b], hz.Ac.data[a:b]
        pool.charge('c89_original_native_word_decode', 8*len(cs))
        words = [native_word(v) for v in vs]
        positions = np.flatnonzero(cs == pivot)
        if len(positions) != 1 or words[int(positions[0])][0] != 1:
            raise NativeUnproved('direct source root is not an unpacked dyadic definition')
        power = words[int(positions[0])][1]
        mapped = compose_row(cs, [w[0] for w in words], [w[1] for w in words],
            pivot, power, roots, weights, pool=pool)
        total += len(mapped['columns'])
    return total


def bill(packet, *, new_factors, direct_nnz):
    """Exact numeric-array bill for the declared prospective native schema."""
    rows = len(packet['rhs'])
    outputs = rows-new_factors
    if not 0 <= new_factors <= rows or direct_nnz < outputs:
        raise ValueError('complete original output/auxiliary partition required')
    # Both original continuous/binary CSR pointers and RHS are replaced.
    old_bytes = 12*direct_nnz+16*outputs+8
    old_entries = 2*direct_nnz+3*outputs+2
    routing_entries = outputs+8*new_factors+8
    new_bytes = sum(v.nbytes for v in packet.values())+8*routing_entries
    new_entries = sum(v.size for v in packet.values())+routing_entries
    nnz = len(packet['native'])
    return dict(new_factors=new_factors, rows=rows, output_rows=outputs,
        nnz=nnz, direct_nnz=direct_nnz, nnz_saving=direct_nnz-nnz,
        declared_old_bytes=old_bytes, declared_new_bytes=new_bytes,
        byte_saving=old_bytes-new_bytes,
        entry_delta=int(new_entries-old_entries),
        new_emission_work=64*rows+16*(nnz+rows),
        diagnostic_native_bytes=sum(v.nbytes for v in packet.values()),
        declared_auxiliary_and_routing_bytes=8*routing_entries,
        full_LIVE_physical_gate_proved=False, fresh_runtime_work_gate_proved=False)


def select(records, *, existing_aux, existing_work, existing_entries, pool, enabled=False):
    """One greedy structural plan; cumulative reserves never reset between blocks."""
    if not enabled:
        return None
    if not (0 <= existing_aux <= 16384 and 0 <= existing_work <= 16_000_000
            and 0 <= existing_entries <= 131072):
        raise MemoryError('existing complete reserve already exhausted')
    pool.charge('c89_complete_budget_order', 64*len(records)*(1+max(1,len(records)).bit_length()))
    candidates = [i for i, r in enumerate(records) if r.get('mapped_native_pass')
        and r['bill']['nnz_saving'] > 0 and r['bill']['byte_saving'] > 0 and r['bill']['entry_delta'] < 0]
    candidates.sort(key=lambda i: (-records[i]['bill']['byte_saving'],
        records[i]['bill']['new_factors'], i))
    aux, work, entries = existing_aux, existing_work, existing_entries
    chosen, decisions = [], []
    for i in candidates:
        b = records[i]['bill']
        fits = aux+b['new_factors'] <= 16384 and work+b['new_emission_work'] <= 16_000_000
        fits = fits and entries+max(0,b['entry_delta']) <= 131072
        decisions.append(dict(position=i, reserve_fit=fits))
        if fits:
            chosen.append(i)
            aux += b['new_factors']; work += b['new_emission_work']
            entries += max(0,b['entry_delta'])
    return dict(selected_positions=chosen, ordered_decisions=decisions,
        whole_auxiliary_reserve_used=aux, whole_declared_emission_reserve_used=work,
        whole_positive_entry_reserve_used=entries,
        nnz_saving=sum(records[i]['bill']['nnz_saving'] for i in chosen),
        declared_byte_saving=sum(records[i]['bill']['byte_saving'] for i in chosen),
        declared_entry_saving=-sum(records[i]['bill']['entry_delta'] for i in chosen),
        old_scalar_factors_restored=0, proposed_not_admitted=True,
        full_source_equivalence_proved=False, full_LIVE_physical_gate_proved=False,
        fresh_runtime_work_gate_proved=False, formal_gain=0)
