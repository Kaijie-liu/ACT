"""Denominator-carrying exact F(4, 3) tile definitions, never an admission.

This is an opt-in, isolated constructor.  Every old input/output coordinate and
every external old HZ predicate survives.  Only normalized continuous V and M
definitions are added; no binary coordinate is inspected, pivoted, or deleted.
In particular an M definition carries its odd denominator on its positive
pivot.  Its semantic unit is 2**semantic_power, NOT that defining pivot.
"""
from fractions import Fraction as F
import numpy as np

from experiments.neural_hz_20260831.c86_complete_tile_hz_v1 import native_row


BT = ((4, 0, -5, 0, 1, 0), (0, -4, -4, 1, 1, 0),
      (0, 4, -4, -1, 1, 0), (0, -2, -1, 2, 1, 0),
      (0, 2, -1, -2, 1, 0), (0, 4, 0, -5, 0, 1))
GNUM = ((1, 0, 0), (-1, -1, -1), (-1, 1, -1),
        (1, 2, 4), (1, -2, 4), (0, 0, 1))
D = (4, 6, 6, 24, 24, 1)
AT = ((1, 1, 1, 1, 1, 0), (0, 1, -1, 2, -2, 0),
      (0, 1, 1, 4, 4, 0), (0, 1, -1, 8, -8, 1))
RATIONAL_BITS = 512


def _bounded(value):
    value = F(value)
    if (abs(value.numerator).bit_length() > RATIONAL_BITS
            or value.denominator.bit_length() > RATIONAL_BITS):
        raise ValueError('exact F4 coefficient outside 512-bit rational domain')
    return value


def _power(exponent):
    exponent = int(exponent)
    if not -511 <= exponent <= 511:
        raise ValueError('semantic power outside 512-bit rational domain')
    return _bounded(F(2) ** exponent)


def _normalization(norm):
    """Smallest nonnegative integer e with the exact norm <= 2**e."""
    norm = _bounded(norm)
    if norm <= 1:
        return 0
    exponent = max(0, norm.numerator.bit_length()-norm.denominator.bit_length())
    if _power(exponent) < norm:
        exponent += 1
    _power(exponent)
    return exponent


def _l1(values):
    total = F(0)
    for value in values:
        total = _bounded(total+abs(value))
    return total


def _native(coefficients, pivot, semantic_power, denominator, role):
    coefficients = [(int(col), _bounded(value)) for col, value in coefficients if value]
    coefficients.sort(key=lambda item: item[0])
    row = native_row(coefficients, F(0), slot=int(pivot))
    row.update(semantic_power=int(semantic_power), denominator=int(denominator),
               role=tuple(map(int, role)))
    return row


def construct(weights, parent_ids, parent_powers, output_ids, output_powers,
              base_n_cont, *, pool, enabled=False):
    """Emit complete exact single-tile rows, retaining all needed nonzero V/M.

    Inputs have shapes K,C,3,3; C,6,6; and K,4,4, respectively.  An input -1
    denotes structural zero; an output -1 denotes an unused output.  All offsets
    are zero.  Caller retains the original source predicates and performs every
    whole-HZ source, physical, numeric, resource, replay and admission gate.

    Return None when disabled, otherwise (JSON-safe report, native packet).
    Rejection raises; no partial packet is returned or installed anywhere.
    """
    if not enabled:
        return None
    weights = np.asarray(weights)
    if (weights.ndim != 4 or weights.shape[-2:] != (3, 3)
            or min(weights.shape[:2]) < 1 or weights.dtype.kind not in 'iuf'
            or weights.dtype.itemsize > 8):
        raise ValueError('ordinary finite native 3x3 filter block required')
    kcount, ccount = map(int, weights.shape[:2])
    pool.charge('c119_complete_tile_topology',
                1024+128*(ccount+kcount)+16*(36*ccount+16*kcount))
    ids, powers = np.asarray(parent_ids), np.asarray(parent_powers)
    outputs, outpowers = np.asarray(output_ids), np.asarray(output_powers)
    if (not isinstance(base_n_cont, (int, np.integer))
            or isinstance(base_n_cont, (bool, np.bool_))
            or not 0 <= int(base_n_cont) <= np.iinfo(np.int32).max-36*(ccount+kcount)
            or ids.shape != (ccount, 6, 6) or powers.shape != ids.shape
            or outputs.shape != (kcount, 4, 4) or outpowers.shape != outputs.shape
            or any(a.dtype.kind not in 'iu' for a in (ids, powers, outputs, outpowers))
            or np.any(ids < -1) or np.any(outputs < -1)
            or np.any(ids >= base_n_cont) or np.any(outputs >= base_n_cont)
            or np.any(powers < -511) or np.any(powers > 511)
            or np.any(outpowers < -511) or np.any(outpowers > 511)):
        raise ValueError('complete bounded original coordinate maps and powers required')
    base_n_cont = int(base_n_cont)
    input_set = set(map(int, ids[ids >= 0]))
    output_list = list(map(int, outputs[outputs >= 0]))
    if len(set(output_list)) != len(output_list) or input_set.intersection(output_list):
        raise ValueError('distinct original outputs disjoint from original inputs required')

    # This entire tariff precedes all exact kernel conversion and transforms.
    kernel_fee = 64*(9*36*kcount*ccount)
    pool.charge('c119_exact_integer_filter_transform', kernel_fee)
    if not np.all(np.isfinite(weights)):
        raise ValueError('finite original filter coefficients required')
    numerators = np.empty((kcount, ccount, 36), dtype=object)
    for k in range(kcount):
        for c in range(ccount):
            kernel = [[_bounded(F(int(weights[k, c, i, j])) if weights.dtype.kind in 'iu'
                                else F(float(weights[k, c, i, j])))
                       for j in range(3)] for i in range(3)]
            for t in range(6):
                for u in range(6):
                    value = F(0)
                    for i in range(3):
                        for j in range(3):
                            term = _bounded(kernel[i][j]*GNUM[t][i]*GNUM[u][j])
                            value = _bounded(value+term)
                    numerators[k, c, 6*t+u] = value

    ids, powers = ids.reshape(ccount, 36), powers.reshape(ccount, 36)
    outputs, outpowers = outputs.reshape(kcount, 16), outpowers.reshape(kcount, 16)
    users = {}
    for k in range(kcount):
        for t in range(6):
            for u in range(6):
                needed = [s for s in range(16) if outputs[k, s] >= 0
                          and AT[s//4][t]*AT[s % 4][u]]
                if needed:
                    users[k, 6*t+u] = needed
    candidate_v = {(c, index) for k, index in users for c in range(ccount)
                   if numerators[k, c, index]}
    forms, prepaid_row_fee = {}, 0
    for c, index in sorted(candidate_v):
        t, u = divmod(index, 6)
        positions = [p for p in range(36)
                     if ids[c, p] >= 0 and BT[t][p//6]*BT[u][p % 6]]
        # Raw terms include the defining pivot; the extra field is the RHS.
        row_fee = 64+16*(len(positions)+2)
        pool.charge('c119_complete_row_terms_and_native', row_fee)
        prepaid_row_fee += row_fee
        merged = {}
        for p in positions:
            col = int(ids[c, p])
            value = _bounded(BT[t][p//6]*BT[u][p % 6]*_power(powers[c, p]))
            merged[col] = _bounded(merged.get(col, F(0))+value)
        merged = {col: value for col, value in merged.items() if value}
        if merged:
            forms[c, index] = merged
    channels = {(k, index): [c for c in range(ccount)
                            if (c, index) in forms and numerators[k, c, index]]
                for k, index in users}
    channels = {key: cs for key, cs in channels.items() if cs}
    keep_v = {(c, index) for (k, index), cs in channels.items() for c in cs}
    rows, vmap, mmap = [], {}, {}
    for c, index in sorted(keep_v):
        form = forms[c, index]
        exponent = _normalization(_l1(form.values()))
        slot = base_n_cont+len(rows)
        rows.append(_native([(slot, _power(exponent)),
                             *((col, -value) for col, value in form.items())],
                            slot, exponent, 1, (0, c, index)))
        vmap[c, index] = (slot, exponent)

    for k, index in sorted(channels):
        cs = channels[k, index]
        row_fee = 64+16*(len(cs)+2)
        pool.charge('c119_complete_row_terms_and_native', row_fee)
        prepaid_row_fee += row_fee
        terms = [(vmap[c, index][0],
                  _bounded(numerators[k, c, index]*_power(vmap[c, index][1])))
                 for c in cs]
        denominator = D[index//6]*D[index % 6]
        exponent = _normalization(_bounded(_l1(v for _, v in terms)/denominator))
        slot = base_n_cont+len(rows)
        pivot = _bounded(denominator*_power(exponent))
        rows.append(_native([(slot, pivot), *((col, -value) for col, value in terms)],
                            slot, exponent, denominator, (1, k, index)))
        mmap[k, index] = (slot, exponent)
    auxiliary_count = len(rows)
    for k in range(kcount):
        for s in range(16):
            if outputs[k, s] < 0:
                continue
            used = [index for index in range(36) if (k, index) in mmap
                    and AT[s//4][index//6]*AT[s % 4][index % 6]]
            row_fee = 64+16*(len(used)+2)
            pool.charge('c119_complete_row_terms_and_native', row_fee)
            prepaid_row_fee += row_fee
            exponent = int(outpowers[k, s])
            slot = int(outputs[k, s])
            coefficients = [(slot, _power(exponent))]
            coefficients += [(mmap[k, index][0],
                              _bounded(-AT[s//4][index//6]*AT[s % 4][index % 6]
                                       *_power(mmap[k, index][1]))) for index in used]
            rows.append(_native(coefficients, slot, exponent, 1, (2, k, s)))

    sizes = np.array([len(row['coefficients']) for row in rows], dtype=np.int64)
    ptr = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    entries = [term for row in rows for term in row['coefficients']]
    packet = dict(
        indptr=ptr,
        columns=np.array([col for col, value in entries], dtype=np.int32),
        native=np.array([value for col, value in entries], dtype=np.float64),
        pivots=np.array([row['slot'] for row in rows], dtype=np.int32),
        gauges=np.array([row['gauge'] for row in rows], dtype=np.int32),
        rhs=np.zeros(len(rows), dtype=np.float64),
        ab_indptr=np.zeros(len(rows)+1, dtype=np.int64),
        roles=np.array([row['role'] for row in rows], dtype=np.int32).reshape(-1, 3),
        semantic_powers=np.array([row['semantic_power'] for row in rows], dtype=np.int32),
        defining_denominators=np.array([row['denominator'] for row in rows], dtype=np.int64))
    emission_fee = 64*len(rows)+16*(int(ptr[-1])+len(rows))
    if prepaid_row_fee < emission_fee:
        raise ValueError('complete row construction tariff was not prepaid')
    report = dict(
        native_coefficients_pass=True, auxiliary_boxes_redundant=True,
        kept_v=len(vmap), kept_m=len(mmap), new_factors=auxiliary_count,
        inlined_v=0, inlined_m=0, rows=len(rows), nnz=int(ptr[-1]),
        base_n_cont=base_n_cont, n_cont=base_n_cont+auxiliary_count,
        kernel_transform_prepaid=kernel_fee, row_construction_prepaid=prepaid_row_fee,
        whole_circuit_emission=emission_fee,
        original_input_output_ids_retained=True, original_predicates_retained_externally=True,
        semantic_units_are_not_defining_pivots=True,
        source_bound=False, global_auxiliary_gate_proved=False,
        complete_physical_reduction_proved=False, live_admission=False,
        score_gain=0, all_gates_not_admitted=True)
    return report, packet
