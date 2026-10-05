"""Complete original convolution-row binding using support-aware dyadic words.

This is a new proof algorithm, not a cheaper tariff for C119's Fraction loops.
Every original kernel coefficient is decoded once.  Parent support is scanned
once per spatial output, shared across filters.  Every demanded native source
row is compared in full against exact, coalesced signed-integer dyadic words.
Unrelated old equations remain part of the caller's qualified unchanged HZ;
this routine does not replace them or issue a source/solver admission.
"""

import math

import numpy as np
import scipy.sparse as sp

from experiments.neural_hz_20260831.c94_raw_mask_plan_v1 import selected_source_check


RATIONAL_BITS = 512
_NATIVE_LOW = 2.**-20
_NATIVE_HIGH = 2.**40


def _word(number, exponent):
    """Canonical signed odd mantissa/exponent in the complete 512-bit domain."""
    number, exponent = int(number), int(exponent)
    if number == 0:
        return 0, 0
    magnitude = abs(number)
    trailing = (magnitude & -magnitude).bit_length()-1
    number >>= trailing
    exponent += trailing
    bits = abs(number).bit_length()
    if (bits > RATIONAL_BITS or exponent < 1-RATIONAL_BITS
            or bits+max(0, exponent) > RATIONAL_BITS):
        raise ValueError('exact original-row word outside 512-bit rational domain')
    return number, exponent


def _decode(value, *, integer=False):
    if integer:
        return _word(int(value), 0)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('finite original/native dyadic coefficient required')
    # Binary32-to-binary64 widening and as_integer_ratio are both exact;
    # no floating multiplication, addition or approximate comparison is used.
    numerator, denominator = value.as_integer_ratio()
    return _word(numerator, -(denominator.bit_length()-1))


def _shift(value, exponent):
    return _word(value[0], value[1]+int(exponent))


def _add(left, right):
    if left[0] == 0:
        return right
    if right[0] == 0:
        return left
    exponent = min(left[1], right[1])
    a, b = left[1]-exponent, right[1]-exponent
    # Check BEFORE either shift.  Temporary aligned integers are at most512
    # bits; their one addition is at most513 bits before exact reduction.
    if (abs(left[0]).bit_length()+a > RATIONAL_BITS
            or abs(right[0]).bit_length()+b > RATIONAL_BITS):
        raise ValueError('exact coalescence alignment outside 512-bit domain')
    return _word((left[0] << a)+(right[0] << b), exponent)


def _integer_vector(value, name):
    if (type(value) is not np.ndarray or value.ndim != 1
            or value.dtype.kind != 'i' or value.dtype.itemsize > 8):
        raise ValueError('complete native signed '+name+' vector required')


def bind_actual_rows(fields, weights, maps, *, pool, enabled=False):
    """Return the old binder's (report, originals, gauges, by_pivot) contract.

    Maps cover C,H,W parents and K,H-2,W-2 outputs; -1 denotes a structural
    parent zero or undemanded output.  Aliased parent IDs are exactly coalesced,
    including cancellation.  Demanded output IDs must be unique and disjoint
    from all parent IDs.  Original selected output RHS/binary rows must be zero,
    as required by the unchanged C94/C91 ordinary-source contract.

    Fees:1024+16 per map entry;64 per original kernel coefficient; support
    scan16*C*H*W+16*C*9*OH*OW; each demanded row128+64 per supported occurrence
    +32 per actual native coefficient; then the unchanged C94 source check.
    All value scans, exact arithmetic and returned literal retention are paid.
    """
    if not enabled:
        return None
    if not isinstance(fields, dict) or not isinstance(maps, dict):
        raise ValueError('complete original source and map dictionaries required')
    required_fields = ('hz', 'old_n_cont', 'old_n_eq', 'logical_n_cont',
                       'eq_roots', 'eq_scales')
    if any(name not in fields for name in required_fields):
        raise ValueError('complete original source fields required')
    if any(name not in maps for name in ('ids', 'powers', 'outs', 'output_powers')):
        raise ValueError('complete original coordinate maps required')
    ids, powers, outputs, opowers = (maps[name] for name in
                                    ('ids', 'powers', 'outs', 'output_powers'))
    if (any(type(value) is not np.ndarray or value.ndim != 3
            or value.dtype.kind != 'i' or value.dtype.itemsize > 8
            for value in (ids, powers, outputs, opowers))
            or powers.shape != ids.shape or outputs.shape != opowers.shape
            or min(ids.shape) < 1 or min(outputs.shape) < 1
            or ids.shape[1] < 3 or ids.shape[2] < 3
            or outputs.shape[1:] != (ids.shape[1]-2, ids.shape[2]-2)):
        raise ValueError('complete signed C,H,W and K,H-2,W-2 maps required')
    channels, height, width = map(int, ids.shape)
    filters, outheight, outwidth = map(int, outputs.shape)
    if (type(weights) is not np.ndarray or weights.shape != (filters, channels, 3, 3)
            or weights.dtype.kind not in 'iuf' or weights.dtype.itemsize > 8):
        raise ValueError('complete ordinary native original K,C,3,3 kernel required')
    roots, scales = fields['eq_roots'], fields['eq_scales']
    _integer_vector(roots, 'equation root')
    _integer_vector(scales, 'equation gauge')
    if roots.shape != scales.shape:
        raise ValueError('complete original root/gauge domains differ')
    scalars = [fields[name] for name in ('old_n_cont', 'old_n_eq', 'logical_n_cont')]
    if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
           for value in scalars):
        raise ValueError('complete original scalar coordinate domain required')
    old, old_eq, logical = map(int, scalars)
    hz = fields['hz']
    if (not sp.isspmatrix_csr(hz.Ac) or not sp.isspmatrix_csr(hz.Ab)
            or hz.Ac.dtype != np.dtype(np.float64) or hz.Ab.dtype != np.dtype(np.float64)
            or type(hz.b) is not np.ndarray or hz.b.dtype != np.dtype(np.float64)
            or hz.b.shape != (hz.n_eq,) or hz.Ac.shape != (hz.n_eq, hz.n_cont)
            or hz.Ab.shape != (hz.n_eq, hz.n_bin) or not hz.exact
            or not 0 <= old <= logical <= hz.n_cont or old_eq < 0
            or len(roots) != old_eq+logical-old):
        raise ValueError('qualified complete exact native source domain required')
    for matrix in (hz.Ac, hz.Ab):
        _integer_vector(matrix.indptr, 'CSR pointer')
        _integer_vector(matrix.indices, 'CSR column')
        if (matrix.indptr.shape != (hz.n_eq+1,)
                or matrix.data.ndim != 1 or matrix.indices.shape != matrix.data.shape):
            raise ValueError('complete native source CSR headers required')

    start = int(pool.used)
    header_fee = 1024+16*sum(int(value.size) for value in (ids, powers, outputs, opowers))
    pool.charge('c123_complete_original_map_and_source_headers', header_fee)
    for matrix in (hz.Ac, hz.Ab):
        if int(matrix.indptr[0]) != 0 or int(matrix.indptr[-1]) != len(matrix.data):
            raise ValueError('complete native source CSR endpoint domain differs')
    if (np.any(ids < -1) or np.any(outputs < -1)
            or np.any(powers < -511) or np.any(powers > 511)
            or np.any(opowers < -511) or np.any(opowers > 511)):
        raise ValueError('complete original masks or semantic powers outside domain')
    parent_ids = [int(value) for value in ids.flat if int(value) >= 0]
    output_ids = [int(value) for value in outputs.flat if int(value) >= 0]
    if (len(set(output_ids)) != len(output_ids) or set(parent_ids).intersection(output_ids)
            or any(not old <= value < logical for value in (*parent_ids, *output_ids))):
        raise ValueError('distinct original MAIN outputs disjoint from original parents required')
    for value in (*parent_ids, *output_ids):
        rank = old_eq+value-old
        if not 0 <= int(roots[rank]) < hz.n_eq:
            raise ValueError('original coordinate removed or original physical root invalid')

    decode_fee = 64*int(weights.size)
    pool.charge('c123_complete_original_kernel_dyadic_decode', decode_fee)
    integer_kernel = weights.dtype.kind in 'iu'
    words = [[[_decode(weights[k, c, dy, dx], integer=integer_kernel)
               for dy in range(3) for dx in range(3)]
              for c in range(channels)] for k in range(filters)]

    support_fee = 16*channels*height*width+16*channels*9*outheight*outwidth
    pool.charge('c123_complete_once_per_spatial_parent_support', support_fee)
    supports = []
    for y in range(outheight):
        for x in range(outwidth):
            support = []
            for channel in range(channels):
                for dy in range(3):
                    for dx in range(3):
                        source = int(ids[channel, y+dy, x+dx])
                        if source >= 0:
                            support.append((source, int(powers[channel, y+dy, x+dx]),
                                            channel, 3*dy+dx))
            supports.append(support)

    originals, gauges, by_pivot = [], [], {}
    direct_nnz = compared = supported_terms = zero_kernel_terms = additions = cancellations = 0
    row_fees = 0
    for k in range(filters):
        for y in range(outheight):
            for x in range(outwidth):
                pivot = int(outputs[k, y, x])
                if pivot < 0:
                    continue
                rank = old_eq+pivot-old
                physical, gauge = int(roots[rank]), int(scales[rank])
                if not -511 <= gauge <= 511:
                    raise ValueError('original native row gauge outside 512-bit domain')
                a, b = int(hz.Ac.indptr[physical]), int(hz.Ac.indptr[physical+1])
                ba, bb = int(hz.Ab.indptr[physical]), int(hz.Ab.indptr[physical+1])
                if not 0 <= a < b <= len(hz.Ac.data) or not 0 <= ba <= bb <= len(hz.Ab.data):
                    raise ValueError('complete demanded native CSR row domain differs')
                support = supports[y*outwidth+x]
                fee = 128+64*len(support)+32*(b-a)
                pool.charge('c123_complete_supported_original_row_and_actual_native_comparison', fee)
                row_fees += fee
                if ba != bb or not math.isfinite(float(hz.b[physical])) or hz.b[physical] != 0:
                    raise ValueError('original offset or binary output predicate cannot be replaced')
                expected = {pivot: _word(1, int(opowers[k, y, x]))}
                supported_terms += len(support)
                for source, power, channel, offset in support:
                    value = words[k][channel][offset]
                    if value[0] == 0:
                        zero_kernel_terms += 1
                        continue
                    value = _shift((-value[0], value[1]), power)
                    if source in expected:
                        additions += 1
                        value = _add(expected[source], value)
                        if value[0] == 0:
                            cancellations += 1
                            del expected[source]
                            continue
                    expected[source] = value
                if len(expected) != b-a:
                    raise ValueError('complete original source support differs from convolution')
                previous = -1
                coefficients = []
                for index in range(a, b):
                    column, value = int(hz.Ac.indices[index]), float(hz.Ac.data[index])
                    if (not previous < column < hz.n_cont or column not in expected
                            or not math.isfinite(value)
                            or not _NATIVE_LOW <= abs(value) <= _NATIVE_HIGH):
                        raise ValueError('canonical complete original native row/window differs')
                    previous = column
                    if _decode(value) != _shift(expected[column], gauge):
                        raise ValueError('actual original row differs from exact dyadic convolution')
                    if column == pivot and value <= 0:
                        raise ValueError('positive original output pivot required')
                    coefficients.append((column, value))
                    compared += 1
                literal = dict(coefficients=coefficients, rhs=float(hz.b[physical]), pivot=pivot)
                originals.append(literal)
                gauges.append(gauge)
                by_pivot[pivot] = (literal, gauge)
                direct_nnz += b-a

    survival = selected_source_check(fields, maps, direct_nnz, pool=pool, enabled=True)
    report = dict(
        all_actual_source_output_rows_bound=len(originals),
        actual_direct_output_nnz=direct_nnz,
        exact_coefficients_and_original_row_gauges_equal=True,
        source_survival=survival,
        algorithm='once_per_spatial_support_and_exact_canonical_dyadic_words',
        original_kernel_coefficients_decoded=int(weights.size),
        spatial_support_lists=len(supports),
        complete_parent_support_positions_scanned=channels*9*outheight*outwidth,
        spatial_supported_occurrences=sum(len(support) for support in supports),
        demanded_supported_kernel_occurrences=supported_terms,
        zero_kernel_occurrences=zero_kernel_terms,
        exact_alias_additions=additions, exact_alias_cancellations=cancellations,
        all_actual_native_coefficients_compared=compared,
        complete_demanded_native_rows_canonical=True,
        every_original_kernel_coefficient_decoded_once=True,
        every_spatial_parent_support_constructed_once=True,
        no_fraction_arithmetic=True, no_approximate_coefficient_comparison=True,
        header_work=header_fee, kernel_decode_work=decode_fee,
        support_scan_work=support_fee, complete_row_work=row_fees,
        complete_observed_work=int(pool.used)-start,
        unrelated_original_rows_remain_in_qualified_source=True,
        original_source_not_mutated=True, numeric_admission=False,
        actual_global_admission=False, formal_gain=0)
    return report, originals, gauges, by_pivot
