"""Exact extended HZ tile equations; no real-source/native admission claim."""
from fractions import Fraction as F
import math
import numpy as np

from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import (
    R, equation, tile_input_equations)
from experiments.neural_hz_20260831.c85_exact_filter_transform_v1 import transform


def native_row(coefficients, rhs, *, slot=None):
    """One exact positive row gauge, including every actual pivot and RHS."""
    values = [(int(c), F(v)) for c, v in coefficients if v]
    if not values or len({c for c, _ in values}) != len(values):
        raise ValueError('nonempty coalesced row required')
    low, high = F(2)**-20, F(2)**40
    minimum, maximum = min(abs(v) for _, v in values), max(abs(v) for _, v in values)
    shift = 0
    while minimum * F(2)**shift < low:
        shift += 1
    while maximum * F(2)**shift > high:
        shift -= 1
    gauge = F(2)**shift
    if any(not low <= abs(v*gauge) <= high for _, v in values):
        raise ValueError('complete coefficient window incompatible')
    scaled = [v*gauge for _, v in values] + [F(rhs)*gauge]
    literals = [float(v) for v in scaled]
    if any(not math.isfinite(x) or F(x) != v for x, v in zip(literals, scaled, strict=True)):
        raise ValueError('complete row is not exactly binary64')
    return dict(coefficients=tuple((c, x) for (c, _), x in zip(values, literals[:-1], strict=True)),
                rhs=literals[-1], slot=slot, gauge=shift)


def build_tile(source, ids, centers, scales, weights, bias, output_ids,
               output_centers, output_scales, *, pool, enabled=False, first_aux=None):
    """Retain all source constraints and output IDs; add only exact definitions."""
    if not enabled:
        return None
    ids = np.asarray(ids)
    centers, scales = np.asarray(centers, object), np.asarray(scales, object)
    weights = np.asarray(weights)
    if weights.ndim != 4 or weights.shape[-2:] != (3, 3):
        raise ValueError('ordinary complete 3x3 filter block required')
    outputs, channels = weights.shape[:2]
    pool.charge('c86_complete_tile_construction_bound', 8192*(outputs*channels+channels+outputs+1))
    base = source['n_cont']
    start = base if first_aux is None else first_aux
    if start < base or ids.shape != (channels, 4, 4) or centers.shape != ids.shape or scales.shape != ids.shape:
        raise ValueError('complete original input geometry required')
    output_ids = np.asarray(output_ids)
    output_centers, output_scales = np.asarray(output_centers, object), np.asarray(output_scales, object)
    if (output_ids.shape != (outputs, 2, 2) or output_centers.shape != output_ids.shape
            or output_scales.shape != output_ids.shape or len(bias) != outputs
            or len(set(map(int, output_ids.flat))) != output_ids.size
            or np.any(output_ids < 0) or np.any(output_ids >= base)
            or np.any(ids < -1) or np.any(ids >= base)
            or set(map(int, ids.flat)) & set(map(int, output_ids.flat))
            or any(F(v) <= 0 for v in output_scales.flat)):
        raise ValueError('distinct original output coordinates with positive scales required')
    proof, transformed = transform(weights, pool=pool, enabled=True)
    if transformed is None or not proof['all_coefficients_exact_binary64']:
        raise ValueError('complete original filter transform rejected')
    auxiliary = []
    v_ids, m_ids = np.empty((channels, 4, 4), np.int64), np.empty((outputs, 4, 4), np.int64)
    v_units, m_units = np.empty_like(v_ids, dtype=object), np.empty_like(m_ids, dtype=object)
    for c in range(channels):
        rows = tile_input_equations(ids[c], centers[c], scales[c], start+len(auxiliary))
        for index, row in enumerate(rows):
            a, b = divmod(index, 4)
            v_ids[c, a, b], v_units[c, a, b] = row['slot'], row['pivot']
            auxiliary.append(native_row(row['coefficients'], row['rhs'], slot=row['slot']))
    for k, a, b in np.ndindex(outputs, 4, 4):
        terms = [(int(v_ids[c, a, b]), F(float(transformed['native'][k, c, a, b]))*v_units[c, a, b])
                 for c in range(channels)]
        row = equation(terms, F(0), start+len(auxiliary))
        m_ids[k, a, b], m_units[k, a, b] = row['slot'], row['pivot']
        auxiliary.append(native_row(row['coefficients'], row['rhs'], slot=row['slot']))
    output_rows = []
    for k, i, j in np.ndindex(outputs, 2, 2):
        coefficients = [(int(output_ids[k, i, j]), F(output_scales[k, i, j]))]
        coefficients += [(int(m_ids[k, a, b]), -int(R[i, a])*int(R[j, b])*m_units[k, a, b])
                         for a in range(4) for b in range(4) if R[i, a] and R[j, b]]
        output_rows.append(native_row(coefficients, F(bias[k])-F(output_centers[k, i, j])))
    return dict(source=source, auxiliary_rows=tuple(auxiliary), output_rows=tuple(output_rows),
                base_n_cont=base, n_cont=start+len(auxiliary), binary_ids=source['binary_ids'],
                new_factors=len(auxiliary), total_new_equation_nnz=sum(len(r['coefficients'])
                    for r in [*auxiliary, *output_rows]), complete_HZ_physical_gate_proved=False,
                actual_network_source_bound=False)


def extend_actual(auxiliary_rows, original):
    """Recover the unique auxiliary extension from the actual emitted rows."""
    point = list(map(F, original))
    for row in auxiliary_rows:
        slot = row['slot']
        if slot != len(point):
            raise ValueError('complete topological actual row sequence required')
        entries = {c: F(v) for c, v in row['coefficients']}
        pivot = entries.pop(slot)
        if pivot <= 0 or any(c >= slot for c in entries):
            raise ValueError('positive topological actual pivot required')
        bound = abs(F(row['rhs']))+sum(map(abs, entries.values()), F(0))
        if bound > pivot:
            raise ValueError('actual new box is not redundant')
        value = (F(row['rhs'])-sum((v*point[c] for c, v in entries.items()), F(0)))/pivot
        if abs(value) > 1:
            raise ValueError('actual extension outside normalized box')
        point.append(value)
    return point


def project_outputs(auxiliary_rows, output_rows, base_n_cont):
    """Independent exact substitution of ACTUAL literals to old coordinates.

    Key -1 denotes a constant. No forward tile/filter matrices are consulted.
    Returns LHS-RHS polynomials, with original output coefficients retained.
    """
    expressions = [{i: F(1)} for i in range(base_n_cont)]
    for row in auxiliary_rows:
        slot = row['slot']
        if slot != len(expressions):
            raise ValueError('complete auxiliary polynomial prefix required')
        actual = {c: F(v) for c, v in row['coefficients']}
        pivot = actual.pop(slot)
        if pivot <= 0 or any(c >= slot for c in actual):
            raise ValueError('topological positive polynomial pivot required')
        poly = {-1: F(row['rhs'])/pivot}
        for col, value in actual.items():
            for old, coefficient in expressions[col].items():
                poly[old] = poly.get(old, F(0))-value*coefficient/pivot
        expressions.append({c: v for c, v in poly.items() if v})
    result = []
    for row in output_rows:
        poly = {-1: -F(row['rhs'])}
        for col, value in row['coefficients']:
            for old, coefficient in expressions[col].items():
                poly[old] = poly.get(old, F(0))+F(value)*coefficient
        # Undo the actual positive output row gauge for independent comparison.
        result.append({c: v/F(2)**row['gauge'] for c, v in poly.items() if v})
    return result
