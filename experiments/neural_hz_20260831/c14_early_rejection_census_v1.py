"""Deterministic complete-population early-rejection affine proof census."""

from fractions import Fraction
import math
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from experiments.neural_hz_20260831.c10_fused_rows_v1 import aliases
from experiments.neural_hz_20260831.c10_predicate_census_v1 import exact_products, histogram
from experiments.neural_hz_20260831.c12_shared_affine_census_v1 import row, overlaps
from experiments.neural_hz_20260831.c13_separated_affine_census_v1 import coefficient_l1_numerator, exact_box
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest

BATCH_TERMS = 8
GUARDS = ('offset_product_exact', 'rhs_exact', 'coefficient_products_exact',
          'products_window_safe', 'redundant_box', 'collision_sums_exact')
REASONS = {0: 'admissible', 1: 'offset_product_inexact', 2: 'rhs_sum_inexact',
           3: 'coefficient_product_inexact', 4: 'coefficient_window',
           5: 'nonredundant_box', 6: 'collision_sum_inexact'}


class WorkPool:
    def __init__(self, cap):
        if type(cap) is not int or not 0 <= cap <= 256_000_000:
            raise ValueError('invalid/increased whole work ceiling')
        self.cap, self.used, self.parts = cap, 0, {}

    def charge(self, name, amount):
        if type(amount) is not int or amount < 0:
            raise ValueError('invalid work charge')
        if self.used + amount > self.cap:
            raise MemoryError(f'whole census budget exhausted before {name}: used={self.used}, requested={amount}, cap={self.cap}')
        self.used += amount
        self.parts[name] = self.parts.get(name, 0) + amount


def prove_definition(cc, cv, bc, bv, constant, pivot, consumer_value, tc, tv, tb, tw, rhs, pool):
    """Return one decision; -1 means UNKNOWN guard, never a passed guard."""
    pool.charge('individual_scalar_and_metadata', 32)
    power = math.frexp(pivot)[1] - 1
    ratio = math.ldexp(-consumer_value, -power)
    if math.ldexp(ratio, power) != -consumer_value:
        raise ValueError('consumer multiplier is not exactly reversible')
    result = {key: -1 for key in GUARDS}
    result.update(first_failure=-1, products_checked=0, failed_term=-1,
        collision_terms=0, cancelled_columns=0, individual_nnz_delta=0,
        delta_evaluated=False, consumer_multiplier=ratio)

    def finish(reason):
        result['first_failure'] = reason
        factors = (result['coefficient_products_exact'], result['offset_product_exact'])
        result['all_products_exact'] = 0 if 0 in factors else (1 if factors == (1, 1) else -1)
        if reason == 0 and any(result[key] != 1 for key in GUARDS):
            raise ValueError('attempted acceptance with an unproved guard')
        return result

    change = 0.
    if constant:
        pool.charge('offset_products', 64)
        exact, products = exact_products(np.array([constant]), ratio)
        result['offset_product_exact'] = int(exact[0])
        if not exact[0]:
            return finish(1)
        change = float(products[0])
    else:
        result['offset_product_exact'] = 1
    new_rhs = float(rhs) + change
    rhs_ok = math.isfinite(new_rhs) and F_equal_rhs(new_rhs, rhs, ratio, constant)
    result['rhs_exact'] = int(rhs_ok)
    if not rhs_ok:
        return finish(2)

    # No all-row product buffer is allocated before early necessary rejection.
    chunks = ([], [])
    logical_position = 0
    for kind, values in enumerate((cv, bv)):
        for begin in range(0, len(values), BATCH_TERMS):
            block = values[begin:begin + BATCH_TERMS]
            pool.charge('coefficient_products', 64 * len(block))
            precise, products = exact_products(block, ratio)
            window = (np.abs(products) >= 2.**-20) & (np.abs(products) <= 2.**40)
            result['products_checked'] += len(block)
            if not precise.all() or not window.all():
                if not precise.all():
                    result['coefficient_products_exact'] = 0
                if not window.all():
                    result['products_window_safe'] = 0
                failed = ~precise if not precise.all() else ~window
                result['failed_term'] = logical_position + begin + int(np.flatnonzero(failed)[0])
                return finish(3 if not precise.all() else 4)
            chunks[kind].append(products)
        logical_position += len(values)
    result['coefficient_products_exact'] = result['products_window_safe'] = 1

    coefficient_count = len(cv) + len(bv)
    pool.charge('exact_coefficient_norm', 16 * coefficient_count)
    norm = coefficient_l1_numerator(cv) + coefficient_l1_numerator(bv)
    result['redundant_box'] = int(exact_box(norm, constant, pivot))
    if not result['redundant_box']:
        return finish(5)
    pool.charge('accepted_product_workspace', 2 * coefficient_count)
    cproducts = np.concatenate(chunks[0]) if chunks[0] else np.zeros(0)
    bproducts = np.concatenate(chunks[1]) if chunks[1] else np.zeros(0)
    pool.charge('consumer_collision_checks', 32 * (len(tc) + len(tb)))
    cok, cn, cz = overlaps(cc, cproducts, (tc, tv))
    bok, bn, bz = overlaps(bc, bproducts, (tb, tw))
    result['collision_sums_exact'] = int(cok and bok)
    result['collision_terms'], result['cancelled_columns'] = cn + bn, cz + bz
    if not (cok and bok):
        return finish(6)
    result['individual_nnz_delta'] = -2 - cn - bn - cz - bz
    result['delta_evaluated'] = True
    return finish(0)


def F_equal_rhs(new_rhs, old_rhs, ratio, constant):
    # Zero offset is an exact identity, not a numerical tolerance shortcut.
    return (new_rhs == float(old_rhs) if constant == 0. else
        Fraction(new_rhs) == Fraction(float(old_rhs)) + Fraction(ratio) * Fraction(constant))


def census(hz, *, old_n_cont, logical_n_cont, old_n_eq, eq_roots, eq_scales, def_rows,
           max_work=256_000_000, max_entries=64_000_000, observe=None):
    pool = WorkPool(max_work)
    if type(max_entries) is not int or not 0 <= max_entries <= 64_000_000:
        raise ValueError('invalid/increased entry ceiling')
    if (any(type(v) is not int or v < 0 for v in (old_n_cont, logical_n_cont, old_n_eq))
            or not old_n_cont <= logical_n_cont <= hz.n_cont or not hz.exact or hz.frame_id is None):
        raise ValueError('invalid protected MAIN prefix')
    matrices = [getattr(hz, k) for k in ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub')]
    if any(not sp.isspmatrix_csr(m) for m in matrices):
        raise ValueError('canonical CSR required')
    nnz = sum(m.nnz for m in matrices)
    if nnz > max_entries:
        raise MemoryError('input coefficient entry ceiling exceeded')
    main = logical_n_cont - old_n_cont
    pool.charge('complete_structural_validation', int(8 * nnz + 32 * main))
    frozen = source_digest(hz)
    if any(not m.has_canonical_format or np.any(m.data == 0.) for m in matrices):
        raise ValueError('finite canonical zero-free CSR required')
    if any(np.any((np.abs(m.data) < 2.**-20) | (np.abs(m.data) > 2.**40)) for m in matrices[2:]):
        raise ValueError('input predicate outside fixed coefficient window')
    eq_roots, eq_scales, def_rows = map(np.asarray, (eq_roots, eq_scales, def_rows))
    if eq_roots.shape != (old_n_eq + main,) or eq_scales.shape != eq_roots.shape:
        raise ValueError('incomplete logical MAIN mapping')
    if def_rows.ndim != 1 or logical_n_cont + def_rows.size > hz.n_cont:
        raise ValueError('invalid radix frame')
    holder = SimpleNamespace(old_n_cont=old_n_cont, logical_n_cont=logical_n_cont,
        old_n_eq=old_n_eq, eq_roots=eq_roots, eq_scales=eq_scales)
    removed, _, _, _ = aliases(holder)
    roots = eq_roots[eq_roots >= 0]
    if (np.asarray(def_rows).dtype != np.dtype(np.int64)
            or not np.array_equal(np.sort(np.r_[roots, def_rows]), np.arange(roots.size + len(def_rows)))
            or roots.size + len(def_rows) > hz.n_eq):
        raise ValueError('invalid complete surviving MAIN/radix prefix partition')
    value_degree = np.bincount(hz.Gc.indices, minlength=hz.n_cont)
    degree = np.bincount(hz.Ac.indices, minlength=hz.n_cont) + np.bincount(hz.Auc.indices, minlength=hz.n_cont)
    if np.any(value_degree[removed] + degree[removed]):
        raise ValueError('tagged removed coordinate still active')
    cols = np.arange(old_n_cont, logical_n_cont, dtype=np.int64)
    defs = eq_roots[old_n_eq:]
    remaining = defs >= 0
    valid = np.flatnonzero(remaining & (value_degree[cols] == 0) & (degree[cols] == 2))
    direct = []
    for index in valid:
        cc, cv = row(hz.Ac, int(defs[index]))
        if cc.size and cc[-1] == cols[index] and cv[-1] > 0. and math.frexp(float(cv[-1]))[0] == .5:
            direct.append(index)
    chosen = np.asarray(direct, dtype=np.int64)
    eqc, ineqc = hz.Ac.tocsc(), hz.Auc.tocsc()
    consumers, inequalities, replacements, consumer_widths = [], [], [], []
    for index in chosen:
        col, definition = int(cols[index]), int(defs[index])
        start, stop = eqc.indptr[col:col + 2]
        found = [int(r) for r in eqc.indices[start:stop] if r != definition]
        inequality = not found
        if inequality:
            start, stop = ineqc.indptr[col:col + 2]
            found = [int(r) for r in ineqc.indices[start:stop]]
        if len(found) != 1:
            raise ValueError('single remaining predicate occurrence accounting failed')
        consumer = found[0]
        cmat, bmat = (hz.Auc, hz.Aub) if inequality else (hz.Ac, hz.Ab)
        width = int(cmat.indptr[consumer + 1] - cmat.indptr[consumer]
                    + bmat.indptr[consumer + 1] - bmat.indptr[consumer])
        terms = int(hz.Ac.indptr[definition + 1] - hz.Ac.indptr[definition] - 1
                    + hz.Ab.indptr[definition + 1] - hz.Ab.indptr[definition] + (hz.b[definition] != 0.))
        consumers.append(consumer)
        inequalities.append(inequality)
        replacements.append(terms)
        consumer_widths.append(width)
    offset_count = int(np.count_nonzero(hz.b[defs[chosen]]))
    coefficient_count = sum(replacements) - offset_count
    eager_upper = (pool.used + 32 * len(chosen) + 64 * sum(replacements)
        + 18 * coefficient_count + 32 * sum(consumer_widths))
    preliminary = {'event': 'complete_structural_preflight',
        'remaining_main_definitions': int(remaining.sum()), 'already_eliminated_main_aliases': int(removed.size),
        'direct_dead_single_use_definitions': int(chosen.size),
        'replacement_and_rhs_terms': sum(replacements), 'coefficient_terms': coefficient_count,
        'nonzero_definition_offsets': offset_count, 'consumer_coefficients': sum(consumer_widths),
        'eager_all_guard_work_upper': eager_upper, 'precharged_structural_work': pool.used,
        'max_work': max_work, 'algorithm': 'deterministic_full_population_short_circuit',
        'batch_terms': BATCH_TERMS, 'partial_population_acceptance_allowed': False}
    if observe:
        observe(preliminary)
    n = len(chosen)
    table = {'column': cols[chosen].copy(), 'defining_row': defs[chosen].copy(),
        'consumer_row': np.asarray(consumers, np.int64), 'consumer_inequality': np.asarray(inequalities, bool),
        'definition_width': np.zeros(n, np.int64), 'replacement_terms': np.asarray(replacements, np.int64),
        'consumer_multiplier': np.zeros(n), 'power_two_multiplier': np.zeros(n, bool),
        'nonzero_offset': np.zeros(n, bool), 'binary_definition': np.zeros(n, bool),
        **{key: np.full(n, -1, np.int8) for key in (*GUARDS, 'all_products_exact', 'first_failure')},
        'products_checked': np.zeros(n, np.int64), 'failed_term': np.full(n, -1, np.int64),
        'collision_terms': np.zeros(n, np.int64), 'cancelled_columns': np.zeros(n, np.int64),
        'individual_nnz_delta': np.zeros(n, np.int64), 'delta_evaluated': np.zeros(n, bool)}
    processed = 0
    try:
        for out, index in enumerate(chosen):
            col, definition = int(cols[index]), int(defs[index])
            cc, cv = row(hz.Ac, definition)
            bc, bv = row(hz.Ab, definition)
            inequality, consumer = inequalities[out], consumers[out]
            cmat, bmat, rhs = (hz.Auc, hz.Aub, hz.ub) if inequality else (hz.Ac, hz.Ab, hz.b)
            tc, tv = row(cmat, consumer)
            tb, tw = row(bmat, consumer)
            pivot, constant = float(cv[-1]), float(hz.b[definition])
            value = float(tv[np.searchsorted(tc, col)])
            proof = prove_definition(cc[:-1], cv[:-1], bc, bv, constant, pivot, value,
                tc, tv, tb, tw, float(rhs[consumer]), pool)
            for key, value in proof.items():
                table[key][out] = value
            table['definition_width'][out] = len(cv) + len(bv)
            table['power_two_multiplier'][out] = math.frexp(abs(proof['consumer_multiplier']))[0] == .5
            table['nonzero_offset'][out] = constant != 0.
            table['binary_definition'][out] = len(bv) > 0
            processed += 1
            if observe and processed % 2048 == 0:
                observe({'event': 'whole_population_proof_progress', 'processed': processed,
                    'total': n, 'charged_work': pool.used, 'acceptance_published': False})
    except MemoryError:
        if observe:
            observe({'event': 'whole_population_budget_rejected', 'processed': processed,
                'total': n, 'charged_work': pool.used, 'work_parts': dict(pool.parts),
                'acceptance_published': False, 'partial_table_returned': False})
        raise
    if processed != n or np.any(table['first_failure'] < 0):
        raise ValueError('incomplete population cannot publish an admissible subset')
    accepted = table['first_failure'] == 0
    if any(np.any(table[key][accepted] != 1) for key in GUARDS):
        raise ValueError('unknown guard cannot certify acceptance')
    table['individually_admissible'] = accepted
    if source_digest(hz) != frozen:
        raise ValueError('read-only census mutated original HZ')
    report = {**preliminary, 'status': 'COMPLETE_EARLY_REJECTION_AFFINE_CENSUS',
        'completed_population': processed, 'logical_work_used': pool.used, 'work_parts': dict(pool.parts),
        'decision_counts': {name: int(np.count_nonzero(table['first_failure'] == code)) for code, name in REASONS.items()},
        'guard_counts': {key: {'passed': int(np.count_nonzero(table[key] == 1)),
            'failed': int(np.count_nonzero(table[key] == 0)), 'not_evaluated': int(np.count_nonzero(table[key] == -1))}
            for key in GUARDS},
        'coefficient_products_actually_checked': int(table['products_checked'].sum()),
        'individually_admissible': int(accepted.sum()),
        'admissible_width_histogram': histogram(table['definition_width'][accepted]),
        'definition_width_histogram': histogram(table['definition_width']),
        'admissible_nonzero_offsets': int(np.count_nonzero(accepted & table['nonzero_offset'])),
        'admissible_binary_definitions': int(np.count_nonzero(accepted & table['binary_definition'])),
        'admissible_power_two_multipliers': int(np.count_nonzero(accepted & table['power_two_multiplier'])),
        'admissible_inequality_consumers': int(np.count_nonzero(accepted & table['consumer_inequality'])),
        'n_bin': hz.n_bin, 'protected_original_continuous': old_n_cont,
        'formal_gain': 0, 'candidate_transformation_constructed': False,
        'simultaneous_substitution_proved': False, 'solver_executed': False,
        'partial_population_acceptance': False}
    return report, table
