"""Complete affine census with byte-proved repeated coefficient certificates."""

from fractions import Fraction
import math
import hashlib
import struct
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp

from experiments.neural_hz_20260831.c10_fused_rows_v1 import aliases
from experiments.neural_hz_20260831.c10_predicate_census_v1 import exact_products, histogram
from experiments.neural_hz_20260831.c10_alias_quotient_v1 import exact_sum
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def row(matrix, index):
    start, stop = matrix.indptr[index:index + 2]
    return matrix.indices[start:stop], matrix.data[start:stop]


def redundant_box(coefficients, binary, rhs, pivot):
    values = np.abs(np.r_[coefficients, binary, [rhs] if rhs else []])
    if not np.isfinite(values).all() or pivot <= 0. or math.frexp(float(pivot))[0] != .5:
        raise ValueError('invalid finite positive-dyadic box definition')
    values = values[values != 0.]
    if not values.size:
        return True
    exponents = np.frexp(values)[1].astype(np.int64)
    floor = int(exponents.max()) - 26
    total = int(np.left_shift(np.ones(values.size, dtype=np.int64), np.maximum(exponents - floor, 0)).sum())
    # Same integer upward envelope used to define the original C9 boxes.
    return floor + (total - 1).bit_length() <= math.frexp(float(pivot))[1] - 1


def overlaps(cols, products, consumer):
    # Probe the (usually narrow) consumer in the definition, not every wide
    # definition coefficient in the consumer. No arithmetic proof is reused
    # across column identities: every actual collision is checked separately.
    cc, cv = consumer
    positions = np.searchsorted(cols, cc)
    match = positions < cols.size
    match[match] &= cols[positions[match]] == cc[match]
    ok, count, cancellations = True, int(match.sum()), 0
    for pos, value in zip(positions[match], cv[match]):
        try:
            result = exact_sum([float(products[pos]), float(value)])
            cancellations += result == 0.
        except ValueError:
            ok = False
    return ok, count, cancellations


def payload_digest(payload):
    return hashlib.blake2b(payload, digest_size=16).digest()


class CertificateGroups:
    """Hash lookup is only an accelerator; complete bytes prove equivalence."""

    def __init__(self, max_work=256_000_000):
        if type(max_work) is not int or not 0 <= max_work <= 256_000_000:
            raise ValueError('invalid/increased signature work ceiling')
        self.buckets, self.groups = {}, []
        self.replacement_terms = 0
        self.unique_terms = 0
        self.extra_collision_work = 0
        self.signature_work, self.max_work = 0, max_work

    def charge(self, amount):
        if self.signature_work + amount > self.max_work:
            raise MemoryError('byte-certificate grouping exceeds fixed work ceiling')
        self.signature_work += amount

    def intern(self, continuous, binary, constant, pivot, ratio):
        terms = len(continuous) + len(binary) + int(constant != 0.)
        self.charge(8 * terms + 16)
        self.replacement_terms += terms
        payload = (struct.pack('<QQddd', len(continuous), len(binary), pivot, constant, ratio)
                   + continuous.tobytes() + binary.tobytes())
        bucket = self.buckets.setdefault(payload_digest(payload), [])
        for attempt, index in enumerate(bucket):
            # The first full comparison is already in the8/term charge.
            # Even adversarial hash collisions are charged before comparison.
            if attempt:
                self.charge(4 * terms + 16)
                self.extra_collision_work += 4 * terms + 16
            if self.groups[index]['payload'] == payload:
                return index
        index = len(self.groups)
        self.groups.append(dict(payload=payload, continuous=continuous, binary=binary,
            constant=constant, pivot=pivot, ratio=ratio, terms=terms))
        self.unique_terms += terms
        bucket.append(index)
        return index

    def prove(self):
        for group in self.groups:
            constant = group['constant']
            values = np.r_[group['continuous'], group['binary'], [constant] if constant else []]
            precise, products = exact_products(values, group['ratio'])
            count = len(group['continuous']) + len(group['binary'])
            coefficients = products[:count]
            group.update(products=products, precise=bool(precise.all()),
                window=bool(np.all((np.abs(coefficients) >= 2.**-20) & (np.abs(coefficients) <= 2.**40))),
                box=redundant_box(group['continuous'], group['binary'], constant, group['pivot']))


def census(hz, *, old_n_cont, logical_n_cont, old_n_eq, eq_roots, eq_scales, def_rows,
           max_work=256_000_000, max_entries=64_000_000, observe=None):
    if (type(max_work) is not int or not 0 <= max_work <= 256_000_000
            or type(max_entries) is not int or not 0 <= max_entries <= 64_000_000):
        raise ValueError('invalid/increased census ceiling')
    frozen = source_digest(hz)
    if (any(type(v) is not int or v < 0 for v in (old_n_cont, logical_n_cont, old_n_eq))
            or not old_n_cont <= logical_n_cont <= hz.n_cont or not hz.exact or hz.frame_id is None):
        raise ValueError('invalid protected MAIN prefix')
    matrices = [getattr(hz, k) for k in ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub')]
    if any(not sp.isspmatrix_csr(m) or not m.has_canonical_format or np.any(m.data == 0.) for m in matrices):
        raise ValueError('finite canonical zero-free CSR required')
    nnz = sum(m.nnz for m in matrices)
    if nnz > max_entries:
        raise MemoryError('input coefficient entry ceiling exceeded')
    if any(np.any((np.abs(m.data) < 2.**-20) | (np.abs(m.data) > 2.**40)) for m in matrices[2:]):
        raise ValueError('input predicate outside fixed coefficient window')
    main = logical_n_cont - old_n_cont
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
    work = 8 * nnz + 32 * main
    if work > max_work:
        raise MemoryError('structural census work ceiling exceeded')
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
    groups = CertificateGroups(max_work=max_work - work)
    group_ids = []
    for index, consumer, inequality in zip(chosen, consumers, inequalities):
        cc, cv = row(hz.Ac, int(defs[index]))
        _, bv = row(hz.Ab, int(defs[index]))
        tc, tv = row(hz.Auc if inequality else hz.Ac, consumer)
        value = float(tv[np.searchsorted(tc, int(cols[index]))])
        power = math.frexp(float(cv[-1]))[1] - 1
        ratio = math.ldexp(-value, -power)
        if math.ldexp(ratio, power) != -value:
            raise ValueError('consumer multiplier is not exactly reversible')
        group_ids.append(groups.intern(cv[:-1], bv, float(hz.b[int(defs[index])]), float(cv[-1]), ratio))
    if groups.replacement_terms != sum(replacements):
        raise ValueError('complete grouped term coverage mismatch')
    work += (8 * groups.replacement_terms + 16 * len(chosen) + groups.extra_collision_work
             + 64 * groups.unique_terms + 32 * sum(consumer_widths))
    preliminary = {'event': 'single_use_structural_preflight', 'remaining_main_definitions': int(remaining.sum()),
        'already_eliminated_main_aliases': int(removed.size), 'direct_dead_single_use_definitions': int(chosen.size),
        'replacement_and_rhs_terms': sum(replacements), 'consumer_coefficients_inspected': sum(consumer_widths),
        'logical_work_upper': work, 'arithmetic_cap_fits': work <= max_work,
        'coefficient_certificate_groups': len(groups.groups), 'unique_replacement_and_rhs_terms': groups.unique_terms,
        'repeated_certificate_occurrences': int(chosen.size) - len(groups.groups),
        'byte_signature_work': 8 * groups.replacement_terms + 16 * len(chosen) + groups.extra_collision_work,
        'generic_unique_arithmetic_work': 64 * groups.unique_terms,
        'all_uses_byte_compared': True}
    if observe:
        observe(preliminary)
    if work > max_work:
        raise MemoryError(f'complete individual arithmetic exceeds fixed census ceiling: {preliminary}')
    groups.prove()
    n = chosen.size
    table = {'certificate_group': np.asarray(group_ids, dtype=np.int64), 'column': cols[chosen].copy(), 'defining_row': defs[chosen].copy(),
        'consumer_row': np.asarray(consumers, dtype=np.int64), 'consumer_inequality': np.asarray(inequalities, dtype=bool),
        'definition_width': np.zeros(n, np.int64), 'replacement_terms': np.asarray(replacements, dtype=np.int64),
        'consumer_multiplier': np.zeros(n), 'power_two_multiplier': np.zeros(n, bool),
        'nonzero_offset': np.zeros(n, bool), 'binary_definition': np.zeros(n, bool),
        'redundant_box': np.zeros(n, bool), 'all_products_exact': np.zeros(n, bool),
        'products_window_safe': np.zeros(n, bool), 'rhs_exact': np.zeros(n, bool),
        'collision_sums_exact': np.zeros(n, bool), 'collision_terms': np.zeros(n, np.int64),
        'cancelled_columns': np.zeros(n, np.int64), 'individual_nnz_delta': np.zeros(n, np.int64)}
    for output_index, index in enumerate(chosen):
        col, definition = int(cols[index]), int(defs[index])
        cc, cv = row(hz.Ac, definition)
        bc, bv = row(hz.Ab, definition)
        inequality, consumer = inequalities[output_index], consumers[output_index]
        cmat, bmat, rhs = (hz.Auc, hz.Aub, hz.ub) if inequality else (hz.Ac, hz.Ab, hz.b)
        tc, tv = row(cmat, consumer)
        tb, tw = row(bmat, consumer)
        pivot, constant = float(cv[-1]), float(hz.b[definition])
        consumer_value = float(tv[np.searchsorted(tc, col)])
        power = math.frexp(pivot)[1] - 1
        ratio = math.ldexp(-consumer_value, -power)
        if math.ldexp(ratio, power) != -consumer_value:
            raise ValueError('consumer multiplier is not exactly reversible')
        proof = groups.groups[group_ids[output_index]]
        products = proof['products']
        coef_products = products[:len(cv) - 1 + len(bv)]
        table['definition_width'][output_index] = len(cv) + len(bv)
        table['consumer_multiplier'][output_index] = ratio
        table['power_two_multiplier'][output_index] = math.frexp(abs(ratio))[0] == .5
        table['nonzero_offset'][output_index] = constant != 0.
        table['binary_definition'][output_index] = len(bv) > 0
        table['redundant_box'][output_index] = proof['box']
        table['all_products_exact'][output_index] = proof['precise']
        window = proof['window']
        table['products_window_safe'][output_index] = window
        change = float(products[-1]) if constant else 0.
        new_rhs = float(rhs[consumer]) + change
        table['rhs_exact'][output_index] = (math.isfinite(new_rhs) and Fraction(new_rhs)
            == Fraction(float(rhs[consumer])) + Fraction(ratio) * Fraction(constant))
        cok, cn, cz = overlaps(cc[:-1], coef_products[:len(cc) - 1], (tc, tv))
        bok, bn, bz = overlaps(bc, coef_products[len(cc) - 1:], (tb, tw))
        table['collision_sums_exact'][output_index] = cok and bok
        table['collision_terms'][output_index] = cn + bn
        table['cancelled_columns'][output_index] = cz + bz
        table['individual_nnz_delta'][output_index] = -2 - cn - bn - cz - bz
    accepted = np.ones(n, dtype=bool)
    guards = ('redundant_box', 'all_products_exact', 'products_window_safe', 'rhs_exact', 'collision_sums_exact')
    for key in guards:
        accepted &= table[key]
    table['individually_admissible'] = accepted
    report = {**preliminary, 'status': 'READ_ONLY_SHARED_AFFINE_CERTIFICATE_CENSUS', 'formal_gain': 0,
        'total_coefficient_nnz': nnz, 'protected_original_continuous': old_n_cont, 'n_bin': hz.n_bin,
        'individual_guard_counts': {key: int(table[key].sum()) for key in guards},
        'individually_admissible': int(accepted.sum()),
        'admissible_nonzero_offsets': int(np.count_nonzero(accepted & table['nonzero_offset'])),
        'admissible_binary_definitions': int(np.count_nonzero(accepted & table['binary_definition'])),
        'admissible_power_two_multipliers': int(np.count_nonzero(accepted & table['power_two_multiplier'])),
        'admissible_inequality_consumers': int(np.count_nonzero(accepted & table['consumer_inequality'])),
        'definition_width_histogram': histogram(table['definition_width']),
        'admissible_width_histogram': histogram(table['definition_width'][accepted]),
        'total_collision_terms': int(table['collision_terms'].sum()),
        'candidate_transformation_constructed': False, 'simultaneous_substitution_proved': False,
        'solver_executed': False}
    if source_digest(hz) != frozen:
        raise ValueError('read-only census mutated original HZ')
    return report, table
