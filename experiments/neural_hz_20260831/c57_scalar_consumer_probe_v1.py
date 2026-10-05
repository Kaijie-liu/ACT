"""Exact diagnostic of the full raw singleton cohort; never an HZ writer."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import numpy as np
from experiments.neural_hz_20260831.c56_gauged_carrier_v1 import digit_rows

ONE = (1, 0)


def word(m, e):
    """Keep an exact diagnostic dyadic; out-of-window words are NOT admitted."""
    m, e = int(m), int(e)
    if not m:
        return (0, 0)
    shift = (abs(m) & -abs(m)).bit_length() - 1
    m >>= shift
    e += shift
    if abs(m).bit_length() > 512:
        raise MemoryError('unchanged512-bit diagnostic precision cap')
    return m, e


def native_word(value):
    numerator, denominator = float(value).as_integer_ratio()
    return word(numerator, -(denominator.bit_length() - 1))


def fraction(value):
    m, e = value
    return F(m << e) if e >= 0 else F(m, 1 << -e)


def in_window(value):
    m, e = value
    if not m:
        return True
    floor = abs(m).bit_length() - 1 + e
    return -20 <= floor <= 40 and (floor != 40 or abs(m) == 1)


def product(a, b, pool):
    wa = max(1, (abs(a[0]).bit_length() + 63) // 64)
    wb = max(1, (abs(b[0]).bit_length() + 63) // 64)
    pool.charge('c57_exact_dyadic_product', 64 * wa * wb)
    return word(a[0] * b[0], a[1] + b[1])


def plus(a, b, pool):
    if not a[0]:
        return b
    if not b[0]:
        return a
    e = min(a[1], b[1])
    span = max(abs(a[0]).bit_length() + a[1] - e,
               abs(b[0]).bit_length() + b[1] - e)
    pool.charge('c57_exact_shared_root_sum', 32 * max(1, (span + 63) // 64))
    return word((a[0] << (a[1] - e)) + (b[0] << (b[1] - e)), e)


class Probe:
    """Cumulative exact root map and streamed consumer cost ledger."""
    def __init__(self, n_cont, n_eq, pool):
        pool.charge('c57_complete_diagnostic_maps', 8 * n_cont + n_eq + 1024)
        self.pool = pool
        self.roots = np.arange(n_cont, dtype=np.int64)
        self.weights = [ONE] * n_cont
        self.depth = np.zeros(n_cont, np.int32)
        self.chosen = np.zeros(n_cont, bool)
        self.removed = np.zeros(n_eq, bool)
        self.counts = Counter()
        self.depth_hist = Counter()
        self.bits_hist = Counter()
        self.floor_hist = Counter()
        self.chain_hash = hashlib.sha256()
        self.consumer_hash = hashlib.sha256()
        self.plans = set()
        self.consumer = Counter()

    def append(self, slot, parent, ratio, physical_row, *, packed=False):
        if (packed or not 0 <= parent < slot < len(self.roots)
                or self.chosen[slot] or not 0 <= physical_row < len(self.removed)
                or self.removed[physical_row] or not 0 < abs(ratio) <= 1):
            raise ValueError('unproved raw singleton birth/physical definition')
        d = ratio.denominator
        if d & (d - 1):
            raise ValueError('positive dyadic pivot required')
        local = word(ratio.numerator, -(d.bit_length() - 1))
        current = product(local, self.weights[parent], self.pool)
        self.pool.charge('c57_independent_Fraction_chain_induction_and_record', 160)
        if fraction(current) != ratio * fraction(self.weights[parent]):
            raise ValueError('composed original-coordinate inverse differs')
        if not 0 < abs(fraction(current)) <= 1:
            raise ValueError('composed factor box is not redundant')
        root = int(self.roots[parent])
        self.roots[slot] = root
        self.weights[slot] = current
        self.depth[slot] = self.depth[parent] + 1
        self.chosen[slot] = True
        self.removed[physical_row] = True
        self.counts['singletons'] += 1
        self.counts['local_general_scalar'] += abs(local[0]) != 1
        self.counts['negative_composed'] += current[0] < 0
        self.counts['local_window_violations'] += not in_window(local)
        self.counts['composed_window_violations'] += not in_window(current)
        self.counts['composed_longer_than_binary64'] += abs(current[0]).bit_length() > 53
        self.depth_hist[str(int(self.depth[slot]))] += 1
        self.bits_hist[str(abs(current[0]).bit_length())] += 1
        self.floor_hist[str(abs(current[0]).bit_length() - 1 + current[1])] += 1
        self.chain_hash.update(json.dumps([slot, parent, root, local, current, physical_row]).encode())

    def translated(self, cols, values):
        """Return only changed-root terms; unchanged native entries stay implicit."""
        selected = self.chosen[cols]
        positions = np.flatnonzero(selected)
        self.pool.charge('c57_affected_consumer_row', 32 + 3 * len(cols) + 16 * len(positions))
        updates = {}
        for i in positions:
            col = int(cols[i])
            root = int(self.roots[col])
            value = product(native_word(values[i]), self.weights[col], self.pool)
            updates[root] = plus(updates.get(root, (0, 0)), value, self.pool)
        collisions = 0
        self.pool.charge('c57_changed_root_ordering', len(updates) * max(1, len(updates).bit_length()))
        for root in sorted(updates):
            self.pool.charge('c57_original_root_binary_search', max(1, len(cols).bit_length()) + 8)
            j = int(np.searchsorted(cols, root))
            if j < len(cols) and int(cols[j]) == root:
                if self.chosen[root]:
                    raise ValueError('composed root was not retained')
                updates[root] = plus(updates[root], native_word(values[j]), self.pool)
                collisions += 1
        self.consumer['changed_terms'] += len(positions)
        self.consumer['coalesced_terms'] += len(positions) - len(updates) + collisions
        self.consumer['cancelled_roots'] += sum(v[0] == 0 for v in updates.values())
        nonzero = {root: value for root, value in updates.items() if value[0]}
        delta = -len(positions) - collisions + len(nonzero)
        return nonzero, delta

    def carrier_groups(self, updates, *, predicate):
        groups = {}
        self.pool.charge('c57_consumer_root_ordering', len(updates) * max(1, len(updates).bit_length()))
        for root, value in sorted(updates.items()):
            m, e = value
            self.consumer['derived_window_violations'] += not in_window(value)
            self.consumer['derived_nonzero_coefficients'] += 1
            if abs(m).bit_length() > 53:
                groups.setdefault((abs(m), e), []).append((root, 1 if m > 0 else -1))
        row_delta = 0
        self.pool.charge('c57_consumer_group_ordering', len(groups) * max(1, len(groups).bit_length()))
        for (m, e), terms in sorted(groups.items()):
            key = (m, e, tuple(terms))
            self.pool.charge('c57_carrier_key_and_digest', 64 + 16 * len(terms))
            self.plans.add(key)
            self.consumer['long_group_occurrences'] += 1
            self.consumer['long_coefficient_terms'] += len(terms)
            row_delta += 1 - len(terms)
            self.consumer_hash.update(json.dumps(key).encode())
        if predicate:
            self.consumer['predicate_carrier_replacement_nnz_delta'] += row_delta

    def finish(self, hz, *, observe=None):
        totals = {}
        for name, matrix in [('Ac', hz.Ac), ('Auc', hz.Auc), ('Gc', hz.Gc)]:
            nr = matrix.shape[0]
            self.pool.charge('c57_full_consumer_incidence_scan', 3 * matrix.nnz + 8 * nr + 1024)
            selected = self.chosen[matrix.indices]
            prefix = np.empty(matrix.nnz + 1, np.int64)
            prefix[0] = 0
            np.cumsum(selected, out=prefix[1:])
            hits = prefix[matrix.indptr[1:]] - prefix[matrix.indptr[:-1]]
            affected = hits > 0
            removed_nnz = 0
            if name == 'Ac':
                removed_nnz = int(np.diff(matrix.indptr)[self.removed].sum())
                affected[self.removed] = False
            chosen_rows = np.flatnonzero(affected)
            after = matrix.nnz - removed_nnz
            del selected, prefix, hits, affected
            for index in chosen_rows:
                a, b = map(int, matrix.indptr[index:index + 2])
                updates, delta = self.translated(matrix.indices[a:b], matrix.data[a:b])
                after += delta
                self.carrier_groups(updates, predicate=name != 'Gc')
            totals[name] = dict(original_nnz=int(matrix.nnz), symbolically_removed_nnz=removed_nnz,
                affected_surviving_rows=len(chosen_rows), symbolic_after_nnz=int(after))
            if observe:
                observe(dict(event='complete_consumer_matrix_censused',matrix=name,
                    **totals[name],partial_consumer_counts=dict(self.consumer)))
        auxiliaries = aux_nnz = multiplier_failures = 0
        group_size_hist = Counter()
        for m, e, terms in self.plans:
            k = (len(terms) - 1).bit_length()
            if k >= 60:
                raise MemoryError('unchanged carrier width cap')
            digits = digit_rows((m, e), min(53, 60 - k))
            self.pool.charge('c57_complete_carrier_digit_cost', 128 + 32 * (len(digits) + len(terms)))
            auxiliaries += len(digits)
            aux_nnz += sum(1 + (i > 0) + (len(terms) if d else 0)
                           for i, (d, _) in enumerate(digits))
            multiplier_failures += not -20 <= m.bit_length() + e + k <= 40
            group_size_hist[str(len(terms))] += 1
        self.pool.charge('c57_complete_inverse_value_liveness', 12 * len(self.weights))
        distinct = set(self.weights)
        self.pool.charge('c57_distinct_inverse_word_statistics', 32 * len(distinct))
        limbs = sum((abs(m).bit_length() + 63) // 64 for m, e in distinct)
        inverse_table_bytes = 8 * limbs + 4 * (len(distinct) + 1) + 5 * len(distinct)
        removed = int(self.chosen.sum())
        old_pred = hz.Ac.nnz + hz.Ab.nnz + hz.Auc.nnz + hz.Aub.nnz
        exact_pred = totals['Ac']['symbolic_after_nnz'] + totals['Auc']['symbolic_after_nnz'] + hz.Ab.nnz + hz.Aub.nnz
        native_pred = exact_pred + aux_nnz + self.consumer['predicate_carrier_replacement_nnz_delta']
        retained_rows = hz.n_eq - removed + hz.n_ineq + hz.Gc.shape[0]
        # C56 charges64 for every compact coefficient, scalar and RHS. Dropping
        # the nonnegative scalar-table term gives a valid standalone lower bound.
        full_scan_lower = 64 * (exact_pred + totals['Gc']['symbolic_after_nnz'] + hz.Gb.nnz + retained_rows)
        added_entries = 2 * aux_nnz + 3 * auxiliaries
        flags = dict(chain_window=self.counts['local_window_violations'] == 0 and self.counts['composed_window_violations'] == 0,
            derived_consumer_window=self.consumer['derived_window_violations'] == 0,
            consumer_multiplier_window=multiplier_failures == 0,
            auxiliary_cap=auxiliaries <= 16384,added_entry_cap=added_entries <= 131072,
            unchanged_C56_full_scan_lower_bound_fits=full_scan_lower <= 16_000_000,
            symbolic_gauged_predicate_nnz_strict=int(native_pred) < int(old_pred))
        return dict(schema='c57_complete_raw_general_scalar_diagnostic_v1',counts=dict(self.counts),
            depth_histogram=dict(self.depth_hist),composed_bits_histogram=dict(self.bits_hist),
            composed_floor_exponent_histogram=dict(self.floor_hist),chain_identity_sha256=self.chain_hash.hexdigest(),
            consumer_counts=dict(self.consumer),consumer_group_sha256=self.consumer_hash.hexdigest(),
            matrices=totals,distinct_inverse_scalars=len(distinct),inverse_scalar_table_estimated_numeric_bytes=inverse_table_bytes,
            full_packed_inverse_map_estimated_numeric_bytes=8 * len(self.weights),
            carriers=len(self.plans),carrier_size_histogram=dict(group_size_hist),
            auxiliary_continuous=auxiliaries,auxiliary_predicate_nnz=aux_nnz,added_radix_numeric_entries=added_entries,
            consumer_multiplier_window_failures=multiplier_failures,original_n_cont=hz.n_cont,
            symbolic_native_n_cont=hz.n_cont - removed + auxiliaries,unchanged_n_binary=hz.n_bin,
            original_predicate_nnz=int(old_pred),symbolic_exact_predicate_nnz=int(exact_pred),
            symbolic_gauged_predicate_nnz=int(native_pred),unchanged_C56_full_scan_work_lower_bound=int(full_scan_lower),
            necessary_flags=flags,all_necessary_flags_pass=all(flags.values()),
            complete_physical_ledger_or_new_HZ_proved=False,full_C54_coalescing_closure_proved=False,
            actual_source_writer_native_witness_or_solver_executed=False,formal_gain=0)
