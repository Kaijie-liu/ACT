"""Authenticate a complete current native relation bank, then optionally use it.

Both public operations take only the actual quiescent SparseHZono and a hard
caller pool. No caller-provided bank, owner id, seal, external bound or graph
certificate can replace authentication. The extended template is identical
to D088/D064, but canonical CSR permits linear ordered extraction: each
candidate EQ is converted once, and each retained g/q is materialized once
without a per-row coefficient dictionary or support sort. Only actual catalog,
parent-bank and observation-index sorts incur logarithmic charges.

The complete schema, every LE/EQ, all original bits, duplicates and ambiguous
templates are accounted for. Guard/equation ambiguity suppresses that bit as
in D088; different retained gates sharing eta reject the whole plan as in
D093. An unsupported/absent relation is not SAFE, a deleted gate or a gain.
The global catalog has no 65536 aggregate cap: selected rows and each planned
relay retain the frozen local guards and 512-bit exact arithmetic.

Dependencies mean native eta incidence, not certified ONNX temporal edges.
Every source bank is returned, including those with no successor. The append
entry builds afresh and applies every applicable bank using D097's private
mathematical body and this call's common actual g/q, never raw re-extraction.
It preserves H0 and has no partial-success return. The allocator must not
resume after this closed-bank operation without a separate lifecycle bridge.

All scans, conversions, output populations, real sorts and installations are
prepaid through the same pool. Local accounting does not certify all live
Python roots, RSS, timing, model/decoder provenance, GPU or formal solves.
"""

from dataclasses import dataclass, fields
from fractions import Fraction
from types import MappingProxyType

from experiments.neural_hz_20260831.definition_first_20260928.d097_certified_observation_relay_20261001 import certified_relay as cr

nb, nt, pb = cr.nb, cr.nt, cr.pb
NativeAffine, Graph, KernelError = nb.NativeAffine, nb.Graph, nb.KernelError
MAX_BITS, MAX_SUPPORT = nb.MAX_BITS, nb.MAX_SUPPORT
ZERO, ONE, TWO = Fraction(0), Fraction(1), Fraction(2)


@dataclass(frozen=True)
class NativeRelation:
    graph: Graph
    preactivation: NativeAffine
    readout: NativeAffine
    footprint: int


@dataclass(frozen=True)
class NativeRelayGroup:
    parents: tuple
    consumers: tuple
    observations: tuple
    next_consumers: tuple


@dataclass(frozen=True)
class NativeBank:
    relations: tuple
    groups: tuple
    dependencies: tuple
    summary: object


@dataclass(frozen=True)
class RelayReceipt:
    old_n_cont: int
    base_n_cont: int
    base_row_count: int
    shared_columns: tuple
    base_residual_bindings: tuple
    observation_bindings: tuple
    observation_readouts: tuple
    observation_slots: tuple
    child_slot_bounds: tuple
    slot_bounds: tuple
    next_residual_bindings: tuple
    next_slot_bounds: tuple
    observation_row_ranges: tuple
    next_upper_row_indices: tuple
    exact_rows: tuple
    row_errors: tuple
    installed_rhs: tuple
    extra_residual_count: int
    physical_bytes: dict
    nnz: dict
    local_cost: dict

    @property
    def rows_added(self):
        return len(self.exact_rows)


@dataclass(frozen=True)
class AppliedNativeBank:
    hz: nb.SparseHZono
    bank: NativeBank
    receipts: tuple
    summary: object


def _key(graph):
    return (*graph.slots, graph.eq_row, *graph.le_rows)


def _sort_charge(meter, name, count):
    # Graph keys have fixed length; matching keys are one original index.
    meter.charge(name, 128 * (count + 1) * (1 + (count + 1).bit_length()))


def _guards(hz, nbin, meter):
    population = len(hz.Auc.data) + len(hz.Aub.data) + len(hz.ub) + nbin + 1
    meter.charge('bank_complete_guard_scan', 32 * population)
    guards, duplicates, negative, positive = {}, 0, 0, 0
    for row, rhs in enumerate(hz.ub):
        if rhs != 0.0:
            continue
        cs, ce = int(hz.Auc.indptr[row]), int(hz.Auc.indptr[row + 1])
        bs, be = int(hz.Aub.indptr[row]), int(hz.Aub.indptr[row + 1])
        if ce - cs != 1 or be - bs != 1 or hz.Auc.data[cs] != -1.0:
            continue
        sign = float(hz.Aub.data[bs])
        if sign not in (-1.0, 1.0):
            continue
        bit, column = int(hz.Aub.indices[bs]), int(hz.Auc.indices[cs])
        position = 0 if sign < 0 else 1
        negative += int(position == 0)
        positive += int(position == 1)
        pair = guards.setdefault(bit, ({}, {}))
        if column in pair[position]:
            duplicates += 1
        else:
            pair[position][column] = row
    usable, ambiguous, missing = {}, 0, 0
    for bit in range(nbin):
        minus, plus = guards.get(bit, ({}, {}))
        if len(minus) > 1 or len(plus) > 1:
            ambiguous += 1
        elif len(minus) != 1 or len(plus) != 1:
            missing += 1
        else:
            s, negrow = next(iter(minus.items()))
            eta, posrow = next(iter(plus.items()))
            if s == eta:
                ambiguous += 1
            else:
                usable[bit] = (s, eta, negrow, posrow)
    return usable, dict(negative_guard_rows=negative, positive_guard_rows=positive,
                        duplicate_guard_rows=duplicates, missing_guard_bits=missing,
                        ambiguous_guard_bits=ambiguous)


def _relations(hz, nc, nbin, usable, meter):
    """Match all EQs linearly and construct each retained source exactly once.

    For a row, wanted maps only its eligible binary positions to their two
    guard columns. The continuous CSR is scanned once and never converted to
    a coefficient dictionary. Each wanted entry is visited at most once.
    Full Fraction EQ conversion occurs once per matched row, even if several
    original bits match that row. Literal duplicate comparisons pay for their
    actual support; the first equation needs no full comparison pass.
    """
    eq_nnz = len(hz.Ac.data) + len(hz.Ab.data)
    meter.charge('bank_complete_equality_matching',
                 64 * (eq_nnz + len(hz.b) + nbin + 1))
    first, ambiguous = {}, set()
    candidate_count = duplicate_count = converted_rows = converted_terms = 0
    for row in range(len(hz.b)):
        cs, ce = int(hz.Ac.indptr[row]), int(hz.Ac.indptr[row + 1])
        bs, be = int(hz.Ab.indptr[row]), int(hz.Ab.indptr[row + 1])
        candidates, wanted = {}, {}
        for position in range(bs, be):
            bit, lower = int(hz.Ab.indices[position]), float(hz.Ab.data[position])
            binding = usable.get(bit)
            if binding is None or lower >= 0.0:
                continue
            nb._limit(len(candidates) + 1)
            s, eta, negrow, posrow = binding
            candidates[bit] = [lower, None, None, binding]
            wanted.setdefault(s, []).append((bit, 1))
            wanted.setdefault(eta, []).append((bit, 2))
        if not candidates:
            continue
        for position in range(cs, ce):
            column, amount = int(hz.Ac.indices[position]), float(hz.Ac.data[position])
            for bit, role in wanted.get(column, ()):
                candidates[bit][role] = amount
        matched = []
        for bit, (lower, source_coefficient, eta_coefficient, binding) in candidates.items():
            if (source_coefficient == lower and eta_coefficient is not None
                    and eta_coefficient < 0.0):
                matched.append((bit, lower, eta_coefficient, binding))
        if not matched:
            continue
        support = nb._limit(ce - cs + be - bs)
        # Four literal guard coefficients participate in the same local
        # graph footprint used by D064/D088 and the relay admission guard.
        nb._limit(support + 4)
        meter.charge('bank_candidate_eq_fraction_conversion',
                     128 * (support + len(matched) + 8))
        continuous = tuple((int(hz.Ac.indices[pos]), nb._fraction(hz.Ac.data[pos]))
                           for pos in range(cs, ce))
        binary = tuple((int(hz.Ab.indices[pos]), nb._fraction(hz.Ab.data[pos]))
                       for pos in range(bs, be))
        equation = NativeAffine(nb._fraction(hz.b[row]), continuous, binary)
        converted_rows += 1
        converted_terms += support
        for bit, lower, eta_coefficient, (s, eta, negrow, posrow) in matched:
            meter.charge('bank_candidate_descriptor', 256)
            candidate_count += 1
            graph = Graph('extended', (s, eta, bit), row, (negrow, posrow))
            L, Q = nb._fraction(lower), nb._neg(nb._fraction(eta_coefficient))
            previous = first.get(bit)
            if previous is None:
                first[bit] = (graph, equation, L, Q)
            else:
                meter.charge('bank_literal_duplicate_comparison', 64 * (support + 8))
                if previous[1] == equation:
                    duplicate_count += 1
                else:
                    ambiguous.add(bit)

    meter.charge('bank_catalog_filter_headers', 128 * (len(first) + 1))
    entries = [record for bit, record in first.items() if bit not in ambiguous]
    _sort_charge(meter, 'bank_actual_catalog_sort', len(entries))
    entries.sort(key=lambda entry: _key(entry[0]))
    relations, by_graph, by_eta = [], {}, {}
    materialized_terms = 0
    for graph, equation, L, Q in entries:
        support = len(equation.continuous) + len(equation.binary)
        meter.charge('bank_ordered_relation_materialization', 128 * (support + 8))
        s, eta, bit = graph.slots
        if eta in by_eta:
            raise KernelError('ambiguous catalog eta: the complete native bank is rejected')
        # Source tuples retain their canonical order after filtering. The
        # copied source coefficients, rhs+Q and negations are exact Fractions.
        g = NativeAffine(nb._add(equation.bias, Q),
            tuple((column, nb._neg(amount)) for column, amount in equation.continuous
                  if column != s and column != eta),
            tuple((column, nb._neg(amount)) for column, amount in equation.binary if column != bit))
        q = NativeAffine(Q, ((eta, nb._neg(Q)),), ())
        nb._checked(g, nc, nbin), nb._checked(q, nc, nbin)
        if not L < ZERO < Q:
            raise KernelError('extended relation requires L<0<Q')
        relation = NativeRelation(graph, g, q, support + 4)
        relations.append(relation)
        by_graph[graph], by_eta[eta] = relation, relation
        materialized_terms += len(g.continuous) + len(g.binary)
    return tuple(relations), by_graph, by_eta, dict(
        candidate_relations=candidate_count, duplicate_equation_rows=duplicate_count,
        ambiguous_relation_bits=len(ambiguous), candidate_eq_rows_converted=converted_rows,
        candidate_eq_terms_converted=converted_terms,
        native_relation_authentications=len(relations), raw_graph_extractions=len(relations),
        generic_graph_parser_calls=0, materialized_preactivation_terms=materialized_terms)


def _dependencies_and_sources(relations, by_eta, meter):
    # Pay the relation-header walk before inspecting individual supports.
    meter.charge('bank_dependency_support_headers', 128 * (len(relations) + 1))
    population = sum(len(r.preactivation.continuous) for r in relations)
    meter.charge('bank_complete_native_dependency_scan',
                 128 * (population + len(relations) + 1))
    dependencies, incoming, grouped = [], {}, {}
    covered = uncovered = 0
    for relation in relations:
        child, g = relation.graph, relation.preactivation
        parents = []
        for column, coefficient in g.continuous:
            parent_relation = by_eta.get(column)
            if parent_relation is not None:
                parent = parent_relation.graph
                if parent.slots[2] != child.slots[2]:
                    parents.append((parent, coefficient))
                    dependencies.append((parent, child, coefficient))
        incoming[child] = tuple(parents)
        # Unique catalog eta/phase identities and ordered CSR ensure this is
        # exactly D088's first two distinct parents in original column order.
        if len(parents) < 2:
            uncovered += 1
            continue
        pair = (parents[0][0], parents[1][0])
        grouped.setdefault(pair, []).append((child,
            nb._mul(-TWO, parents[0][1]), nb._mul(-TWO, parents[1][1])))
        covered += 1
    _sort_charge(meter, 'bank_actual_parent_group_sort', len(grouped))
    sources = tuple((parents, tuple(children)) for parents, children in
                    sorted(grouped.items(), key=lambda item: tuple(_key(g) for g in item[0])))
    return tuple(dependencies), incoming, sources, covered, uncovered


def _group_metadata(group, by_graph, meter):
    """The complete D097 input/role guard, using only this call's catalog."""
    k, p, jn = len(group.consumers), len(group.observations), len(group.next_consumers)
    # These header passes precede the term-population-dependent payment.
    # The latter separately covers role tuples, source lookups and all
    # sparse-entry validation; neither pass inspects uncharged coefficients.
    meter.charge('bank_relay_admission_headers', 128 * (k + p + jn + 3))
    terms = sum(len(row) for row in group.observations)
    terms += sum(len(item[1]) for item in group.next_consumers)
    meter.charge('bank_complete_relay_admission', 128 * (k + p + jn + terms + 8))
    parameter_entries, index_entries = nb._limit(2 * k), 0
    occurrences = nb._limit(2 + k + p + jn + parameter_entries)
    for _, a, b in group.consumers:
        nb._f(a), nb._f(b)
    for row in group.observations:
        size = len(row)
        parameter_entries = nb._limit(parameter_entries + size)
        index_entries = nb._limit(index_entries + size)
        occurrences = nb._limit(occurrences + 2 * size)
        cr._sparse_row(row, k)
    for _, row in group.next_consumers:
        size = len(row)
        parameter_entries = nb._limit(parameter_entries + size)
        index_entries = nb._limit(index_entries + size)
        occurrences = nb._limit(occurrences + 2 * size)
        cr._sparse_row(row, p)
    graphs = (*group.parents, *(item[0] for item in group.consumers),
              *(item[0] for item in group.next_consumers))
    phases, etas, footprints, extracted = set(), set(), [], []
    for graph in graphs:
        relation = by_graph[graph]
        phase, eta = graph.slots[2], graph.slots[1]
        if phase in phases or eta in etas:
            raise KernelError('relay roles must have distinct original phase and eta identities')
        phases.add(phase)
        etas.add(eta)
        occurrences = nb._limit(occurrences + relation.footprint)
        footprints.append(relation.footprint)
        extracted.append((relation.preactivation, relation.readout))
    return tuple(extracted), tuple(footprints), (
        occurrences, parameter_entries, index_entries, len(graphs))


def _plans(relations, by_graph, incoming, sources, meter):
    groups, admissions = [], []
    ng, ns = len(relations), len(sources)
    excluded_count = scanned_count = insufficient = without = applicable = 0
    recipients = consumer_count = nnz_w = added_cont = added_rows = 0
    for parents, consumers in sources:
        meter.charge('bank_complete_source_group_headers', 128 * (len(consumers) + 3))
        excluded = {graph.slots[2] for graph in parents}
        child_lookup = {}
        for index, (child, _, _) in enumerate(consumers):
            if child.slots[2] in excluded:
                raise KernelError('source bank parent and child roles overlap')
            excluded.add(child.slots[2])
            child_lookup[child] = (index, by_graph[child].readout.bias)
        observations, successors = [], []
        group_nnz = 0
        meter.charge('bank_complete_recipient_headers', 128 * (ng + 1))
        for relation in relations:
            graph = relation.graph
            if graph.slots[2] in excluded:
                excluded_count += 1
                continue
            edges = incoming[graph]
            meter.charge('bank_recipient_dependency_matches', 128 * (len(edges) + 8))
            scanned_count += 1
            matches = []
            for parent, coefficient in edges:
                match = child_lookup.get(parent)
                if match is not None:
                    index, Q = match
                    weight = nb._f(nb._neg(coefficient) / Q)
                    if weight == ZERO:
                        raise KernelError('a native nonzero dependency became zero')
                    matches.append((index, weight))
            if len(matches) < 2:
                insufficient += 1
                continue
            nb._limit(len(matches))
            _sort_charge(meter, 'bank_actual_observation_index_sort', len(matches))
            matches.sort(key=lambda item: item[0])
            index = len(observations)
            observations.append(tuple(matches))
            successors.append((graph, ((index, ONE),)))
            group_nnz += len(matches)
        group = NativeRelayGroup(parents, consumers, tuple(observations), tuple(successors))
        groups.append(group)
        if not successors:
            without += 1
            admissions.append(None)
            continue
        admissions.append(_group_metadata(group, by_graph, meter))
        jn, k = len(successors), len(consumers)
        applicable += 1
        recipients += jn
        consumer_count += k
        nnz_w += group_nnz
        added_cont += 3 + 3 * k + 6 * jn
        added_rows += 12 + 9 * k + 17 * jn
    if (excluded_count + scanned_count != ng * ns
            or insufficient + recipients != scanned_count or without + applicable != ns):
        raise KernelError('complete native relay population accounting failed')
    return tuple(groups), tuple(admissions), dict(
        catalog_graphs=ng, source_groups=ns, catalog_group_visits=ng * ns,
        excluded_recipients=excluded_count, scanned_recipients=scanned_count,
        insufficient_child_matches=insufficient, applicable_groups=applicable,
        groups_without_relay=without, recipients=recipients, observations=recipients,
        consumer_occurrences=consumer_count, nnz_W=nnz_w, nnz_C=recipients,
        added_cont_upper=added_cont, added_rows_upper=added_rows)


def _build(hz, meter):
    meter.charge('bank_metadata', 128)
    nc, nbin, population = pb._headers(hz)
    meter.charge('bank_complete_native_schema', 32 * (population + nc + nbin + 1))
    nb._hz_schema(hz)
    usable, guard_summary = _guards(hz, nbin, meter)
    relations, by_graph, by_eta, relation_summary = _relations(hz, nc, nbin, usable, meter)
    dependencies, incoming, sources, covered, uncovered = _dependencies_and_sources(
        relations, by_eta, meter)
    groups, admissions, plan_summary = _plans(relations, by_graph, incoming, sources, meter)
    meter.charge('bank_complete_result_receipt',
                 128 * (len(relations) + len(groups) + len(dependencies) + 64))
    summary = dict(extended_only=True, eq_rows_scanned=hz.n_eq, le_rows_scanned=hz.n_ineq,
        graphs=len(relations), groups=len(groups), covered_children=covered,
        uncovered_children=uncovered, dependency_edges=len(dependencies),
        original_n_cont=nc, original_n_bin=nbin,
        actual_model_qualified=False, whole_physical_qualified=False,
        source_census_qualified=False, gpu_qualified=False, formal_gain=0)
    summary.update(guard_summary)
    summary.update(relation_summary)
    summary.update(plan_summary)
    summary['work_charged'] = meter.used
    return NativeBank(relations, groups, dependencies, MappingProxyType(summary)), admissions


def build_native_relation_bank(hz, *, pool, enabled=False):
    """Read the complete current HZ; authenticate and plan without adding rows."""
    if not nb.ms._on(enabled):
        return None
    meter = cr._Meter(pool)
    bank, _ = _build(hz, meter)
    return bank


def append_native_observation_bank(hz, *, pool, enabled=False):
    """Build afresh and append EVERY applicable group; never accept an old bank.

    H0 must remain quiescent throughout. Every application uses the same
    original authenticated source readouts and appends disjoint new factors.
    Receipts retain exact rows and algebraic metadata, never intermediate HZs.
    An exception yields no partially applied bank; the caller's H0 is intact.
    """
    if not nb.ms._on(enabled):
        return None
    meter = cr._Meter(pool)
    bank, admissions = _build(hz, meter)
    bank_work = meter.used
    meter.charge('bank_apply_headers', 128 * (len(bank.groups) + 1))
    current, receipts, reuses = hz, [], 0
    for group, admission in zip(bank.groups, admissions):
        if not group.next_consumers:
            continue
        if admission is None:
            raise KernelError('applicable group is missing complete admission')
        extracted, footprints, metadata = admission
        start_work = meter.used
        result = cr._relay(current, group.parents, group.consumers, group.observations,
                           group.next_consumers, extracted, footprints, metadata, meter)
        meter.charge('bank_hz_free_group_receipt',
                     128 * (len(result.exact_rows) + len(result.observation_bindings)
                            + len(result.next_residual_bindings) + 32))
        costs = dict(result.local_cost)
        costs.update(unique_birth_certificates=0, certificate_verifications=0,
                     native_relation_authentications=0, raw_graph_extractions=0,
                     native_relation_reuses=metadata[3],
                     bank_raw_graph_extractions=len(bank.relations),
                     work_charged=meter.used - start_work,
                     cumulative_work_charged=meter.used)
        values = {field.name: getattr(result, field.name) for field in fields(RelayReceipt)}
        values['local_cost'] = costs
        receipts.append(RelayReceipt(**values))
        current = result.hz
        reuses += metadata[3]
        del result
    meter.charge('bank_applied_result_receipt', 128 * (len(receipts) + 32))
    summary = dict(applied_groups=len(receipts), source_groups=len(bank.groups),
        groups_without_relay=bank.summary['groups_without_relay'],
        bank_work_charged=bank_work, relay_work_charged=meter.used - bank_work,
        work_charged=meter.used, native_relation_authentications=len(bank.relations),
        raw_graph_extractions=len(bank.relations), generic_graph_parser_calls=0,
        birth_certificate_verifications=0, reused_relation_occurrences=reuses,
        added_cont=current.n_cont - hz.n_cont, added_rows=current.n_ineq - hz.n_ineq,
        original_n_cont=hz.n_cont, original_n_bin=hz.n_bin,
        actual_model_qualified=False, whole_physical_qualified=False,
        source_census_qualified=False, gpu_qualified=False, formal_gain=0)
    if len(receipts) != bank.summary['applicable_groups'] or current.n_bin != hz.n_bin:
        raise KernelError('complete native bank application accounting failed')
    return AppliedNativeBank(current, bank, tuple(receipts), MappingProxyType(summary))
