"""Default-off exhaustive EXTENDED native-template discovery and closed-bank use.

This authenticates algebraic ReLU relations in the supplied SparseHZono, NOT
ONNX node names, temporal graph order, or all gates of an external network.
Compact templates are deliberately outside this version. Original predicates,
factors and decoder coordinates are never removed. The HZ must be quiescent
between discover and apply_groups; owner_id is an accidental-object check, not
a content fingerprint or an untrusted-certificate authentication mechanism.

pool.charge(name, amount) is the caller's existing monotonically charged pool.
Each scan, candidate authentication and allocation/sort batch is prepaid.
Full D064 schema checking is included. No D086 call occurs in discover.

For each application group, let k be its consumers, S the sum of their
preactivation support sizes plus two per consumer, and T the largest selected
graph support plus 16. A conservative D086 allocation bound is C=3+3k new
continuous columns, R=12+9k LE rows and D=26+39k+3S row nonzeros. Work charged is
32*(current_nnz+matrix_rows+vector_entries+1) plus
128*(S+32k+32)*(1+bit_length(T)). This covers repeated validation, sparse copies,
row combinations, comparisons/sorts, conversion and outward compensation;
it is a deliberately conservative scalar/container-visit bound, not timing.
All groups are planned together; no partial-prefix success is returned.

The conservative numeric-entry reserve is 32 times the maximum native bank
size (nnz, row/vector counts, widths), plus 64 times all generated row entries
and metadata. It includes overlapping native copies and retained exact-row
receipts; raw archive buffers and caller-owned roots are additional. Entry
allocation work is prepaid too. The caller must additionally enforce its
physical memory/entry cap BEFORE applying, using the discovery summary.
The returned receipts never retain intermediate HZ objects.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import scipy.sparse as sp

from experiments.neural_hz_20260831.definition_first_20260928.d086_native_shared_transfer_20261001 import native_shared_transfer as nt

nb = nt.nb
Graph, KernelError = nb.Graph, nb.KernelError
MAX_BITS, MAX_SUPPORT = nb.MAX_BITS, nb.MAX_SUPPORT
ZERO, ONE, TWO = Fraction(0), Fraction(1), Fraction(2)
_SEAL = object()


@dataclass(frozen=True)
class Discovery:
    graphs: tuple
    groups: tuple
    summary: dict
    owner_id: int
    _seal: object


@dataclass(frozen=True)
class GroupReceipt:
    old_n_cont: int
    shared_columns: tuple
    residual_bindings: tuple
    exact_rows: tuple
    row_errors: tuple
    installed_rhs: tuple
    extra_residual_count: int
    physical_bytes: dict
    nnz: dict

    @property
    def rows_added(self):
        return len(self.exact_rows)


@dataclass(frozen=True)
class Applied:
    hz: nb.SparseHZono
    receipts: tuple
    summary: dict


def _charge(pool, name, amount):
    if type(amount) is not int or amount < 0 or amount.bit_length() > MAX_BITS:
        raise KernelError('invalid work charge')
    charge = getattr(pool, 'charge', None)
    if not callable(charge):
        raise KernelError('a caller-owned charging pool is required')
    charge(name, amount)


def _metadata(hz, pool):
    """Constant-size structural preflight; no array population is read here."""
    _charge(pool, 'discovery_metadata', 128)
    if type(hz) is not nb.SparseHZono:
        raise KernelError('actual native SparseHZono required')
    matrices = (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub)
    vectors = (hz.c, hz.b, hz.ub)
    if any(type(m) is not sp.csr_matrix for m in matrices):
        raise KernelError('stored CSR matrices required')
    if any(type(v) is not np.ndarray or v.ndim != 1 for v in vectors):
        raise KernelError('stored one-dimensional vectors required')
    nnz = sum(len(m.data) for m in matrices)
    rows = sum(m.shape[0] + 1 for m in matrices)
    values = sum(len(v) for v in vectors)
    return nnz, rows, values


def _key(graph):
    return (*graph.slots, graph.eq_row, *graph.le_rows)


def _plan(hz, groups, extracted, metadata):
    """Called only after the complete grouping/plan batch has been prepaid."""
    nnz, rows, values = metadata
    width = hz.n_cont + hz.n_bin
    work, added_rows, added_nnz, added_cont = 0, 0, 0, 0
    for parents, consumers in groups:
        k = len(consumers)
        selected = (*parents, *(item[0] for item in consumers))
        largest = max(len(extracted[g][0].continuous) + len(extracted[g][0].binary)
                      for g in selected) + 16
        support = sum(len(extracted[g][0].continuous) + len(extracted[g][0].binary) + 2
                      for g, _, _ in consumers)
        # Include authentication of both parent equations even for k=1.
        parent_support = sum(len(extracted[g][0].continuous) + len(extracted[g][0].binary)
                             for g in parents)
        work += 32*(nnz+rows+values+1)
        work += 128*(support+parent_support+32*k+32)*(1+largest.bit_length())
        dc, dr, dn = 3+3*k, 12+9*k, 26+39*k+3*support
        added_cont += dc
        added_rows += dr
        added_nnz += dn
        nnz += dn
        rows += 2*dr
        values += dr
        width += dc
    entries = (32*(nnz+rows+values+width+1)
               + 64*(added_nnz+added_rows+len(groups)+1)) if groups else 0
    return dict(apply_work_upper=work, apply_numeric_entries_upper=entries,
                apply_total_charge=work+entries, added_cont_upper=added_cont,
                added_rows_upper=added_rows, added_nnz_upper=added_nnz)


def discover(hz, *, pool, enabled=False):
    """Scan every LE/EQ; return canonical native Graphs and all consumer groups.

    guards -s-z<=0 and -eta+z<=0 are indexed by original binary column.
    Identical guard duplicates use their first row. Multiple different slots
    for either guard, or multiple different authenticated equations for a bit,
    suppress that bit's graph and are counted. They never imply UNSAT.
    For each authenticated child, the first two parents in (eta,bit,slots)
    order whose eta occurs in its complete g and whose bits are distinct from
    the child and each other are selected. a=-2*g_etaA and b=-2*g_etaB multiply
    normalized parent amplitudes (1-eta)/2. Everything else remains residual.
    """
    if not nb.ms._on(enabled):
        return None
    metadata = _metadata(hz, pool)
    nnz, matrix_rows, vector_entries = metadata
    _charge(pool, 'discovery_full_native_schema',
            16*(nnz+matrix_rows+vector_entries+1))
    nc, nbin = nb._hz_schema(hz)
    le_nnz = len(hz.Auc.data)+len(hz.Aub.data)
    eq_nnz = len(hz.Ac.data)+len(hz.Ab.data)
    _charge(pool, 'discovery_all_guards', 32*(le_nnz+len(hz.ub)+nbin+1))
    guards, duplicate_guards, negative_guards, positive_guards = {}, 0, 0, 0
    for row, rhs in enumerate(hz.ub):
        if rhs != 0.0:
            continue
        cs, ce = int(hz.Auc.indptr[row]), int(hz.Auc.indptr[row+1])
        bs, be = int(hz.Aub.indptr[row]), int(hz.Aub.indptr[row+1])
        if ce-cs != 1 or be-bs != 1 or hz.Auc.data[cs] != -1.0:
            continue
        sign = float(hz.Aub.data[bs])
        if sign not in (-1.0, 1.0):
            continue
        bit, column = int(hz.Aub.indices[bs]), int(hz.Auc.indices[cs])
        position = 0 if sign < 0 else 1
        if position == 0:
            negative_guards += 1
        else:
            positive_guards += 1
        pair = guards.setdefault(bit, ({}, {}))
        if column in pair[position]:
            duplicate_guards += 1
        else:
            pair[position][column] = row
    usable, ambiguous_guards, missing_guards = {}, 0, 0
    for bit in range(nbin):
        minus, plus = guards.get(bit, ({}, {}))
        if len(minus) > 1 or len(plus) > 1:
            ambiguous_guards += 1
        elif len(minus) != 1 or len(plus) != 1:
            missing_guards += 1
        else:
            s, negrow = next(iter(minus.items()))
            eta, posrow = next(iter(plus.items()))
            if s == eta:
                ambiguous_guards += 1
            else:
                usable[bit] = (s, eta, negrow, posrow)

    _charge(pool, 'discovery_all_equalities', 32*(eq_nnz+len(hz.b)+nbin+1))
    relations, ambiguous_relations = {}, set()
    candidates, duplicate_equations, auth_support = 0, 0, 0
    for row in range(len(hz.b)):
        cs, ce = int(hz.Ac.indptr[row]), int(hz.Ac.indptr[row+1])
        bs, be = int(hz.Ab.indptr[row]), int(hz.Ab.indptr[row+1])
        continuous = {int(hz.Ac.indices[p]): float(hz.Ac.data[p]) for p in range(cs, ce)}
        for position in range(bs, be):
            bit = int(hz.Ab.indices[position])
            if bit not in usable:
                continue
            s, eta, negrow, posrow = usable[bit]
            lower = float(hz.Ab.data[position])
            if (lower >= 0.0 or continuous.get(s) != lower
                    or continuous.get(eta, 0.0) >= 0.0):
                continue
            population = ce-cs+be-bs+8
            _charge(pool, 'discovery_authenticate_candidate',
                    128*population*(1+population.bit_length()))
            graph = Graph('extended', (s, eta, bit), row, (negrow, posrow))
            nb._graph_shape(graph, hz, nc, nbin)
            g, q, error = nb._extract_graph(hz, graph)
            nb._checked(g, nc, nbin), nb._checked(q, nc, nbin)
            if error != ZERO:
                raise KernelError('nonzero error in extended graph authentication')
            equation = nb._row(hz.Ac, hz.Ab, row, nb._fraction(hz.b[row]))
            candidates += 1
            auth_support += population
            previous = relations.get(bit)
            if previous is None:
                relations[bit] = (graph, g, q, equation)
            elif previous[3] == equation:
                duplicate_equations += 1
            else:
                ambiguous_relations.add(bit)

    graph_count = len(relations)
    _charge(pool, 'discovery_catalog_group_and_plan',
            128*(auth_support+graph_count+1)*(1+(graph_count+1).bit_length()))
    entries = [item for bit, item in relations.items() if bit not in ambiguous_relations]
    entries.sort(key=lambda item: _key(item[0]))
    graphs = tuple(item[0] for item in entries)
    extracted = {item[0]: (item[1], item[2]) for item in entries}
    by_eta = {}
    for graph in sorted(graphs, key=lambda g: (g.slots[1], g.slots[2], _key(g))):
        by_eta.setdefault(graph.slots[1], []).append(graph)
    grouped, uncovered, covered = {}, 0, 0
    # Each child is visited once. Every matching parent is still scanned; the
    # canonical prefix is a structural choice, never a relaxation-status menu.
    for child in graphs:
        g = extracted[child][0]
        possible = sum(len(by_eta.get(column, ())) for column, _ in g.continuous)
        _charge(pool, 'discovery_child_parent_matches', 32*(possible+len(g.continuous)+1))
        chosen, used_bits = [], {child.slots[2]}
        for column, amount in g.continuous:
            for parent in by_eta.get(column, ()):
                bit = parent.slots[2]
                if bit not in used_bits and len(chosen) < 2:
                    chosen.append((parent, nb._mul(-TWO, amount)))
                    used_bits.add(bit)
        if len(chosen) < 2:
            uncovered += 1
            continue
        parents = (chosen[0][0], chosen[1][0])
        grouped.setdefault(parents, []).append((child, chosen[0][1], chosen[1][1]))
        covered += 1
    groups = tuple((parents, tuple(consumers)) for parents, consumers in
                   sorted(grouped.items(), key=lambda item: tuple(_key(g) for g in item[0])))
    summary = dict(extended_only=1, eq_rows_scanned=len(hz.b), le_rows_scanned=len(hz.ub),
                   negative_guard_rows=negative_guards, positive_guard_rows=positive_guards,
                   duplicate_guard_rows=duplicate_guards, missing_guard_bits=missing_guards,
                   ambiguous_guard_bits=ambiguous_guards,
                   ambiguous_relation_bits=len(ambiguous_relations),
                   candidate_relations=candidates, duplicate_equation_rows=duplicate_equations,
                   graphs=len(graphs), groups=len(groups), covered_children=covered,
                   uncovered_children=uncovered, original_n_cont=nc, original_n_bin=nbin,
                   discovery_numeric_entries_upper=32*(nnz+matrix_rows+vector_entries+nc+nbin
                                                       +auth_support+candidates+1))
    summary.update(_plan(hz, groups, extracted, metadata))
    return Discovery(graphs, groups, summary, id(hz), _SEAL)


def apply_groups(hz, discovered, *, pool, enabled=False):
    """Append the COMPLETE discovered bank on a closed, quiescent native HZ.

    A Discovery is trusted output of this module, not an external proof format.
    No frame allocator may resume after this terminal/snapshot-only operation.
    A failure returns no transformed HZ. Caller original arrays remain intact.
    Caller must check summary['apply_numeric_entries_upper'] against its own
    physical cap before entry; charging is not a substitute for that cap.
    """
    if not nb.ms._on(enabled):
        return None
    _charge(pool, 'apply_binding_preflight', 32)
    if (type(discovered) is not Discovery or discovered._seal is not _SEAL
            or discovered.owner_id != id(hz)):
        raise KernelError('discovery must belong to this same quiescent HZ')
    _charge(pool, 'apply_entire_bank_work_and_allocation',
            discovered.summary['apply_total_charge'])
    current, receipts = hz, []
    for parents, consumers in discovered.groups:
        result = nt.append_shared_upper(current, parents, consumers, enabled=True)
        receipts.append(GroupReceipt(result.old_n_cont, result.shared_columns,
                                     result.residual_bindings, result.exact_rows,
                                     result.row_errors, result.installed_rhs,
                                     result.extra_residual_count, result.physical_bytes,
                                     result.nnz))
        current = result.hz
        # result would otherwise keep the preceding result's HZ until the next
        # assignment; receipts intentionally carry no HZ/frame reference.
        del result
    summary = dict(groups_applied=len(receipts), consumers_applied=discovered.summary['covered_children'],
                   added_cont=current.n_cont-hz.n_cont, added_rows=current.n_ineq-hz.n_ineq,
                   original_bits_preserved=int(current.n_bin == hz.n_bin),
                   apply_work_upper=discovered.summary['apply_work_upper'],
                   apply_numeric_entries_upper=discovered.summary['apply_numeric_entries_upper'])
    return Applied(current, tuple(receipts), summary)
