"""Default-off sparse relay planning on a closed, quiescent original HZ.

Only the frozen D088 Discovery for this same H0 is accepted. Its seal and
owner_id prevent accidental misbinding; neither authenticates mutable content
or concurrent changes. H0 and Discovery must remain unchanged from discovery
through planning and any later application. Every catalog graph is nevertheless
reauthenticated against the complete native predicates here. This is native
algebraic binding, not ONNX topology, external-model or decoder certification.

For every original D088 parent-pair bank, EVERY catalog gate is considered.
The group's parent/child phase columns are excluded. Every other gate whose
full preactivation contains at least two distinct child eta columns receives
one sparse observation. If q_j=Q_j(1-eta_j), its coefficient is -g_eta_j/Q_j.
All remaining terms are left to the relay kernel's complete residual. No HZ
copy, solver call, state-dependent selection or candidate installation occurs.
An eta with multiple catalog meanings rejects the whole plan.

The caller's monotonically charged pool pays BEFORE every population scan,
authentication, sparse merge/sort and retained-container batch. Global catalog
support is NOT subject to the local 65536 guard. Each graph and each planned
kernel input retain their original local guards. The output row/column bounds
assume all residuals and observations need their three continuous coordinates;
they are not an application-work, nnz, memory, or timing receipt.

Before entry, planner_reserve must be included with all existing live roots
against the caller's retained-numeric-entry cap. For a hard remaining work
upper bound W its additional reserve is
64*(nc+nb+65536+catalog_count+group_count+1)+floor(W/4).
The fixed part covers graph headers, lookup containers and one checked native
row/affine conversion. A population item p saved during authentication or
matching prepays at least 128*p visits, while its Fraction numerators and
denominators, indices, graph references, sparse row copies and sorting overlap
use at most 32*p numeric entries. Repeated group/graph appearances are counted,
not silently deduplicated. Hence their retained/temporary portion is at most
W/4. Caller HZ/Discovery, archive bytes, later kernel copies and receipts are
additional roots. This bound does not prove bytes, RSS, allocator or runtime
qualification. No resource cap is increased by this helper.
"""

from dataclasses import dataclass
from fractions import Fraction

from experiments.neural_hz_20260831.definition_first_20260928.d088_native_structure_discovery_20261001 import native_discovery as nd

nb = nd.nb
Graph, KernelError = nb.Graph, nb.KernelError
MAX_BITS, MAX_SUPPORT, WORK_CAP = nd.MAX_BITS, nd.MAX_SUPPORT, nd.WORK_CAP
ZERO, ONE, TWO = Fraction(0), Fraction(1), Fraction(2)
_SEAL = object()


@dataclass(frozen=True)
class NativeRelayGroup:
    parents: tuple
    consumers: tuple
    observations: tuple
    next_consumers: tuple


@dataclass(frozen=True)
class NativeRelayPlan:
    owner_id: int
    groups: tuple
    summary: dict
    seal: object


def _binding(hz, discovered, pool):
    nd._charge(pool, 'relay_plan_binding', 128)
    if (type(discovered) is not nd.Discovery
            or discovered._seal is not nd._SEAL
            or discovered.owner_id != id(hz)
            or type(discovered.graphs) is not tuple
            or type(discovered.groups) is not tuple):
        raise KernelError('the same quiescent H0 and its frozen Discovery are required')


def planner_reserve(hz, discovered, work_limit, *, pool, enabled=False):
    """Pre-entry additional numeric-entry bound, NOT a whole-physical receipt.

    W must upper-bound all remaining work on the caller's hard pool, normally
    min(branch_remaining, whole_remaining) BEFORE this helper's own charges.
    A later smaller remaining balance cannot invalidate the bound. The caller
    admits existing roots plus this reserve before plan_native_relays.
    """
    if not nb.ms._on(enabled):
        return None
    _binding(hz, discovered, pool)
    nd._charge(pool, 'relay_plan_reserve', 64)
    if type(work_limit) is not int or not 0 <= work_limit <= WORK_CAP:
        raise KernelError('remaining hard work must be an integer within the whole cap')
    nd._metadata(hz, pool)
    return (64*(hz.Gc.shape[1]+hz.Gb.shape[1]+MAX_SUPPORT
                +len(discovered.graphs)+len(discovered.groups)+1)
            +work_limit//4)


def _catalog(hz, discovered, pool, nc, nbin):
    """All catalog authentication and identity checks, never a selected prefix."""
    count = len(discovered.graphs)
    nd._charge(pool, 'relay_plan_catalog_headers', 128*(count+1))
    extracted, supports, by_eta, bits = {}, {}, {}, set()
    previous = None
    for graph in discovered.graphs:
        # Header work is prepaid above. _graph_shape reads only row pointers
        # and fixed Graph fields; coefficient population is charged below.
        support = nb._graph_shape(graph, hz, nc, nbin)
        if graph.kind != 'extended':
            raise KernelError('only the complete D088 extended catalog is supported')
        key = nd._key(graph)
        if previous is not None and key <= previous:
            raise KernelError('catalog graphs must retain their canonical unique order')
        previous = key
        eta, bit = graph.slots[1], graph.slots[2]
        if eta in by_eta:
            raise KernelError('ambiguous catalog eta: the complete relay plan is rejected')
        if bit in bits:
            raise KernelError('catalog original phase columns must be distinct')
        population = support+16
        nd._charge(pool, 'relay_plan_authenticate_graph',
                   128*population*(1+population.bit_length()))
        g, q, error = nb._extract_graph(hz, graph)
        nb._checked(g, nc, nbin)
        nb._checked(q, nc, nbin)
        Q = nb._f(q.bias)
        if (error != ZERO or Q <= ZERO or q.binary
                or q.continuous != ((eta, nb._neg(Q)),)):
            raise KernelError('catalog graph must certify exact q=Q*(1-eta), Q>0')
        extracted[graph] = (g, Q)
        supports[graph] = support
        by_eta[eta] = graph
        bits.add(bit)
    return extracted, supports


def _group_inputs(entry, extracted, pool):
    """Validate a trusted-construction group's full bank before lookup creation."""
    nd._charge(pool, 'relay_plan_group_header', 128)
    if type(entry) is not tuple or len(entry) != 2:
        raise KernelError('immutable D088 parent/consumer group required')
    parents, consumers = entry
    if (type(parents) is not tuple or len(parents) != 2
            or type(consumers) is not tuple or not consumers):
        raise KernelError('two parents and a nonempty complete consumer bank required')
    nd._charge(pool, 'relay_plan_complete_group_inputs', 128*(len(consumers)+3))
    for parent in parents:
        if type(parent) is not Graph or parent not in extracted:
            raise KernelError('every parent must belong to the original catalog')
    if parents[0].slots[2] == parents[1].slots[2]:
        raise KernelError('parent original bits must be distinct')
    excluded = {parent.slots[2] for parent in parents}
    children, previous = {}, None
    for index, item in enumerate(consumers):
        if type(item) is not tuple or len(item) != 3:
            raise KernelError('consumer entries must retain Graph,a,b')
        child, a, b = item
        nb._f(a), nb._f(b)
        if type(child) is not Graph or child not in extracted:
            raise KernelError('every consumer must belong to the original catalog')
        key = nd._key(child)
        if previous is not None and key <= previous:
            raise KernelError('consumer bank must retain canonical unique order')
        previous = key
        if child.slots[2] in excluded:
            raise KernelError('parent and consumer original bits must all be distinct')
        excluded.add(child.slots[2])
        children[child.slots[1]] = (index, extracted[child][1])
    return parents, consumers, excluded, children


def plan_native_relays(hz, discovered, *, pool, enabled=False):
    """Return ALL applicable sparse relay groups without applying any group.

    Returned groups preserve every child in the corresponding D088 bank. A
    missing successor makes that source group inapplicable, not a gain or an
    UNSAT conclusion. Each considered recipient is counted, including those
    excluded by existing group bits and those with fewer than two matches.
    Every prospective next graph is inspected against the original H0, never
    against previously augmented predicates. No bank-application cost is
    inherited from D088, whose apply path implements a different operation.
    """
    if not nb.ms._on(enabled):
        return None
    _binding(hz, discovered, pool)
    nnz, matrix_rows, vector_entries = nd._metadata(hz, pool)
    nd._charge(pool, 'relay_plan_full_native_schema',
               16*(nnz+matrix_rows+vector_entries+1))
    nc, nbin = nb._hz_schema(hz)
    extracted, supports = _catalog(hz, discovered, pool, nc, nbin)
    ng, ns = len(discovered.graphs), len(discovered.groups)
    nd._charge(pool, 'relay_plan_population_and_summary', 256*(ng+ns+1))
    result = []
    excluded_count = scanned_count = insufficient = without = 0
    recipients = consumer_count = nnz_w = added_cont = added_rows = 0
    previous_group = None
    for entry in discovered.groups:
        parents, consumers, excluded, children = _group_inputs(entry, extracted, pool)
        key = tuple(nd._key(graph) for graph in parents)
        if previous_group is not None and key <= previous_group:
            raise KernelError('source groups must retain their canonical unique order')
        previous_group = key
        k = len(consumers)
        observations, successors = [], []
        group_nnz = 0
        # No catalog population is filtered before charging/visiting it.
        nd._charge(pool, 'relay_plan_all_recipient_headers', 128*(ng+1))
        for graph in discovered.graphs:
            if graph.slots[2] in excluded:
                excluded_count += 1
                continue
            g, _ = extracted[graph]
            population = len(g.continuous)+len(g.binary)+8
            nd._charge(pool, 'relay_plan_recipient_support_and_sparse_storage',
                       128*population*(1+(k+1).bit_length()))
            scanned_count += 1
            matches = []
            # The complete authenticated g is retained in extracted; its
            # binary/other continuous terms are deliberately NOT dropped from
            # kernel residuals. Here only child eta coordinates define W.
            for column, coefficient in g.continuous:
                matched = children.get(column)
                if matched is not None:
                    index, Q = matched
                    weight = nb._f(nb._neg(coefficient)/Q)
                    if weight == ZERO:
                        raise KernelError('a canonical native nonzero coefficient became zero')
                    matches.append((index, weight))
            if len(matches) < 2:
                insufficient += 1
                continue
            nb._limit(len(matches))
            matches.sort(key=lambda item: item[0])
            observation_index = len(observations)
            observations.append(tuple(matches))
            successors.append((graph, ((observation_index, ONE),)))
            group_nnz += len(matches)
        if not successors:
            without += 1
            continue
        jn = len(successors)
        # Match the sparse kernel's aggregate input guard per ONE group. Do
        # not impose it on the complete catalog or on the sum across groups.
        nd._charge(pool, 'relay_plan_complete_group_admission',
                   128*(k+jn+group_nnz+8))
        nb._limit(2*k+group_nnz+jn)
        nb._limit(group_nnz+jn)
        occurrences = nb._limit(2+k+jn+jn+2*k+2*group_nnz+2*jn)
        for graph in parents:
            occurrences = nb._limit(occurrences+supports[graph])
        for graph, _, _ in consumers:
            occurrences = nb._limit(occurrences+supports[graph])
        for graph, _ in successors:
            occurrences = nb._limit(occurrences+supports[graph])
        result.append(NativeRelayGroup(parents, consumers,
                                       tuple(observations), tuple(successors)))
        recipients += jn
        consumer_count += k
        nnz_w += group_nnz
        added_cont += 3+3*k+6*jn
        added_rows += 12+9*k+17*jn
    summary = dict(catalog_graphs=ng, source_groups=ns,
                   catalog_group_visits=ng*ns, excluded_recipients=excluded_count,
                   scanned_recipients=scanned_count,
                   insufficient_child_matches=insufficient,
                   applicable_groups=len(result), groups_without_relay=without,
                   recipients=recipients, observations=recipients,
                   consumer_occurrences=consumer_count, nnz_W=nnz_w,
                   nnz_C=recipients, added_cont_upper=added_cont,
                   added_rows_upper=added_rows, original_n_cont=nc,
                   original_n_bin=nbin, actual_model_qualified=0,
                   whole_physical_qualified=0, formal_gain=0)
    if (excluded_count+scanned_count != ng*ns
            or insufficient+recipients != scanned_count
            or without+len(result) != ns):
        raise KernelError('complete relay population accounting failed')
    return NativeRelayPlan(id(hz), tuple(result), summary, _SEAL)
