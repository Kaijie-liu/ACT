"""Scalar-only, default-off certificate for C97's existing node-local omissions.

This does not reduce any tariff of an operation that is still executed. C97
already proves the four omitted comparisons per original logical coefficient
for its whole-graph bound. Here the same producer counters are partitioned at
every original node, permitting a fresh maximum-over-parents branch proof.

The new tariff is 2048 + 512*N + 64*P, where P counts ALL parent incidences,
including repeated parents. It is included in BOTH conservative source bases.
The complete-source caller prepays its full source reservation before lift;
this planner is not an executed coupled-pool charge. Its header check only
establishes that the proof fee itself fits both caps before parent walks. The
complete graph-plus-proof bases are checked after planning, before emission.
The fixed/node terms cover metadata checks, both old/new recurrences at planning
and final validation, path records, four-counter snapshots/deltas, and the
predicate partition; these are composite scalar charges, not arithmetic-only
counts. They also cover the small temporary dictionaries and comparison walks.
the incidence term covers all parent validation/copy/recurrence traversals.
External uses of verify_certificate must separately budget their own reads.
"""

HEADER_FEE = 2048
NODE_FEE = 512
PARENT_FEE = 64
COUNTERS = ('logical_rows', 'logical_coefficients',
            'once_checked_logical_power_elements',
            'omitted_post_magnitude_elements')
COUNT_FIELDS = ('width', 'auxiliaries', 'continuous_edges', 'binary_edges',
                'center_edges', 'support_work')
SCHEMA = 'c130_complete_nodewise_original_emission_v1'


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError('invalid nonnegative integer: ' + name)
    return value


def _enabled(enabled):
    if type(enabled) is not bool:
        raise ValueError('enabled must be an explicit Boolean')
    return enabled


def _path(records, end, parent_key):
    result = []
    while end is not None:
        result.append(end)
        end = records[end][parent_key]
    result.reverse()
    return result


def _recurrence(records):
    old_costs, new_costs = [], []
    for index, record in enumerate(records):
        parents = record['parents']
        old_parent = max(parents, key=lambda p: old_costs[p]) if parents else None
        new_parent = max(parents, key=lambda p: new_costs[p]) if parents else None
        old_cost = record['original_encoding_work_upper'] + record['support_work']
        new_cost = record['encoding_work_upper'] + record['support_work']
        old_cost += 0 if old_parent is None else old_costs[old_parent]
        new_cost += 0 if new_parent is None else new_costs[new_parent]
        old_costs.append(old_cost)
        new_costs.append(new_cost)
        record.update(old_path_parent=old_parent, new_path_parent=new_parent,
                      old_path_work=old_cost, new_path_work=new_cost)
    old_end = max(range(len(records)), key=lambda p: old_costs[p])
    new_end = max(range(len(records)), key=lambda p: new_costs[p])
    return (old_costs[old_end], new_costs[new_end],
            _path(records, old_end, 'old_path_parent'),
            _path(records, new_end, 'new_path_parent'))


def plan(nodes, node_counts, old_entries, old_rows, *, enabled=False,
         max_work=256_000_000, max_branch_work=200_000_000):
    """Plan from the unchanged graph's pre-C97 (16/entry) counts.

    False is a literal no-inspection return. The header-only N/P fee envelope
    is checked before visiting parent elements or allocating node records.
    This planner has no authority to skip or replace original emission checks.
    """
    if not _enabled(enabled):
        return None
    for value, limit, name in ((max_work, 256_000_000, 'whole cap'),
                               (max_branch_work, 200_000_000, 'branch cap')):
        if _integer(value, name) > limit:
            raise ValueError('increased original cap')
    if (type(nodes) not in (list, tuple) or not nodes
            or type(node_counts) not in (list, tuple)
            or len(nodes) != len(node_counts)):
        raise ValueError('incomplete graph/count population')
    # len(parent-list) is a metadata-only quantity; no parent value is read yet.
    parent_entries = 0
    for node in nodes:
        if type(node) is not dict or type(node.get('parents')) not in (list, tuple):
            raise ValueError('missing complete parent-list header')
        parent_entries += len(node['parents'])
    proof_work = HEADER_FEE + NODE_FEE*len(nodes) + PARENT_FEE*parent_entries
    if proof_work > max_work or proof_work > max_branch_work:
        raise MemoryError('nodewise certificate prepayment exceeds original caps')
    old_entries = _integer(old_entries, 'old predicate entries')
    old_rows = _integer(old_rows, 'old predicate rows')
    if old_rows > old_entries:
        raise ValueError('old predicate rows exceed complete entries')
    records = []
    for index, (node, count) in enumerate(zip(nodes, node_counts)):
        if type(count) is not dict or node.get('kind') not in ('source', 'op', 'sum'):
            raise ValueError('invalid original node/count record')
        if count.get('kind') != node['kind'] or count.get('width') != node.get('width'):
            raise ValueError('node/count identity or width differs')
        values = {key: _integer(count[key], key) for key in COUNT_FIELDS}
        if values['auxiliaries'] > values['width'] or values['center_edges'] > values['auxiliaries']:
            raise ValueError('invalid complete original row/center counts')
        parents = []
        for parent in node['parents']:
            if type(parent) is not int or not 0 <= parent < index:
                raise ValueError('non-topological original parent')
            parents.append(parent)
        if ((node['kind'] == 'source' and parents)
                or (node['kind'] == 'op' and len(parents) != 1)
                or (node['kind'] == 'sum' and not parents)):
            raise ValueError('original node kind/parents differ')
        coefficients = values['continuous_edges'] + values['binary_edges'] + values['auxiliaries']
        entries = coefficients + values['center_edges']
        if _integer(count['encoding_work_upper'], 'original encoding') != 16*entries:
            raise ValueError('graph count is not the unchanged original 16/entry plan')
        records.append(dict(node=index, kind=node['kind'], parents=parents,
            **values, planned_rows=values['auxiliaries'],
            planned_coefficients=coefficients, planned_power_elements=coefficients,
            minimum_post_omissions=coefficients,
            original_encoding_work_upper=12*entries,
            encoding_work_upper=12*entries-4*coefficients,
            before=None, after=None, observed=None))
    old_max, new_max, old_path, new_path = _recurrence(records)
    whole = sum(r['encoding_work_upper'] + r['support_work'] for r in records)
    whole += 12*old_entries - 4*(old_entries-old_rows)
    result = dict(schema=SCHEMA, proof_work=proof_work, node_count=len(records),
        parent_entries=parent_entries, old_predicate_entries=old_entries,
        old_predicate_rows=old_rows, old_predicate_branch_work=12*old_entries,
        old_predicate_whole_work=12*old_entries-4*(old_entries-old_rows),
        whole_base_without_proof_fee=whole,
        old_branch_base_without_proof_fee=old_max+12*old_entries,
        new_branch_base_without_proof_fee=new_max+12*old_entries,
        old_maximizing_path=old_path, new_maximizing_path=new_path,
        nodes=records, predicate_before=None, predicate_after=None,
        predicate_observed=None, observed_node_count=0,
        complete_original_emission_proved=False,
        all_parent_incidences_retained=True, coupled_work_unchanged=True,
        old_predicate_branch_price_retained=True,
        runtime_speedup_claimed=False, formal_gain=0)
    if whole+proof_work > max_work or result['new_branch_base_without_proof_fee']+proof_work > max_branch_work:
        raise MemoryError('complete nodewise bases exceed original caps')
    return result


def snapshot(encoder):
    """Read the actual fresh PreparedOwnedEncoder only at an idle row boundary."""
    if encoder.pending_uid is not None or encoder.in_auxiliary or encoder.heads_discarded:
        raise ValueError('node certificate outside original unpublished row boundary')
    return {key: _integer(getattr(encoder, key), key) for key in COUNTERS}


def _counter_dict(values):
    if type(values) is not dict or set(values) != set(COUNTERS):
        raise ValueError('incomplete original four-counter snapshot')
    return {key: _integer(values[key], key) for key in COUNTERS}


def _delta(before, after, rows, coefficients):
    # Both callers have already validated/copied these four-counter records.
    delta = {key: after[key]-before[key] for key in COUNTERS}
    if (delta['logical_rows'] != rows
            or delta['logical_coefficients'] != coefficients
            or delta['once_checked_logical_power_elements'] != coefficients
            or delta['omitted_post_magnitude_elements'] < coefficients):
        raise ValueError('actual node emission does not prove its complete local credit')
    return delta


def observe_predicates(certificate, before, after):
    if certificate['predicate_observed'] is not None or certificate['observed_node_count']:
        raise ValueError('old predicate partition must be observed exactly once first')
    before, after = _counter_dict(before), _counter_dict(after)
    if any(before.values()):
        raise ValueError('original predicate producer is not fresh')
    observed = _delta(before, after, certificate['old_predicate_rows'],
                      certificate['old_predicate_entries']-certificate['old_predicate_rows'])
    certificate.update(predicate_before=before, predicate_after=after,
                       predicate_observed=observed)


def observe_node(certificate, index, before, after):
    if (type(index) is not int or index != certificate['observed_node_count']
            or not 0 <= index < certificate['node_count']
            or certificate['predicate_observed'] is None):
        raise ValueError('missing, duplicate or reordered original node observation')
    before, after = _counter_dict(before), _counter_dict(after)
    previous = certificate['predicate_after'] if index == 0 else certificate['nodes'][index-1]['after']
    if before != previous:
        raise ValueError('unattributed original emission between node boundaries')
    record = certificate['nodes'][index]
    observed = _delta(before, after, record['planned_rows'], record['planned_coefficients'])
    record.update(before=before, after=after, observed=observed)
    certificate['observed_node_count'] += 1


def finish(certificate, prepared_report):
    if certificate['observed_node_count'] != certificate['node_count']:
        raise ValueError('incomplete original node emission partition')
    last = certificate['nodes'][-1]['after']
    expected = dict(logical_rows=prepared_report['logical_input_rows'],
        logical_coefficients=prepared_report['logical_input_coefficients'],
        once_checked_logical_power_elements=prepared_report['once_checked_logical_power_elements'],
        omitted_post_magnitude_elements=prepared_report['actual_omitted_post_magnitude_coefficients'])
    if last != _counter_dict(expected):
        raise ValueError('global original prepared report differs from node partition')
    certificate['complete_original_emission_proved'] = True


def verify_certificate(report, nodes_metadata=None):
    """Re-derive the complete saved scalar certificate; no numeric imports/reads.

    This verifies a certificate's accounting, not the truth of unauthenticated
    external scalar receipts. Runtime emission creates it; the full source
    qualification separately authenticates source arrays and saved metadata.
    """
    certificate = report['nodewise_branch_certificate']
    if type(certificate) is not dict or certificate.get('schema') != SCHEMA:
        raise ValueError('missing nodewise original-emission schema')
    records = certificate['nodes']
    if type(records) is not list or not records or len(records) != certificate['node_count']:
        raise ValueError('incomplete node certificate population')
    if nodes_metadata is not None and len(nodes_metadata) != len(records):
        raise ValueError('source node metadata population differs')
    counts, nodes = [], []
    if len(report['node_counts']) != len(records):
        raise ValueError('report node population differs')
    for index, record in enumerate(records):
        if record['node'] != index:
            raise ValueError('node certificate identity reordered')
        node = dict(kind=record['kind'], width=record['width'], parents=record['parents'])
        if nodes_metadata is not None:
            actual = nodes_metadata[index]
            if (actual['kind'] != node['kind'] or list(actual['parents']) != node['parents']
                    or ('width' in actual and actual['width'] != node['width'])
                    or ('support_work' in actual and actual['support_work'] != record['support_work'])):
                raise ValueError('original source node/parent metadata differs')
        count = dict(kind=record['kind'], **{key: record[key] for key in COUNT_FIELDS})
        count['encoding_work_upper'] = 16*(record['continuous_edges']+record['binary_edges']
                                         +record['center_edges']+record['auxiliaries'])
        expected_report_count = dict(count, encoding_work_upper=record['encoding_work_upper'])
        if report['node_counts'][index] != expected_report_count:
            raise ValueError('saved full node count differs from certificate')
        nodes.append(node)
        counts.append(count)
    rebuilt = plan(nodes, counts, certificate['old_predicate_entries'],
                   certificate['old_predicate_rows'], enabled=True)
    observe_predicates(rebuilt, certificate['predicate_before'], certificate['predicate_after'])
    for index, record in enumerate(records):
        observe_node(rebuilt, index, record['before'], record['after'])
    finish(rebuilt, report['prepared_encoding'])
    if rebuilt != certificate:
        raise ValueError('complete nodewise accounting/observation certificate differs')
    prepared = report['prepared_encoding']
    if (prepared['omitted_post_finite_coefficient_checks'] < prepared['logical_input_coefficients']
            or prepared['once_checked_logical_power_elements'] != prepared['logical_input_coefficients']
            or prepared['duplicate_comparison_credit_per_logical_coefficient'] != 2
            or not all(prepared[key] is True for key in (
                'original_input_finite_and_complete_inverse_checks_retained',
                'generic_RHS_finite_check_retained',
                'original_power_bounds_checked_before_signed_copy',
                'generic_emit_radix_power_validation_retained'))):
        raise ValueError('unchanged complete C97 producer guards are missing')
    fee = certificate['proof_work']
    whole = certificate['whole_base_without_proof_fee'] + fee
    branch = certificate['new_branch_base_without_proof_fee'] + fee
    if (report['whole_base_work'] != whole or report['branch_base_work'] != branch
            or report['original_branch_encoding_price_retained'] is not False
            or report['nodewise_original_emission_credit_proved'] is not True
            or report['old_predicate_work_upper'] != certificate['old_predicate_whole_work']
            or report['support_work'] != sum(r['support_work'] for r in records)
            or report['affine_work_upper'] != sum(r['encoding_work_upper']+r['support_work'] for r in records)):
        raise ValueError('complete saved bases/flags differ from local proof')
    coupled = _integer(report['alias_quotient']['coupled_extra_work'], 'complete coupled work')
    if (report['total_work_upper'] != whole+coupled
            or report['largest_branch_work_upper'] != branch+coupled):
        raise ValueError('complete unchanged coupled work omitted from a cap')
    return True
