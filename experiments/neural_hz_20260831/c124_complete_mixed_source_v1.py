"""One uniform mixed-channel rule on complete fresh ordinary nonconvex HZs.

The original C120 dense and historical masked fixtures are unchanged.  Two
additional geometrically heterogeneous/no-hit sources exercise routing; no
target model, solver, production admission or archived HZ is used here.
"""
from dataclasses import asdict
from fractions import Fraction as F
import numpy as np

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c120_complete_source_fixture_v1 import expression, lift, _binding, _receipt
from experiments.neural_hz_20260831.c123_support_word_binding_v1 import bind_actual_rows
from experiments.neural_hz_20260831.c123_source_binding_worker_v1 import state_identity
from experiments.neural_hz_20260831.c122_channel_route_v1 import route
from experiments.neural_hz_20260831.c124_mixed_f4_v1 import construct
from experiments.neural_hz_20260831.c124_mixed_f4_oracle_v1 import prove
from experiments.neural_hz_20260831.c91_physical_circuit_v1 import make, audit, extend, recover
from experiments.neural_hz_20260831.c62_local_equations_v1 import reconstruct, SCHEMA
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout, metadata


MODES = ('dense', 'masked', 'heterogeneous', 'noop')
SOURCE_RESERVES = dict(dense=32000000, masked=32000000, heterogeneous=32000000, noop=4000000)
CATEGORY_CAPS = dict(source=100000000, setup=4000000, binding=24000000,
    construction=10000000, proof=60000000, physical=24000000, ledger=24000000, evidence=8000000)
COMPLETE_WORK_UPPER = sum(CATEGORY_CAPS.values())


def source_maps(nodes, pool):
    pool.charge('c124_complete_source_geometry_and_maps',
                1024+64*len(nodes)+16*sum(int(n['width']) for n in nodes))
    matches = [n for n in nodes if n['kind'] == 'op' and type(n['op']) is ImplicitConv2DOp]
    if len(matches) != 1:
        raise ValueError('one complete ordinary convolution required')
    node = matches[0]
    op = node['op']
    if (op.input_shape != (1, 16, 6, 6) or op.output_shape != (1, 32, 4, 4)
        or op._padding != (0, 0) or op._stride != (1, 1)
        or op._dilation != (1, 1) or op._groups != 1 or len(node['parents']) != 1):
        raise ValueError('complete original ordinary geometry differs')
    parent = nodes[node['parents'][0]]
    return op._kernel, dict(
        ids=np.where(parent['needed'], parent['slots'], -1).reshape(16, 6, 6).copy(),
        powers=parent['exponents'].reshape(16, 6, 6).copy(),
        outs=np.where(node['needed'], node['slots'], -1).reshape(32, 4, 4).copy(),
        output_powers=node['exponents'].reshape(32, 4, 4).copy())


def source_route(weights, maps, binding, pool):
    """Use actual source support; no fixture name is visible to this rule."""
    ids, outs = maps['ids'], maps['outs']
    pool.charge('c124_complete_dense_injective_source_route_binding',
                1024+16*(int(weights.size)+int(ids.size)+int(outs.size)))
    if not np.all(np.isfinite(weights)) or np.any(weights == 0):
        raise ValueError('C122 dense original kernel premise not established')
    used = ids[ids >= 0]
    if len(set(map(int, used))) != len(used):
        raise ValueError('C122 raw direct count requires injective surviving parent IDs')
    inputs = np.array([sum(1 << i for i, v in enumerate(a.flat) if v >= 0)
                       for a in ids], np.uint64)
    outputs = np.array([sum(1 << i for i, v in enumerate(a.flat) if v >= 0)
                        for a in outs], np.uint16)
    report, evidence = route(inputs, outputs, int(binding['actual_direct_output_nnz']),
                             pool=pool, enabled=True)
    return report, dict(input_masks=inputs, output_masks=outputs, **evidence)


def physical(fields, packet, binding, proof, point, expected_point, budgets):
    pool = budgets['physical']
    count, nnz = int(packet['new_factors']), len(packet['native'])
    nout = len(packet['rhs'])-count
    pool.charge('c124_complete_mixed_source_cost_preflight', 1024+32*nout)
    pivots = packet['pivots'][count:]
    ranks = int(fields['old_n_eq'])+pivots.astype(np.int64)-int(fields['old_n_cont'])
    positions = fields['eq_roots'][ranks]
    old_nnz = int((fields['hz'].Ac.indptr[positions+1]-fields['hz'].Ac.indptr[positions]).sum())
    route_entries = int(packet['selected_channels'].size)
    entry_delta = 2*(nnz-old_nnz)+13*count+3*nout+8+route_entries
    emission = int(fields['report']['actual_radix_work'])+64*len(packet['rhs'])+16*(nnz+len(packet['rhs']))
    reasons = []
    if nnz >= binding['actual_direct_output_nnz']:
        reasons.append('complete_predicate_nnz_not_strictly_smaller')
    if count+len(fields['def_rows']) > 16384:
        reasons.append('whole_shared_auxiliary_cap')
    if 131072+max(0, entry_delta) > 131072:
        reasons.append('whole_shared_positive_entry_cap')
    if emission > 16000000:
        reasons.append('whole_shared_emission_cap')
    report = dict(rejection_reasons=reasons, new_factors=count, native_packet_nnz=nnz,
        complete_packet_entry_delta_including_route_mask=entry_delta,
        route_mask_bytes_and_entries=route_entries, original_radix_entry_reserve=131072,
        whole_positive_entry_reserve_used=131072+max(0, entry_delta),
        whole_shared_emission=emission, actual_source_constructed=False,
        complete_physical_reduction_proved=False, source_or_LIVE_admitted=False, formal_gain=0)
    if reasons:
        return report, None, None
    state = make(fields, [packet], _receipt(binding, pool), _receipt(proof, pool), pool=pool, enabled=True)
    # This retained route is part of the actual reachable candidate, not only
    # diagnostic evidence outside the full-state physical comparison.
    state['selected_channels'] = packet['selected_channels']
    checked = audit(fields, state, [packet], pool=pool, enabled=True)
    checked['source_box_theorem_in_this_comparison'] = (
        'C123 every original row binding plus C124 independent complete mixed native proof')
    expanded = extend(state, point, pool=pool)
    recovered = recover(state, expanded, pool=pool)
    if recovered != expected_point:
        raise ValueError('complete original inverse differs after mixed circuit extension')
    pool.charge('c124_complete_exact_inverse_point_encoding', 16*(len(expanded)+len(recovered)))
    inverse = dict(expanded=[(v.numerator, v.denominator) for v in expanded],
                   recovered=[(v.numerator, v.denominator) for v in recovered])
    before, after = numeric_layout(fields, budgets['ledger']), numeric_layout(state, budgets['ledger'])
    before_meta, after_meta = metadata(fields, budgets['ledger']), metadata(state, budgets['ledger'])
    if max(before.resident_entries, after.resident_entries) > 64000000:
        raise MemoryError('complete actual source exceeds64M entries')
    won = after.resident_bytes < before.resident_bytes and after.resident_entries < before.resident_entries
    if not won:
        reasons.append('complete_actual_numeric_storage_not_strictly_smaller')
    report.update(actual_source_constructed=True, complete_original_audit=checked,
        actual_complete_inverse_equal=True, point_is_not_a_network_witness=True,
        whole_nnz_before=int(fields['hz'].Ac.nnz), whole_nnz_after=int(state['fields']['hz'].Ac.nnz),
        before_numeric_bytes=before.resident_bytes, after_numeric_bytes=after.resident_bytes,
        before_numeric_entries=before.resident_entries, after_numeric_entries=after.resident_entries,
        numeric_byte_delta=after.resident_bytes-before.resident_bytes,
        numeric_entry_delta=after.resident_entries-before.resident_entries,
        before_metadata=before_meta, after_metadata=after_meta,
        complete_physical_reduction_proved=bool(won))
    return report, state, inverse


def complete_source(mode, budgets, held, emit, retain):
    if mode not in MODES:
        raise ValueError('unregistered complete source fixture')
    reserve = SOURCE_RESERVES[mode]
    budgets['source'].charge('c124_fresh_complete_source_reservation', reserve)
    expr = expression(c=16, k=32, h=6)
    if mode != 'dense':
        if mode == 'masked':
            removed = (np.indices((16, 6, 6)).sum(axis=0).reshape(-1) % 4) == 0
        else:
            removed = np.ones((16, 6, 6), bool)
            if mode == 'heterogeneous':
                removed[:12] = False
                removed[:, 0, 0] = False
            else:
                # All four central positions occur across channels, so every
                # original output remains nonzero while each channel is a
                # genuine sparse/no-hit route under the same structural rule.
                for channel in range(16):
                    removed[channel, 2+channel % 2, 2+(channel//2) % 2] = False
            removed = removed.reshape(-1)
        source = expr.terms[0].source
        source.Gc.data[removed] = 0
        source.Gc.eliminate_zeros()
        source.Gb.data[removed] = 0
        source.Gb.eliminate_zeros()
        source.c[removed] = 0
    before_expression = _binding(expr, budgets['setup'])
    keep = np.ones(expr.n_out, bool)
    direct = lift(expr, keep, enabled=True, max_work=reserve, max_branch_work=reserve)
    fields = direct['fields']
    if fields['hz'].n_bin != 1 or fields['hz'].n_ineq != 1:
        raise ValueError('original nonconvex binary/inequality partition lost')
    weights, maps = source_maps(direct['construction']['nodes'], budgets['setup'])
    case = dict(expression=expr, direct=direct, keep=keep, weights=weights, maps=maps)
    held[mode] = case
    before = state_identity(case, budgets['ledger'])
    binding = bind_actual_rows(fields, weights, maps, pool=budgets['binding'], enabled=True)
    bound = binding[0]
    route_report, route_evidence = source_route(weights, maps, bound, budgets['setup'])
    selected = route_evidence['selected']
    construction, packet = construct(weights, maps['ids'], maps['powers'], maps['outs'],
        maps['output_powers'], fields['hz'].n_cont, selected_channels=selected,
        pool=budgets['construction'], enabled=True)
    # Complete original state identity is checked before adding any candidate
    # proof or diagnostic payload to its retained case dictionary.
    case_report = dict(mode=mode, binding=bound, route=route_report, constructor=construction,
        original_source_reservation=reserve,
        original_generation_work_upper=int(fields['report']['total_work_upper']),
        original_nonconvex_binary_factors=fields['hz'].n_bin,
        original_inequalities=fields['hz'].n_ineq, formal_gain=0, source_or_LIVE_admitted=False)
    if packet is None:
        if np.any(selected) or not route_report['literal_noop']:
            raise ValueError('empty route did not remain a literal no-op')
        state, inverse, proof = fields, None, None
        physical_report = dict(literal_noop=True, exact_original_object_retained=True,
            complete_physical_reduction_proved=False, formal_gain=0)
    else:
        retain(mode, 'constructed', packet, construction)
        proof = prove(packet, weights, maps['ids'], maps['powers'], maps['outs'],
            maps['output_powers'], fields['hz'].n_cont, selected_channels=selected,
            pool=budgets['proof'], enabled=True)
        budgets['physical'].charge('c124_complete_original_inverse_reference',
            32*fields['hz'].n_cont+128*int(np.count_nonzero(fields['eq_roots'] < 0)))
        point = [F((i % 5)-2, 8) for i in range(fields['hz'].n_cont)]
        expected = reconstruct(point, fields['eq_roots'], fields['eq_scales'],
            old_n_cont=fields['old_n_cont'], old_n_eq=fields['old_n_eq'],
            n_cont=fields['hz'].n_cont, schema=SCHEMA)
        physical_report, state, inverse = physical(fields, packet, bound, proof, point, expected, budgets)
        budgets['physical'].charge('c124_complete_original_point_evidence', 16*(len(point)+len(expected)))
        case_report['reference_points'] = dict(original=[(v.numerator, v.denominator) for v in point],
            expected=[(v.numerator, v.denominator) for v in expected])
    after = state_identity(case, budgets['ledger'])
    if before != after or _binding(expr, budgets['setup']) != before_expression:
        raise ValueError('complete original source/frame/owners/maps/expression changed')
    case.update(binding=binding, routing=route_evidence, packet=packet, state=state,
                inverse=inverse, proof=proof, original_identity=before)
    case_report.update(proof=proof, physical=physical_report, original_source_unchanged=True,
                       original_numeric_identity=before)
    emit(dict(event='complete_mixed_source_checked', mode=mode,
              selected_channels=int(np.count_nonzero(selected)), physical=physical_report))
    return case_report, case
