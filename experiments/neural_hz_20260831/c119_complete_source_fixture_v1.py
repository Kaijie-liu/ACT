"""Complete ordinary nonconvex source comparison: direct, four F2 tiles, one F4.

No solver, model archive, production path or admission is reached.  Every
candidate is bound to the actual unchanged C97 source equations before its
complete SparseHZ, ownership, inverse and physical storage can be compared.
"""
from fractions import Fraction as F
import hashlib
import json
import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.test_c98_fresh_circuit_v1 import expression
from experiments.neural_hz_20260831.c97_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c7_factored_hz_v1 import expression_binding
from experiments.neural_hz_20260831.c94_raw_mask_plan_v1 import selected_source_check
from experiments.neural_hz_20260831.c95_word_filter_v1 import prepare_words
from experiments.neural_hz_20260831.c96_word_row_v1 import construct as f2_construct
from experiments.neural_hz_20260831.c90_actual_circuit_proof_v1 import prove as f2_prove
from experiments.neural_hz_20260831.c91_physical_circuit_v1 import make, audit, row, extend, recover
from experiments.neural_hz_20260831.c62_local_equations_v1 import reconstruct, SCHEMA
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout, metadata
from experiments.neural_hz_20260831.c119_denominator_f4_v1 import construct as f4_construct
from experiments.neural_hz_20260831.c119_f4_oracle_v1 import prove as f4_prove


def _binding(expr, pool):
    entries = int(expr.bias.size)
    for term in expr.terms:
        source = term.source
        entries += sum(int(getattr(source, k).size) for k in ('c', 'b', 'ub'))
        for key in ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub'):
            matrix = getattr(source, key)
            entries += int(matrix.data.size+matrix.indices.size+matrix.indptr.size)
        for op in term.operators:
            if type(op) is ImplicitConv2DOp:
                entries += int(op._kernel.size)
            elif sp.isspmatrix_csr(op):
                entries += int(op.data.size+op.indices.size+op.indptr.size)
            else:
                raise ValueError('unregistered ordinary source operator')
    pool.charge('c119_complete_original_expression_binding', 1024+4*entries)
    return expression_binding(expr)


def _whole_maps(nodes, pool):
    pool.charge('c119_complete_source_geometry_and_maps',
                1024+64*len(nodes)+16*sum(int(n['width']) for n in nodes))
    selected = [n for n in nodes if n['kind'] == 'op'
                and type(n['op']) is ImplicitConv2DOp]
    if len(selected) != 1:
        raise ValueError('exactly one ordinary complete convolution required')
    node = selected[0]
    op = node['op']
    if (op.input_shape != (1, 16, 6, 6) or op.output_shape != (1, 32, 4, 4)
            or op._padding != (0, 0) or op._stride != (1, 1)
            or op._dilation != (1, 1) or op._groups != 1
            or len(node['parents']) != 1):
        raise ValueError('registered full 6x6-to-4x4 source geometry differs')
    parent = nodes[node['parents'][0]]
    ids = np.where(parent['needed'], parent['slots'], -1).reshape(16, 6, 6).copy()
    powers = parent['exponents'].reshape(16, 6, 6).copy()
    outs = np.where(node['needed'], node['slots'], -1).reshape(32, 4, 4).copy()
    opowers = node['exponents'].reshape(32, 4, 4).copy()
    if np.any(outs < 0):
        raise ValueError('all registered source outputs must be present')
    return op._kernel, dict(ids=ids, powers=powers, outs=outs, output_powers=opowers)


def _bind_actual_rows(fields, weights, maps, pool):
    """All original native rows equal the independently mapped convolution."""
    hz = fields['hz']
    ids, powers = maps['ids'], maps['powers']
    outs, opowers = maps['outs'], maps['output_powers']
    maximum_terms = int(outs.size)*(9*int(ids.shape[0])+1)
    pool.charge('c119_all_actual_source_equations_exact_binding',
                1024+64*(3*maximum_terms+int(outs.size)))
    originals, gauges, by_pivot = [], [], {}
    direct_nnz = 0
    for k, y, x in np.ndindex(outs.shape):
        pivot = int(outs[k, y, x])
        expected = {pivot: F(2)**int(opowers[k, y, x])}
        for c, dy, dx in np.ndindex(ids.shape[0], 3, 3):
            source = int(ids[c, y+dy, x+dx])
            if source < 0:
                continue
            coefficient = F(float(weights[k, c, dy, dx]))*F(2)**int(powers[c, y+dy, x+dx])
            expected[source] = expected.get(source, F(0))-coefficient
        expected = {c: v for c, v in expected.items() if v}
        rank = int(fields['old_n_eq'])+pivot-int(fields['old_n_cont'])
        physical = int(fields['eq_roots'][rank])
        gauge = int(fields['eq_scales'][rank])
        if physical < 0:
            raise ValueError('original convolution output was removed')
        columns, values = row(hz.Ac, physical)
        actual = {int(c): F(float(v))/F(2)**gauge
                  for c, v in zip(columns, values, strict=True)}
        if (actual != expected or hz.b[physical] != 0
                or hz.Ab.indptr[physical+1] != hz.Ab.indptr[physical]):
            raise ValueError('actual original row does not equal supplied convolution')
        literal = dict(coefficients=[(int(c), float(v)) for c, v in zip(columns, values, strict=True)],
                       rhs=float(hz.b[physical]), pivot=pivot)
        by_pivot[pivot] = (literal, gauge)
        originals.append(literal)
        gauges.append(gauge)
        direct_nnz += len(columns)
    survival = selected_source_check(fields, maps, direct_nnz, pool=pool, enabled=True)
    return dict(all_actual_source_output_rows_bound=len(originals),
                actual_direct_output_nnz=int(direct_nnz),
                exact_coefficients_and_original_row_gauges_equal=True,
                source_survival=survival), originals, gauges, by_pivot


def _receipt(value, pool):
    pool.charge('c119_completed_proof_receipt_bytes_reserve', 131072)
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    if len(encoded)+1024 > 131072:
        raise MemoryError('complete proof receipt exceeds prepaid encoding reserve')
    return hashlib.sha256(encoded).digest()


def _physical(name, fields, packets, proofs, binding, point, expected_point, pool):
    """Only construct an eligible full source; retain explicit cost rejections."""
    factors = sum(int(p['new_factors']) for p in packets)
    emitted = sum(int(p['native'].size) for p in packets)
    outputs = sum(len(p['rhs'])-int(p['new_factors']) for p in packets)
    emission = int(fields['report']['actual_radix_work'])+sum(
        64*len(p['rhs'])+16*(len(p['native'])+len(p['rhs'])) for p in packets)
    pool.charge('c119_whole_source_candidate_cost_preflight',
                1024+64*len(packets)+32*outputs)
    # Unchanged C92 contract reserves the complete old radix allowance first;
    # positive increments are not financed by another packet's negative delta.
    entry_deltas = []
    for packet in packets:
        count = int(packet['new_factors'])
        pivots = packet['pivots'][count:]
        ranks = fields['old_n_eq']+pivots.astype(np.int64)-fields['old_n_cont']
        physical = fields['eq_roots'][ranks]
        direct = int((fields['hz'].Ac.indptr[physical+1]-fields['hz'].Ac.indptr[physical]).sum())
        entry_deltas.append(2*(len(packet['native'])-direct)+13*count+3*len(pivots)+8)
    positive_entries = sum(max(0, value) for value in entry_deltas)
    whole_entry_reserve = 131072+positive_entries
    reasons = []
    if emitted >= binding['actual_direct_output_nnz']:
        reasons.append('whole_predicate_nnz_not_strictly_smaller')
    if factors+len(fields['def_rows']) > 16384:
        reasons.append('whole_shared_auxiliary_cap')
    if whole_entry_reserve > 131072:
        reasons.append('whole_shared_extra_entry_cap')
    if emission > 16_000_000:
        reasons.append('whole_shared_extra_emission_cap')
    result = dict(implementation=name, native_packet_nnz=emitted, new_factors=factors,
                  replaced_output_rows=outputs, whole_shared_emission=emission,
                  original_radix_entry_reserve=131072,
                  C92_complete_packet_entry_deltas=entry_deltas,
                  positive_extra_entry_charge=positive_entries,
                  whole_positive_entry_reserve_used=whole_entry_reserve,
                  rejection_reasons=reasons, actual_source_constructed=False,
                  complete_physical_reduction_proved=False, formal_gain=0,
                  source_runtime_LIVE_admitted=False)
    if reasons:
        return result, None, None
    state = make(fields, packets, _receipt(binding, pool), _receipt(proofs, pool),
                 pool=pool, enabled=True)
    checked = audit(fields, state, packets, pool=pool, enabled=True)
    # C91's generic structural audit does not authenticate theorem receipts.
    # Here its source/box precondition is discharged by the actual-row binding
    # and, respectively, C90's full polynomial or C119's independent F4 proof.
    checked['source_box_theorem_in_this_comparison'] = (
        'C90 complete native polynomial proofs plus exact original row binding'
        if name == 'F2' else 'C119 independent compositional proof plus every actual original row binding')
    expanded = extend(state, point, pool=pool)
    recovered = recover(state, expanded, pool=pool)
    if recovered != expected_point:
        raise ValueError('complete original inverse differs after circuit extension')
    pool.charge('c119_complete_exact_inverse_point_encoding', 16*(len(expanded)+len(recovered)))
    point_evidence = dict(expanded=[(v.numerator, v.denominator) for v in expanded],
                          recovered=[(v.numerator, v.denominator) for v in recovered])
    before = numeric_layout(fields, pool)
    after = numeric_layout(state, pool)
    before_meta, after_meta = metadata(fields, pool), metadata(state, pool)
    if max(before.resident_entries, after.resident_entries) > 64_000_000:
        raise MemoryError('complete physical source exceeds 64M numeric entries')
    numeric_win = (after.resident_entries < before.resident_entries
                   and after.resident_bytes < before.resident_bytes)
    if not numeric_win:
        reasons.append('complete_actual_numeric_storage_not_strictly_smaller')
    result.update(actual_source_constructed=True, complete_original_audit=checked,
                  actual_complete_inverse_equal=True, point_is_not_a_network_witness=True,
                  whole_nnz_before=int(fields['hz'].Ac.nnz), whole_nnz_after=int(state['fields']['hz'].Ac.nnz),
                  before_numeric_bytes=int(before.resident_bytes), after_numeric_bytes=int(after.resident_bytes),
                  before_numeric_entries=int(before.resident_entries), after_numeric_entries=int(after.resident_entries),
                  numeric_byte_delta=int(after.resident_bytes-before.resident_bytes),
                  numeric_entry_delta=int(after.resident_entries-before.resident_entries),
                  before_metadata=before_meta, after_metadata=after_meta,
                  complete_physical_reduction_proved=bool(numeric_win))
    return result, state, point_evidence


def complete_source(mode, pool, *, enabled=False, observe=None, retain=None):
    """Complete source comparison; caller owns one shared whole-batch pool."""
    if not enabled:
        return None
    if mode not in ('dense', 'masked'):
        raise ValueError('registered ordinary dense or historical masked source required')
    def event(name):
        if observe is not None:
            observe(dict(event=name, mode=mode, work=int(pool.used)))
    start = pool.used
    pool.charge('c119_unchanged_complete_C97_source_reserve', 32_000_000)
    expr = expression(c=16, k=32, h=6)
    if mode == 'masked':
        source = expr.terms[0].source
        removed = (np.indices((16, 6, 6)).sum(axis=0).reshape(-1) % 4) == 0
        source.Gc.data[removed] = 0
        source.Gc.eliminate_zeros()
        source.Gb.data[removed] = 0
        source.Gb.eliminate_zeros()
        source.c[removed] = 0
    identity = _binding(expr, pool)
    keep = np.ones(expr.n_out, bool)
    direct = lift(expr, keep, enabled=True, max_work=32_000_000, max_branch_work=32_000_000)
    event('source_built')
    fields = direct['fields']
    if fields['hz'].n_bin != 1 or fields['hz'].n_ineq != 1:
        raise ValueError('ordinary nonconvex source binary or inequality lost')
    weights, maps = _whole_maps(direct['construction']['nodes'], pool)
    binding, originals, gauges, by_pivot = _bind_actual_rows(fields, weights, maps, pool)
    event('original_rows_bound')
    f4_report, f4_packet = f4_construct(weights, maps['ids'], maps['powers'], maps['outs'],
        maps['output_powers'], fields['hz'].n_cont, pool=pool, enabled=True)
    f4_packet['new_factors'] = int(f4_report['new_factors'])
    if retain is not None:
        retain('f4_constructed', f4_packet,
               dict(mode=mode, proof_status='UNPROVED', constructor=f4_report))
    f4_proof = f4_prove(f4_packet, weights, maps['ids'], maps['powers'], maps['outs'],
        maps['output_powers'], fields['hz'].n_cont, pool=pool, enabled=True)
    event('F4_proved')
    pool.charge('c119_complete_original_kernel_binary32_check', 4*int(weights.size))
    binary32 = weights.astype(np.float32)
    if not np.array_equal(binary32.astype(np.float64), weights):
        raise ValueError('unchanged F2 original filter domain differs')
    transform_report, transformed = prepare_words(binary32, pool=pool, enabled=True)
    if transformed is None:
        raise ValueError('unchanged complete F2 transform rejected')
    f2_packets, f2_reports, f2_proofs, f2_maps = [], [], [], []
    offset = 0
    for y, x in ((0, 0), (0, 2), (2, 0), (2, 2)):
        pool.charge('c119_unchanged_F2_complete_subtile_maps', 1024+16*(16*16+32*4))
        tile = dict(ids=maps['ids'][:, y:y+4, x:x+4].copy(),
                    powers=maps['powers'][:, y:y+4, x:x+4].copy(),
                    outs=maps['outs'][:, y:y+2, x:x+2].copy(),
                    output_powers=maps['output_powers'][:, y:y+2, x:x+2].copy())
        report, packet = f2_construct(transformed, tile['ids'], tile['powers'], tile['outs'],
            tile['output_powers'], fields['hz'].n_cont+offset, pool=pool, enabled=True)
        pool.charge('c119_complete_F2_native_packet_pointer_adaptation', 16*(len(packet['rhs'])+1))
        packet['new_factors'] = int(report['new_factors'])
        packet['indptr'] = packet['indptr'].astype(np.int32)
        packet['ab_indptr'] = np.zeros(len(packet['rhs'])+1, np.int32)
        if retain is not None:
            retain('f2_'+str(len(f2_packets))+'_constructed', packet,
                   dict(mode=mode, y=y, x=x, proof_status='UNPROVED', constructor=report))
        source_rows, source_gauges = zip(*(by_pivot[int(p)] for p in tile['outs'].reshape(-1)))
        proof = f2_prove(packet, source_rows, source_gauges, old_n_cont=fields['hz'].n_cont,
                         first_aux=fields['hz'].n_cont+offset, new_factors=report['new_factors'],
                         pool=pool, enabled=True)
        f2_packets.append(packet)
        f2_reports.append(dict(y=y, x=x, **report))
        f2_proofs.append(proof)
        f2_maps.append(tile)
        offset += int(report['new_factors'])
    event('F2_proved')
    pool.charge('c119_complete_original_inverse_reference',
                32*fields['hz'].n_cont+128*int(np.count_nonzero(fields['eq_roots'] < 0)))
    point = [F((i % 5)-2, 8) for i in range(fields['hz'].n_cont)]
    expected_point = reconstruct(point, fields['eq_roots'], fields['eq_scales'],
        old_n_cont=fields['old_n_cont'], old_n_eq=fields['old_n_eq'],
        n_cont=fields['hz'].n_cont, schema=SCHEMA)
    f4_physical, f4_state, f4_points = _physical('F4', fields, [f4_packet], f4_proof,
        binding, point, expected_point, pool)
    event('F4_source_audited_or_cost_rejected')
    f2_physical, f2_state, f2_points = _physical('F2', fields, f2_packets, f2_proofs,
        binding, point, expected_point, pool)
    event('F2_source_audited_or_cost_rejected')
    pool.charge('c119_complete_reference_point_encoding', 16*(len(point)+len(expected_point)))
    encoded_points = dict(original=[(v.numerator, v.denominator) for v in point],
                          expected=[(v.numerator, v.denominator) for v in expected_point])
    if _binding(expr, pool) != identity:
        raise ValueError('original expression changed during complete comparison')
    held = dict(original=expr, direct=direct, keep=keep, maps=maps,
                original_binary32_kernel=binary32,
                original_rows=originals, original_gauges=gauges,
                f4_packet=f4_packet, f2_packets=f2_packets, f2_maps=f2_maps,
                f2_transform=dict(transformed=transformed.transformed, dense=transformed.dense),
                f4_state=f4_state, f2_state=f2_state, encoded_points=encoded_points,
                f4_inverse=f4_points, f2_inverse=f2_points)
    layout = numeric_layout(held, pool)
    full_metadata = metadata(held, pool)
    event('complete_source_ledger')
    if layout.resident_entries > 64_000_000:
        raise MemoryError('all retained direct, F2, F4 and proof arrays exceed 64M')
    report = dict(mode=mode, channels=16, outputs=32, input_shape=[16, 6, 6],
                  output_shape=[32, 4, 4], historical_F2_tile_offsets=[[0, 0], [0, 2], [2, 0], [2, 2]],
                  original_expression_unchanged=True, complete_source_binding=binding,
                  old_generation_work_upper=int(fields['report']['total_work_upper']),
                  source_reserved_work=32_000_000, source_reservation_not_refunded=True,
                  f4_constructor=f4_report, f4_proof=f4_proof, f4_physical=f4_physical,
                  f2_transform=transform_report, f2_constructors=f2_reports,
                  f2_proofs=f2_proofs, f2_physical=f2_physical,
                  complete_numeric_bytes=int(layout.resident_bytes),
                  complete_numeric_entries=int(layout.resident_entries),
                  complete_metadata=full_metadata, actual_work=pool.used-start,
                  cumulative_work=pool.used, formal_gain=0, source_runtime_LIVE_admitted=False)
    return report, held
