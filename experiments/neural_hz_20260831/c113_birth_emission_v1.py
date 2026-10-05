# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fresh original-expression generation and in-stream exact Conv circuits.

Unchanged C31 source graph, exact original encoding, budgets and UID allocation.
Only the quotient is replaced; new local-equation tags never enter old readers.
"""

from experiments.neural_hz_20260831.c106_defining_rows_v1 import bind_node
from experiments.neural_hz_20260831.c113_circuit_stream_v1 import install, matrices
from collections import Counter
from dataclasses import dataclass
import hashlib

import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.hybridz_tf.tf_cnn import _sparse_hz_storage_entries
from experiments.neural_hz_20260831.c7_factored_hz_v1 import Lifted, expression_binding, scaled_exact, restricted_row, _source_row
from experiments.neural_hz_20260831.c8_dyadic_balance_v1 import box_exponent
from experiments.neural_hz_20260831.c24_dense_graph_v1 import graph
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import Ledger, RADIX
from experiments.neural_hz_20260831.c9_radix_predicate_v1 import _row
from experiments.neural_hz_20260831.c24_dense_rows_v1 import WorkPool
from experiments.neural_hz_20260831.c107_packed_frontier_v1 import BirthTracker
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA
from experiments.neural_hz_20260831.c107_prepared_owned_rows_v1 import PreparedOwnedEncoder as OwnedRowEncoder
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import build_slabs
from experiments.neural_hz_20260831.c5_zero_suffix_audit_v1 import zero_transfer_reference
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import digest_arrays





def lift(expr, keep_rows, *, enabled=False, frame_widths=None, max_work=256_000_000,
         max_branch_work=200_000_000, max_entries=64_000_000, observe=None, before_fold=None):
    if not enabled:
        return None
    for cap, limit in ((max_work, 256_000_000), (max_branch_work, 200_000_000), (max_entries, 64_000_000)):
        if type(cap) is not int or not 0 <= cap <= limit:
            raise ValueError('invalid or increased global cap')
    frozen = expression_binding(expr)
    keep = np.asarray(keep_rows)
    if keep.dtype != np.dtype(bool) or keep.shape != (expr.n_out,):
        raise ValueError('invalid output mask')
    keep = keep.copy()
    base = zero_transfer_reference(expr)
    old_nc, old_nb = base.n_cont, base.n_bin
    if frame_widths is not None:
        if (len(frame_widths) != 2 or any(type(v) is not int for v in frame_widths)
                or frame_widths[0] < old_nc or frame_widths[1] < old_nb):
            raise ValueError('old shared frame cannot shrink')
        old_nc, old_nb = frame_widths
    nodes, root, report, owner_nodes, uid_bases = graph(expr, keep, max_work,
        uid_start=base.n_eq + base.n_ineq)
    main_aux = report['auxiliaries']
    logical_nc = old_nc + main_aux
    old_entries = sum(getattr(base, key).nnz for key in ('Ac', 'Ab', 'Auc', 'Aub')) + base.n_eq + base.n_ineq
    old_coefficients = old_entries-base.n_eq-base.n_ineq
    old_work = 12*old_entries-4*old_coefficients
    costs = []
    logical_coefficients = old_coefficients
    for node, count in zip(nodes, report['node_counts']):
        original_encoding = count['encoding_work_upper']//16*12
        removed = count['continuous_edges']+count['binary_edges']+count['auxiliaries']
        count['encoding_work_upper'] = original_encoding-4*removed
        logical_coefficients += removed
        # Keep the original conservative branch price; do not subtract the
        # full graph credit from an unrelated single branch.
        cost = original_encoding+count['support_work']
        costs.append(cost + max((costs[p] for p in node['parents']), default=0))
    report['affine_work_upper'] = report['support_work'] + sum(n['encoding_work_upper'] for n in report['node_counts'])
    report['total_work_upper'] = int(report['affine_work_upper'] + old_work)
    report['largest_branch_work_upper'] = int(max(costs)+12*old_entries)
    predicate_entries = old_entries + report['continuous_edges'] + report['binary_edges'] + 2 * main_aux
    estimate = _sparse_hz_storage_entries(base) + report['continuous_edges'] + report['binary_edges'] + 2 * main_aux + int(keep.sum()) + 131_072 + main_aux + len(nodes)
    report.update(formal_gain=0, source_count=len({id(t.source) for t in expr.terms}), terms=len(expr.terms),
        node_count=len(nodes), old_predicate_work_upper=old_work, radix_work_reserve=16_000_000,
        radix_auxiliary_reserve=16_384, radix_entry_reserve=131_072, estimated_hz_entries=estimate,
        default_off=True, input_binding_unchanged=False, live_publication_executed=False,
        encoding_operations_per_entry=None, whole_state_reduction_proved=False,
        encoding_price_rule='C24_prices_minus_post_abs_implied_finite_and_two_duplicate_power_comparisons',
        logical_coefficient_credit_preflight=int(logical_coefficients),
        original_branch_encoding_price_retained=True)
    if observe is not None:
        observe('complete_preflight', dict(report))
    if report['total_work_upper'] > max_work or report['largest_branch_work_upper'] > max_branch_work or estimate > max_entries:
        raise MemoryError('integrated complete plus reserves preflight rejected')
    if (4 * 16 * (predicate_entries + 131_072) + 256 * (main_aux + 16_384)
            + 16 * (sum(n['width'] for n in nodes) + main_aux)
            + 48 * (base.n_eq + base.n_ineq + main_aux + 16_384)
            + 96 * (logical_nc + 16_384)
            + 32 * (base.n_eq+base.n_ineq+main_aux+16_384) > 1024**3):
        raise MemoryError('integrated numeric preallocation envelope exceeds 1 GiB')
    cursor = old_nc
    for node in nodes:
        rows = np.flatnonzero(node['needed'])
        node['slots'] = np.full(node['width'], -1, dtype=np.int64)
        node['slots'][rows] = cursor + np.arange(rows.size)
        node['exponents'] = np.zeros(node['width'], dtype=np.int32)
        cursor += rows.size
    if cursor != logical_nc:
        raise ValueError('complete main-slot reserve mismatch')
    pool = WorkPool(report['total_work_upper'], report['largest_branch_work_upper'], max_work, max_branch_work, observe=observe)
    pool.charge('c67_direct_final_index_assembly',1024)
    pool.charge('c69_finite_inverse_producer_binding',128)
    pool.charge('c97_once_power_producer_binding',128)
    pool.charge('prepared_report_accounting',8*len(nodes)+32)
    pool.charge('main_metadata_frontier', 32 * main_aux)
    main_columns = np.arange(old_nc, logical_nc, dtype=np.int64)
    slabs = build_slabs(report['node_counts'], base.n_eq + base.n_ineq, pool=pool)
    pool.charge('ownership_MAIN_materialization', 3 * main_aux)
    words = np.empty(main_aux, np.int64)
    cursor = 0
    for n, owners, uid in zip(nodes, owner_nodes, uid_bases):
        coords = np.flatnonzero(n['needed'])
        np.add(owners[coords], RADIX + uid - old_nc - cursor + main_columns[cursor:cursor + len(coords)],
            out=words[cursor:cursor + len(coords)])
        cursor += len(coords)
    if cursor != main_aux:
        raise ValueError('complete MAIN ownership emission mismatch')
    del owner_nodes
    ledger = Ledger(words, old_nc, pool=pool)
    encoder = OwnedRowEncoder(logical_nc, old_nb, predicate_entries, pool=pool,
        ledger=ledger, radix_uid_base=report['radix_uid_base'])
    output_slots = nodes[root]['slots'][nodes[root]['needed']]
    tracker = BirthTracker(encoder,old_nc,base.n_eq,output_slots)
    eq_roots, eq_scales, ineq_roots, ineq_scales = [], [], [], []
    for cmat, bmat, rhs, inequality, roots, scales in (
            (base.Ac, base.Ab, base.b, False, eq_roots, eq_scales),
            (base.Auc, base.Aub, base.ub, True, ineq_roots, ineq_scales)):
        for index in range(cmat.shape[0]):
            cc, cv = _row(cmat, index)
            bc, bv = _row(bmat, index)
            uid = base.n_eq + index if inequality else index
            physical, shift = encoder.encode_uid(uid, cc, cv, bc, bv, float(rhs[index]), inequality=inequality)
            roots.append(physical)
            scales.append(shift)
    for node_index, node in enumerate(nodes):
        coordinates=np.flatnonzero(node['needed'])
        tracker.begin(node,coordinates)
        define = bind_node(node,nodes)
        for rank, coordinate in enumerate(coordinates):
            bc, bv = np.empty(0, dtype=np.int64), np.empty(0)
            constant = 0.
            if node['kind'] == 'source':
                source = node['source']
                (cc, cv), (bc, bv) = _source_row(source, coordinate)
                constant = float(source.c[coordinate])
                powers = np.zeros(cv.size, dtype=np.int64)
                summands = np.concatenate((cv, bv, [constant] if constant != 0. else []))
                unit = box_exponent(summands)
                cc = np.append(cc,node['slots'][coordinate])
                cv = np.append(-cv,1.)
                powers = np.append(powers,unit)
            else:
                cc,cv,powers,unit = define(int(coordinate),int(node['slots'][coordinate]))
            node['exponents'][coordinate] = unit
            first_physical=len(encoder.eq)
            physical, shift = encoder.encode_uid(int(uid_bases[node_index] + rank),
                cc, cv, bc, -bv, constant, cp=powers)
            eq_roots.append(physical)
            eq_scales.append(shift)
            tracker.born(int(node['slots'][coordinate]),physical,first_physical)
        tracker.end()
        if observe is not None:
            observe('encoded_node', {'node': node_index, 'kind': node['kind'], 'logical_equalities': len(eq_roots),
                'radix_auxiliaries': len(encoder.def_rows), 'radix_extra_work': encoder.extra_work})
    prepared_report = encoder.discard_unpublished_heads()
    if (prepared_report['logical_input_coefficients'] != logical_coefficients
            or prepared_report['logical_input_rows'] != base.n_eq+base.n_ineq+main_aux):
        raise ValueError('actual logical emission differs from source-derived preflight credit')
    if (prepared_report['omitted_post_finite_coefficient_checks'] < logical_coefficients
            or not prepared_report['original_input_finite_and_complete_inverse_checks_retained']):
        raise ValueError('actual finite-scan omission is not backed by complete coefficient emission')
    if (prepared_report['once_checked_logical_power_elements'] != logical_coefficients
            or not prepared_report['original_power_bounds_checked_before_signed_copy']
            or not prepared_report['generic_emit_radix_power_validation_retained']):
        raise ValueError('complete original power omission lacks its producer condition')
    report['prepared_encoding'] = prepared_report
    if before_fold is not None:
        # OFFLINE full source qualification only: the observer cannot select a
        # boundary or supply a price/receipt to the production construction.
        before_fold(encoder,nodes,root,eq_roots,eq_scales,ineq_roots,ineq_scales,tracker)
    eq_roots,eq_scales,radix_gauges,alias_report=tracker.fold(eq_roots,eq_scales,ineq_scales,observe=observe)
    report.update(alias_quotient=alias_report, work_allocation='coupled_fixed_whole_and_branch_caps',
        whole_base_work=pool.whole_base, branch_base_work=pool.branch_base,
        total_work_upper=pool.whole_base + pool.used,
        largest_branch_work_upper=pool.branch_base + pool.used)
    # Retire ONLY new construction metadata before CSR allocation. All source,
    # graph, inverse, ownership and UID proof inputs remain reachable.
    del tracker
    if observe:observe('c69_frontier_retired',{})
    stream=install(encoder,nodes,eq_roots,eq_scales,old_nc=old_nc,old_neq=base.n_eq,
        observe=observe,enabled=True)
    report.update(total_work_upper=pool.whole_base+pool.used,
        largest_branch_work_upper=pool.branch_base+pool.used,
        complete_generation_work_parts=dict(pool.parts),circuit_generation=stream.summary)
    ac, ab, b = matrices(stream,encoder.eq)
    if observe:observe('c69_EQ_CSR_assembled',{})
    auc, aub, ub = matrices(stream,encoder.ineq)
    if observe:observe('c69_all_CSR_assembled',{})
    root_node = nodes[root]
    output_rows = np.flatnonzero(root_node['needed'])
    output_values = scaled_exact(np.ones(output_rows.size), root_node['exponents'][output_rows])
    if np.any(output_values <= 1e-9) or np.any(output_values >= 1e15):
        raise ValueError('output coefficient outside unchanged native thresholds')
    nc = logical_nc + len(encoder.def_rows)+len(stream.auxiliary_records)
    gc = sp.csr_matrix((output_values, (output_rows, root_node['slots'][output_rows])), shape=(expr.n_out, nc))
    hz = SparseHZono(expr.bias.copy(), gc, sp.csr_matrix((expr.n_out, old_nb)), ac, ab, b, auc, aub, ub,
        frame_id=expr.frame_id, exact=True)
    actual = (_sparse_hz_storage_entries(hz) + words.size + slabs.size + radix_gauges.size
        +stream.auxiliary_records.size+stream.output_routes.size+stream.block_records.size)
    if actual > max_entries or actual > estimate:
        raise MemoryError('integrated final storage exceeds complete preflight')
    report.update(actual_hz_and_owner_entries=actual, ownership_words=words.size,
        ownership_numeric_bytes=words.nbytes, ownership_event_updates=ledger.updates, radix_auxiliaries=len(encoder.def_rows),
        uid_slab_entries=len(slabs), uid_slab_bytes=slabs.nbytes,
        packed_logical_rows=encoder.packed_rows, radix_relays=encoder.relays, actual_radix_work=encoder.extra_work,
        n_cont=hz.n_cont, n_bin=hz.n_bin, n_eq=hz.n_eq, n_ineq=hz.n_ineq,
        input_binding_unchanged=True, exact_power_two_coefficients=True,
        real_affine_program_identity=True, rounded_materialized_coefficient_identity_claimed=False)
    if expression_binding(expr)!=frozen:raise ValueError('complete original expression changed during generation')
    if observe:observe('c69_final_source_rebound',{})
    report.update(new_original_boundary_quotient=True,new_lineage_schema=SCHEMA,
                  source_first_native_or_LIVE_admission=False)
    fields=dict(expression=expr,hz=hz,old_n_cont=old_nc,old_n_bin=old_nb,old_n_eq=base.n_eq,
        logical_n_cont=logical_nc,keep=keep,report=report,eq_roots=np.asarray(eq_roots,np.int64),
        eq_scales=np.asarray(eq_scales,np.int64),ineq_roots=np.asarray(ineq_roots,np.int64),
        ineq_scales=np.asarray(ineq_scales,np.int64),def_rows=np.asarray(encoder.def_rows,np.int64),
        owners=words,uid_slabs=slabs,radix_gauges=radix_gauges)
    construction=dict(nodes=nodes,root=root,origin_binding=frozen,
        eq_uids=np.asarray(encoder.eq_uids,np.int64),ineq_uids=np.asarray(encoder.ineq_uids,np.int64))
    report.update(source_representation_schema='c91_complete_circuit_source_v1',
        circuit_auxiliaries=len(stream.auxiliary_records),circuit_rows_replaced=len(stream.output_routes),
        total_shared_auxiliaries=len(encoder.def_rows)+len(stream.auxiliary_records),
        declared_circuit_emission_reserve_work=stream.summary['plan']['whole_emission_work_upper'],
        fresh_generation_work_proved=True,inherited_report_scope='full fresh C98 source construction')
    state=dict(schema='c91_complete_circuit_source_v1',fields=fields,
        old_source_n_cont=stream.old_nc,old_source_n_eq=stream.old_neq,
        auxiliary_records=stream.auxiliary_records,output_routes=stream.output_routes,
        block_records=stream.block_records,original_source_proof=b'',original_circuit_proof=b'',
        native_or_LIVE_admission=False,formal_gain=0)
    construction.update(circuits=stream.packets,selected=stream.selected)
    return dict(schema='c98_fresh_circuit_source_draft_v1',state=state,
        construction=construction,independent_qualification_pending=True)
