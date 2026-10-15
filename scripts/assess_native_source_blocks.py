"""Static native-replacement count and source-block representation assessment.

Only source/record reads, AST inspection and integer arithmetic. This is not an
operator implementation, candidate control, capacity execution or quota setter.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = 'docs/hz_native_source_blocks_20261016.json'
SHAPE = 'docs/hz_model_scaled_resources_20261013.json'
WORK = 'docs/hz_shared_execution_envelope_20261015.json'
PRIOR = 'docs/hz_full_path_decision_20261016.json'
NATIVE = 'act/back_end/moe/scaled_forest_endpoints.py'
FILES = (SHAPE, WORK, PRIOR, NATIVE,
    'scoped_source/shared_source.py', 'scoped_source/shared_build.py',
    'scoped_source/check_shared_source.py', 'scoped_source/scaled_references.py',
    'scoped_source/affine_references.py', 'scoped_source/source_dag.py',
    'scoped_source/direct_lifecycle.py', 'scoped_source/pair_workspace.py',
    'scoped_source/bounded_runtime.py', 'scoped_source/direct_candidates.py',
    'scoped_source/scaled_endpoints.py', 'scoped_source/direct_method.py',
    'scoped_source/forest_endpoints.py', 'act/back_end/moe/block_support.py',
    'scoped_source/direct_source_format.py',
    'act/back_end/moe/store_endpoints.py', 'scripts/analyze_full_resource_ledger.py',
    'scripts/assess_native_source_blocks.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(name):
    return json.loads((ROOT / name).read_bytes())


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def nodes(name):
    return {ast.unparse(n) for n in ast.walk(ast.parse((ROOT/name).read_text()))}


def counts(rows, coefficients, exact=0):
    return dict(row_visits=rows, coefficient_visits=coefficients, arithmetic=exact)


def analyze():
    shape, work, prior = map(load, (SHAPE, WORK, PRIOR))
    for name, expected in prior['anchors'].items():
        require(sha(name) == expected, 'prior cross-contract anchor drift: ' + name)
    v = shape['inventory']
    require(work['inventory'] == v, 'registered shape binding')
    require(v['router_widths'] == [3072,128,8] and v['expert_widths'] == [3072,256,128,10]
            and v['experts'] == 8, 'registered depth and natural first blocks')
    require('range(129)' in nodes(NATIVE) and 'step == 128' in nodes(NATIVE), 'native schedule')
    require('len(values) <= 4' in nodes('scoped_source/pair_workspace.py'), 'root cardinality')
    require('len(tail) <= 2' in nodes('scoped_source/shared_source.py'), 'complete private tails')
    n, a, u = v['inputs'], v['affine_outputs'], v['relu_units']
    N, F, L, R = (v[k] for k in ('nodes_upper','factors_upper','local_entries_upper','references_upper'))
    E, T, M = (v[k] for k in ('endpoints_upper','pair_local_tiles_upper','pair_dual_rows_upper'))
    w1, J, k = v['first_weight_positions'], 2, 8
    natural = [v['router_widths'][1]] + [v['expert_widths'][1]]*v['experts']
    s, q = sum(natural), n+2*a+12*u
    require(s*n == w1 and q+w1 == L and (E,T) == (504,84), 'complete roster')
    setup = counts(N, L+R+F+w1)
    generation = counts(3*E+T*(M+129+257*J), 4*E+T*M+E*M)
    # RootMap delegates to SparseRoots. Its successive key lookups scan
    # 1+...+h Block.get calls, in addition to map and key enumeration charges.
    lease_per_root_size = [counts(2+h, 3*h+h*(h+1)//2) for h in range(5)]
    lease = {c:E*lease_per_root_size[4][c] for c in setup}
    native_counts = {c:setup[c]+generation[c]+lease[c] for c in setup}
    native_upper = sum(native_counts.values())
    old_fixed = E*(33+2*M)
    require(native_upper == 15632268 and old_fixed == 2738232, 'reviewed replacement arithmetic')
    postcapture = {}
    for arm, row in work['consumers'].items():
        require(row['fixed_proposer_upper'] == old_fixed, 'replace, do not double count fixed producer')
        postcapture[arm] = row['postcapture_counted_work_upper']-old_fixed+native_upper
    calls = 257*T
    named_old = 48*L+24*R+8*F
    named_block = 8*w1+48*q+24*R+8*F
    return dict(schema='NATIVE_SOURCE_BLOCK_STATIC_ASSESSMENT_V1',
        based_on_head='5989618ff3c9db9c4aa164392b806c272d787cbc',
        anchors={name:sha(name) for name in FILES},
        unchanged=work['unchanged'],
        roster=dict(pairs=v['pairs'], duties=v['property_duties'], endpoints=E,
                    role_candidates=v['candidates_upper'], tiles=T, candidate_width=k, check_width=2),
        conditional_existing_operator_join=dict(
            scope='same shared Driver/frame/replay; replace fixed producer with current CPU Operator only',
            setup_upper=setup, tuple_generation_upper=generation, lease_additional_upper=lease,
            lease_per_endpoint_by_root_size=lease_per_root_size,
            producer_counted_upper=native_counts, producer_work_upper=native_upper,
            removed_fixed_producer_upper=old_fixed, delta=native_upper-old_fixed,
            postcapture_counted_upper=postcapture, nonempty_reference_levels_upper=J,
            adjoint_calls_upper=129*T, forward_calls_upper=128*T,
            future_extra_adapter_charges_included=False, old_produce_wrapper_used=False,
            implemented=False, includes_flops_or_runtime_upper=False),
        proposed_local_algebra=dict(identity='L = E_aux + S W P',
            full_operator='C = inverse(I-R) L D', natural_block_rows=natural,
            first_rows=s, first_positions=w1, aux_local_upper=q,
            first_private_tails_upper=2*s, remaining_aux_local_upper=q-2*s,
            shared_factor_ids_preserved=True, all_centers_scales_tails_refs_preserved=True,
            changes_pair_closure=False, changes_exact_consumer=False,
            numerical_trajectory_bitwise_required=False, same_candidate_exact_acceptance_required=True,
            mathematical_decomposition_not_program_proof=True, implemented=False),
        lifecycle_choices=dict(
            repeated_transcode=dict(selected_as_next_implementation=False,
                largest_block_bytes=8*max(natural)*n, position_reads_upper=calls*w1,
                source_payload_bytes_upper=8*calls*w1, natural_block_calls_upper=calls*len(natural),
                source_reads_are_physical_io=False),
            owned_bank=dict(preferred_for_further_static_contract=True, implemented=False,
                native_weights_bytes=8*w1, source_first_rows_retained_bytes=8*(w1+s),
                source_full_snapshot_bytes=8*v['parameter_values'],
                mutable_alias_into_checked_source_allowed=False,
                implied_new_resource_cap=None)),
        conditional_named_payload=dict(old_operator_tensor_formula_upper=named_old,
            proposed_operator_tensor_formula_upper=named_block, formula_difference=named_old-named_block,
            proposed_plus_reachable_source_rows=named_block+8*(w1+s), node_map_if_required=8*s,
            largest_block_transcode_two_copies=16*max(natural)*n,
            one_node_tile=8*N*k, one_factor_tile=8*F*k,
            dual_z_and_best=16*M*k, dual_indices_rhs=16*M,
            reference_select_and_product_upper=16*R*k,
            block_forward_output=8*max(natural)*k, block_adjoint_output=8*n*k,
            not_full_simultaneous_peak=True, actual_bytes=None, measured_saving=None),
        candidate_boundary=dict(native_still_returns_string_records=True,
            proposed_scalar_text_upper=k*M*342, two_role_scalar_text_upper=k*M*343,
            two_role_list_pointer_slots=2*k*M*8,
            binary64_frame_upper=128+k*(2152+16*M),
            binary_frame_does_not_remove_prior_string_records=True,
            full_binary64_stream_upper=prior['inherited_archive_upper']['binary64']['candidate_stream_upper'],
            stream_cap=prior['inherited_archive_upper']['binary64']['candidate_stream_cap']),
        missing=dict(live_capture_upper=None, proposed_row_block_work_upper=None,
            full_heap_upper=None, full_rss_upper=None, dense_gemm_workspace_upper=None,
            full_request_upper=None),
        decision='LOCAL_DECOMPOSITION_AND_CONDITIONAL_NATIVE_COUNTS_ONLY_NO_FULL_ADMISSION',
        real_source_selected=False, quota_changed=False, runtime_changed=False,
        model_runs=0, gpu_runs=0, solver_calls=0, candidate_calls=0, source_builds=0,
        full_capacity_admitted=False, scientific_goals_closed=[])


if __name__ == '__main__':
    report = analyze()
    if sys.argv[1:] == ['--check']:
        require(load(REPORT) == report, 'native source-block static record drift')
        print('PASS: static counts and conditional payloads only; no native calls')
    else:
        require(not sys.argv[1:], 'only --check supported')
        print(json.dumps(report, indent=2, sort_keys=True))
