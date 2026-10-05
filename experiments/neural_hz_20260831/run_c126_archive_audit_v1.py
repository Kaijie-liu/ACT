"""Saved-JSON/XML and full-file hash audit of ordinary C126 qualification.

No numerical array is loaded, no source is restored, and no proof or test is
rerun.  Reuse the frozen supervisor's complete saved-evidence checker.  C125
stays qualified only for ordinary HZ; C124 stays authoritatively failed.
"""
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c125_exact_json_v1 import JsonAllowance, publish_bytes
from experiments.neural_hz_20260831.run_c126_mixed_source_supervisor_v1 import verify_saved_evidence
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256

EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c126_support_word_20260927_v1'
EXIT_SHA256 = '4ba7922145351478584cedb3666d03b1d2ec0668346adf746c2da33af615b573'
RESULT_SHA256 = '2d8f8c40cab47fccd8f150bf3c228ca382e12896fe80b441b892e963d2f27227'
C125_EXIT_SHA256 = '2757a975900e6281c1e6a8ef6fb3cfc65ade8cf48aec76fb2b2d623c8ede6bb3'
C125_SEAL_SHA256 = '39f0fd9b8b86bbed54e87cde9726d573cffa339fc43b9c1a7af1786eee6f82dd'
C124_SEAL_SHA256 = '8287afa172e1b9dcc4cac76a1529f6e777dbf7ec64130cdad6fbc0ef318d053f'
FROZEN_NAMES = (
    'C126_SUPPORT_WORD_PREREG_20260927.md', 'c126_support_word_mixed_v1.py',
    'c126_support_word_oracle_v1.py', 'c126_complete_mixed_source_v1.py',
    'c126_mixed_source_worker_v1.py', 'test_c126_support_word_mixed_v1.py',
    'run_c126_mixed_source_supervisor_v1.py')
NEW_DOCUMENTS = (
    'C126_SUPPORT_WORD_AUDIT_20260927.md', 'C126_SOURCE_BIRTH_HANDOFF_20260927.md',
    'CHECKPOINT_C126_SUPPORT_WORD_20260927.md')
CATEGORY_CAPS = dict(source=100000000, setup=2000000, binding=21000000,
    control=7000000, construction=15000000, proof=53000000,
    comparison=5000000, physical=17000000, ledger=18000000, evidence=16000000)
CATEGORY_WORK = dict(source=100000000, setup=1102144, binding=20085472,
    control=6895616, construction=14264064, proof=52379904,
    comparison=4507488, physical=16167744, ledger=16790846, evidence=15145664)


def main():
    started = time.monotonic()
    output = EXP/'C126_TERMINAL_INTEGRITY_20260927.json'
    seal = EXP/'CHECKPOINT_C126_SUPPORT_WORD_20260927_SHA256SUMS'
    if output.exists() or output.is_symlink() or seal.exists() or seal.is_symlink():
        raise FileExistsError('exclusive C126 terminal record or seal already exists')
    if (_sha256(RUN/'exit.json') != EXIT_SHA256
        or _sha256(RUN/'result.json') != RESULT_SHA256):
        raise ValueError('frozen completed C126 run identity differs')
    for name in (*FROZEN_NAMES, *NEW_DOCUMENTS):
        if not (EXP/name).is_file():
            raise ValueError('complete checkpoint source/document missing: '+name)
    freeze = json.loads((RUN/'preregistered.json').read_text())
    exited = json.loads((RUN/'exit.json').read_text())
    result = json.loads((RUN/'result.json').read_text())
    data, measurement = result['data'], result['measurement']
    if result['work'] != 247338942 or exited['work'] != result['work']:
        raise ValueError('complete frozen C126 paid work differs')
    pool = WorkPool(256000000)
    pool.charge('carried_complete_C126_paid_generation_and_evidence_work', result['work'])
    allowance = JsonAllowance(pool, 65536, 'c126_saved_JSON_archive_exact_terminal_reservation')
    if pool.used != 247404478:
        raise ValueError('complete run plus terminal archive reservation differs')
    if (not exited['all_stages_passed'] or not result['completed']
        or exited.get('failure') or result.get('failure')
        or not data['qualification_pass'] or data['qualification_failures']
        or exited['tests_exit'] or exited['worker_exit'] or exited['tests_count'] != 3556
        or freeze['required_test_count'] != 3556 or len(freeze['tests']) != 154
        or len(set(freeze['tests'])) != 154 or exited['test_wall_s'] > 60
        or freeze['complete_test_wall_cap_s'] != 60 or freeze['stage_worker_wall_cap_s'] != 240
        or result['wall_s'] > 240 or data['complete_source_count'] != 4
        or exited['complete_source_count'] != 4
        or sum(result['work_parts'].values()) != result['work']
        or sum(data['category_work'].values()) != result['work']
        or data['category_work'] != CATEGORY_WORK
        or data['category_limits'] != CATEGORY_CAPS or freeze['category_caps'] != CATEGORY_CAPS
        or sum(CATEGORY_CAPS.values()) != 254000000
        or any(v > CATEGORY_CAPS[k] for k,v in data['category_work'].items())
        or freeze['complete_work_upper'] != 254000000 or result['work'] > freeze['complete_work_upper']
        or freeze['whole_work_cap'] != 256000000 or freeze['branch_work_cap'] != 200000000
        or not measurement['measured_transient_gate'] or not measurement['build_returned']
        or measurement['transient_cap_bytes'] != 1073741824 or freeze['transient_bytes'] != 1073741824
        or measurement['resident_growth_upper_bound_bytes'] > 1073741824
        or measurement['traced_peak_bytes']+measurement['tracer_metadata_bytes'] > 1073741824
        or any(result[k] or exited[k] for k in ('source_drift','input_drift','provenance_drift'))
        or result['solver_calls'] or result['formal_gain'] or exited['formal_gain']
        or any(data[k] for k in ('original_network_loaded','archived_HZ_loaded',
                                 'source_or_LIVE_admitted','solver_calls','formal_gain'))
        or freeze['prior_C125_complete_qualification_passed'] is not True
        or exited['prior_C125_complete_qualification_passed'] is not True
        or freeze['prior_C124_complete_qualification_passed'] is not False
        or exited['prior_C124_complete_qualification_passed'] is not False):
        raise ValueError('complete frozen qualification/resource/provenance scope differs')
    if (freeze['fixed_complete_source_modes'] != ['dense','masked','heterogeneous','noop']
        or freeze['fixed_source_geometry'] != dict(C=16,K=32,input=[6,6],output=[4,4])
        or freeze['child_source_reserved_work'] != dict(dense=32000000,masked=32000000,
                                                       heterogeneous=32000000,noop=4000000)
        or freeze['shared_radix_caps'] != [16384,131072,16000000]
        or freeze['entries_cap'] != 64000000 or freeze['cpu_threads'] != 1
        or freeze['gpu_enabled'] or freeze['address_space_bytes'] != 16*1024**3
        or not freeze['complete_old_new_native_packets_required']
        or not freeze['complete_JSON_receipts_and_actual_sizes_required']
        or not freeze['full_owned_kernel_and_four_point_artifacts_required']
        or any(freeze[k] for k in ('original_network_run_authorized','solver_authorized',
             'actual_network_source_or_LIVE_admitted','kernel_cache_or_reuse_authorized',
             'promotion_authorized','archived_HZ_restore_authorized','formal_gain',
             'numeric_hash_traffic_in_token_pool','all_CPU_work_in_generation_cap'))):
        raise ValueError('fixed complete source or unchanged authority boundary differs')

    saved = verify_saved_evidence(result)
    if (saved != exited['complete_evidence_payment_checks']
        or saved['complete_prepaid_JSON_receipt_count'] != 17
        or len(saved['JSON_checks']) != 18 or len(saved['complete_numeric_artifacts']) != 9
        or saved['numeric_arrays_reanalysed']):
        raise ValueError('frozen complete saved-evidence verification differs')
    exact_result = (json.dumps(result, sort_keys=True, separators=(',', ':'),
                               allow_nan=False, ensure_ascii=True)+'\n').encode('utf-8')
    if (RUN/'result.json').read_bytes() != exact_result:
        raise ValueError('terminal success artifact is not its exact checked compact encoding')
    layout, metadata = data['ledger']['numeric'], data['ledger']['known_metadata']
    if (layout['resident_entries'] != 1316545 or layout['resident_bytes'] != 10284196
        or layout['numeric_storage_count'] != 474 or layout['resident_entries'] > 64000000
        or metadata['nonoverlapping_known_metadata_bytes'] != 26523476
        or len(metadata['opaque_inherited_ids']) != 8
        or json.loads((RUN/'complete_held_ledger.json').read_text()) != data['ledger']):
        raise ValueError('complete held control/native/kernel/point metadata observation differs')
    inventory = json.loads((RUN/'inventory.json').read_text())
    nodeids = inventory['nodeids']
    expected_files = {str((EXP/name).resolve().relative_to(ROOT)) for name in freeze['tests']}
    inherited = json.loads((EXP/'results/c125_kernel_proof_20260927_v1/preregistered.json').read_text())
    if (freeze['tests'] != inherited['tests']+['test_c126_support_word_mixed_v1.py']
        or inherited['required_test_count'] != 3524 or len(inherited['tests']) != 153
        or inventory['count'] != 3556 or len(nodeids) != 3556 or len(set(nodeids)) != 3556
        or {name.split('::',1)[0] for name in nodeids} != expected_files):
        raise ValueError('complete 154-file 3556-test inherited/new inventory differs')
    xml_cases = ET.parse(RUN/'tests.xml').findall('.//testcase')
    xml_ids = [case.get('classname','').replace('.','/')+'.py::'+case.get('name','')
               for case in xml_cases]
    if (sorted(xml_ids) != sorted(nodeids)
        or any(case.find(name) is not None for case in xml_cases
               for name in ('failure','error','skipped'))):
        raise ValueError('saved complete JUnit population or zero-failure/skip requirement differs')

    expected_cases = dict(dense=(16,74240,-274864,-18344,1728,38784,2688,99986),
        masked=(16,55808,-76912,-1848,1728,36848,2544,96114),
        heterogeneous=(12,55936,-145456,-7896,1584,32224,2548,85426))
    artifacts = {item['file']:item for item in saved['complete_numeric_artifacts']}
    cases = []
    for mode in ('dense','masked','heterogeneous','noop'):
        report = data['reports'][mode]
        binding, constructor, physical = report['binding'], report['constructor'], report['physical']
        control, native = report['control_constructor'], report['complete_native_comparison']
        if (json.loads((RUN/(mode+'_complete_source.json')).read_text()) != report
            or report['mode'] != mode or not report['original_source_unchanged']
            or report['original_source_reservation'] != freeze['child_source_reserved_work'][mode]
            or binding['all_actual_source_output_rows_bound'] != 512
            or binding['all_actual_native_coefficients_compared'] != binding['actual_direct_output_nnz']
            or not binding['exact_coefficients_and_original_row_gauges_equal']
            or not binding['source_survival']['all_selected_original_coordinates_survive']
            or report['original_nonconvex_binary_factors'] != 1 or report['original_inequalities'] != 1
            or not native['all_native_arrays_bitwise_equal']
            or report['source_or_LIVE_admitted'] or report['formal_gain']):
            raise ValueError('complete original source binding or native comparison differs: '+mode)
        if mode == 'noop':
            if (report['proof'] is not None or report['kernel_comparison'] is not None
                or report['complete_point_evidence'] is not None
                or not physical['exact_original_object_retained'] or not physical['literal_noop']
                or not constructor['literal_noop'] or constructor['kernel_transform_prepaid']
                or not control['literal_noop'] or control['kernel_transform_prepaid']
                or not native['literal_noop'] or native['compared_arrays']
                or native['compared_numeric_entries'] or native['work']
                or report['route']['selected_channels'] or binding['actual_direct_output_nnz'] != 5120):
                raise ValueError('complete no-hit source/control did not remain literal')
            continue
        selected,direct,byte_delta,entry_delta,aux,nnz,original_points,packet_entries = expected_cases[mode]
        proof, comparison = report['proof'], report['kernel_comparison']
        audited, kernel = physical['complete_original_audit'], proof['kernel_proof']
        counts = report['complete_point_evidence']['counts']
        if (report['route']['selected_channels'] != selected or not report['route']['conditional_candidate']
            or binding['actual_direct_output_nnz'] != direct
            or native['literal_noop'] or native['compared_arrays'] != 11
            or native['compared_numeric_entries'] != packet_entries
            or native['work'] != 1024+16*packet_entries
            or any(artifacts[mode+suffix]['numeric_entries'] != packet_entries
                   for suffix in ('_control_arrays.npz','_constructed_arrays.npz'))
            or any(constructor[key] != control[key] for key in
                   ('kept_v','kept_m','new_factors','rows','nnz','base_n_cont','n_cont','whole_circuit_emission'))
            or not constructor['exact_word_row_algorithm']
            or not constructor['row_hotpath_has_no_fraction_arithmetic']
            or not constructor['support_plan_is_owned'] or not constructor['source_words_are_owned']
            or not constructor['support_and_source_workspaces_ephemeral']
            or physical['numeric_byte_delta'] != byte_delta or physical['numeric_entry_delta'] != entry_delta
            or not physical['complete_physical_reduction_proved'] or physical['rejection_reasons']
            or not physical['actual_complete_inverse_equal'] or physical['route_mask_bytes_and_entries'] != 16
            or physical['whole_nnz_after'] >= physical['whole_nnz_before']
            or physical['whole_positive_entry_reserve_used'] != 131072
            or physical['whole_shared_emission'] > 16000000
            or constructor['new_factors'] != aux or constructor['nnz'] != nnz
            or proof['all_native_nnz_proved'] != nnz or proof['all_native_rows_proved'] != aux+512
            or proof['all_transformed_kernel_coefficients_rebuilt'] != 18432
            or proof['all_original_output_equations_proved'] != 512
            or proof['all_auxiliary_equations_and_redundant_boxes_proved'] != aux
            or not proof['exact_required_support_coverage'] or not proof['universal_unique_box_extension']
            or not proof['full_native_row_proof_scope_retained']
            or not proof['exact_row_arithmetic_without_Fraction']
            or not proof['independent_word_arithmetic_without_constructor_or_C123_helpers']
            or not proof['complete_reduced_numerator_and_denominator_512_bit_guards']
            or not proof['old_full_materialization_reservation_retained_without_refund']
            or proof['all36_fraction_materialization_completed']
            or not audited['all_original_maps_and_other_predicates_preserved']
            or not audited['independent_complete_owner_delta_proved'] or audited['original_binary_factors'] != 1
            or not kernel['complete_all36_proved'] or not kernel['all_preparation_arrays_owned']
            or not kernel['no_prepared_cache_or_reuse'] or kernel['current_source_reuse_authorized']
            or comparison['all_coefficients_compared'] != 18432
            or not comparison['complete_Fraction_agreement'] or not comparison['all_original_channels_included']
            or comparison['once_per_operator_reuse_proved']
            or comparison['reference_report']['fraction_contraction_prepaid'] != 6422528
            or not comparison['reference_report']['exact_lossless_numeric_reference_retained']
            or counts != dict(original=original_points,expected=original_points,
                               expanded=original_points+aux,recovered=original_points)
            or not report['complete_point_evidence']['all_four_full_populations_saved']):
            raise ValueError('complete actual-HZ/control/word/kernel/inverse qualification differs: '+mode)
        cases.append(dict(mode=mode,selected_channels=selected,original_nnz=direct,
            native_packet_nnz=nnz,new_factors=aux,complete_numeric_byte_delta=byte_delta,
            complete_numeric_entry_delta=entry_delta,all36_coefficients_compared=18432,
            exact_original_inverse=True,full_point_counts=counts,
            complete_old_new_native_comparison=native))
    if (sum(case['complete_old_new_native_comparison']['compared_arrays'] for case in cases) != 33
        or sum(case['complete_old_new_native_comparison']['compared_numeric_entries'] for case in cases) != 281526):
        raise ValueError('complete old/new literal comparison population differs')

    expected = {}
    def bind(path, digest):
        name = str(path.resolve())
        if name in expected and expected[name] != digest:
            raise ValueError('conflicting inherited file identity: '+name)
        expected[name] = digest
    for name,digest in freeze['source_sha256'].items():
        bind(EXP/name,digest)
    for name,digest in exited['artifacts'].items():
        bind(RUN/name,digest)
    bind(RUN/'result.json',RESULT_SHA256)
    bind(RUN/'exit.json',EXIT_SHA256)
    previous = EXP/'CHECKPOINT_C125_KERNEL_PROOF_20260927_SHA256SUMS'
    if _sha256(previous) != C125_SEAL_SHA256:
        raise ValueError('complete previous C125 qualified-checkpoint seal differs')
    bind(previous,C125_SEAL_SHA256)
    for line in previous.read_text().splitlines():
        digest,name = line.split('  ',1)
        bind(ROOT/name,digest)
    older = EXP/'CHECKPOINT_C124_MIXED_SOURCE_20260927_SHA256SUMS'
    if _sha256(older) != C124_SEAL_SHA256:
        raise ValueError('complete previous C124 failed-checkpoint seal differs')
    bind(older,C124_SEAL_SHA256)
    for line in older.read_text().splitlines():
        digest,name = line.split('  ',1)
        bind(ROOT/name,digest)
    prior = json.loads((EXP/'C125_TERMINAL_INTEGRITY_20260927.json').read_text())
    if (prior['complete_qualification_passed'] is not True
        or prior['ordinary_qualification_passed'] is not True or not prior['completed']
        or prior['qualification_scope'] != 'fixed_complete_ordinary_HZ_only'
        or prior['prior_C124_complete_qualification_passed'] is not False
        or not prior['prior_C124_failure_not_repaired_or_requalified']
        or prior['exit_sha256'] != C125_EXIT_SHA256 or prior['formal_gain']
        or prior['mismatches'] or prior['input_mismatches'] or not prior['provenance_unchanged']):
        raise ValueError('C125 ordinary-only qualification must remain immutable')
    prior_failed = json.loads((EXP/'C124_TERMINAL_INTEGRITY_20260927.json').read_text())
    if (prior_failed['complete_qualification_passed'] is not False
        or not prior_failed['candidate_version_closed'] or not prior_failed['completed']
        or not prior_failed['completion_means_archive_integrity_only']
        or not prior_failed['this_terminal_audit_overrides_raw_complete_qualification_claim']
        or not prior_failed['raw_wrapper_green_flags_preserved_not_rewritten']
        or prior_failed['mismatches'] or prior_failed['input_mismatches']
        or not prior_failed['provenance_unchanged']):
        raise ValueError('C124 must remain explicitly failed and immutable')
    run_files = sorted(path for path in RUN.rglob('*') if path.is_file())
    if (len(run_files) != 36 or {str(path.relative_to(RUN)) for path in run_files}
        != set(exited['artifacts']) | {'exit.json'}):
        raise ValueError('complete 36-file exclusive run archive population differs')
    mismatches = [name for name,digest in expected.items() if _sha256(Path(name)) != digest]
    input_mismatches = [name for name,digest in freeze['input_sha256'].items()
                        if _sha256(Path(name)) != digest]
    provenance = _provenance(ROOT)
    for name in FROZEN_NAMES:
        if name not in freeze['source_sha256']:
            raise ValueError('new frozen source omitted from authentication: '+name)
    complete = not mismatches and not input_mismatches and provenance == freeze['provenance']
    record = dict(completed=complete,ordinary_qualification_passed=complete,
        complete_qualification_passed=complete,qualification_scope='fixed_complete_ordinary_HZ_only',
        checked_unique_files=len(expected),checked_file_bytes=sum(Path(name).stat().st_size for name in expected),
        mismatches=mismatches,input_mismatches=input_mismatches,provenance=provenance,
        provenance_unchanged=provenance == freeze['provenance'],
        prior_C125_complete_qualification_passed=True,prior_C125_ordinary_only_qualification_preserved=True,
        prior_C124_complete_qualification_passed=False,prior_C124_failure_not_repaired_or_requalified=True,
        complete_inherited_tests=3524,new_tests=32,total_tests=3556,total_test_files=154,
        test_collection_and_execution_wall_s=exited['test_wall_s'],
        complete_source_cases=cases,complete_nohit_literal_source_preserved=True,
        all_original_output_rows=2048,all_actual_original_coefficients=191104,
        complete_native_rows_proved=6576,complete_native_coefficients_proved=107856,
        complete_old_new_native_arrays_bitwise_equal=True,complete_native_arrays_compared=33,
        complete_native_array_entries_compared=281526,retained_control_native_array_entries=281526,
        retained_new_native_array_entries=281526,complete_control_and_new_native_arrays_saved=True,
        complete_all36_kernel_coefficients_compared=55296,retained_new_kernel_array_entries=264192,
        complete_four_point_populations_saved=True,exact_saved_evidence_checks=saved,
        complete_numeric_bytes=layout['resident_bytes'],complete_numeric_entries=layout['resident_entries'],
        complete_numeric_storage_count=layout['numeric_storage_count'],
        known_metadata_bytes=metadata['nonoverlapping_known_metadata_bytes'],
        inherited_opaque_id_count=len(metadata['opaque_inherited_ids']),measurement=measurement,
        both_transient_gates_passed=True,measurement_scope='C41_build_not_later_terminal_encoding',
        category_work=data['category_work'],category_limits=data['category_limits'],
        paid_run_work=result['work'],paid_run_plus_archive_reporting=pool.used,
        archive_reporting_work_parts=pool.parts,archive_JSON_reservation=65536,
        numerical_proofs_rerun=False,archived_numeric_arrays_reanalysed=False,
        target_source_or_LIVE_admitted=False,original_network_loaded=False,
        reusable_preparation_or_cache_proved=False,solver_calls=0,formal_gain=0,speedup_claimed=False,
        baseline_formal_results_changed=False,hash_traffic_separate_from_generation_tokens=True,
        all_CPU_work_in_generation_cap=False,wall_s=time.monotonic()-started,
        exit_sha256=EXIT_SHA256,result_sha256=RESULT_SHA256)
    receipt = allowance.write(output,record)
    if not complete:
        raise ValueError('terminal archive integrity failed; failed record retained')
    names = [EXP/name for name in (*FROZEN_NAMES,*NEW_DOCUMENTS,'run_c126_archive_audit_v1.py')]
    names += [output,*run_files]
    if len(names) != 48 or len(set(names)) != 48:
        raise ValueError('complete 48-file C126 checkpoint seal population differs')
    seal_bytes = ''.join(_sha256(path)+'  '+str(path.relative_to(ROOT))+'\n'
                         for path in sorted(names)).encode('utf-8')
    publish_bytes(seal,seal_bytes)
    print(json.dumps(dict(completed=True,ordinary_qualification_passed=True,
        checked_unique_files=len(expected),checked_file_bytes=record['checked_file_bytes'],
        paid_run_plus_archive_reporting=pool.used,terminal_receipt=receipt,
        sealed_files=len(names),seal_sha256=_sha256(seal),formal_gain=0)),flush=True)


if __name__ == '__main__':
    main()
