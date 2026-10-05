"""Saved-JSON/XML and full-file hash audit of qualified ordinary C125 evidence.

No numerical arrays are loaded, no source is restored, and no proof or test is
rerun.  The frozen supervisor's saved-evidence checker is reused verbatim.
C124 remains an authoritatively failed reporting version throughout this audit.
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
from experiments.neural_hz_20260831.run_c125_mixed_source_supervisor_v1 import verify_saved_evidence
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256

EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c125_kernel_proof_20260927_v1'
EXIT_SHA256 = '2757a975900e6281c1e6a8ef6fb3cfc65ade8cf48aec76fb2b2d623c8ede6bb3'
RESULT_SHA256 = '223699c44e063c2d6e357c6b4105e02856c394669699a229f2991b4a12bb371a'
C124_SEAL_SHA256 = '8287afa172e1b9dcc4cac76a1529f6e777dbf7ec64130cdad6fbc0ef318d053f'
FROZEN_NAMES = (
    'C125_KERNEL_PROOF_PREREG_20260927.md', 'c125_exact_kernel_proof_v1.py',
    'c125_exact_json_v1.py', 'c125_mixed_f4_oracle_v1.py',
    'c125_complete_mixed_source_v1.py', 'c125_mixed_source_worker_v1.py',
    'test_c125_exact_kernel_proof_v1.py', 'test_c125_exact_json_v1.py',
    'run_c125_mixed_source_supervisor_v1.py')
NEW_DOCUMENTS = (
    'C125_KERNEL_PROOF_AUDIT_20260927.md', 'C125_SUPPORT_WORD_HANDOFF_20260927.md',
    'CHECKPOINT_C125_KERNEL_PROOF_20260927.md')


def main():
    started = time.monotonic()
    output = EXP/'C125_TERMINAL_INTEGRITY_20260927.json'
    seal = EXP/'CHECKPOINT_C125_KERNEL_PROOF_20260927_SHA256SUMS'
    if output.exists() or output.is_symlink() or seal.exists() or seal.is_symlink():
        raise FileExistsError('exclusive C125 terminal record or seal already exists')
    if (_sha256(RUN/'exit.json') != EXIT_SHA256
        or _sha256(RUN/'result.json') != RESULT_SHA256):
        raise ValueError('frozen completed C125 run identity differs')
    for name in (*FROZEN_NAMES, *NEW_DOCUMENTS):
        if not (EXP/name).is_file():
            raise ValueError('complete checkpoint source/document missing: '+name)
    freeze = json.loads((RUN/'preregistered.json').read_text())
    exited = json.loads((RUN/'exit.json').read_text())
    result = json.loads((RUN/'result.json').read_text())
    data, measurement = result['data'], result['measurement']
    if result['work'] != 242927910:
        raise ValueError('complete frozen C125 paid work differs')
    pool = WorkPool(256000000)
    pool.charge('carried_complete_C125_paid_generation_and_evidence_work', result['work'])
    allowance = JsonAllowance(pool, 65536, 'c125_saved_JSON_archive_exact_terminal_reservation')
    if (not exited['all_stages_passed'] or not result['completed']
        or exited.get('failure') or result.get('failure')
        or not data['qualification_pass'] or data['qualification_failures']
        or exited['tests_exit'] or exited['worker_exit'] or exited['tests_count'] != 3524
        or freeze['required_test_count'] != 3524 or len(freeze['tests']) != 153
        or len(set(freeze['tests'])) != 153 or exited['test_wall_s'] > 60
        or result['wall_s'] > 240 or data['complete_source_count'] != 4
        or sum(result['work_parts'].values()) != result['work']
        or sum(data['category_work'].values()) != result['work']
        or data['category_limits'] != freeze['category_caps']
        or sum(data['category_limits'].values()) != 254000000
        or any(v > data['category_limits'][k] for k,v in data['category_work'].items())
        or result['work'] > freeze['complete_work_upper']
        or freeze['whole_work_cap'] != 256000000 or freeze['branch_work_cap'] != 200000000
        or not measurement['measured_transient_gate'] or not measurement['build_returned']
        or measurement['transient_cap_bytes'] != 1073741824
        or measurement['resident_growth_upper_bound_bytes'] > 1073741824
        or measurement['traced_peak_bytes']+measurement['tracer_metadata_bytes'] > 1073741824
        or any(result[k] or exited[k] for k in ('source_drift','input_drift','provenance_drift'))
        or result['solver_calls'] or result['formal_gain'] or exited['formal_gain']
        or any(data[k] for k in ('original_network_loaded','archived_HZ_loaded',
                                 'source_or_LIVE_admitted','solver_calls','formal_gain'))
        or freeze['prior_C124_complete_qualification_passed'] is not False
        or exited['prior_C124_complete_qualification_passed'] is not False
        or not freeze['prior_C124_raw_green_is_not_qualification']):
        raise ValueError('complete frozen qualification/resource/provenance scope differs')
    if (freeze['fixed_complete_source_modes'] != ['dense','masked','heterogeneous','noop']
        or freeze['fixed_source_geometry'] != dict(C=16,K=32,input=[6,6],output=[4,4])
        or freeze['child_source_reserved_work'] != dict(dense=32000000,masked=32000000,
                                                       heterogeneous=32000000,noop=4000000)
        or freeze['shared_radix_caps'] != [16384,131072,16000000]
        or freeze['entries_cap'] != 64000000 or freeze['cpu_threads'] != 1
        or freeze['gpu_enabled'] or freeze['address_space_bytes'] != 16*1024**3
        or any(freeze[k] for k in ('original_network_run_authorized','solver_authorized',
             'actual_network_source_or_LIVE_admitted','kernel_cache_or_reuse_authorized',
             'promotion_authorized','archived_HZ_restore_authorized','formal_gain'))):
        raise ValueError('fixed complete source or unchanged authority boundary differs')

    saved = verify_saved_evidence(result)
    if saved != exited['complete_evidence_payment_checks']:
        raise ValueError('frozen complete saved-evidence verification differs')
    exact_result = (json.dumps(result, sort_keys=True, separators=(',', ':'),
                               allow_nan=False, ensure_ascii=True)+'\n').encode('utf-8')
    if (RUN/'result.json').read_bytes() != exact_result:
        raise ValueError('terminal success artifact is not its exact checked compact encoding')
    layout, metadata = data['ledger']['numeric'], data['ledger']['known_metadata']
    if (layout['resident_entries'] != 1035019 or layout['resident_bytes'] != 8621572
        or layout['numeric_storage_count'] != 441 or layout['resident_entries'] > 64000000
        or metadata['nonoverlapping_known_metadata_bytes'] != 26489092
        or len(metadata['opaque_inherited_ids']) != 8
        or json.loads((RUN/'complete_held_ledger.json').read_text()) != data['ledger']):
        raise ValueError('complete held numeric/kernel/point metadata observation differs')
    inventory = json.loads((RUN/'inventory.json').read_text())
    nodeids = inventory['nodeids']
    expected_files = {str((EXP/name).resolve().relative_to(ROOT)) for name in freeze['tests']}
    if (inventory['count'] != 3524 or len(nodeids) != 3524 or len(set(nodeids)) != 3524
        or {name.split('::',1)[0] for name in nodeids} != expected_files):
        raise ValueError('complete153-file3524-test inventory differs')
    xml_cases = ET.parse(RUN/'tests.xml').findall('.//testcase')
    xml_ids = [case.get('classname','').replace('.','/')+'.py::'+case.get('name','')
               for case in xml_cases]
    if (sorted(xml_ids) != sorted(nodeids)
        or any(case.find(name) is not None for case in xml_cases
               for name in ('failure','error','skipped'))):
        raise ValueError('saved complete JUnit population or zero-failure/skip requirement differs')

    expected_cases = dict(dense=(16,74240,-274864,-18344,1728,38784,2688),
        masked=(16,55808,-76912,-1848,1728,36848,2544),
        heterogeneous=(12,55936,-145456,-7896,1584,32224,2548))
    cases = []
    for mode in ('dense','masked','heterogeneous','noop'):
        report = data['reports'][mode]
        binding, constructor, physical = report['binding'], report['constructor'], report['physical']
        if (json.loads((RUN/(mode+'_complete_source.json')).read_text()) != report
            or not report['original_source_unchanged']
            or binding['all_actual_source_output_rows_bound'] != 512
            or binding['all_actual_native_coefficients_compared'] != binding['actual_direct_output_nnz']
            or not binding['exact_coefficients_and_original_row_gauges_equal']
            or not binding['source_survival']['all_selected_original_coordinates_survive']
            or report['original_nonconvex_binary_factors'] != 1 or report['original_inequalities'] != 1
            or report['source_or_LIVE_admitted'] or report['formal_gain']):
            raise ValueError('complete original source binding differs: '+mode)
        if mode == 'noop':
            if (report['proof'] is not None or report['kernel_comparison'] is not None
                or report['complete_point_evidence'] is not None
                or not physical['exact_original_object_retained'] or not physical['literal_noop']
                or not constructor['literal_noop'] or constructor['kernel_transform_prepaid']
                or report['route']['selected_channels'] or binding['actual_direct_output_nnz'] != 5120):
                raise ValueError('complete no-hit source did not remain literal')
            continue
        selected,direct,byte_delta,entry_delta,aux,nnz,original_points = expected_cases[mode]
        proof, comparison = report['proof'], report['kernel_comparison']
        audited, kernel = physical['complete_original_audit'], proof['kernel_proof']
        counts = report['complete_point_evidence']['counts']
        if (report['route']['selected_channels'] != selected or not report['route']['conditional_candidate']
            or binding['actual_direct_output_nnz'] != direct
            or physical['numeric_byte_delta'] != byte_delta or physical['numeric_entry_delta'] != entry_delta
            or not physical['complete_physical_reduction_proved'] or physical['rejection_reasons']
            or not physical['actual_complete_inverse_equal'] or physical['route_mask_bytes_and_entries'] != 16
            or physical['whole_nnz_after'] >= physical['whole_nnz_before']
            or physical['whole_positive_entry_reserve_used'] != 131072
            or physical['whole_shared_emission'] > 16000000
            or constructor['new_factors'] != aux or constructor['nnz'] != nnz
            or proof['all_native_nnz_proved'] != nnz
            or proof['all_transformed_kernel_coefficients_rebuilt'] != 18432
            or proof['all_original_output_equations_proved'] != 512
            or proof['all_auxiliary_equations_and_redundant_boxes_proved'] != aux
            or not proof['exact_required_support_coverage'] or not proof['universal_unique_box_extension']
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
            raise ValueError('complete actual-HZ/kernel/inverse qualification differs: '+mode)
        cases.append(dict(mode=mode,selected_channels=selected,original_nnz=direct,
            native_packet_nnz=nnz,new_factors=aux,complete_numeric_byte_delta=byte_delta,
            complete_numeric_entry_delta=entry_delta,all36_coefficients_compared=18432,
            exact_original_inverse=True,full_point_counts=counts))

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
    previous = EXP/'CHECKPOINT_C124_MIXED_SOURCE_20260927_SHA256SUMS'
    if _sha256(previous) != C124_SEAL_SHA256:
        raise ValueError('complete previous C124 failed-checkpoint seal differs')
    bind(previous,C124_SEAL_SHA256)
    for line in previous.read_text().splitlines():
        digest,name = line.split('  ',1)
        bind(ROOT/name,digest)
    prior = json.loads((EXP/'C124_TERMINAL_INTEGRITY_20260927.json').read_text())
    if (prior['complete_qualification_passed'] is not False or not prior['candidate_version_closed']
        or not prior['completed'] or not prior['completion_means_archive_integrity_only']
        or not prior['this_terminal_audit_overrides_raw_complete_qualification_claim']
        or not prior['raw_wrapper_green_flags_preserved_not_rewritten']
        or prior['mismatches'] or prior['input_mismatches'] or not prior['provenance_unchanged']):
        raise ValueError('C124 must remain explicitly failed and immutable')
    run_files = sorted(path for path in RUN.rglob('*') if path.is_file())
    if (len(run_files) != 30 or {str(path.relative_to(RUN)) for path in run_files}
        != set(exited['artifacts']) | {'exit.json'}):
        raise ValueError('complete30-file exclusive run archive population differs')
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
        prior_C124_complete_qualification_passed=False,prior_C124_failure_not_repaired_or_requalified=True,
        complete_inherited_tests=3495,new_tests=29,total_tests=3524,total_test_files=153,
        test_collection_and_execution_wall_s=exited['test_wall_s'],
        complete_source_cases=cases,complete_nohit_literal_source_preserved=True,
        all_original_output_rows=2048,all_actual_original_coefficients=191104,
        complete_native_rows_proved=6576,complete_native_coefficients_proved=107856,
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
        hash_traffic_separate_from_generation_tokens=True,all_CPU_work_in_generation_cap=False,
        wall_s=time.monotonic()-started,exit_sha256=EXIT_SHA256,result_sha256=RESULT_SHA256)
    receipt = allowance.write(output,record)
    if not complete:
        raise ValueError('terminal archive integrity failed; failed record retained')
    names = [EXP/name for name in (*FROZEN_NAMES,*NEW_DOCUMENTS,'run_c125_archive_audit_v1.py')]
    names += [output,*run_files]
    if len(names) != 44 or len(set(names)) != 44:
        raise ValueError('complete44-file C125 checkpoint seal population differs')
    seal_bytes = ''.join(_sha256(path)+'  '+str(path.relative_to(ROOT))+'\n'
                         for path in sorted(names)).encode('utf-8')
    publish_bytes(seal,seal_bytes)
    print(json.dumps(dict(completed=True,ordinary_qualification_passed=True,
        checked_unique_files=len(expected),checked_file_bytes=record['checked_file_bytes'],
        paid_run_plus_archive_reporting=pool.used,terminal_receipt=receipt,
        sealed_files=len(names),seal_sha256=_sha256(seal),formal_gain=0)),flush=True)


if __name__ == '__main__':
    main()
