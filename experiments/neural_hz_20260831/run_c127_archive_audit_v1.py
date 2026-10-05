"""Authenticate complete C127 failed-v1 and corrected-v2 saved evidence only.

No numerical array is loaded and no source, proof, model or test is executed.
A raw v2 failure is archived as failure, never converted into qualification.
Both version outcomes and the final v2 hashes are fixed by their saved runs.
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
from experiments.neural_hz_20260831.run_c127_mixed_source_supervisor_v2 import verify_saved_evidence
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256

EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c127_systematic_kernel_20260927_v2'
FAILED_RUN = EXP/'results/c127_systematic_kernel_20260927_v1'
V1_EXIT_SHA256 = 'eaccb8b1b3848833a7405fd1e9c199f00784ec8dad0b10465184c3fec2235c7c'
EXIT_SHA256 = '1b1ba7fa7a281e255a8f435b268a9b790b1d2c385611029251b477460b631238'
RESULT_SHA256 = '519ca415a32b406be523779519bee8d21b0288bfd911130604e9437b2bbb8bea'
PRIOR_SEALS = {
    'CHECKPOINT_C126_SUPPORT_WORD_20260927_SHA256SUMS':
        '2d71126243a0ca4e74385c56641b7e6d465c61c2b8098e25135d4bc99a7cbc7a',
    'CHECKPOINT_C125_KERNEL_PROOF_20260927_SHA256SUMS':
        '39f0fd9b8b86bbed54e87cde9726d573cffa339fc43b9c1a7af1786eee6f82dd',
    'CHECKPOINT_C124_MIXED_SOURCE_20260927_SHA256SUMS':
        '8287afa172e1b9dcc4cac76a1529f6e777dbf7ec64130cdad6fbc0ef318d053f'}
PRIOR_EXITS = {
    'C126': '4ba7922145351478584cedb3666d03b1d2ec0668346adf746c2da33af615b573',
    'C125': '2757a975900e6281c1e6a8ef6fb3cfc65ade8cf48aec76fb2b2d623c8ede6bb3'}
FROZEN_V1 = (
    'C127_SOURCE_BIRTH_PREFLIGHT_20260927.md', 'C127_SYSTEMATIC_KERNEL_PREREG_20260927.md',
    'c127_systematic_kernel_proof_v1.py', 'c127_systematic_native_oracle_v1.py',
    'c127_complete_mixed_source_v1.py', 'c127_mixed_source_worker_v1.py',
    'test_c127_systematic_kernel_proof_v1.py', 'run_c127_mixed_source_supervisor_v1.py')
FROZEN_V2 = (
    'C127_SYSTEMATIC_KERNEL_PREREG_20260927_v2.md', 'C127_V1_FAILURE_AUDIT_20260927.md',
    'c127_systematic_kernel_proof_v2.py', 'c127_systematic_native_oracle_v2.py',
    'c127_complete_mixed_source_v2.py', 'c127_mixed_source_worker_v2.py',
    'test_c127_systematic_kernel_proof_v2.py', 'run_c127_mixed_source_supervisor_v2.py')
NEW_DOCUMENTS = (
    'C127_SYSTEMATIC_KERNEL_AUDIT_20260927.md', 'CHECKPOINT_C127_SYSTEMATIC_KERNEL_20260927.md',
    'C127_DENSE_SUPPORT_HANDOFF_20260927.md')
CATEGORY_CAPS = dict(source=100000000, setup=2000000, binding=21000000,
    control=7000000, construction=15000000, proof=53000000,
    comparison=5000000, physical=17000000, ledger=18000000, evidence=16000000)
FIXED_CATEGORY_WORK = dict(source=100000000, setup=1102144, binding=20085472,
    control=6895616, construction=14264064, proof=48515328,
    comparison=4507488, physical=16167744, ledger=16795070, evidence=15145664)


def _qualified_v2(result, exited, freeze):
    """Complete C126 qualification obligations plus full new systematic proof."""
    data, measurement = result['data'], result['measurement']
    if (not exited['all_stages_passed'] or not result['completed']
        or exited.get('failure') or result.get('failure')
        or not data['qualification_pass'] or data['qualification_failures']
        or exited['tests_exit'] or exited['worker_exit'] or exited['tests_count'] != 3584
        or freeze['required_test_count'] != 3584 or len(freeze['tests']) != 155
        or len(set(freeze['tests'])) != 155 or exited['test_wall_s'] > 60
        or freeze['complete_test_wall_cap_s'] != 60 or freeze['stage_worker_wall_cap_s'] != 240
        or result['wall_s'] > 240 or data['complete_source_count'] != 4
        or exited['complete_source_count'] != 4
        or sum(result['work_parts'].values()) != result['work']
        or sum(data['category_work'].values()) != result['work']
        or data['category_work'] != FIXED_CATEGORY_WORK
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
        or freeze['prior_C126_complete_qualification_passed'] is not True
        or exited['prior_C126_complete_qualification_passed'] is not True
        or freeze['prior_C127_v1_complete_qualification_passed'] is not False
        or exited['prior_C127_v1_complete_qualification_passed'] is not False
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
        or metadata['nonoverlapping_known_metadata_bytes'] != 26539063
        or len(metadata['opaque_inherited_ids']) != 8
        or json.loads((RUN/'complete_held_ledger.json').read_text()) != data['ledger']):
        raise ValueError('complete held control/native/kernel/point metadata observation differs')
    inventory = json.loads((RUN/'inventory.json').read_text())
    nodeids = inventory['nodeids']
    expected_files = {str((EXP/name).resolve().relative_to(ROOT)) for name in freeze['tests']}
    inherited = json.loads((EXP/'results/c126_support_word_20260927_v1/preregistered.json').read_text())
    if (freeze['tests'] != inherited['tests']+['test_c127_systematic_kernel_proof_v2.py']
        or inherited['required_test_count'] != 3556 or len(inherited['tests']) != 154
        or inventory['count'] != 3584 or len(nodeids) != 3584 or len(set(nodeids)) != 3584
        or {name.split('::',1)[0] for name in nodeids} != expected_files):
        raise ValueError('complete 155-file 3584-test inherited/new inventory differs')
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
            or not kernel['systematic_source_derived_program']
            or kernel['source_decoder'] != 'independent_vector_frexp_ldexp'
            or kernel['source_anchor_indices'] != [0,1,5]
            or kernel['source_anchor_cells_per_kernel'] != 9
            or kernel['source_derived_remaining_cells_per_kernel'] != 27
            or kernel['all_source_basis_output_coefficients_proved'] != 324
            or not kernel['full_basis_theorem_executed_each_call']
            or kernel['complete_prepaid_work'] != 1755136
            or kernel['external_transformed_candidate_or_receipt_accepted']
            or kernel['constructor_transform_or_inverse_used']
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


    return dict(complete_source_cases=cases, exact_saved_evidence_checks=saved,
        complete_numeric_bytes=layout['resident_bytes'],
        complete_numeric_entries=layout['resident_entries'],
        complete_numeric_storage_count=layout['numeric_storage_count'],
        known_metadata_bytes=metadata['nonoverlapping_known_metadata_bytes'],
        inherited_opaque_id_count=len(metadata['opaque_inherited_ids']),
        measurement=measurement, both_transient_gates_passed=True,
        measurement_scope='C41_build_not_later_terminal_encoding',
        category_work=data['category_work'], category_limits=data['category_limits'],
        all_original_output_rows=2048, all_actual_original_coefficients=191104,
        complete_native_rows_proved=6576, complete_native_coefficients_proved=107856,
        complete_native_arrays_compared=33, complete_native_array_entries_compared=281526,
        complete_old_new_native_arrays_bitwise_equal=True,
        complete_all36_kernel_coefficients_compared=55296,
        retained_new_kernel_array_entries=264192, retained_control_native_array_entries=281526,
        retained_new_native_array_entries=281526,
        complete_four_point_populations_saved=True, complete_nohit_literal_source_preserved=True,
        full_independent_systematic_recomputation=True, source_basis_equalities_per_call=324)


def _test_partition(run, freeze, expected_new_name):
    inventory = json.loads((run/'inventory.json').read_text())
    nodeids = inventory['nodeids']
    cases = ET.parse(run/'tests.xml').findall('.//testcase')
    actual = [case.get('classname','').replace('.','/')+'.py::'+case.get('name','')
              for case in cases]
    previous = EXP/'results/c126_support_word_20260927_v1'
    inherited = set(json.loads((previous/'inventory.json').read_text())['nodeids'])
    expected_files = {str((EXP/name).resolve().relative_to(ROOT)) for name in freeze['tests']}
    if (inventory['count'] != 3584 or len(nodeids) != 3584 or len(set(nodeids)) != 3584
        or sorted(actual) != sorted(nodeids) or len(inherited) != 3556
        or len(freeze['tests']) != 155 or len(set(freeze['tests'])) != 155
        or freeze['required_test_count'] != 3584
        or {name.split('::',1)[0] for name in nodeids} != expected_files
        or not inherited.issubset(nodeids)
        or any(not nodeid.split('::',1)[0].endswith('/'+expected_new_name)
               for nodeid in set(nodeids)-inherited)):
        raise ValueError('complete saved C127 inherited/new test population differs')
    old_passes = new_passes = failed = errors = skipped = 0
    failure_details = []
    for case,nodeid in zip(cases,actual,strict=True):
        failure, error, skip = case.find('failure'), case.find('error'), case.find('skipped')
        if error is not None:
            errors += 1
        if skip is not None:
            skipped += 1
        if failure is not None:
            failed += 1
            failure_details.append(dict(nodeid=nodeid, inherited=nodeid in inherited,
                                        message=failure.get('message',''), text=failure.text or ''))
        elif error is None and skip is None:
            if nodeid in inherited:
                old_passes += 1
            else:
                new_passes += 1
    return dict(total=3584, files=155, passed=old_passes+new_passes, failed=failed,
        errors=errors, skipped=skipped, inherited_passed=old_passes, new_passed=new_passes), failure_details


def _verify_v1(freeze, exited):
    if (_sha256(FAILED_RUN/'exit.json') != V1_EXIT_SHA256
        or exited['all_stages_passed'] is not False or exited['tests_exit'] != 1
        or exited['tests_count'] != 3584 or exited.get('worker_exit') is not None
        or (FAILED_RUN/'result.json').exists() or (FAILED_RUN/'worker.log').exists()
        or any(exited[k] for k in ('formal_gain','source_drift','input_drift','provenance_drift'))):
        raise ValueError('immutable C127 v1 must remain failed before worker execution')
    partition, failures = _test_partition(FAILED_RUN, freeze, 'test_c127_systematic_kernel_proof_v1.py')
    if partition != dict(total=3584, files=155, passed=3569, failed=15,
                         errors=0, skipped=0, inherited_passed=3556, new_passed=13):
        raise ValueError('complete C127 v1 failure partition differs')
    for failure in failures:
        if (failure['inherited']
            or failure['message'] != 'ValueError: either both or neither of x and y should be given'
            or 'c127_systematic_kernel_proof_v1.py:210' not in failure['text']
            or 'np.where(minimum == 300, 0).astype(np.int32)' not in failure['text']):
            raise ValueError('C127 v1 failure root cause differs')
    old = (EXP/'c127_systematic_kernel_proof_v1.py').read_text()
    new = (EXP/'c127_systematic_kernel_proof_v2.py').read_text()
    missing = 'np.where(minimum == 300, 0).astype(np.int32)'
    fixed = 'np.where(minimum == 300, 0, minimum).astype(np.int32)'
    if old.count(missing) != 1 or new != old.replace(missing, fixed):
        raise ValueError('v2 core differs beyond the registered missing operand')
    old = (EXP/'test_c127_systematic_kernel_proof_v1.py').read_text()
    new = (EXP/'test_c127_systematic_kernel_proof_v2.py').read_text()
    if new != old.replace('c127_systematic_kernel_proof_v1 as systematic',
                          'c127_systematic_kernel_proof_v2 as systematic'):
        raise ValueError('v2 test assertions or population changed')
    return dict(partition=partition, qualification_passed=False, candidate_version_closed=True,
        worker_launched=False, complete_source_qualification_evaluated=False,
        sole_root_cause='missing_np_where_false_operand_at_core_v1_line210',
        original_failure_preserved=True, v2_is_changed_program_not_unchanged_retry=True,
        failures=[{k:v for k,v in failure.items() if k != 'text'} for failure in failures],
        exit_sha256=V1_EXIT_SHA256)


def main():
    started = time.monotonic()
    output = EXP/'C127_TERMINAL_INTEGRITY_20260927.json'
    seal = EXP/'CHECKPOINT_C127_SYSTEMATIC_KERNEL_20260927_SHA256SUMS'
    if output.exists() or output.is_symlink() or seal.exists() or seal.is_symlink():
        raise FileExistsError('exclusive C127 terminal record or checkpoint seal exists')
    for name in (*FROZEN_V1,*FROZEN_V2,*NEW_DOCUMENTS):
        if not (EXP/name).is_file():
            raise ValueError('complete C127 checkpoint source/document missing: '+name)
    if _sha256(RUN/'exit.json') != EXIT_SHA256:
        raise ValueError('completed C127 v2 exit differs from pinned identity')
    result_path = RUN/'result.json'
    if _sha256(result_path) != RESULT_SHA256:
        raise ValueError('completed C127 v2 result differs from pinned identity')
    freeze = json.loads((RUN/'preregistered.json').read_text())
    exited = json.loads((RUN/'exit.json').read_text())
    failed_freeze = json.loads((FAILED_RUN/'preregistered.json').read_text())
    failed_exit = json.loads((FAILED_RUN/'exit.json').read_text())
    result = json.loads(result_path.read_text())
    carried_work = result.get('work')
    if type(carried_work) is not int or carried_work != 243478590:
        raise ValueError('complete raw v2 paid work differs from completed run')
    pool = WorkPool(256000000)
    pool.charge('carried_complete_C127_v2_paid_generation_and_evidence_work', carried_work)
    allowance = JsonAllowance(pool, 65536, 'c127_saved_JSON_archive_exact_terminal_reservation')
    if pool.used != 243544126:
        raise ValueError('complete run plus terminal archive reservation differs')
    v1 = _verify_v1(failed_freeze, failed_exit)
    if (freeze['prior_C127_v1_complete_qualification_passed'] is not False
        or freeze['prior_C127_v1_exit_sha256'] != V1_EXIT_SHA256
        or freeze['sole_v2_core_change'] != 'restore_missing_np_where_false_operand'
        or not freeze['unchanged_v1_test_assertions_and_cases']):
        raise ValueError('frozen corrected-version authority or failed history differs')

    expected = {}
    def bind(path, digest):
        name = str(path.resolve())
        if name in expected and expected[name] != digest:
            raise ValueError('conflicting inherited/frozen file identity: '+name)
        expected[name] = digest
    run_populations = []
    for run, frozen, done, pinned, own_names in (
        (FAILED_RUN,failed_freeze,failed_exit,V1_EXIT_SHA256,FROZEN_V1),
        (RUN,freeze,exited,EXIT_SHA256,FROZEN_V2)):
        for name,digest in frozen['source_sha256'].items():
            bind(EXP/name,digest)
        for name,digest in done['artifacts'].items():
            bind(run/name,digest)
        bind(run/'exit.json',pinned)
        for name in own_names:
            if name not in frozen['source_sha256']:
                raise ValueError('complete version-specific frozen source omitted: '+name)
        files = sorted(path for path in run.rglob('*') if path.is_file())
        if {str(path.relative_to(run)) for path in files} != set(done['artifacts']) | {'exit.json'}:
            raise ValueError('complete saved exclusive run file population differs')
        run_populations.append(files)
    if len(run_populations[0]) != 6:
        raise ValueError('failed v1 must retain exactly its six complete run files')
    bind(result_path,RESULT_SHA256)
    for name,digest in PRIOR_SEALS.items():
        path = EXP/name
        if _sha256(path) != digest:
            raise ValueError('complete prior checkpoint seal differs: '+name)
        bind(path,digest)
        for line in path.read_text().splitlines():
            item_hash, relative = line.split('  ',1)
            bind(ROOT/relative,item_hash)
    for version in ('C125','C126'):
        prior = json.loads((EXP/(version+'_TERMINAL_INTEGRITY_20260927.json')).read_text())
        if (prior['completed'] is not True or prior['complete_qualification_passed'] is not True
            or prior['ordinary_qualification_passed'] is not True
            or prior['qualification_scope'] != 'fixed_complete_ordinary_HZ_only'
            or prior['exit_sha256'] != PRIOR_EXITS[version]
            or prior['prior_C124_complete_qualification_passed'] is not False
            or prior['mismatches'] or prior['input_mismatches'] or not prior['provenance_unchanged']
            or prior['formal_gain']):
            raise ValueError('prior ordinary-only qualification must remain immutable: '+version)
    prior = json.loads((EXP/'C124_TERMINAL_INTEGRITY_20260927.json').read_text())
    if (prior['complete_qualification_passed'] is not False or not prior['candidate_version_closed']
        or not prior['completed'] or not prior['completion_means_archive_integrity_only']
        or not prior['this_terminal_audit_overrides_raw_complete_qualification_claim']
        or not prior['raw_wrapper_green_flags_preserved_not_rewritten']
        or prior['mismatches'] or prior['input_mismatches'] or not prior['provenance_unchanged']):
        raise ValueError('C124 must remain failed and immutable')
    inputs = {}
    for frozen in (failed_freeze,freeze):
        for name,digest in frozen['input_sha256'].items():
            if name in inputs and inputs[name] != digest:
                raise ValueError('conflicting original input identity')
            inputs[name] = digest
    mismatches = [name for name,digest in expected.items() if _sha256(Path(name)) != digest]
    input_mismatches = [name for name,digest in inputs.items() if _sha256(Path(name)) != digest]
    provenance = _provenance(ROOT)
    unchanged = provenance == freeze['provenance'] == failed_freeze['provenance']
    intact = not mismatches and not input_mismatches and unchanged
    qualification_errors = []
    details, v2_tests = {}, None
    try:
        v2_tests, _ = _test_partition(RUN,freeze,'test_c127_systematic_kernel_proof_v2.py')
        if exited.get('work') != result['work']:
            raise ValueError('raw exit/result paid work differs')
        if len(run_populations[1]) != 36:
            raise ValueError('complete successful v2 archive requires all36 run files')
        details = _qualified_v2(result,exited,freeze)
    except Exception as error:
        qualification_errors.append(dict(type=type(error).__name__, reason=str(error)))
    qualified = intact and not qualification_errors
    if exited.get('all_stages_passed') is not True:
        qualified = False
        qualification_errors.append(dict(type='RawRunFailure', reason=exited.get('failure')))
    record = dict(completed=intact, archive_integrity_passed=intact,
        ordinary_qualification_passed=qualified, complete_qualification_passed=qualified,
        qualification_scope='fixed_complete_ordinary_HZ_only',
        completion_means_archive_integrity_only=not qualified,
        candidate_version_closed=True, qualification_errors=qualification_errors,
        raw_v2_all_stages_passed=exited.get('all_stages_passed'),
        raw_v2_failure=exited.get('failure'), raw_v2_result_failure=result.get('failure'),
        raw_v2_flags_and_files_preserved_not_rewritten=True,
        this_terminal_audit_overrides_raw_complete_qualification_claim=True,
        prior_C127_v1_complete_qualification_passed=False, complete_v1_failed_history=v1,
        prior_C126_complete_qualification_passed=True, prior_C125_complete_qualification_passed=True,
        prior_C124_complete_qualification_passed=False, prior_C124_failure_not_repaired_or_requalified=True,
        complete_inherited_tests=3556, new_tests=28, total_tests=3584, total_test_files=155,
        v2_test_partition=v2_tests, test_collection_and_execution_wall_s=exited.get('test_wall_s'),
        checked_unique_files=len(expected), checked_file_bytes=sum(Path(name).stat().st_size for name in expected),
        mismatches=mismatches, input_mismatches=input_mismatches,
        provenance=provenance, provenance_unchanged=unchanged,
        paid_run_work=carried_work, paid_run_plus_archive_reporting=pool.used,
        archive_reporting_work_parts=pool.parts, archive_JSON_reservation=65536,
        numerical_proofs_rerun=False, archived_numeric_arrays_reanalysed=False,
        target_source_or_LIVE_admitted=False, original_network_loaded=False,
        reusable_preparation_or_cache_proved=False, solver_calls=0, formal_gain=0,
        speedup_claimed=False, baseline_formal_results_changed=False,
        hash_traffic_separate_from_generation_tokens=True, all_CPU_work_in_generation_cap=False,
        v1_exit_sha256=V1_EXIT_SHA256, exit_sha256=EXIT_SHA256,
        result_sha256=RESULT_SHA256, wall_s=time.monotonic()-started,
        **details)
    receipt = allowance.write(output,record)
    if not intact:
        raise ValueError('terminal C127 archive integrity failed; failed record retained')
    names = [EXP/name for name in (*FROZEN_V1,*FROZEN_V2,*NEW_DOCUMENTS,'run_c127_archive_audit_v1.py')]
    names += [output,*run_populations[0],*run_populations[1]]
    expected_count = 16+3+1+1+sum(map(len,run_populations))
    if len(names) != expected_count or len(set(names)) != expected_count:
        raise ValueError('complete combined C127 checkpoint seal population differs')
    if qualified and expected_count != 63:
        raise ValueError('qualified C127 must seal all63 current source/document/run files')
    seal_bytes = ''.join(_sha256(path)+'  '+str(path.relative_to(ROOT))+'\n'
                         for path in sorted(names)).encode('utf-8')
    publish_bytes(seal,seal_bytes)
    print(json.dumps(dict(completed=True, ordinary_qualification_passed=qualified,
        prior_C127_v1_complete_qualification_passed=False, sealed_files=len(names),
        paid_run_plus_archive_reporting=pool.used, terminal_receipt=receipt,
        seal_sha256=_sha256(seal), formal_gain=0)),flush=True)
    if not qualified:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
