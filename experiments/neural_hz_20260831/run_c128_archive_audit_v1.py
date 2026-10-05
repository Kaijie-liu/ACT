"""Authenticate failed C128v1 and the complete C128v2 saved disposition.

Only saved JSON/XML, complete file hashes and provenance are inspected.  No
NumPy/SciPy, model, source constructor, array archive or proof is imported or
executed.  Raw failure remains failure, with all completed stage files retained.
Final run identities and scalar observations must be pinned before execution.
"""
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831.c125_exact_json_v1 import JsonAllowance, publish_bytes
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256

EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c128_channel_support_20260927_v2'
FAILED_RUN = EXP/'results/c128_channel_support_20260927_v1'
PRIOR_RUN = EXP/'results/c127_systematic_kernel_20260927_v2'
V1_EXIT_SHA256 = '243e8a544c877553227c418ee84e32850eb5a2777a34445a28dc6b682934720d'
V1_RESULT_SHA256 = 'ba666689328184398facfd4164399c58b39a323b65580fc1022b3c52d8320584'
V1_DENSE_PACKET_SHA256 = '71a357583a941cdb4e0a85dc7eebd0105fa42e824537024d1e5bbaaa055da71f'
EXIT_SHA256 = '53d2b6bd379a9b33313956fa762feaa92aee5681a537535cd03f90035f8b6c36'
RESULT_SHA256 = 'dd22494013bd014693a1b2c0e129c55206c60eda3c8c81d2e775c7903c7cc925'
FIXED_PAID_WORK = 234693760
FIXED_CATEGORY_WORK = dict(source=200000000,setup=1083872,comparison=7911728,
                           proof=4802560,evidence=19303264,ledger=1592336)
PRIOR_EXIT_SHA256 = '1b1ba7fa7a281e255a8f435b268a9b790b1d2c385611029251b477460b631238'
PRIOR_RESULT_SHA256 = '519ca415a32b406be523779519bee8d21b0288bfd911130604e9437b2bbb8bea'
PRIOR_V1_EXIT_SHA256 = 'eaccb8b1b3848833a7405fd1e9c199f00784ec8dad0b10465184c3fec2235c7c'
PRIOR_SEALS = {
    'CHECKPOINT_C127_SYSTEMATIC_KERNEL_20260927_SHA256SUMS':
        '11ff89c1b98349adf91f281213edb057258b1cfd2bd76775c812cd692e81298f',
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
    'c128_channel_support_v1.py', 'c128_channel_graph_v1.py',
    'c128_birth_emission_v1.py', 'c128_complete_source_v1.py',
    'c128_source_worker_v1.py', 'test_c128_channel_support_v1.py',
    'run_c128_source_supervisor_v1.py', 'C128_CHANNEL_SUPPORT_PREREG_20260927.md',
    'C128_SOURCE_BUDGET_PREFLIGHT_20260927.md')
FROZEN_V2 = (
    'c128_complete_source_v2.py', 'c128_source_worker_v2.py',
    'test_c128_budget_metadata_v2.py', 'run_c128_source_supervisor_v2.py',
    'C128_CHANNEL_SUPPORT_PREREG_20260927_v2.md', 'C128_V1_FAILURE_AUDIT_20260927.md')
FROZEN_FILES = (*FROZEN_V1,*FROZEN_V2)
NEW_DOCUMENTS = (
    'C128_CHANNEL_SUPPORT_AUDIT_20260927.md',
    'CHECKPOINT_C128_CHANNEL_SUPPORT_20260927.md',
    'C128_COUPLED_WORK_HANDOFF_20260927.md',
    'C128_EVIDENCE_CUSTODY_HANDOFF_20260927.md')
MODES = ('dense', 'masked', 'heterogeneous', 'noop')
STAGES = ('source', 'old', 'new', 'owners')
CATEGORY_CAPS = dict(source=200000000, setup=2000000, comparison=10000000,
                     proof=6000000, evidence=20000000, ledger=16000000)
CASE_INVENTORY = dict(dense=(2688,1601,2112,12246,178267),
    masked=(2544,1457,1968,11670,139532),
    heterogeneous=(2548,1461,1972,11686,139840),
    noop=(2128,1041,1552,10006,32748))


class ArchivePool:
    """Integer-only C14-compatible monotone ledger; no numerical imports."""
    def __init__(self, cap):
        if type(cap) is not int or not 0 <= cap <= 256000000:
            raise ValueError('invalid or increased complete archive work cap')
        self.cap, self.used, self.parts = cap, 0, {}

    def charge(self, name, amount):
        if type(amount) is not int or amount < 0:
            raise ValueError('invalid complete archive work charge')
        if self.used + amount > self.cap:
            raise MemoryError('complete run plus archive exceeds unchanged whole cap')
        self.used += amount
        self.parts[name] = self.parts.get(name, 0) + amount


def _test_partition(run, freeze, *, v2):
    inventory = json.loads((run/'inventory.json').read_text())
    inherited_freeze = json.loads((PRIOR_RUN/'preregistered.json').read_text())
    inherited = set(json.loads((PRIOR_RUN/'inventory.json').read_text())['nodeids'])
    nodeids = inventory['nodeids']
    expected_files = {str((EXP/name).resolve().relative_to(ROOT)) for name in freeze['tests']}
    core_prefix = str((EXP/'test_c128_channel_support_v1.py').relative_to(ROOT))+'::'
    new_prefix = str((EXP/'test_c128_budget_metadata_v2.py').relative_to(ROOT))+'::'
    new_ids = set(nodeids)-inherited
    core_ids = {name for name in new_ids if name.startswith(core_prefix)}
    budget_ids = {name for name in new_ids if name.startswith(new_prefix)}
    count, files = (3636,157) if v2 else (3632,156)
    new_files = ['test_c128_channel_support_v1.py']+(['test_c128_budget_metadata_v2.py'] if v2 else [])
    if (freeze['tests'] != inherited_freeze['tests']+new_files
        or inherited_freeze['required_test_count'] != 3584 or len(inherited_freeze['tests']) != 155
        or len(inherited) != 3584 or freeze['required_test_count'] != count
        or len(freeze['tests']) != files or len(set(freeze['tests'])) != files
        or inventory['count'] != count or len(nodeids) != count or len(set(nodeids)) != count
        or {name.split('::',1)[0] for name in nodeids} != expected_files
        or not inherited.issubset(nodeids) or len(core_ids) != 48
        or len(budget_ids) != (4 if v2 else 0) or new_ids != core_ids|budget_ids):
        raise ValueError('complete3584 plus unchanged48 and v2-only4 test inventory differs')
    if v2:
        failed_ids = set(json.loads((FAILED_RUN/'inventory.json').read_text())['nodeids'])
        if failed_ids != inherited|core_ids or set(nodeids) != failed_ids|budget_ids:
            raise ValueError('all3632 passing v1 tests must remain literal v2 inherited cases')
    cases = ET.parse(run/'tests.xml').findall('.//testcase')
    actual = [case.get('classname','').replace('.','/')+'.py::'+case.get('name','') for case in cases]
    if sorted(actual) != sorted(nodeids):
        raise ValueError('complete JUnit and collected node populations differ')
    old_passes = core_passes = new_passes = failures = errors = skipped = 0
    for case, nodeid in zip(cases, actual, strict=True):
        failed, error, skip = (case.find(key) is not None for key in ('failure','error','skipped'))
        failures += int(failed)
        errors += int(error)
        skipped += int(skip)
        if not failed and not error and not skip:
            if nodeid in inherited:
                old_passes += 1
            elif nodeid in core_ids:
                core_passes += 1
            else:
                new_passes += 1
    return dict(total=count, files=files, inherited_passed=old_passes,
        unchanged_core_passed=core_passes,new_budget_passed=new_passes,
        passed=old_passes+core_passes+new_passes, failed=failures, errors=errors, skipped=skipped)


def _history():
    prior = json.loads((EXP/'C127_TERMINAL_INTEGRITY_20260927.json').read_text())
    exited = json.loads((PRIOR_RUN/'exit.json').read_text())
    if (prior['completed'] is not True or prior['ordinary_qualification_passed'] is not True
        or prior['complete_qualification_passed'] is not True
        or prior['qualification_scope'] != 'fixed_complete_ordinary_HZ_only'
        or prior['exit_sha256'] != PRIOR_EXIT_SHA256 or prior['result_sha256'] != PRIOR_RESULT_SHA256
        or _sha256(PRIOR_RUN/'exit.json') != PRIOR_EXIT_SHA256
        or _sha256(PRIOR_RUN/'result.json') != PRIOR_RESULT_SHA256
        or not exited['all_stages_passed'] or exited['tests_count'] != 3584
        or exited['tests_exit'] or exited['worker_exit']
        or prior['prior_C127_v1_complete_qualification_passed'] is not False
        or prior['prior_C126_complete_qualification_passed'] is not True
        or prior['prior_C125_complete_qualification_passed'] is not True
        or prior['prior_C124_complete_qualification_passed'] is not False
        or prior['qualification_errors'] or prior['mismatches'] or prior['input_mismatches']
        or not prior['provenance_unchanged'] or prior['formal_gain']):
        raise ValueError('qualified ordinary-only C127v2 lineage differs')
    failed_run = EXP/'results/c127_systematic_kernel_20260927_v1'
    failed = json.loads((failed_run/'exit.json').read_text())
    v1 = prior['complete_v1_failed_history']
    if (_sha256(failed_run/'exit.json') != PRIOR_V1_EXIT_SHA256
        or failed['all_stages_passed'] is not False or failed['tests_exit'] != 1
        or failed.get('worker_exit') is not None or (failed_run/'result.json').exists()
        or v1['qualification_passed'] is not False or v1['worker_launched']
        or not v1['original_failure_preserved'] or not v1['candidate_version_closed']
        or v1['partition'] != dict(total=3584,files=155,passed=3569,failed=15,
                                  errors=0,skipped=0,inherited_passed=3556,new_passed=13)
        or len(v1['failures']) != 15):
        raise ValueError('failed original C127v1 disposition differs')
    for version, digest in PRIOR_EXITS.items():
        item = json.loads((EXP/(version+'_TERMINAL_INTEGRITY_20260927.json')).read_text())
        if (item['completed'] is not True or item['ordinary_qualification_passed'] is not True
            or item['complete_qualification_passed'] is not True
            or item['qualification_scope'] != 'fixed_complete_ordinary_HZ_only'
            or item['exit_sha256'] != digest or item['prior_C124_complete_qualification_passed'] is not False
            or item['mismatches'] or item['input_mismatches'] or not item['provenance_unchanged']
            or item['formal_gain']):
            raise ValueError('inherited ordinary-only qualification differs: '+version)
    failed124 = json.loads((EXP/'C124_TERMINAL_INTEGRITY_20260927.json').read_text())
    if (failed124['complete_qualification_passed'] is not False or not failed124['completed']
        or not failed124['candidate_version_closed'] or not failed124['completion_means_archive_integrity_only']
        or not failed124['this_terminal_audit_overrides_raw_complete_qualification_claim']
        or not failed124['raw_wrapper_green_flags_preserved_not_rewritten']
        or failed124['mismatches'] or failed124['input_mismatches']
        or not failed124['provenance_unchanged'] or failed124['formal_gain']):
        raise ValueError('C124 must remain archived failure, not requalified')
    return dict(prior_C127_v2_exit_sha256=PRIOR_EXIT_SHA256,
        prior_C127_v2_result_sha256=PRIOR_RESULT_SHA256,
        prior_C127_v1_exit_sha256=PRIOR_V1_EXIT_SHA256,
        prior_C127_v1_failed_test_partition=v1['partition'])


def _saved_semantics(value, reserve):
    """JSON-only validation of the exact original coupled-work diagnostic."""
    report = value['report']
    whole,branch = report['whole_base_work'],report['branch_base_work']
    coupled = report['alias_quotient']
    capacity,used = coupled['coupled_extra_capacity'],coupled['coupled_extra_work']
    if (any(type(v) is not int or v < 0 for v in (whole,branch,capacity,used))
        or max(whole,branch) > reserve
        or capacity != min(reserve-whole,reserve-branch) or used > capacity
        or report['total_work_upper'] != whole+used
        or report['largest_branch_work_upper'] != branch+used):
        raise ValueError('saved original coupled-work pool formula differs')
    work_fields = {'support_work','total_work_upper','largest_branch_work_upper',
                   'affine_work_upper','whole_base_work','branch_base_work'}
    normalized = {key:entry for key,entry in report.items() if key not in work_fields}
    normalized['alias_quotient'] = dict(coupled,coupled_extra_capacity='exact_original_pool_formula_checked')
    normalized['node_counts'] = [{key:entry for key,entry in count.items() if key != 'support_work'}
                                  for count in report['node_counts']]
    nodes = [{key:entry for key,entry in node.items() if key != 'support_work'} for node in value['nodes']]
    return dict(value,report=normalized,nodes=nodes)


def _failed_v1(freeze, exited):
    result_path = FAILED_RUN/'result.json'
    result = json.loads(result_path.read_text())
    if (_sha256(FAILED_RUN/'exit.json') != V1_EXIT_SHA256 or _sha256(result_path) != V1_RESULT_SHA256
        or exited['all_stages_passed'] is not False or exited['tests_exit'] != 0
        or exited['worker_exit'] != 1 or exited['tests_count'] != 3632 or exited['test_wall_s'] > 60
        or result['completed'] is not False or result['measurement']['build_returned'] is not False
        or result['failure'] != dict(type='ValueError',reason='full source graph/frame/predicate/UID metadata differs')
        or result['work'] != 73970808 or sum(result['work_parts'].values()) != 73970808
        or result['formal_gain'] or result['solver_calls'] or exited['formal_gain']
        or any(result[key] or exited[key] for key in ('source_drift','input_drift','provenance_drift'))):
        raise ValueError('frozen C128v1 must remain failed at the original metadata comparison')
    partition = _test_partition(FAILED_RUN,freeze,v2=False)
    if partition != dict(total=3632,files=156,inherited_passed=3584,
                         unchanged_core_passed=48,new_budget_passed=0,passed=3632,failed=0,errors=0,skipped=0):
        raise ValueError('all3632 passing v1 tests must remain distinct from failed source qualification')
    exact = (json.dumps(result,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode()
    if (result_path.read_bytes() != exact or len(exact)+1024 > 524288
        or result['terminal_success_reservation'] != 524288
        or result['terminal_failure_reservation'] != 65536 or result['complete_ledger_reservation'] != 131072
        or not result['terminal_allowances_prepaid_before_build']):
        raise ValueError('failed v1 exact terminal payment differs')
    stage_records, stages = {}, []
    for stage, arrays, entries in (('source',29,12246),('old',48,178267),('new',48,178267)):
        path = FAILED_RUN/('dense_'+stage+'.json')
        raw = path.read_bytes()
        record = json.loads(raw)
        canonical = (json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode()
        artifact = record['artifact']
        numeric_path = FAILED_RUN/('dense_'+stage+'_arrays.npz')
        if (raw != canonical or len(raw)+1024 > 65536 or record['mode'] != 'dense'
            or record['stage'] != stage or record['proof_completed_at_save'] is not False or record['formal_gain']
            or artifact['file'] != numeric_path.name or artifact['sha256'] != _sha256(numeric_path)
            or artifact['stored_bytes'] != numeric_path.stat().st_size
            or artifact['arrays'] != arrays or len(artifact['complete_array_manifest']) != arrays
            or artifact['numeric_entries'] != entries
            or sum(item['entries'] for item in artifact['complete_array_manifest'].values()) != entries
            or artifact['encoding_work'] != 1024+16*entries):
            raise ValueError('complete retained failed-v1 stage or exact payment differs: '+stage)
        stage_records[stage] = record
        stages.append(dict(stage=stage,arrays=arrays,entries=entries,
                           artifact_sha256=artifact['sha256'],numeric_payload_loaded=False))
    old,new = (stage_records[version]['metadata'] for version in ('old','new'))
    if (stage_records['old']['artifact']['sha256'] != V1_DENSE_PACKET_SHA256
        or stage_records['new']['artifact']['sha256'] != V1_DENSE_PACKET_SHA256
        or stage_records['old']['artifact']['complete_array_manifest'] != stage_records['new']['artifact']['complete_array_manifest']
        or old['report']['alias_quotient']['coupled_extra_capacity'] != 30897272
        or new['report']['alias_quotient']['coupled_extra_capacity'] != 30983608
        or old['report']['alias_quotient']['coupled_extra_work'] != 247860
        or new['report']['alias_quotient']['coupled_extra_work'] != 247860
        or _saved_semantics(old,32000000) != _saved_semantics(new,32000000)):
        raise ValueError('saved failed-v1 capacity-only comparator mismatch differs')
    expected_files = {'preregistered.json','collection.log','inventory.json','tests.log','tests.xml',
                      'worker.log','fatal.log','phases.jsonl','result.json','exit.json'}
    expected_files.update('dense_'+stage+suffix for stage in ('source','old','new')
                          for suffix in ('.json','_arrays.npz'))
    if {str(path.relative_to(FAILED_RUN)) for path in FAILED_RUN.rglob('*') if path.is_file()} != expected_files:
        raise ValueError('failed C128v1 must preserve exactly its sixteen original artifacts')
    return dict(qualification_passed=False,candidate_version_closed=True,original_failure_preserved=True,
        test_partition=partition,complete_four_source_qualification_reached=False,
        independent_complete_owner_inverse_proof_reached=False,actual_complete_held_ledger_reached=False,
        retained_raw_run_files=16,retained_stage_arrays=125,retained_numeric_entries=368780,
        complete_saved_stages=stages,all_saved_dense_packet_bytes_identical=True,
        packet_byte_equality_does_not_override_failed_gate=True,paid_incomplete_work=73970808,
        sole_remaining_metadata_difference='report.alias_quotient.coupled_extra_capacity',
        raw_failure=result['failure'],exit_sha256=V1_EXIT_SHA256,result_sha256=V1_RESULT_SHA256)


def _failed_v2(result, exited, freeze, partition):
    """Validate the saved completed prefix without inventing the missing ledger."""
    measurement = result['measurement']
    failure = dict(type='WholeStateReject',
        reason="incompatible_storage_alias:active['508'].terms[0].source.value.Gc.data")
    if (exited['all_stages_passed'] is not False or result['completed'] is not False
        or exited['tests_exit'] != 0 or exited['worker_exit'] != 1 or exited['tests_count'] != 3636
        or result['failure'] != failure or 'data' in result
        or (RUN/'complete_held_ledger.json').exists()
        or 'complete_evidence_payment_checks' in exited
        or partition != dict(total=3636,files=157,inherited_passed=3584,
            unchanged_core_passed=48,new_budget_passed=4,passed=3636,failed=0,errors=0,skipped=0)
        or exited['test_wall_s'] > 60 or result['wall_s'] > 240
        or freeze['complete_test_wall_cap_s'] != 60 or freeze['stage_worker_wall_cap_s'] != 240
        or result['work'] != FIXED_PAID_WORK or sum(result['work_parts'].values()) != FIXED_PAID_WORK
        or freeze['category_caps'] != CATEGORY_CAPS or sum(CATEGORY_CAPS.values()) != 254000000
        or freeze['complete_work_upper'] != 254000000 or FIXED_PAID_WORK > 254000000
        or freeze['whole_work_cap'] != 256000000 or freeze['branch_work_cap'] != 200000000
        or freeze['entries_cap'] != 64000000 or freeze['cpu_threads'] != 1 or freeze['gpu_enabled']
        or freeze['address_space_bytes'] != 16*1024**3 or freeze['transient_bytes'] != 1073741824
        or measurement['build_returned'] is not False or not measurement['measured_transient_gate']
        or measurement['transient_cap_bytes'] != 1073741824
        or measurement['resident_growth_upper_bound_bytes'] != 23756800
        or measurement['traced_peak_bytes'] != 13812298 or measurement['tracer_metadata_bytes'] != 5775296
        or measurement['resident_growth_upper_bound_bytes'] > 1073741824
        or measurement['traced_peak_bytes']+measurement['tracer_metadata_bytes'] > 1073741824
        or any(result[key] or exited[key] for key in ('source_drift','input_drift','provenance_drift'))
        or result['solver_calls'] or result['formal_gain'] or exited['formal_gain']
        or result['numeric_hash_traffic_in_token_pool'] or result['all_CPU_work_in_generation_cap']):
        raise ValueError('failed C128v2 disposition, actual-prefix work or resource scope differs')
    if (freeze['fixed_complete_source_modes'] != list(MODES)
        or freeze['fixed_source_geometry'] != dict(C=16,K=32,input=[6,6],output=[4,4])
        or freeze['source_reservation_each'] != dict(dense=32000000,masked=32000000,
                                                    heterogeneous=32000000,noop=4000000)
        or freeze['independent_original_and_candidate_builds_per_source'] != 2
        or not freeze['full_source_graph_UID_owner_inverse_bitwise_comparison_required']
        or not freeze['full_numeric_source_artifacts_and_point_populations_required']
        or freeze['shared_radix_caps'] != [16384,131072,16000000]
        or freeze['prior_C128_v1_qualification_passed'] is not False
        or freeze['exact_original_coupled_capacity_formula_required'] is not True
        or freeze['prior_C127_v2_qualification_passed'] is not True
        or freeze['prior_C127_v1_qualification_passed'] is not False
        or freeze['prior_C124_qualification_passed'] is not False
        or any(freeze[key] for key in ('original_network_run_authorized','archived_HZ_restore_authorized',
            'solver_authorized','actual_network_source_or_LIVE_admitted',
            'kernel_cache_or_reuse_authorized','promotion_authorized','formal_gain'))):
        raise ValueError('full fixed source scope or original authority boundary differs')
    parts = result['work_parts']
    expected_parts = dict(c119_complete_original_expression_binding=559584,
        c128_all_source_graph_inverse_arrays_and_scalar_comparison=7846192,
        c128_complete_independent_sparse_owner_and_inverse=4802560,
        c128_complete_new_source_reservation=100000000,c128_complete_old_source_reservation=100000000,
        c128_complete_original_fixture_creation=524288,c128_complete_packet_source_and_archive_headers=65536,
        c128_complete_source_exact_JSON_reservation=2621440,
        c128_complete_source_graph_inverse_array_encoding=16681824,
        c128_upfront_complete_ledger_JSON_reservation=131072,
        c128_upfront_terminal_failure_JSON_reservation=65536,
        c128_upfront_terminal_success_JSON_reservation=524288,
        c62_complete_numeric_header_walk=801296,c62_complete_numeric_owner_layout=70144)
    category_work = dict(
        source=parts['c128_complete_old_source_reservation']+parts['c128_complete_new_source_reservation'],
        setup=parts['c119_complete_original_expression_binding']+parts['c128_complete_original_fixture_creation'],
        comparison=parts['c128_all_source_graph_inverse_arrays_and_scalar_comparison']
                   +parts['c128_complete_packet_source_and_archive_headers'],
        proof=parts['c128_complete_independent_sparse_owner_and_inverse'],
        evidence=parts['c128_complete_source_exact_JSON_reservation']
                 +parts['c128_complete_source_graph_inverse_array_encoding'],
        ledger=parts['c128_upfront_complete_ledger_JSON_reservation']
               +parts['c128_upfront_terminal_failure_JSON_reservation']
               +parts['c128_upfront_terminal_success_JSON_reservation']
               +parts['c62_complete_numeric_header_walk']+parts['c62_complete_numeric_owner_layout'])
    if (parts != expected_parts or category_work != FIXED_CATEGORY_WORK
        or sum(category_work.values()) != FIXED_PAID_WORK
        or any(value > CATEGORY_CAPS[key] for key,value in category_work.items())):
        raise ValueError('complete actually-paid failed-prefix category/work-parts accounting differs')
    checks, artifacts, cases = [], [], []
    def saved_json(name, reserve):
        path = RUN/name
        raw = path.read_bytes()
        payload = json.loads(raw)
        canonical = (json.dumps(payload,sort_keys=True,separators=(',',':'),
                                ensure_ascii=True,allow_nan=False)+'\n').encode()
        if raw != canonical or len(raw)+1024 > reserve or _sha256(path) != exited['artifacts'][name]:
            raise ValueError('actual retained compact JSON bytes exceed their fixed payment: '+name)
        checks.append(dict(file=name,actual_bytes=len(raw),prepaid_bytes=reserve,
                           sha256=exited['artifacts'][name],passed=True))
        return payload
    packet_names = {'keep','eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows',
        'owners','uid_slabs','radix_gauges','eq_uids','ineq_uids','hz_c','hz_b','hz_ub'}
    packet_names.update('hz_'+matrix+'_'+part for matrix in ('Gc','Gb','Ac','Ab','Auc','Aub')
                        for part in ('data','indices','indptr'))
    packet_names.update('node_'+str(index)+'_'+part for index in range(4)
                        for part in ('support','needed','slots','exponents'))
    source_names = {'expression_bias','source_c','source_b','source_ub','operator_0_kernel'}
    source_names.update('source_'+matrix+'_'+part for matrix in ('Gc','Gb','Ac','Ab','Auc','Aub')
                        for part in ('data','indices','indptr'))
    source_names.update('operator_'+str(index)+'_'+part for index in (1,2)
                        for part in ('data','indices','indptr'))
    total_arrays = total_entries = old_support = new_support = 0
    for mode in MODES:
        report = saved_json(mode+'_complete_source.json',131072)
        points = saved_json(mode+'_complete_points.json',262144)
        nc,ne,owners,source_entries,packet_entries = CASE_INVENTORY[mode]
        reserve = freeze['source_reservation_each'][mode]
        if (report['mode'] != mode or report['source_reservation_each'] != reserve
            or report['source_or_LIVE_admitted'] or report['formal_gain']
            or report['nonconvex_binary_factors'] != 1 or report['inequalities'] != 1
            or report['arrays_per_version'] != 48 or report['entries_both_versions'] != 2*packet_entries
            or not all(report[key] for key in ('complete_source_arrays_bitwise_equal',
                'complete_semantic_metadata_equal','complete_original_source_sharing_equal',
                'full_sparse_owner_oracles_equal','complete_inverse_arrays_and_points_equal',
                'original_source_unchanged','exact_original_coupled_capacity_formula_verified'))
            or set(report['artifacts']) != set(STAGES)):
            raise ValueError('complete saved case proof scope differs: '+mode)
        header = report['complete_original_source_header']
        source = header['source']
        if (header['frame_id'] != 17 or header['n_out'] != 512 or header['term_count'] != 1
            or header['source_count'] != 1 or source['frame_id'] != 17 or not source['exact']
            or source['n_cont'] != 576 or source['n_out'] != 576 or source['n_bin'] != 1
            or source['n_eq'] != 1 or source['n_ineq'] != 1
            or not header['binding_object_ids_are_process_local_not_portable_authority']
            or len(header['operators']) != 3
            or header['operators'][0] != dict(shape=[512,576],type='ImplicitConv2DOp',
                input_shape=[1,16,6,6],output_shape=[1,32,4,4],stride=[1,1],padding=[0,0],
                dilation=[1,1],groups=1,kernel_shape=[32,16,3,3],output_row_mask_present=False)
            or any(item != dict(shape=[512,512],type='csr_matrix') for item in header['operators'][1:])):
            raise ValueError('complete original source/operator/header geometry differs: '+mode)
        manifests, records = {}, {}
        for stage in STAGES:
            stage_record = saved_json(mode+'_'+stage+'.json',65536)
            artifact = report['artifacts'][stage]
            name = mode+'_'+stage+'_arrays.npz'
            path = RUN/name
            manifest = artifact['complete_array_manifest']
            count = 29 if stage=='source' else 2 if stage=='owners' else 48
            entries = source_entries if stage=='source' else 2*owners if stage=='owners' else packet_entries
            if (stage_record['mode'] != mode or stage_record['stage'] != stage
                or stage_record['artifact'] != artifact or stage_record['formal_gain']
                or stage_record['proof_completed_at_save'] is not False
                or artifact['file'] != name or artifact['sha256'] != _sha256(path)
                or artifact['sha256'] != exited['artifacts'][name]
                or artifact['stored_bytes'] != path.stat().st_size or artifact['arrays'] != count
                or len(manifest) != count or artifact['numeric_entries'] != entries
                or sum(item['entries'] for item in manifest.values()) != entries
                or artifact['encoding_work'] != 1024+16*entries):
                raise ValueError('complete staged numeric archive differs: '+mode+'/'+stage)
            for item in manifest.values():
                size = 1
                for dimension in item['shape']:
                    if type(dimension) is not int or dimension < 0:
                        raise ValueError('invalid saved numeric array shape')
                    size *= dimension
                if size != item['entries'] or type(item['dtype']) is not str:
                    raise ValueError('complete array shape/entry metadata differs')
            manifests[stage],records[stage] = manifest,stage_record
            total_arrays += count
            total_entries += entries
            artifacts.append(dict(file=name,arrays=count,numeric_entries=entries,
                stored_bytes=artifact['stored_bytes'],sha256=artifact['sha256'],
                full_file_authenticated=True,numeric_payload_reanalysed=False))
        if (set(manifests['source']) != source_names or set(manifests['old']) != packet_names
            or manifests['old'] != manifests['new'] or set(manifests['owners']) != {'old','new'}
            or any(item['entries'] != owners for item in manifests['owners'].values())
            or records['source']['metadata'] != header
            or report['artifacts']['old']['sha256'] != report['artifacts']['new']['sha256']):
            raise ValueError('full original/control/candidate/owner archive population differs: '+mode)
        for version in ('old','new'):
            metadata = report['complete_source_metadata'][version]
            if (records[version]['metadata'] != metadata or metadata['origin_binding'] != header['origin_binding']
                or metadata['hz']['n_cont'] != nc or metadata['hz']['n_eq'] != ne
                or metadata['hz']['n_bin'] != 1 or metadata['hz']['n_ineq'] != 1
                or metadata['hz']['frame_id'] != 17 or not metadata['hz']['exact']
                or metadata['report']['source_first_native_or_LIVE_admission']
                or report['whole_work'][version] != metadata['report']['total_work_upper']
                or report['branch_work'][version] != metadata['report']['largest_branch_work_upper']
                or report['whole_work'][version] > reserve or report['branch_work'][version] > reserve):
                raise ValueError('complete source native semantics or budget differs: '+mode+'/'+version)
        if (_saved_semantics(report['complete_source_metadata']['old'],reserve) !=
            _saved_semantics(report['complete_source_metadata']['new'],reserve)
            or report['whole_work']['old']-report['whole_work']['new'] !=
                report['support_work']['old']-report['support_work']['new']):
            raise ValueError('semantic equality after exact coupled-capacity check differs: '+mode)
        receipt = report['complete_points_receipt']
        point_raw = (RUN/(mode+'_complete_points.json')).read_bytes()
        if (points['mode'] != mode or points['formal_gain'] or not points['points_are_not_network_witnesses']
            or not points['complete_old_new_original_and_recovered_populations']
            or set(points['points']) != {'old','new'} or points['points']['old'] != points['points']['new']
            or report['full_point_counts'] != {version:dict(original=nc,recovered=nc) for version in ('old','new')}
            or any(set(population) != {'original','recovered'}
                or any(type(values) is not list or len(values) != nc for values in population.values())
                for population in points['points'].values())
            or receipt['file'] != mode+'_complete_points.json' or receipt['encoded_bytes'] != len(point_raw)
            or receipt['sha256'] != exited['artifacts'][receipt['file']]
            or receipt['prepaid_bytes'] != 262144 or receipt['overhead_bytes'] != 1024
            or not receipt['exact_checked_bytes_published']
            or receipt['encoding'] != 'sorted_compact_ascii_JSON_newline'):
            raise ValueError('complete saved old/new original/recovered inverse populations differ: '+mode)
        owner_metadata = records['owners']['metadata']
        if (owner_metadata['complete_owner_counts'] != dict(old=owners,new=owners)
            or owner_metadata['full_point_counts'] != report['full_point_counts']
            or not all(owner_metadata[key] for key in ('full_sparse_owner_oracles_equal',
                'complete_inverse_arrays_and_points_equal','points_are_not_network_witnesses',
                'complete_case_qualification_pending'))):
            raise ValueError('complete saved independent owner/inverse proof record differs: '+mode)
        old_support += report['support_work']['old']
        new_support += report['support_work']['new']
        cases.append(dict(mode=mode,source_n_cont=nc,source_n_eq=ne,MAIN_owners_each=owners,
            original_source_entries=source_entries,complete_packet_entries_each=packet_entries,
            arrays=127,full_point_counts=report['full_point_counts'],support_work=report['support_work'],
            whole_work=report['whole_work'],branch_work=report['branch_work'],
            saved_case_proofs_completed=True,complete_candidate_qualification_passed=False))
    if (len(checks) != 24 or len(artifacts) != 16 or total_arrays != 508 or total_entries != 1041590
        or old_support <= new_support or parts['c128_complete_source_exact_JSON_reservation'] !=
            16*65536+4*131072+4*262144
        or parts['c128_complete_source_graph_inverse_array_encoding'] != 16*1024+16*total_entries):
        raise ValueError('full saved case/archive population or exact prepaid evidence cost differs')
    exact_result = (json.dumps(result,sort_keys=True,separators=(',',':'),
                               ensure_ascii=True,allow_nan=False)+'\n').encode()
    if ((RUN/'result.json').read_bytes() != exact_result or len(exact_result)+1024 > 524288
        or result['terminal_success_reservation'] != 524288 or result['terminal_failure_reservation'] != 65536
        or result['complete_ledger_reservation'] != 131072 or not result['terminal_allowances_prepaid_before_build']):
        raise ValueError('actual failed terminal bytes or original reservations differ')
    return dict(complete_source_cases=cases,saved_JSON_checks=checks,complete_numeric_artifacts=artifacts,
        saved_source_case_count=4,saved_independent_source_build_count=8,
        saved_numeric_artifact_count=16,saved_named_array_count=508,
        saved_archive_numeric_entries=1041590,saved_JSON_artifact_count=24,
        planned_success_receipt_count=25,complete_success_receipt_population_not_completed=True,
        full_original_control_candidate_owner_inverse_evidence_retained=True,
        all_old_new_source_archive_bytes_equal=True,saved_complete_point_populations=True,
        diagnostic_support_work=dict(old=old_support,new=new_support,saved=old_support-new_support),
        diagnostic_counters_do_not_establish_complete_qualification=True,
        actual_complete_numeric_ledger_available=False,actual_complete_known_metadata_ledger_available=False,
        actual_complete_64M_entry_gate_proved=False,complete_source_custody_qualified=False,
        failed_ledger_not_relaxed_or_filtered=True,
        raw_measurement=measurement,aborted_prefix_transient_gates_passed=True,
        measurement_scope='C41_failed_build_prefix_not_completed_whole_state_or_terminal_encoding',
        actual_paid_prefix_category_work=category_work,category_limits=CATEGORY_CAPS,
        complete_generation_payment_certificate=False,raw_failure=failure)


def main():
    started = time.monotonic()
    output = EXP/'C128_TERMINAL_INTEGRITY_20260927.json'
    seal = EXP/'CHECKPOINT_C128_CHANNEL_SUPPORT_20260927_SHA256SUMS'
    if output.exists() or output.is_symlink() or seal.exists() or seal.is_symlink():
        raise FileExistsError('exclusive C128 failed-candidate terminal record or seal exists')
    if (_sha256(RUN/'exit.json') != EXIT_SHA256 or _sha256(RUN/'result.json') != RESULT_SHA256
        or _sha256(FAILED_RUN/'exit.json') != V1_EXIT_SHA256
        or _sha256(FAILED_RUN/'result.json') != V1_RESULT_SHA256):
        raise ValueError('immutable failed C128v1/v2 raw terminal identities differ')
    for name in (*FROZEN_FILES,*NEW_DOCUMENTS):
        if not (EXP/name).is_file():
            raise ValueError('complete C128 frozen source or closing document missing: '+name)
    freeze = json.loads((RUN/'preregistered.json').read_text())
    exited = json.loads((RUN/'exit.json').read_text())
    result = json.loads((RUN/'result.json').read_text())
    v1_freeze = json.loads((FAILED_RUN/'preregistered.json').read_text())
    v1_exit = json.loads((FAILED_RUN/'exit.json').read_text())
    carried_work = result.get('work')
    if type(carried_work) is not int or carried_work != FIXED_PAID_WORK:
        raise ValueError('actual failed-v2 prefix work differs from its final pin')
    pool = ArchivePool(256000000)
    pool.charge('carried_actual_C128_v2_failed_generation_and_evidence_prefix',carried_work)
    allowance = JsonAllowance(pool,65536,'c128_saved_JSON_archive_exact_terminal_reservation')
    if pool.used != 234759296:
        raise ValueError('failed-v2 prefix plus terminal archive reservation differs')
    history = _history()
    expected = {}
    def bind(path, digest):
        name = str(path.resolve())
        if name in expected and expected[name] != digest:
            raise ValueError('conflicting inherited/frozen/archive file identity: '+name)
        expected[name] = digest
    run_populations = []
    for run,frozen,done,exit_hash,result_hash,own_sources,expected_count in (
        (FAILED_RUN,v1_freeze,v1_exit,V1_EXIT_SHA256,V1_RESULT_SHA256,FROZEN_V1,16),
        (RUN,freeze,exited,EXIT_SHA256,RESULT_SHA256,FROZEN_V2,50)):
        for name,digest in frozen['source_sha256'].items():
            bind(EXP/name,digest)
        for name,digest in done['artifacts'].items():
            bind(run/name,digest)
        bind(run/'exit.json',exit_hash)
        bind(run/'result.json',result_hash)
        if any(name not in frozen['source_sha256'] for name in own_sources):
            raise ValueError('a complete version-specific frozen source/document was omitted')
        files = sorted(path for path in run.rglob('*') if path.is_file())
        if (len(files) != expected_count
            or {str(path.relative_to(run)) for path in files} != set(done['artifacts'])|{'exit.json'}):
            raise ValueError('complete failed run file population differs from its pinned manifest')
        run_populations.append(files)
    for name in FROZEN_V1:
        if freeze['source_sha256'].get(name) != v1_freeze['source_sha256'][name]:
            raise ValueError('a frozen failed-v1 source was changed for v2: '+name)
    for name,digest in PRIOR_SEALS.items():
        path = EXP/name
        if _sha256(path) != digest:
            raise ValueError('prior immutable checkpoint seal differs: '+name)
        bind(path,digest)
        for line in path.read_text().splitlines():
            item_hash,relative = line.split('  ',1)
            bind(ROOT/relative,item_hash)
    inputs = dict(freeze['input_sha256'])
    prior_freeze = json.loads((PRIOR_RUN/'preregistered.json').read_text())
    if inputs != prior_freeze['input_sha256'] or inputs != v1_freeze['input_sha256']:
        raise ValueError('original input population changed across qualified/failed predecessors')
    mismatches = [name for name,digest in expected.items() if _sha256(Path(name)) != digest]
    input_mismatches = [name for name,digest in inputs.items() if _sha256(Path(name)) != digest]
    provenance = _provenance(ROOT)
    unchanged = provenance == freeze['provenance'] == v1_freeze['provenance'] == prior_freeze['provenance']
    intact = not mismatches and not input_mismatches and unchanged
    errors,details,v1,partition = [],{},None,None
    try:
        v1 = _failed_v1(v1_freeze,v1_exit)
        partition = _test_partition(RUN,freeze,v2=True)
        details = _failed_v2(result,exited,freeze,partition)
    except Exception as error:
        errors.append(dict(type=type(error).__name__,reason=str(error)))
    completed = intact and not errors
    names = [EXP/name for name in (*FROZEN_FILES,*NEW_DOCUMENTS,'run_c128_archive_audit_v1.py')]
    names += [output,*run_populations[0],*run_populations[1]]
    if len(names) != 87 or len(set(names)) != 87:
        raise ValueError('complete failed-v1/v2 C128 seal must contain all87 files')
    record = dict(completed=completed,archive_integrity_passed=intact,
        ordinary_qualification_passed=False,complete_qualification_passed=False,
        qualification_scope='failed_complete_ordinary_source_HZ_candidates',
        completion_means_archive_integrity_only=True,candidate_version_closed=True,
        both_C128_candidate_versions_closed_failed=True,archive_audit_errors=errors,
        qualification_errors=[dict(version='v1',reason='nested_coupled_capacity_comparator_mismatch'),
                              dict(version='v2',reason='incompatible_storage_alias_in_unchanged_complete_ledger')],
        raw_v1_all_stages_passed=v1_exit.get('all_stages_passed'),
        raw_v2_all_stages_passed=exited.get('all_stages_passed'),
        raw_v2_supervisor_failure=exited.get('failure'),
        raw_v2_result_failure=result.get('failure'),
        raw_flags_and_files_preserved_not_rewritten=True,
        this_terminal_audit_overrides_raw_complete_qualification_claim=False,
        raw_failure_disposition_confirmed=True,
        prior_C128_v1_complete_qualification_passed=False,complete_v1_failed_history=v1,
        prior_C127_v2_complete_qualification_passed=True,prior_C127_v1_complete_qualification_passed=False,
        prior_C126_complete_qualification_passed=True,prior_C125_complete_qualification_passed=True,
        prior_C124_complete_qualification_passed=False,prior_failures_not_repaired_or_requalified=True,
        qualified_inherited_tests=3584,unchanged_C128_v1_tests=48,new_C128_v2_tests=4,
        total_tests=3636,total_test_files=157,test_partition=partition,
        test_collection_and_execution_wall_s=exited.get('test_wall_s'),
        passing_tests_do_not_override_failed_complete_candidate=True,
        checked_unique_files=len(expected),checked_file_bytes=sum(Path(name).stat().st_size for name in expected),
        retained_v1_run_files=16,retained_v2_run_files=50,sealed_files=87,
        mismatches=mismatches,input_mismatches=input_mismatches,
        provenance=provenance,provenance_unchanged=unchanged,
        paid_run_work=carried_work,paid_run_plus_archive_reporting=pool.used,
        prior_v1_paid_incomplete_work=73970808,prior_v1_work_not_recharged_to_new_attempt=True,
        archive_reporting_work_parts=pool.parts,archive_JSON_reservation=65536,
        numerical_proofs_rerun=False,archived_numeric_arrays_reanalysed=False,
        frozen_success_verifier_not_called_without_missing_ledger=True,
        missing_success_data_or_ledger_not_synthesized=True,
        target_source_or_LIVE_admitted=False,original_network_loaded=False,
        combined_source_F4_path_qualified=False,reusable_preparation_or_cache_proved=False,
        solver_calls=0,formal_gain=0,speedup_claimed=False,baseline_formal_results_changed=False,
        hash_traffic_separate_from_generation_tokens=True,all_CPU_work_in_generation_cap=False,
        v1_exit_sha256=V1_EXIT_SHA256,v1_result_sha256=V1_RESULT_SHA256,
        exit_sha256=EXIT_SHA256,result_sha256=RESULT_SHA256,wall_s=time.monotonic()-started,
        **history,**details)
    receipt = allowance.write(output,record)
    if not completed:
        raise ValueError('C128 archive integrity/evidence audit failed; failed terminal record retained')
    seal_bytes = ''.join(_sha256(path)+'  '+str(path.relative_to(ROOT))+'\n'
                         for path in sorted(names)).encode('utf-8')
    publish_bytes(seal,seal_bytes)
    print(json.dumps(dict(completed=True,ordinary_qualification_passed=False,
        both_C128_candidate_versions_closed_failed=True,sealed_files=87,
        paid_run_plus_archive_reporting=pool.used,terminal_receipt=receipt,
        seal_sha256=_sha256(seal),formal_gain=0)),flush=True)


if __name__ == '__main__':
    main()
