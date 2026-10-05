"""Saved-only C130 terminal audit: JSON/XML, full hashes and provenance.

No numerical module, NPZ payload, source constructor, test or numerical proof is run.
The frozen verifier rechecks complete saved scalar accounting certificates.
Success reuses the frozen supervisor's complete saved-evidence verifier; raw
failure remains failure. Actual terminal identities and observations must be
pinned after the sole run, before this unexecuted helper may publish anything.
"""
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c125_exact_json_v1 import JsonAllowance, publish_bytes
from experiments.neural_hz_20260831.run_c130_source_supervisor_v1 import verify_saved_evidence, header_occurrences
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256

EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c130_nodewise_branch_20260927_v1'
PRIOR_RUN = EXP/'results/c129_owned_evidence_20260927_v1'
# Actual observations pinned from saved JSON and exact file hashes only,
# after the sole frozen run terminated; no numerical artifacts were decoded.
EXIT_SHA256 = '852f0912305fcee0c13fd4cdb9a1ad32b7e05b37b138c3f38f3b7bbfb596b32b'
RESULT_SHA256 = 'db25a599663447af1b3b608d86809b25a15130c0ffc59d0dc3bbb8bd7353dee3'
FIXED_PAID_WORK = 244830488
FIXED_CATEGORY_WORK = dict(source=200000000,setup=1083872,comparison=7928880,
    proof=4802560,evidence=19303264,custody=8223344,ledger=3488568)
FIXED_NUMERIC = dict(resident_entries=1620452,resident_bytes=12290496,numeric_storage_count=1012)
FIXED_METADATA_BYTES = 3737875
FIXED_OPAQUE_COUNT = 8
FIXED_TRANSIENT_OBSERVATIONS = dict(resident_growth_upper_bound_bytes=40939520,
                                    traced_peak_bytes=24030708,tracer_metadata_bytes=8143264)
FIXED_SUPPORT_WORK = dict(old=292864,new=292864,saved=0)
FIXED_BRANCH_BOUND = dict(old=3689676,new=2892124,saved=797552)
FIXED_NODEWISE_PROOF_WORK = 17152
FIXED_CERTIFICATE_COMPARISON_WORK = 17152
FIXED_RESULT_BYTES = 402716
FIXED_LEDGER_BYTES = 173894
PRIOR_EXIT_SHA256 = '86e7347db5e4f24dfbe2a3b6271231f44d4c09ec1c825179f01ae93efcac5499'
PRIOR_TERMINAL_SHA256 = '6577a5d9039c246dc9b440f6dcc0b212fbf70241f2d0c9b23f7a994cd4b73aec'
PRIOR_SEAL_SHA256 = 'a03a2eebe67f16b62eec66ea7496bdad68ab65a20a2aea0e81912660260c1e50'
FROZEN_FILES = (
    'c130_nodewise_branch_v1.py','c130_birth_emission_v1.py',
    'c130_complete_source_v1.py','c130_source_worker_v1.py',
    'test_c130_nodewise_branch_v1.py','run_c130_source_supervisor_v1.py',
    'C130_NODEWISE_BRANCH_PREREG_20260927.md','C130_NODEWISE_BRANCH_PREFLIGHT_20260927.md',
    'C130_DEMAND_BLOCK_NATIVE_HANDOFF_20260927.md')
NEW_DOCUMENTS = (
    'C130_NODEWISE_BRANCH_AUDIT_20260927.md','CHECKPOINT_C130_NODEWISE_BRANCH_20260927.md')
CATEGORY_CAPS = dict(source=200000000,setup=2000000,comparison=10000000,
                     proof=6000000,evidence=20000000,custody=9000000,ledger=7000000)


class ArchivePool:
    """Integer-only monotone C14-compatible accounting, without its imports."""
    def __init__(self,cap):
        if type(cap) is not int or not 0 <= cap <= 256000000:
            raise ValueError('invalid or increased complete archive cap')
        self.cap,self.used,self.parts = cap,0,{}

    def charge(self,name,amount):
        if type(amount) is not int or amount < 0:
            raise ValueError('invalid archive work charge')
        if self.used+amount > self.cap:
            raise MemoryError('actual run plus archive exceeds unchanged whole cap')
        self.used += amount
        self.parts[name] = self.parts.get(name,0)+amount


def _prior_history():
    path = EXP/'C129_TERMINAL_INTEGRITY_20260927.json'
    if _sha256(path) != PRIOR_TERMINAL_SHA256 or _sha256(PRIOR_RUN/'exit.json') != PRIOR_EXIT_SHA256:
        raise ValueError('immutable qualified C129 authority differs')
    terminal = json.loads(path.read_text())
    prior_exit = json.loads((PRIOR_RUN/'exit.json').read_text())
    if (terminal['completed'] is not True or terminal['archive_integrity_passed'] is not True
        or terminal['ordinary_qualification_passed'] is not True
        or terminal['complete_qualification_passed'] is not True
        or terminal['prior_C128_v1_complete_qualification_passed'] is not False
        or terminal['prior_C128_v2_complete_qualification_passed'] is not False
        or terminal['prior_C127_v2_complete_qualification_passed'] is not True
        or terminal['prior_C127_v1_complete_qualification_passed'] is not False
        or terminal['prior_C126_complete_qualification_passed'] is not True
        or terminal['prior_C125_complete_qualification_passed'] is not True
        or terminal['prior_C124_complete_qualification_passed'] is not False
        or terminal['raw_all_stages_passed'] is not True
        or terminal['qualification_errors'] or terminal['mismatches'] or terminal['input_mismatches']
        or not terminal['provenance_unchanged'] or terminal['formal_gain']
        or prior_exit['all_stages_passed'] is not True
        or prior_exit['tests_exit'] != 0 or prior_exit['worker_exit'] != 0
        or prior_exit['tests_count'] != 3644 or prior_exit['test_wall_s'] > 60
        or any(prior_exit[key] for key in ('formal_gain','source_drift','input_drift','provenance_drift'))):
        raise ValueError('C129 qualified / inherited failed historical disposition differs')
    return dict(prior_C129_terminal_sha256=PRIOR_TERMINAL_SHA256,
        prior_C129_seal_sha256=PRIOR_SEAL_SHA256,prior_C129_exit_sha256=PRIOR_EXIT_SHA256,
        prior_C129_complete_qualification_passed=True,
        prior_C128_v1_complete_qualification_passed=False,prior_C128_v2_complete_qualification_passed=False,
        prior_C127_v2_complete_qualification_passed=True,prior_C127_v1_complete_qualification_passed=False,
        prior_C126_complete_qualification_passed=True,prior_C125_complete_qualification_passed=True,
        prior_C124_complete_qualification_passed=False,prior_failures_not_repaired_or_requalified=True)


def _test_partition(freeze):
    inventory = json.loads((RUN/'inventory.json').read_text())
    prior = json.loads((PRIOR_RUN/'preregistered.json').read_text())
    inherited = set(json.loads((PRIOR_RUN/'inventory.json').read_text())['nodeids'])
    nodeids = inventory['nodeids']
    expected_files = {str((EXP/name).resolve().relative_to(ROOT)) for name in freeze['tests']}
    prefix = str((EXP/'test_c130_nodewise_branch_v1.py').relative_to(ROOT))+'::'
    added = set(nodeids)-inherited
    if (freeze['tests'] != prior['tests']+['test_c130_nodewise_branch_v1.py']
        or len(prior['tests']) != 158 or prior['required_test_count'] != 3644 or len(inherited) != 3644
        or freeze['required_test_count'] != 3660 or len(freeze['tests']) != 159 or len(set(freeze['tests'])) != 159
        or inventory['count'] != 3660 or len(nodeids) != 3660 or len(set(nodeids)) != 3660
        or {name.split('::',1)[0] for name in nodeids} != expected_files
        or not inherited.issubset(nodeids) or len(added) != 16
        or any(not name.startswith(prefix) for name in added)):
        raise ValueError('all3644 unchanged plus16 new nodewise-bound cases must be retained')
    cases = ET.parse(RUN/'tests.xml').findall('.//testcase')
    actual = [case.get('classname','').replace('.','/')+'.py::'+case.get('name','') for case in cases]
    if sorted(actual) != sorted(nodeids):
        raise ValueError('complete saved JUnit and collected node populations differ')
    old_passed = new_passed = failed = errors = skipped = 0
    for case,nodeid in zip(cases,actual,strict=True):
        fail,error,skip = (case.find(key) is not None for key in ('failure','error','skipped'))
        failed += int(fail)
        errors += int(error)
        skipped += int(skip)
        if not fail and not error and not skip:
            if nodeid in inherited:
                old_passed += 1
            else:
                new_passed += 1
    return dict(total=3660,files=159,inherited_passed=old_passed,new_passed=new_passed,
        passed=old_passed+new_passed,failed=failed,errors=errors,skipped=skipped)


def _qualified(result,exited,freeze,partition):
    if (exited['all_stages_passed'] is not True or result['completed'] is not True
        or result.get('failure') or exited.get('failure') or result.get('reporting_failure')
        or exited['tests_exit'] or exited['worker_exit'] or exited['tests_count'] != 3660
        or exited['test_wall_s'] > 60 or result['wall_s'] > 240
        or partition != dict(total=3660,files=159,inherited_passed=3644,new_passed=16,
                             passed=3660,failed=0,errors=0,skipped=0)
        or freeze['complete_test_wall_cap_s'] != 60 or freeze['stage_worker_wall_cap_s'] != 240
        or freeze['whole_work_cap'] != 256000000 or freeze['branch_work_cap'] != 200000000
        or freeze['complete_work_upper'] != 254000000 or freeze['category_caps'] != CATEGORY_CAPS
        or freeze['complete_ledger_JSON_reserve'] != 1048576
        or freeze['terminal_success_JSON_reserve'] != 1048576
        or freeze['terminal_failure_JSON_reserve'] != 65536
        or freeze['complete_ledger_work_upper'] != 6752832
        or freeze['complete_fixed_program_work_upper'] != 248094752
        or result['complete_ledger_reservation'] != 1048576
        or result['terminal_success_reservation'] != 1048576
        or result['terminal_failure_reservation'] != 65536
        or freeze['entries_cap'] != 64000000 or freeze['transient_bytes'] != 1073741824
        or freeze['address_space_bytes'] != 16*1024**3 or freeze['cpu_threads'] != 1 or freeze['gpu_enabled']
        or freeze['fixed_complete_source_modes'] != ['dense','masked','heterogeneous','noop']
        or freeze['fixed_source_geometry'] != dict(C=16,K=32,input=[6,6],output=[4,4])
        or freeze['source_reservation_each'] != dict(dense=32000000,masked=32000000,
                                                    heterogeneous=32000000,noop=4000000)
        or freeze['independent_original_and_candidate_builds_per_source'] != 2
        or freeze['owned_evidence_copy_work'] != 8223344 or freeze['owned_evidence_packet_count'] != 12
        or freeze['owned_evidence_entries'] != 1026382
        or len(freeze['source_sha256']) > 3000
        or freeze['freeze_header_occurrence_cap'] != 8192 or header_occurrences(freeze) > 8192
        or freeze['shared_radix_caps'] != [16384,131072,16000000]
        or any(not freeze[key] for key in ('complete_snapshot_roots_in_unchanged_ledger_required',
            'original_live_array_comparison_required','exact_original_coupled_capacity_formula_required',
            'full_source_graph_UID_owner_inverse_bitwise_comparison_required',
            'full_numeric_source_artifacts_and_point_populations_required',
            'all_nodewise_original_emission_counters_required',
            'all_parent_paths_recomputed_with_unchanged_whole_node_tariffs',
            'old_predicate_branch_price_retained','new_proof_fee_in_both_whole_and_branch_required',
            'branch_refinement_is_not_runtime_speedup'))
        or freeze['prior_C129_qualification_passed'] is not True
        or freeze['prior_C128_v1_qualification_passed'] is not False
        or freeze['prior_C128_v2_qualification_passed'] is not False
        or freeze['prior_C127_v2_qualification_passed'] is not True
        or freeze['prior_C127_v1_qualification_passed'] is not False
        or freeze['prior_C124_qualification_passed'] is not False
        or any(freeze[key] for key in ('original_network_run_authorized','archived_HZ_restore_authorized',
            'solver_authorized','actual_network_source_or_LIVE_admitted','kernel_cache_or_reuse_authorized',
            'promotion_authorized','formal_gain'))
        or any(result[key] or exited[key] for key in ('source_drift','input_drift','provenance_drift'))
        or result['solver_calls'] or result['formal_gain'] or exited['formal_gain']
        or result['numeric_hash_traffic_in_token_pool'] or result['all_CPU_work_in_generation_cap']):
        raise ValueError('complete frozen C130 qualification/resources/authority scope differs')
    data = result['data']
    numeric = data['ledger']['numeric']
    metadata = data['ledger']['known_metadata']
    measurement = result['measurement']
    numeric_observation = {key:numeric[key] for key in ('resident_entries','resident_bytes','numeric_storage_count')}
    transient_observation = {key:measurement[key] for key in ('resident_growth_upper_bound_bytes',
                                'traced_peak_bytes','tracer_metadata_bytes')}
    if (exited['work'] != result['work'] or result['work'] != FIXED_PAID_WORK
        or sum(result['work_parts'].values()) != result['work']
        or FIXED_CATEGORY_WORK is None or data['category_work'] != FIXED_CATEGORY_WORK
        or data['category_work']['ledger'] > 6752832 or result['work'] > 248094752
        or numeric_observation != FIXED_NUMERIC
        or metadata['nonoverlapping_known_metadata_bytes'] != FIXED_METADATA_BYTES
        or len(metadata['opaque_inherited_ids']) != FIXED_OPAQUE_COUNT
        or metadata['python_allocator_occupancy_not_measured'] is not True
        or numeric['csr_entry_convention'] != 'data.size_only;indices_and_indptr_bytes_only'
        or numeric['dense_entry_convention'] != 'deduplicated_backing_storage_elements'
        or transient_observation != FIXED_TRANSIENT_OBSERVATIONS
        or data['complete_support_work'] != FIXED_SUPPORT_WORK
        or data['complete_branch_bound'] != FIXED_BRANCH_BOUND
        or data['complete_nodewise_proof_work'] != FIXED_NODEWISE_PROOF_WORK
        or data['branch_bound_is_not_a_runtime_speedup'] is not True
        or result['work_parts'].get('c130_independent_complete_nodewise_certificate_validation') != FIXED_CERTIFICATE_COMPARISON_WORK
        or (RUN/'result.json').stat().st_size != FIXED_RESULT_BYTES
        or (RUN/'complete_held_ledger.json').stat().st_size != FIXED_LEDGER_BYTES):
        raise ValueError('actual complete saved work/ledger/resource observations differ from final pins')
    # The frozen verifier checks ALL25 exact JSON receipts, ALL16 complete NPZ
    # hashes/manifests, all source/owner/inverse populations, both1GiB gates,
    # the64M ledger gate, complete source budgets, custody charges and authority.
    # No array payload or original numerical proof is invoked. The frozen
    # verifier replays the complete saved scalar nodewise certificate only.
    checked = verify_saved_evidence(result)
    if (checked != exited['complete_evidence_payment_checks']
        or checked['complete_prepaid_JSON_receipt_count'] != 25 or len(checked['JSON_checks']) != 25
        or len(checked['complete_numeric_artifacts']) != 16 or checked['numeric_arrays_reanalysed']
        or not checked['all_exact_saved_JSON_payments_passed']
        or not checked['all_complete_source_owner_inverse_artifacts_authenticated']):
        raise ValueError('frozen full saved-evidence verifier disagrees with its original exit receipt')
    expected_stages = {mode+'_'+stage for mode in ('dense','masked','heterogeneous','noop')
                       for stage in ('source','old','new','owners')}
    if set(data['stage_records']) != expected_stages:
        raise ValueError('complete held stage metadata population differs')
    for name,record in data['stage_records'].items():
        if json.loads((RUN/(name+'.json')).read_text()) != record:
            raise ValueError('held and exact-saved stage record differ: '+name)
    return dict(exact_saved_evidence_checks=checked,complete_source_count=4,
        independent_source_build_count=8,complete_numeric_artifact_count=16,
        complete_prepaid_JSON_receipt_count=25,complete_named_array_count=508,
        complete_saved_archive_entries=1041590,owned_snapshot_packet_count=12,
        owned_snapshot_entries=1026382,owned_snapshot_work=8223344,
        actual_live_source_arrays_compared=True,all_original_live_objects_and_owned_snapshots_retained=True,
        strict_C5_C62_ledger_not_filtered_or_relaxed=True,
        complete_numeric_bytes=numeric['resident_bytes'],complete_numeric_entries=numeric['resident_entries'],
        complete_numeric_storage_count=numeric['numeric_storage_count'],
        known_metadata_bytes=metadata['nonoverlapping_known_metadata_bytes'],
        inherited_opaque_id_count=len(metadata['opaque_inherited_ids']),measurement=measurement,
        both_transient_gates_passed=True,measurement_scope='C41_build_not_later_terminal_encoding',
        complete_ledger_JSON_reservation=1048576,terminal_success_JSON_reservation=1048576,
        actual_complete_ledger_JSON_bytes=FIXED_LEDGER_BYTES,actual_result_JSON_bytes=FIXED_RESULT_BYTES,
        complete_ledger_work_upper=6752832,complete_fixed_program_work_upper=248094752,
        category_work=data['category_work'],category_limits=data['category_limits'],
        diagnostic_support_work=data['complete_support_work'],
        complete_nodewise_branch_bound=data['complete_branch_bound'],
        complete_nodewise_proof_work=data['complete_nodewise_proof_work'],
        independent_certificate_comparison_work=FIXED_CERTIFICATE_COMPARISON_WORK,
        full_saved_nodewise_certificates_verified=True,
        existing_omission_accounting_refinement_only=True,branch_bound_is_not_runtime_speedup=True)


def main():
    started = time.monotonic()
    output = EXP/'C130_TERMINAL_INTEGRITY_20260927.json'
    seal = EXP/'CHECKPOINT_C130_NODEWISE_BRANCH_20260927_SHA256SUMS'
    if output.exists() or output.is_symlink() or seal.exists() or seal.is_symlink():
        raise FileExistsError('exclusive C130 terminal record or checkpoint seal exists')
    if type(EXIT_SHA256) is not str or len(EXIT_SHA256) != 64:
        raise ValueError('actual sole-run exit hash must be pinned before this audit')
    if _sha256(RUN/'exit.json') != EXIT_SHA256:
        raise ValueError('completed raw C130 exit differs from pinned identity')
    for name in (*FROZEN_FILES,*NEW_DOCUMENTS):
        if not (EXP/name).is_file():
            raise ValueError('complete C130 frozen source/closing document missing: '+name)
    freeze = json.loads((RUN/'preregistered.json').read_text())
    exited = json.loads((RUN/'exit.json').read_text())
    result_path = RUN/'result.json'
    result = None
    if result_path.exists():
        if type(RESULT_SHA256) is not str or _sha256(result_path) != RESULT_SHA256:
            raise ValueError('completed raw C130 result differs from pinned identity')
        result = json.loads(result_path.read_text())
    elif RESULT_SHA256 is not None or exited.get('all_stages_passed') is True:
        raise ValueError('a qualified run cannot lack its raw result')
    carried_work = 0 if result is None else result.get('work')
    if type(carried_work) is not int or carried_work != FIXED_PAID_WORK:
        raise ValueError('actual raw paid work must match its final pin before reporting')
    pool = ArchivePool(256000000)
    pool.charge('carried_complete_C130_actual_generation_and_evidence_work',carried_work)
    allowance = JsonAllowance(pool,65536,'c130_saved_JSON_archive_exact_terminal_reservation')
    if pool.used != FIXED_PAID_WORK+65536:
        raise ValueError('actual complete C130 run plus terminal archive reservation differs')
    history = _prior_history()
    expected = {}
    def bind(path,digest):
        name = str(path.resolve())
        if name in expected and expected[name] != digest:
            raise ValueError('conflicting inherited/frozen/archive identity: '+name)
        expected[name] = digest
    for name,digest in freeze['source_sha256'].items():
        bind(EXP/name,digest)
    for name,digest in exited['artifacts'].items():
        bind(RUN/name,digest)
    bind(RUN/'exit.json',EXIT_SHA256)
    if result is not None:
        bind(result_path,RESULT_SHA256)
    if any(name not in freeze['source_sha256'] for name in FROZEN_FILES):
        raise ValueError('complete nine-file C130 freeze omitted a source/document')
    prior_seal = EXP/'CHECKPOINT_C129_OWNED_EVIDENCE_20260927_SHA256SUMS'
    if _sha256(prior_seal) != PRIOR_SEAL_SHA256 or len(prior_seal.read_text().splitlines()) != 63:
        raise ValueError('complete63-file qualified-C129 checkpoint seal differs')
    bind(prior_seal,PRIOR_SEAL_SHA256)
    bind(EXP/'C129_TERMINAL_INTEGRITY_20260927.json',PRIOR_TERMINAL_SHA256)
    for line in prior_seal.read_text().splitlines():
        digest,relative = line.split('  ',1)
        bind(ROOT/relative,digest)
    files = sorted(path for path in RUN.rglob('*') if path.is_file())
    if {str(path.relative_to(RUN)) for path in files} != set(exited['artifacts'])|{'exit.json'}:
        raise ValueError('complete raw run file population differs from its terminal manifest')
    prior_freeze = json.loads((PRIOR_RUN/'preregistered.json').read_text())
    inputs = dict(freeze['input_sha256'])
    if inputs != prior_freeze['input_sha256']:
        raise ValueError('original frozen input population changed')
    mismatches = [name for name,digest in expected.items() if _sha256(Path(name)) != digest]
    input_mismatches = [name for name,digest in inputs.items() if _sha256(Path(name)) != digest]
    provenance = _provenance(ROOT)
    unchanged = provenance == freeze['provenance'] == prior_freeze['provenance']
    intact = not mismatches and not input_mismatches and unchanged
    errors,details,partition = [],{},None
    try:
        partition = _test_partition(freeze)
        if result is None or len(files) != 51:
            raise ValueError('complete C130 success requires all51 saved run files and result')
        details = _qualified(result,exited,freeze,partition)
    except Exception as error:
        errors.append(dict(type=type(error).__name__,reason=str(error)))
    if exited.get('all_stages_passed') is not True:
        errors.append(dict(type='RawRunFailure',reason=exited.get('failure'),timeout_s=exited.get('timeout_s')))
    qualified = intact and not errors
    names = [EXP/name for name in (*FROZEN_FILES,*NEW_DOCUMENTS,'run_c130_archive_audit_v1.py')]
    names += [output,*files]
    if len(names) != 13+len(files) or len(set(names)) != len(names):
        raise ValueError('complete C130 checkpoint seal population differs')
    if qualified and len(names) != 64:
        raise ValueError('qualified C130 must seal all64 current files')
    record = dict(completed=intact,archive_integrity_passed=intact,
        ordinary_qualification_passed=qualified,complete_qualification_passed=qualified,
        qualification_scope='fixed_complete_ordinary_source_HZ_owned_evidence_and_nodewise_bound_only',
        completion_means_archive_integrity_only=not qualified,candidate_version_closed=True,
        qualification_errors=errors,raw_all_stages_passed=exited.get('all_stages_passed'),
        raw_supervisor_failure=exited.get('failure'),raw_timeout_s=exited.get('timeout_s'),
        raw_result_failure=None if result is None else result.get('failure'),
        raw_flags_and_files_preserved_not_rewritten=True,
        complete_inherited_tests=3644,new_tests=16,total_tests=3660,total_test_files=159,
        passing_prior_tests_do_not_requalify_failed_C128=True,test_partition=partition,
        test_collection_and_execution_wall_s=exited.get('test_wall_s'),
        checked_unique_files=len(expected),checked_file_bytes=sum(Path(name).stat().st_size for name in expected),
        retained_raw_run_files=len(files),sealed_files=len(names),mismatches=mismatches,
        input_mismatches=input_mismatches,provenance=provenance,provenance_unchanged=unchanged,
        paid_run_work=carried_work,paid_run_plus_archive_reporting=pool.used,
        archive_reporting_work_parts=pool.parts,archive_JSON_reservation=65536,
        numerical_proofs_rerun=False,archived_numeric_arrays_reanalysed=False,
        target_source_or_LIVE_admitted=False,original_network_loaded=False,
        combined_source_F4_path_qualified=False,kernel_factory_custody_or_reuse_qualified=False,
        solver_calls=0,formal_gain=0,speedup_claimed=False,baseline_formal_results_changed=False,
        hash_traffic_separate_from_generation_tokens=True,all_CPU_work_in_generation_cap=False,
        exit_sha256=EXIT_SHA256,result_sha256=RESULT_SHA256,wall_s=time.monotonic()-started,
        **history,**details)
    receipt = allowance.write(output,record)
    if not intact:
        raise ValueError('C130 archive integrity failed; failed terminal record retained')
    seal_bytes = ''.join(_sha256(path)+'  '+str(path.relative_to(ROOT))+'\n'
                         for path in sorted(names)).encode('utf-8')
    publish_bytes(seal,seal_bytes)
    print(json.dumps(dict(completed=True,ordinary_qualification_passed=qualified,sealed_files=len(names),
        paid_run_plus_archive_reporting=pool.used,terminal_receipt=receipt,
        seal_sha256=_sha256(seal),formal_gain=0)),flush=True)


if __name__ == '__main__':
    main()
