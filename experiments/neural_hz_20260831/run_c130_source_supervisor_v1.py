"""Sole frozen C130: full live-source proofs and separately owned evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json

EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c130_nodewise_branch_20260927_v1'
EXPECTED_TESTS = 3660  # All3644 unchanged prior tests plus16 nodewise-bound cases.
MODES = ('dense','masked','heterogeneous','noop')
STAGES = ('source','old','new','owners')
# n_cont, n_eq, complete MAIN owner count, original payload entries,
# full packet entries per version. Derived before execution, not fitted to it.
CASE_INVENTORY = dict(dense=(2688,1601,2112,12246,178267),
    masked=(2544,1457,1968,11670,139532),
    heterogeneous=(2548,1461,1972,11686,139840),
    noop=(2128,1041,1552,10006,32748))
CATEGORY_CAPS = dict(source=200000000,setup=2000000,comparison=10000000,
                     proof=6000000,evidence=20000000,custody=9000000,ledger=7000000)
C129_EXIT = '86e7347db5e4f24dfbe2a3b6271231f44d4c09ec1c825179f01ae93efcac5499'
C129_SEAL = 'a03a2eebe67f16b62eec66ea7496bdad68ab65a20a2aea0e81912660260c1e50'
C129_TERMINAL = '6577a5d9039c246dc9b440f6dcc0b212fbf70241f2d0c9b23f7a994cd4b73aec'
# Complete source/AST/scalar-schema bound fixed before any numerical execution.
LEDGER_WORK_UPPER = 6752832
FIXED_PROGRAM_WORK_UPPER = 248094752
NEW_FILES = ('c130_nodewise_branch_v1.py','c130_birth_emission_v1.py',
    'c130_complete_source_v1.py','c130_source_worker_v1.py',
    'test_c130_nodewise_branch_v1.py','run_c130_source_supervisor_v1.py',
    'C130_NODEWISE_BRANCH_PREREG_20260927.md','C130_NODEWISE_BRANCH_PREFLIGHT_20260927.md',
    'C130_DEMAND_BLOCK_NATIVE_HANDOFF_20260927.md')
from experiments.neural_hz_20260831.c130_nodewise_branch_v1 import verify_certificate



def header_occurrences(value):
    # Scalar/JSON structure only, no model, HZ, numeric array or source read.
    if type(value) is dict:
        return 1+sum(header_occurrences(k)+header_occurrences(v) for k,v in value.items())
    if type(value) in (list,tuple):
        return 1+sum(header_occurrences(v) for v in value)
    if value is None or type(value) in (str,int,float,bool):
        return 1
    raise ValueError('non-JSON frozen header outside the fixed bound')



def verify_saved_evidence(result):
    data = result['data']
    if (data['qualification_pass'] is not True or data['qualification_failures']
        or data['complete_source_count'] != 4 or data['independent_source_build_count'] != 8
        or set(data['reports']) != set(MODES) or data['category_limits'] != CATEGORY_CAPS
        or set(data['category_work']) != set(CATEGORY_CAPS)
        or any(type(value) is not int or not 0 <= value <= CATEGORY_CAPS[key]
               for key,value in data['category_work'].items())
        or sum(data['category_work'].values()) != result['work'] or result['work'] > 254000000
        or not all(data[key] for key in ('unchanged_complete_ordinary_source_geometry',
            'complete_source_graph_owner_inverse_equivalence','complete_all_numeric_source_arrays_saved',
            'complete_all_point_populations_saved','full_original_source_sharing_and_nonconvex_predicates_preserved',
            'original_live_objects_and_all_owned_evidence_retained','unchanged_strict_owner_ledger_used'))
        or any(data[key] for key in ('source_or_LIVE_admitted','original_network_loaded',
                                     'archived_HZ_loaded','solver_calls','formal_gain'))):
        raise ValueError('complete C130 source/proof/payment scope differs')
    if (data['category_work']['custody'] != 8223344
        or data['category_work']['ledger'] > LEDGER_WORK_UPPER or result['work'] > FIXED_PROGRAM_WORK_UPPER
        or result['work_parts'].get('c129_complete_owned_evidence_headers') != 12*1024
        or result['work_parts'].get('c129_complete_owned_evidence_copy') != 8*1026382):
        raise ValueError('complete actual owned evidence prepayment differs')
    expected = {mode+'_complete_source.json':131072 for mode in MODES}
    expected.update({mode+'_complete_points.json':262144 for mode in MODES})
    expected.update({mode+'_'+stage+'.json':65536 for mode in MODES for stage in STAGES})
    expected['complete_held_ledger.json'] = 1048576
    receipts = data['serialization_receipts']
    if len(receipts) != 25 or {item['file'] for item in receipts} != set(expected):
        raise ValueError('complete exclusive exact JSON receipt population differs')
    decoded,by_name,checks = {},{},[]
    for receipt in receipts:
        name = receipt['file']
        if name in decoded:
            raise ValueError('duplicate complete exact JSON receipt')
        path = RUN/name
        raw = path.read_bytes()
        payload = json.loads(raw)
        canonical = (json.dumps(payload,sort_keys=True,separators=(',',':'),
                               ensure_ascii=True,allow_nan=False)+'\n').encode('utf-8')
        if (raw != canonical or receipt['encoded_bytes'] != len(raw)
            or receipt['prepaid_bytes'] != expected[name] or receipt['overhead_bytes'] != 1024
            or len(raw)+1024 > expected[name] or receipt['sha256'] != _sha256(path)
            or not receipt['exact_checked_bytes_published']
            or receipt['encoding'] != 'sorted_compact_ascii_JSON_newline'):
            raise ValueError('actual complete JSON bytes differ from prepaid evidence: '+name)
        decoded[name],by_name[name] = payload,receipt
        checks.append(dict(file=name,stored_bytes=len(raw),prepaid_bytes=expected[name],
                           sha256=receipt['sha256'],passed=True))
    if decoded['complete_held_ledger.json'] != data['ledger']:
        raise ValueError('saved complete root ledger differs')
    numeric = data['ledger']['numeric']
    if numeric['resident_entries'] > 64000000:
        raise ValueError('complete numeric root entry gate failed')
    if (numeric['numeric_storage_count'] > 1024
        or len(numeric['storage_provenance']) != numeric['numeric_storage_count']
        or sum(len(item['roles']) for item in numeric['storage_provenance']) > 2048
        or any(len(role) > 64 or not role.isascii()
               for item in numeric['storage_provenance'] for role in item['roles'])):
        raise ValueError('complete actual storage/role population exceeds the preflight envelope')
    measurement = result['measurement']
    if (not measurement['build_returned'] or not measurement['measured_transient_gate']
        or measurement['transient_cap_bytes'] != 1073741824
        or measurement['resident_growth_upper_bound_bytes'] > 1073741824
        or measurement['traced_peak_bytes']+measurement['tracer_metadata_bytes'] > 1073741824
        or numeric['resident_bytes'] <= 0 or data['ledger']['known_metadata']['nonoverlapping_known_metadata_bytes'] <= 0
        or result['wall_s'] > 240):
        raise ValueError('complete source measured transient/held-root gates differ')
    artifacts,old_support,new_support = [],0,0
    old_branch,new_branch,proof_work,comparison_work = 0,0,0,0
    for mode in MODES:
        report = data['reports'][mode]
        nc,ne,owner_count,source_entries,packet_entries = CASE_INVENTORY[mode]
        if decoded[mode+'_complete_source.json'] != report:
            raise ValueError('complete saved source report differs')
        if (report['mode'] != mode or not all(report[key] for key in (
            'complete_source_arrays_bitwise_equal','complete_semantic_metadata_equal',
            'complete_original_source_sharing_equal','full_sparse_owner_oracles_equal',
            'complete_inverse_arrays_and_points_equal','original_source_unchanged',
            'exact_original_coupled_capacity_formula_verified',
            'complete_nodewise_branch_credit_independently_verified',
            'complete_original_emission_and_coupled_fees_unchanged',
            'branch_bound_is_not_a_runtime_speedup',
            'actual_live_source_arrays_compared_before_local_views_expire',
            'full_saved_evidence_owns_independent_storage'))
            or report['nonconvex_binary_factors'] != 1 or report['inequalities'] != 1
            or report['source_or_LIVE_admitted'] or report['formal_gain']
            or report['source_reservation_each'] != (4000000 if mode=='noop' else 32000000)):
            raise ValueError('complete original nonconvex source equivalence differs: '+mode)
        points = decoded[mode+'_complete_points.json']
        if (points['mode'] != mode or points['formal_gain']
            or not points['points_are_not_network_witnesses']
            or not points['complete_old_new_original_and_recovered_populations']
            or set(points['points']) != {'old','new'}
            or points['points']['old'] != points['points']['new']
            or report['complete_points_receipt'] != by_name[mode+'_complete_points.json']):
            raise ValueError('complete saved inverse evidence differs: '+mode)
        for version in ('old','new'):
            counts = report['full_point_counts'][version]
            if (set(points['points'][version]) != {'original','recovered'}
                or set(counts) != {'original','recovered'}
                or any(type(values) is not list or not values or len(values) != counts[name]
                       for name,values in points['points'][version].items())
                or counts['original'] != report['complete_source_metadata'][version]['hz']['n_cont']
                or counts['recovered'] != counts['original']):
                raise ValueError('complete inverse point population omitted')
            source = report['complete_source_metadata'][version]
            if (source['hz']['n_bin'] != 1 or source['hz']['n_ineq'] != 1
                or source['hz']['n_cont'] != nc or source['hz']['n_eq'] != ne
                or not source['hz']['exact'] or source['hz']['frame_id'] != 17
                or source['report']['source_first_native_or_LIVE_admission']
                or report['whole_work'][version] > report['source_reservation_each']
                or report['branch_work'][version] > report['source_reservation_each']):
                raise ValueError('full source frame or individual source budget differs')
            # Independently check the original WorkPool formula in complete
            # saved metadata; a worker success flag is not sufficient authority.
            base = source['report']
            coupled = base['alias_quotient']
            reserve = report['source_reservation_each']
            whole,branch = base['whole_base_work'],base['branch_base_work']
            capacity,used = coupled['coupled_extra_capacity'],coupled['coupled_extra_work']
            if (any(type(v) is not int or v < 0 for v in (whole,branch,capacity,used))
                or max(whole,branch) > reserve
                or capacity != min(reserve-whole,reserve-branch)
                or used > capacity
                or base['total_work_upper'] != whole+used
                or base['largest_branch_work_upper'] != branch+used
                or report['whole_work'][version] != whole+used
                or report['branch_work'][version] != branch+used):
                raise ValueError('saved source coupled budget formula differs: '+mode+'/'+version)
        old_source,new_source = (report['complete_source_metadata'][name] for name in ('old','new'))
        old_report,new_report = old_source['report'],new_source['report']
        verify_certificate(new_report,new_source['nodes'])
        certificate = new_report['nodewise_branch_certificate']
        independent_fee = 2048+512*len(new_source['nodes'])+64*sum(len(n['parents']) for n in new_source['nodes'])
        if (old_report['original_branch_encoding_price_retained'] is not True
            or 'nodewise_branch_certificate' in old_report
            or old_source['nodes'] != new_source['nodes']
            or old_report['node_counts'] != new_report['node_counts']
            or certificate['whole_base_without_proof_fee'] != old_report['whole_base_work']
            or certificate['old_branch_base_without_proof_fee'] != old_report['branch_base_work']
            or new_report['whole_base_work'] != old_report['whole_base_work']+certificate['proof_work']
            or old_report['alias_quotient']['work_parts'] != new_report['alias_quotient']['work_parts']
            or report['nodewise_proof_work'] != certificate['proof_work']
            or report['independent_certificate_comparison_work'] != independent_fee):
            raise ValueError('full saved nodewise attribution or independent payment differs: '+mode)
        old_branch += report['branch_work']['old']
        new_branch += report['branch_work']['new']
        proof_work += report['nodewise_proof_work']
        comparison_work += independent_fee
        if set(report['artifacts']) != set(STAGES):
            raise ValueError('source/control/candidate/owner stage evidence is incomplete')
        manifests = {}
        for stage in STAGES:
            artifact = report['artifacts'][stage]
            manifest = artifact['complete_array_manifest']
            stage_record = decoded[mode+'_'+stage+'.json']
            name = mode+'_'+stage+'_arrays.npz'
            path = RUN/name
            entries = sum(item['entries'] for item in manifest.values())
            expected_entries = source_entries if stage=='source' else 2*owner_count if stage=='owners' else packet_entries
            expected_arrays = 29 if stage=='source' else 2 if stage=='owners' else 48
            if (stage_record['mode'] != mode or stage_record['stage'] != stage
                or stage_record['artifact'] != artifact or stage_record['formal_gain']
                or stage_record['proof_completed_at_save'] is not False
                or artifact['file'] != name or artifact['sha256'] != _sha256(path)
                or artifact['stored_bytes'] != path.stat().st_size
                or artifact['arrays'] != len(manifest) or len(manifest) != expected_arrays
                or artifact['numeric_entries'] != entries or entries != expected_entries
                or artifact['encoding_work'] != 1024+16*entries):
                raise ValueError('complete retained stage artifact differs: '+mode+'/'+stage)
            if stage in ('old','new') and stage_record['metadata'] != report['complete_source_metadata'][stage]:
                raise ValueError('saved precomparison source metadata differs')
            if stage=='source' and stage_record['metadata'] != report['complete_original_source_header']:
                raise ValueError('complete original geometry/header evidence differs')
            manifests[stage] = manifest
            artifacts.append(dict(file=name,sha256=artifact['sha256'],numeric_entries=entries,
                stored_bytes=artifact['stored_bytes'],full_file_authenticated=True,
                numeric_payload_reanalysed=False))
        packet_names = {'keep','eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows',
            'owners','uid_slabs','radix_gauges','eq_uids','ineq_uids','hz_c','hz_b','hz_ub'}
        packet_names.update('hz_'+matrix+'_'+part for matrix in ('Gc','Gb','Ac','Ab','Auc','Aub')
                            for part in ('data','indices','indptr'))
        packet_names.update('node_'+str(index)+'_'+part for index in range(4)
                            for part in ('support','needed','slots','exponents'))
        if (manifests['old'] != manifests['new'] or set(manifests['old']) != packet_names
            or report['arrays_per_version'] != 48 or report['entries_both_versions'] != 2*packet_entries
            or set(manifests['owners']) != {'old','new'}
            or any(item['entries'] != owner_count for item in manifests['owners'].values())
            or 'expression_bias' not in manifests['source']):
            raise ValueError('complete paired source/owner manifest differs')
        old_support += report['support_work']['old']
        new_support += report['support_work']['new']
    if (new_support != old_support or data['complete_support_work'] !=
        dict(old=old_support,new=new_support,saved=old_support-new_support)):
        raise ValueError('original graph support work must remain unchanged')
    if (new_branch >= old_branch or data['complete_branch_bound'] !=
        dict(old=old_branch,new=new_branch,saved=old_branch-new_branch)
        or data['complete_nodewise_proof_work'] != proof_work
        or data['branch_bound_is_not_a_runtime_speedup'] is not True
        or result['work_parts'].get('c130_independent_complete_nodewise_certificate_validation') != comparison_work):
        raise ValueError('complete all-case branch-bound refinement/payment differs')
    size = (RUN/'result.json').stat().st_size
    if (size+1024 > 1048576 or result['terminal_success_reservation'] != 1048576
        or result['terminal_failure_reservation'] != 65536
        or result['complete_ledger_reservation'] != 1048576
        or not result['terminal_allowances_prepaid_before_build'] or result.get('reporting_failure')):
        raise ValueError('complete actual terminal bytes exceed original reservation')
    canonical = (json.dumps(result,sort_keys=True,separators=(',',':'),ensure_ascii=True,
                            allow_nan=False)+'\n').encode('utf-8')
    if (RUN/'result.json').read_bytes() != canonical:
        raise ValueError('actual result is not its prepaid exact compact encoding')
    return dict(all_exact_saved_JSON_payments_passed=True,JSON_checks=checks,
        complete_prepaid_JSON_receipt_count=25,complete_numeric_artifacts=artifacts,
        actual_result_bytes=size,all_complete_source_owner_inverse_artifacts_authenticated=True,
        numeric_arrays_reanalysed=False,supervisor_check_outside_worker_generation_counter=True)


def main():
    if RUN.exists() or RUN.is_symlink():
        raise FileExistsError(RUN)
    if (EXPECTED_TESTS != 3660 or type(LEDGER_WORK_UPPER) is not int
        or type(FIXED_PROGRAM_WORK_UPPER) is not int):
        raise ValueError('final test inventory and complete fresh bounds must be fixed before run')
    previous = EXP/'results/c129_owned_evidence_20260927_v1'
    prior = json.loads((previous/'preregistered.json').read_text())
    done = json.loads((previous/'exit.json').read_text())
    if (_sha256(previous/'exit.json') != C129_EXIT or done['all_stages_passed'] is not True
        or done['tests_exit'] != 0 or done['worker_exit'] != 0 or done['tests_count'] != 3644
        or done['test_wall_s'] > 60
        or any(done[key] for key in ('formal_gain','source_drift','input_drift','provenance_drift'))):
        raise ValueError('immutable qualified C129 predecessor differs')
    hashes = dict(prior['source_sha256'])
    def bind(name,digest):
        if name in hashes and hashes[name] != digest:
            raise ValueError('conflicting inherited source identity: '+name)
        hashes[name] = digest
    for name,digest in done['artifacts'].items():
        bind(str((previous/name).relative_to(EXP)),digest)
    bind(str((previous/'exit.json').relative_to(EXP)),C129_EXIT)
    seal = EXP/'CHECKPOINT_C129_OWNED_EVIDENCE_20260927_SHA256SUMS'
    if _sha256(seal) != C129_SEAL or len(seal.read_text().splitlines()) != 63:
        raise ValueError('complete qualified C129 seal differs')
    for line in seal.read_text().splitlines():
        digest,name = line.split('  ',1)
        bind(str((ROOT/name).relative_to(EXP)),digest)
    bind(seal.name,C129_SEAL)
    terminal_path = EXP/'C129_TERMINAL_INTEGRITY_20260927.json'
    if _sha256(terminal_path) != C129_TERMINAL:
        raise ValueError('authoritative qualified C129 archive differs')
    terminal = json.loads(terminal_path.read_text())
    if (not terminal['completed'] or not terminal['archive_integrity_passed']
        or terminal['complete_qualification_passed'] is not True
        or terminal['ordinary_qualification_passed'] is not True
        or terminal['prior_C128_v1_complete_qualification_passed'] is not False
        or terminal['prior_C128_v2_complete_qualification_passed'] is not False
        or terminal['prior_C127_v2_complete_qualification_passed'] is not True
        or terminal['prior_C127_v1_complete_qualification_passed'] is not False
        or terminal['prior_C124_complete_qualification_passed'] is not False
        or terminal['qualification_errors'] or terminal['mismatches'] or terminal['input_mismatches']
        or not terminal['provenance_unchanged'] or terminal['formal_gain']):
        raise ValueError('qualified C129 / failed C128 historical disposition differs')
    for name in NEW_FILES:
        bind(name,_sha256(EXP/name))
    if len(hashes) > 3000 or any(_sha256(EXP/name) != digest for name,digest in hashes.items()):
        raise ValueError('complete inherited dependency population or identity differs')
    tests = prior['tests']+['test_c130_nodewise_branch_v1.py']
    if len(tests) != 159 or len(set(tests)) != 159:
        raise ValueError('complete inherited/new test-file population differs')
    inputs,provenance = dict(prior['input_sha256']),_provenance(ROOT)
    if provenance != prior['provenance'] or any(_sha256(Path(n)) != h for n,h in inputs.items()):
        raise ValueError('production/original input provenance changed')
    env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):
        raise ValueError('ordinary assertions required')
    freeze = dict(source_sha256=hashes,
        input_sha256=inputs,provenance=provenance,tests=tests,required_test_count=EXPECTED_TESTS,
        complete_test_wall_cap_s=60,stage_worker_wall_cap_s=240,cpu_threads=1,gpu_enabled=False,
        address_space_bytes=16*1024**3,transient_bytes=1024**3,entries_cap=64000000,
        category_caps=CATEGORY_CAPS,complete_work_upper=254000000,whole_work_cap=256000000,
        branch_work_cap=200000000,fixed_complete_source_modes=list(MODES),
        independent_original_and_candidate_builds_per_source=2,
        fixed_source_geometry=dict(C=16,K=32,input=[6,6],output=[4,4]),
        source_reservation_each=dict(dense=32000000,masked=32000000,heterogeneous=32000000,noop=4000000),
        full_source_graph_UID_owner_inverse_bitwise_comparison_required=True,
        full_numeric_source_artifacts_and_point_populations_required=True,
        prior_C129_qualification_passed=True,
        prior_C128_v1_qualification_passed=False,prior_C128_v2_qualification_passed=False,
        owned_evidence_copy_work=8223344,owned_evidence_packet_count=12,
        owned_evidence_entries=1026382,complete_snapshot_roots_in_unchanged_ledger_required=True,
        original_live_array_comparison_required=True,freeze_header_occurrence_cap=8192,
        complete_ledger_JSON_reserve=1048576,terminal_success_JSON_reserve=1048576,
        terminal_failure_JSON_reserve=65536,complete_ledger_work_upper=LEDGER_WORK_UPPER,
        complete_fixed_program_work_upper=FIXED_PROGRAM_WORK_UPPER,
        exact_original_coupled_capacity_formula_required=True,
        all_nodewise_original_emission_counters_required=True,
        all_parent_paths_recomputed_with_unchanged_whole_node_tariffs=True,
        old_predicate_branch_price_retained=True,
        new_proof_fee_in_both_whole_and_branch_required=True,
        branch_refinement_is_not_runtime_speedup=True,
        prior_C127_v2_qualification_passed=True,prior_C127_v1_qualification_passed=False,
        prior_C124_qualification_passed=False,shared_radix_caps=[16384,131072,16000000],
        original_network_run_authorized=False,archived_HZ_restore_authorized=False,
        solver_authorized=False,actual_network_source_or_LIVE_admitted=False,
        kernel_cache_or_reuse_authorized=False,promotion_authorized=False,formal_gain=0)
    if header_occurrences(freeze) > 8192:
        raise ValueError('complete frozen JSON exceeds the prepaid header envelope')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',freeze)
    started = time.monotonic()
    record = dict(all_stages_passed=False,formal_gain=0)
    try:
        command = [sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',
                   *(str(EXP/name) for name in tests)]
        collected = subprocess.run([*command,'--collect-only'],cwd=ROOT,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        with (RUN/'collection.log').open('x') as stream:
            stream.write(collected.stdout)
        ids = [line for line in collected.stdout.splitlines() if line.startswith(('experiments/','act/')) and '::' in line]
        expected_files = {str((EXP/name).resolve().relative_to(ROOT)) for name in tests}
        if (collected.returncode or len(ids) != EXPECTED_TESTS or len(set(ids)) != EXPECTED_TESTS
            or {item.split('::',1)[0] for item in ids} != expected_files):
            raise ValueError('complete frozen test inventory differs')
        _atomic_exclusive_json(RUN/'inventory.json',dict(nodeids=ids,count=len(ids)))
        left = 60-(time.monotonic()-started)
        if left <= 0:
            raise subprocess.TimeoutExpired(command,60)
        with (RUN/'tests.log').open('x') as stream:
            tested = subprocess.run([*command,'--junitxml='+str(RUN/'tests.xml')],cwd=ROOT,
                env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=left)
        record.update(tests_exit=tested.returncode,tests_count=len(ids),test_wall_s=time.monotonic()-started)
        cases = ET.parse(RUN/'tests.xml').findall('.//testcase')
        actual = [case.get('classname','').replace('.','/')+'.py::'+case.get('name','') for case in cases]
        if (tested.returncode or sorted(actual) != sorted(ids)
            or any(case.find(key) is not None for case in cases for key in ('failure','error','skipped'))):
            raise ValueError('complete inherited/new qualification failed')
        print(json.dumps(dict(event='all_tests_passed',tests=len(ids),wall_s=record['test_wall_s'])),flush=True)
        if any(_sha256(EXP/name) != digest for name,digest in hashes.items()):
            raise ValueError('source drift after qualification')
        with (RUN/'worker.log').open('x') as stream:
            worker = subprocess.run([sys.executable,str(EXP/'c130_source_worker_v1.py')],cwd=ROOT,
                env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit'] = worker.returncode
        result = json.loads((RUN/'result.json').read_text())
        if (worker.returncode or not result['completed'] or result.get('failure')
            or any(result[key] for key in ('solver_calls','source_drift','input_drift','provenance_drift'))
            or not result['data']['qualification_pass']):
            raise ValueError('complete original/candidate source qualification failed')
        checks = verify_saved_evidence(result)
        record.update(all_stages_passed=True,work=result['work'],complete_source_count=4,
                      independent_source_build_count=8,complete_evidence_payment_checks=checks)
    except subprocess.TimeoutExpired as exc:
        record['timeout_s'] = exc.timeout
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/name) != digest for name,digest in hashes.items()),
            input_drift=any(_sha256(Path(name)) != digest for name,digest in inputs.items()),
            provenance_drift=_provenance(ROOT) != provenance,
            artifacts={str(path.relative_to(RUN)):_sha256(path) for path in RUN.rglob('*') if path.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record)
        print(json.dumps(record),flush=True)
    if not record['all_stages_passed'] or record['source_drift'] or record['input_drift'] or record['provenance_drift']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
