"""Exclusive C126 full exact-word native comparison and evidence qualification.

C124's raw green exit is inherited as immutable history, NOT as qualification.
Its sealed terminal audit must explicitly remain failed.  Post-worker checks
read saved JSON and hash complete artifacts; they never load numeric NPZ data.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256, _atomic_exclusive_json

EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c126_support_word_20260927_v1'
EXPECTED_TESTS = 3556
MODES = ('dense', 'masked', 'heterogeneous', 'noop')
NONEMPTY = MODES[:3]
CATEGORY_CAPS = dict(source=100_000_000, setup=2_000_000, binding=21_000_000,
    control=7_000_000, construction=15_000_000, proof=53_000_000,
    comparison=5_000_000, physical=17_000_000, ledger=18_000_000, evidence=16_000_000)
C125_EXIT_SHA256 = '2757a975900e6281c1e6a8ef6fb3cfc65ade8cf48aec76fb2b2d623c8ede6bb3'
C125_SEAL_SHA256 = '39f0fd9b8b86bbed54e87cde9726d573cffa339fc43b9c1a7af1786eee6f82dd'


def verify_saved_evidence(result):
    """Authenticate every prepaid JSON and complete proof/point artifact."""
    data = result['data']
    if (data['qualification_pass'] is not True or data['qualification_failures']
        or data['complete_source_count'] != 4 or set(data['reports']) != set(MODES)
        or data['category_limits'] != CATEGORY_CAPS
        or set(data['category_work']) != set(CATEGORY_CAPS)
        or any(type(data['category_work'][name]) is not int
               or not 0 <= data['category_work'][name] <= cap
               for name, cap in CATEGORY_CAPS.items())
        or sum(data['category_work'].values()) != result['work']
        or result['work'] > 254_000_000
        or not data['full_original_four_source_fixtures_retained']
        or not data['complete_independent_all36_Fraction_word_agreement']
        or not data['complete_point_populations_saved_without_nested_duplication']
        or not data['no_kernel_cache_or_reuse_claim']
        or not data['complete_old_new_native_packet_bitwise_agreement']
        or not data['complete_control_and_new_native_arrays_saved']
        or not data['support_compiled_exact_word_rows_executed']
        or data['source_or_LIVE_admitted'] or data['formal_gain']):
        raise ValueError('complete frozen source/kernel/payment scope differs')
    expected = {mode+'_complete_source.json':131072 for mode in MODES}
    for mode in NONEMPTY:
        expected.update({mode+'_constructed.json':65536, mode+'_control.json':65536,
            mode+'_complete_kernel_proofs.json':65536,
            mode+'_complete_point_evidence.json':262144})
    expected['complete_held_ledger.json'] = 131072
    receipts = data['serialization_receipts']
    if (len(receipts) != len(expected)
        or {receipt['file'] for receipt in receipts} != set(expected)):
        raise ValueError('complete exclusive prepaid JSON inventory differs')
    decoded, checked, by_name = {}, [], {}
    for receipt in receipts:
        name = receipt['file']
        if name in by_name:
            raise ValueError('duplicate prepaid JSON receipt')
        path = RUN/name
        size = path.stat().st_size
        digest = _sha256(path)
        if (receipt['prepaid_bytes'] != expected[name]
            or receipt['overhead_bytes'] != 1024
            or receipt['encoded_bytes'] != size
            or size+1024 > expected[name]
            or receipt['sha256'] != digest
            or receipt['exact_checked_bytes_published'] is not True
            or receipt['encoding'] != 'sorted_compact_ascii_JSON_newline'):
            raise ValueError('actual saved JSON differs from fixed prepaid bytes: '+name)
        raw = path.read_bytes()
        payload = json.loads(raw)
        canonical = (json.dumps(payload, sort_keys=True, separators=(',', ':'),
                               allow_nan=False, ensure_ascii=True)+'\n').encode('utf-8')
        if raw != canonical:
            raise ValueError('actual JSON is not the checked compact byte encoding: '+name)
        decoded[name], by_name[name] = payload, receipt
        checked.append(dict(file=name, stored_bytes=size, overhead_bytes=1024,
            prepaid_bytes=expected[name], sha256=digest, passed=True))
    if decoded['complete_held_ledger.json'] != data['ledger']:
        raise ValueError('saved complete owner ledger differs from terminal ledger')
    artifacts = []
    for mode in MODES:
        report = decoded[mode+'_complete_source.json']
        if report != data['reports'][mode]:
            raise ValueError('saved complete case differs from terminal case: '+mode)
        if mode == 'noop':
            if (report['complete_point_evidence'] is not None
                or report['kernel_comparison'] is not None
                or not report['physical']['literal_noop']
                or not report['physical']['exact_original_object_retained']):
                raise ValueError('zero-hit original source did not remain literal')
            continue
        constructed = decoded[mode+'_constructed.json']
        control = decoded[mode+'_control.json']
        kernel = decoded[mode+'_complete_kernel_proofs.json']
        points = decoded[mode+'_complete_point_evidence.json']
        point_ref = report['complete_point_evidence']
        if (constructed['constructor'] != report['constructor']
            or control['constructor'] != report['control_constructor']
            or not report['complete_native_comparison']['all_native_arrays_bitwise_equal']
            or constructed['artifact']['numeric_entries'] != control['artifact']['numeric_entries']
            or kernel['comparison'] != report['kernel_comparison']
            or not kernel['complete_all36_numeric_evidence_retained']
            or kernel['cache_reuse_authorized'] or kernel['source_or_LIVE_admitted']
            or point_ref['receipt'] != by_name[mode+'_complete_point_evidence.json']
            or not point_ref['all_four_full_populations_saved']
            or not points['complete_original_expected_expanded_recovered']
            or set(points['points']) != {'original','expected','expanded','recovered'}
            or set(point_ref['counts']) != set(points['points'])
            or any(not isinstance(values, list) or not values
                   or len(values) != point_ref['counts'][name]
                   for name, values in points['points'].items())
            or points['points']['expected'] != points['points']['recovered']):
            raise ValueError('complete source/kernel/four-point evidence differs: '+mode)
        for record, suffix in ((constructed, '_constructed_arrays.npz'),
                               (control, '_control_arrays.npz'),
                               (kernel, '_complete_kernel_proofs.npz')):
            artifact = record['artifact']
            name = mode+suffix
            path = RUN/name
            if (artifact['file'] != name or not path.is_file()
                or artifact['stored_bytes'] != path.stat().st_size
                or artifact['sha256'] != _sha256(path)
                or type(artifact['numeric_entries']) is not int
                or artifact['numeric_entries'] <= 0
                or artifact['encoding_work'] != 1024+16*artifact['numeric_entries']
                or (suffix == '_complete_kernel_proofs.npz'
                    and artifact['numeric_entries'] != 172*32*16)):
                raise ValueError('complete retained numeric proof artifact differs: '+name)
            artifacts.append(dict(artifact, full_file_authenticated=True,
                                  numeric_payload_reanalysed=False))
    result_size = (RUN/'result.json').stat().st_size
    if (result_size+1024 > 524288
        or result['terminal_success_reservation'] != 524288
        or result['terminal_failure_reservation'] != 65536
        or result['complete_ledger_reservation'] != 131072
        or not result['terminal_allowances_prepaid_before_build']
        or result.get('reporting_failure')):
        raise ValueError('actual terminal JSON exceeds complete prepaid success allowance')
    checked.append(dict(file='result.json', stored_bytes=result_size,
        overhead_bytes=1024, prepaid_bytes=524288,
        sha256=_sha256(RUN/'result.json'), passed=True))
    return dict(all_exact_saved_JSON_payments_passed=True,
        all_complete_kernel_native_and_point_artifacts_authenticated=True,
        complete_prepaid_JSON_receipt_count=len(receipts), JSON_checks=checked,
        complete_numeric_artifacts=artifacts, numeric_arrays_reanalysed=False,
        supervisor_check_outside_worker_generation_counter=True)


def main():
    if RUN.exists():
        raise FileExistsError(RUN)
    previous = EXP/'results/c125_kernel_proof_20260927_v1'
    prior = json.loads((previous/'preregistered.json').read_text())
    done = json.loads((previous/'exit.json').read_text())
    if (_sha256(previous/'exit.json') != C125_EXIT_SHA256
        or not done['all_stages_passed'] or done['worker_exit'] or done['tests_exit']
        or done['tests_count'] != 3524 or done['source_drift']
        or done['provenance_drift'] or done['input_drift']):
        raise ValueError('immutable qualified C125 run identity differs')
    hashes = dict(prior['source_sha256'])
    hashes.update({str((previous/n).relative_to(EXP)):h for n,h in done['artifacts'].items()})
    hashes[str((previous/'exit.json').relative_to(EXP))] = C125_EXIT_SHA256
    seal = EXP/'CHECKPOINT_C125_KERNEL_PROOF_20260927_SHA256SUMS'
    if _sha256(seal) != C125_SEAL_SHA256:
        raise ValueError('complete C125 qualified-checkpoint seal differs')
    for line in seal.read_text().splitlines():
        digest, name = line.split('  ',1)
        relative = str((ROOT/name).relative_to(EXP))
        if relative in hashes and hashes[relative] != digest:
            raise ValueError('inherited sealed source conflict')
        hashes[relative] = digest
    hashes[seal.name] = C125_SEAL_SHA256
    terminal_name = 'C125_TERMINAL_INTEGRITY_20260927.json'
    if terminal_name not in hashes or _sha256(EXP/terminal_name) != hashes[terminal_name]:
        raise ValueError('authoritative C125 terminal audit not authenticated')
    terminal = json.loads((EXP/terminal_name).read_text())
    if (terminal['complete_qualification_passed'] is not True
        or terminal['ordinary_qualification_passed'] is not True
        or not terminal['completed'] or terminal['prior_C124_complete_qualification_passed'] is not False
        or terminal['exit_sha256'] != C125_EXIT_SHA256
        or terminal['mismatches'] or terminal['input_mismatches']
        or not terminal['provenance_unchanged'] or terminal['formal_gain']):
        raise ValueError('C125 complete ordinary qualification or C124 failure status differs')
    new = ['C126_SUPPORT_WORD_PREREG_20260927.md', 'c126_support_word_mixed_v1.py',
        'c126_support_word_oracle_v1.py', 'c126_complete_mixed_source_v1.py',
        'c126_mixed_source_worker_v1.py', 'test_c126_support_word_mixed_v1.py',
        'run_c126_mixed_source_supervisor_v1.py']
    hashes.update({name:_sha256(EXP/name) for name in new})
    if any(_sha256(EXP/name) != digest for name,digest in hashes.items()):
        raise ValueError('complete inherited dependency/artifact drift')
    tests = prior['tests']+['test_c126_support_word_mixed_v1.py']
    if len(tests) != 154 or len(set(tests)) != 154:
        raise ValueError('complete inherited/new test population differs')
    provenance = _provenance(ROOT)
    inputs = dict(prior['input_sha256'])
    if (provenance != prior['provenance']
        or any(_sha256(Path(name)) != digest for name,digest in inputs.items())):
        raise ValueError('production/original input provenance changed')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):
        raise ValueError('ordinary assertions required')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,
        provenance=provenance, input_sha256=inputs, tests=tests,
        required_test_count=EXPECTED_TESTS, complete_test_wall_cap_s=60,
        stage_worker_wall_cap_s=240, cpu_threads=1, gpu_enabled=False,
        address_space_bytes=16*1024**3, transient_bytes=1024**3, entries_cap=64_000_000,
        category_caps=CATEGORY_CAPS, complete_work_upper=254_000_000,
        whole_work_cap=256_000_000, branch_work_cap=200_000_000,
        fixed_complete_source_modes=list(MODES),
        fixed_source_geometry=dict(C=16,K=32,input=[6,6],output=[4,4]),
        child_source_reserved_work=dict(dense=32_000_000,masked=32_000_000,
            heterogeneous=32_000_000,noop=4_000_000),
        shared_radix_caps=[16384,131072,16_000_000],
        prior_C125_complete_qualification_passed=True,
        prior_C124_complete_qualification_passed=False,
        complete_old_new_native_packets_required=True,
        complete_JSON_receipts_and_actual_sizes_required=True,
        full_owned_kernel_and_four_point_artifacts_required=True,
        numeric_hash_traffic_in_token_pool=False, all_CPU_work_in_generation_cap=False,
        scope='complete_fresh_ordinary_nonconvex_support_compiled_exact_word_mixed_rows',
        archived_HZ_restore_authorized=False, original_network_run_authorized=False,
        solver_authorized=False, actual_network_source_or_LIVE_admitted=False,
        kernel_cache_or_reuse_authorized=False, promotion_authorized=False, formal_gain=0))
    started = time.monotonic()
    record = dict(all_stages_passed=False, formal_gain=0,
                  prior_C125_complete_qualification_passed=True,
                  prior_C124_complete_qualification_passed=False)
    try:
        command = [sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',
                   *(str(EXP/name) for name in tests)]
        collected = subprocess.run([*command,'--collect-only'],cwd=ROOT,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        with (RUN/'collection.log').open('x') as stream:
            stream.write(collected.stdout)
        ids = [line for line in collected.stdout.splitlines()
               if line.startswith(('experiments/','act/')) and '::' in line]
        files = {str((EXP/name).resolve().relative_to(ROOT)) for name in tests}
        if (collected.returncode or len(ids) != EXPECTED_TESTS
            or len(set(ids)) != EXPECTED_TESTS
            or {name.split('::',1)[0] for name in ids} != files):
            raise ValueError('complete exact test inventory differs')
        _atomic_exclusive_json(RUN/'inventory.json',dict(nodeids=ids,count=len(ids)))
        left = 60-(time.monotonic()-started)
        if left <= 0:
            raise subprocess.TimeoutExpired(command,60)
        with (RUN/'tests.log').open('x') as stream:
            tested = subprocess.run([*command,'--junitxml='+str(RUN/'tests.xml')],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=left)
        record.update(tests_exit=tested.returncode,tests_count=len(ids),
                      test_wall_s=time.monotonic()-started)
        cases = ET.parse(RUN/'tests.xml').findall('.//testcase')
        actual = [case.get('classname','').replace('.','/')+'.py::'+case.get('name','')
                  for case in cases]
        if (tested.returncode or sorted(actual) != sorted(ids)
            or any(case.find(name) is not None for case in cases
                   for name in ('failure','error','skipped'))):
            raise ValueError('complete inherited/new qualification failed')
        print(json.dumps(dict(event='all_tests_passed',tests=len(ids),
                              wall_s=record['test_wall_s'])),flush=True)
        if any(_sha256(EXP/name) != digest for name,digest in hashes.items()):
            raise ValueError('source drift after qualification')
        with (RUN/'worker.log').open('x') as stream:
            worker = subprocess.run([sys.executable,str(EXP/'c126_mixed_source_worker_v1.py')],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit'] = worker.returncode
        result = json.loads((RUN/'result.json').read_text())
        if (worker.returncode or not result['completed'] or result.get('failure')
            or result['solver_calls'] or result['source_drift'] or result['input_drift']
            or result['provenance_drift'] or not result['data']['qualification_pass']):
            raise ValueError('complete ordinary support-word native qualification failed')
        saved = verify_saved_evidence(result)
        record.update(all_stages_passed=True,work=result['work'],
            complete_source_count=result['data']['complete_source_count'],
            complete_evidence_payment_checks=saved)
    except subprocess.TimeoutExpired as exc:
        record['timeout_s'] = exc.timeout
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/name) != digest for name,digest in hashes.items()),
            input_drift=any(_sha256(Path(name)) != digest for name,digest in inputs.items()),
            provenance_drift=_provenance(ROOT) != provenance,
            artifacts={str(path.relative_to(RUN)):_sha256(path)
                       for path in RUN.rglob('*') if path.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record)
        print(json.dumps(record),flush=True)
    if (not record['all_stages_passed'] or record['source_drift']
        or record['input_drift'] or record['provenance_drift']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
