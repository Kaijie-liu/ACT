"""One fresh-source diagnostic, after the unchanged D099 mathematical gate.

The 240 second timeout covers worker startup, imports and its complete run;
the worker owns its stricter 235 second internal deadline. Parent identity
reads, finalization and host observations are reported separately, not hidden
inside the worker's scalar-work meter. A completed native-bank diagnostic is
NOT model/native/physical qualification and is never a formal solved gain.
No retry, candidate import, test rerun, source snapshot or old-file write.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import tracemalloc

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d099_fresh_residual_bank_20261001_v1'
OUT = RUN / 'source_probe_supervisor'
WORKER_OUT = RUN / 'source_probe'
FREEZE = HERE / 'freeze.json'
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
FILES = ('CONTRACT.md', 'PREREG.md', 'source_probe.py', 'run_source.py',
         'run_math.py', 'collection_contract.py')
RESERVE, MEMORY_CAP, AS_CAP = 65536, 1024**3, 16*1024**3


def read_json(path, limit=RESERVE):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError('missing, oversized or linked supervisor input: ' + str(path))
    return json.loads(path.read_text())


def sha(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or linked supervision identity: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def merge(identities, path, digest):
    if (type(path) is not str or not Path(path).is_absolute()
            or type(digest) is not str or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('invalid source identity')
    if path in identities and identities[path] != digest:
        raise ValueError('conflicting source identity: ' + path)
    identities[path] = digest


def project_population(manifest):
    record = manifest.get('project_import_closure')
    closure_path = HERE.parent / 'd090_bound_native_discovery_20261001/project_import_closure.json'
    if (type(record) is not dict or record.get('path') != str(closure_path)
            or record.get('file_count') != 116 or record.get('total_bytes') != 2782307
            or record.get('source_hash_only') is not True
            or record.get('outside_act_paths') != []
            or record.get('sha256') != manifest['source_sha256'].get(str(closure_path))
            or sha(closure_path) != record['sha256']):
        raise ValueError('inherited project closure differs')
    closure = read_json(closure_path)
    files = closure.get('files')
    if (closure.get('schema') != 'd090_project_import_closure_v1'
            or closure.get('project_root') != str(ROOT)
            or type(files) is not dict or len(files) != 116
            or record.get('source_files') != list(files)
            or sorted(str(p) for p in (ROOT / 'act').rglob('*.py')) != list(files)):
        raise ValueError('project source population differs')
    for path, item in files.items():
        if (type(item) is not dict
                or manifest['source_sha256'].get(path) != item.get('sha256')
                or Path(path).stat().st_size != item.get('bytes')):
            raise ValueError('project closure member differs: ' + path)


def rss():
    with Path('/proc/self/status').open() as stream:
        for line in stream:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])*1024
    raise ValueError('own RSS unavailable')


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    if RUN.is_symlink() or not RUN.is_dir():
        raise ValueError('the one mathematical run must exist first')
    OUT.mkdir(exist_ok=False)  # This first attempt consumes the source-stage version.
    started = time.monotonic()
    identities, inputs, manifest = {}, {}, None
    record = dict(schema='d099_fresh_residual_bank_supervisor_v1',
        worker_launched=False, timeout=False, worker_exit=None,
        execution_completed=False, diagnostic_completed=False,
        native_closed_bank_diagnostic_completed=False,
        source_component_qualified=False, source_census_qualified=False,
        complete_physical_qualification=False, actual_model_binding_qualified=False,
        production_scalar_work_qualified=False, native_HZ_admitted=False,
        gpu_computation_completed=False, formal_gain=0,
        source_drift=[], input_drift=[], artifacts={},
        accounting_scope='supervisor identity and host costs are separate from the worker meter')
    initial = None
    try:
        initial = rss()
        tracemalloc.start()
        resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        sys.dont_write_bytecode = True
        if (not __debug__ or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0')
                or os.environ.get('LD_PRELOAD')):
            raise ValueError('assertions and the unmodified interpreter environment required')
        if WORKER_OUT.exists() or WORKER_OUT.is_symlink():
            raise ValueError('source worker version is already consumed')
        frozen = read_json(FREEZE)
        done = read_json(RUN / 'exit.json')
        manifest = read_json(RUN / 'preregistered.json', 8*1024**2)
        inventory = read_json(RUN / 'inventory.json', 8*1024**2)
        sources = frozen.get('source_sha256')
        if (frozen.get('schema') != 'd099_fresh_residual_bank_v1'
                or frozen.get('required_tests') != 3845 or frozen.get('required_test_files') != 188
                or frozen.get('new_test_names') != []
                or type(sources) is not dict or set(sources) != {str(HERE / n) for n in FILES}
                or any(done.get(k) is not True for k in ('all_stages_passed',
                    'component_tests_passed', 'mathematical_component_gate_passed',
                    'host_observations_within_caps', 'inventory_validated_before_execution'))
                or done.get('supervisor_exit') != 0 or done.get('tests_exit') != 0
                or done.get('tests_count') != 3845 or done.get('test_files') != 188
                or not 0 <= done.get('test_wall_s', 61) <= 60
                or done.get('source_drift') != [] or done.get('input_drift') != []
                or done.get('provenance_drift') is not False or 'failure' in done):
            raise ValueError('complete frozen mathematical gate has not passed')
        manifest_digest = sha(RUN / 'preregistered.json')
        if (manifest.get('required_tests') != 3845 or manifest.get('required_test_files') != 188
                or manifest.get('inherited_tests') != 3845 or manifest.get('inherited_test_files') != 188
                or manifest.get('new_test_names') != [] or manifest.get('new_test_files') != 0
                or manifest.get('inherited_test_population_unchanged') is not True
                or manifest.get('single_fresh_source_probe_registered') is not True
                or manifest.get('source_stop_layer') != 20
                or manifest.get('freeze_sha256') != sha(FREEZE)
                or done.get('artifacts', {}).get('preregistered.json') != manifest_digest
                or done.get('artifacts', {}).get('inventory.json') != sha(RUN / 'inventory.json')
                or inventory.get('manifest_sha256') != manifest_digest
                or inventory.get('validated_before_execution') is not True
                or inventory.get('count') != 3845 or inventory.get('files') != 188
                or inventory.get('nodeids') != manifest.get('expected_nodeids')
                or len(manifest.get('tests', ())) != 188
                or len(manifest.get('expected_nodeids', ())) != 3845):
            raise ValueError('mathematical provenance or complete inventory differs')
        for path, digest in manifest['source_sha256'].items():
            merge(identities, path, digest)
        for path, digest in sources.items():
            merge(identities, path, digest)
        for path, digest in manifest['input_sha256'].items():
            merge(inputs, path, digest)
            if path in identities and identities[path] != digest:
                raise ValueError('source and input identities conflict')
        for name in ('freeze.json',):
            merge(identities, str(HERE / name), sha(HERE / name))
        for name in ('exit.json', 'inventory.json', 'preregistered.json'):
            merge(identities, str(RUN / name), sha(RUN / name))
        if (len(inputs) != 9 or len(manifest.get('gpu_dependency_files', ())) != 4417
                or len(manifest.get('decoder_dependency_files', ())) != 1011
                or Path(sys.executable).resolve() != PYTHON.resolve()
                or identities.get(str(PYTHON.resolve())) != sha(PYTHON.resolve())):
            raise ValueError('inherited source population or interpreter differs')
        for path, digest in identities.items():
            if sha(Path(path)) != digest:
                raise ValueError('pre-execution source drift: ' + path)
        for path, digest in inputs.items():
            if sha(Path(path)) != digest:
                raise ValueError('pre-execution input drift: ' + path)
        project_population(manifest)
        record.update(source_identity_count=len(identities), input_identity_count=len(inputs),
            source_identity_bytes=sum(Path(p).stat().st_size for p in identities),
            input_identity_bytes=sum(Path(p).stat().st_size for p in inputs),
            manifest_sha256=manifest_digest, freeze_sha256=sha(FREEZE))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
            CUDA_VISIBLE_DEVICES='', CUDA_LOG_FILE='stderr', OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
            VECLIB_MAXIMUM_THREADS='1', NEURAL_HZ_D099_MANIFEST_SHA256=manifest_digest)
        for name in ('TMPDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'CUDA_CACHE_PATH',
                     'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR'):
            path = OUT / name.lower()
            path.mkdir()
            env[name] = str(path)
        command = [sys.executable, '-B', str(HERE / 'source_probe.py'), '--enabled']
        record.update(command=command, cpu_affinity=sorted(os.sched_getaffinity(0)),
            worker_timeout_s=240, worker_internal_deadline_s=235, as_cap_bytes=AS_CAP,
            mathematical_run=str(RUN), prelaunch_wall_s=time.monotonic()-started)
        with (OUT / 'worker.log').open('xb') as stream:
            launch = time.monotonic()
            record['worker_launched'] = True
            try:
                completed = subprocess.run(command, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=240,
                    check=False, cwd=str(ROOT))
                record['worker_exit'] = completed.returncode
            except subprocess.TimeoutExpired:
                # subprocess.run kills and waits for THIS child before raising.
                record['timeout'] = True
            finally:
                record['subprocess_wall_s'] = time.monotonic()-launch
        receipt = WORKER_OUT / 'worker.json'
        if receipt.is_file():
            worker = read_json(receipt)
            if worker.get('schema') != 'd099_fresh_residual_bank_worker_v1':
                raise ValueError('unexpected fresh source receipt schema')
            record['worker_receipt'] = dict(file=str(receipt), sha256=sha(receipt),
                bytes=receipt.stat().st_size, failure=worker.get('failure'))
            completed = (record['worker_exit'] == 0 and not record['timeout']
                and all(worker.get(k) is True for k in ('prefix_completed',
                    'bank_application_completed', 'all_groups_applied',
                    'execution_completed', 'native_closed_bank_diagnostic_completed'))
                and worker.get('stopped_at_layer') == 20 and worker.get('formal_gain') == 0
                and not worker.get('failure'))
            record.update(execution_completed=completed, diagnostic_completed=completed,
                          native_closed_bank_diagnostic_completed=completed)
            for name in ('production_scalar_work_qualified', 'complete_physical_qualification',
                         'actual_model_binding_qualified', 'source_census_qualified'):
                if worker.get(name) is not False:
                    raise ValueError('diagnostic must not upgrade qualification: ' + name)
    except BaseException as exc:
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:2048])
    finally:
        sealing = time.monotonic()
        for label, population in (('source_drift', identities), ('input_drift', inputs)):
            for path, digest in population.items():
                try:
                    if sha(Path(path)) != digest:
                        record[label].append(path)
                except BaseException as exc:
                    record.setdefault('identities_unchecked', []).append(
                        dict(path=path, reason=str(exc)[:256]))
        if manifest is not None:
            try:
                project_population(manifest)
            except BaseException as exc:
                record['project_population_failure'] = str(exc)[:512]
        log = OUT / 'worker.log'
        if log.is_file():
            try:
                record['artifacts']['worker.log'] = dict(bytes=log.stat().st_size, sha256=sha(log))
            except BaseException as exc:
                record['sealing_error'] = str(exc)[:512]
        try:
            if initial is None or not tracemalloc.is_tracing():
                raise ValueError('missing host telemetry')
            current, peak = tracemalloc.get_traced_memory()
            metadata = tracemalloc.get_tracemalloc_memory()
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            growth = max(0, peak_rss-initial)
            memory_ok = (peak_rss+RESERVE <= MEMORY_CAP and growth+RESERVE <= MEMORY_CAP
                         and peak+metadata+RESERVE <= MEMORY_CAP)
            record.update(initial_rss_bytes=initial, peak_rss_bytes=peak_rss,
                rss_growth_bytes=growth, tracemalloc_current_bytes=current,
                tracemalloc_peak_bytes=peak, tracemalloc_metadata_bytes=metadata,
                summary_reserve_bytes=RESERVE, supervisor_memory_gate_passed=memory_ok)
        except BaseException as exc:
            record['supervisor_memory_gate_passed'] = False
            record.setdefault('failure', dict(type=type(exc).__name__, reason=str(exc)[:512]))
        passed = bool(record['diagnostic_completed'] and record['supervisor_memory_gate_passed']
            and not record['source_drift'] and not record['input_drift']
            and not record.get('identities_unchecked') and not record.get('failure')
            and not record.get('sealing_error') and not record.get('project_population_failure'))
        record.update(execution_completed=passed, diagnostic_completed=passed,
            native_closed_bank_diagnostic_completed=passed,
            supervisor_exit=0 if passed else 1, supervisor_wall_s=time.monotonic()-started,
            finalization_wall_s=time.monotonic()-sealing)
        encoded = json.dumps(record, sort_keys=True, indent=2, allow_nan=False)
        if len(encoded.encode())+1 > RESERVE:
            record['diagnostic_completed'] = False
            record['supervisor_exit'] = 1
            encoded = json.dumps(dict(schema=record['schema'], diagnostic_completed=False,
                execution_completed=False, native_closed_bank_diagnostic_completed=False,
                source_component_qualified=False, complete_physical_qualification=False,
                formal_gain=0, supervisor_exit=1, failure={'type': 'SummaryReserveExceeded'},
                worker_exit=record['worker_exit'], timeout=record['timeout']))
        with (OUT / 'supervisor.json').open('x') as stream:
            stream.write(encoded+'\n')
    return record['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
