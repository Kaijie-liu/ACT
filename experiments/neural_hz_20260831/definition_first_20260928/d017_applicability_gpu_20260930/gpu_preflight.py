"""Single-use GPU prerequisite experiment, not a network census or verifier."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import tracemalloc
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
OLD = HERE.parent / 'd015_batch_binding_20260928_v2/run_v2.py'
PRIOR = EXP / 'results/d015_source_shielding_20260928_v2'
RUN = EXP / 'results/d017_applicability_gpu_20260930_v1'
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
SITE = Path('/data1/Kane/miniconda3/lib/python3.13/site-packages')
GPU = 'GPU-f491c2c6-a093-590a-6b8a-13b5f76aadcc'
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
ANCHORS = {
    OLD: '8ff9d296ce56b8dd9481d0c8367484b8ff37148db5dfa522062136d9d04e12ce',
    PRIOR / 'preregistered.json': '4697d2bbc1b86e7732e825e0e8a3b6d9c595fc275516688d746e738d2bba50cd',
    PRIOR / 'inventory.json': '542946f332ce716a0b6274e43587d3154da9c5a2d02698c956716bfa6a9bbf75',
    PRIOR / 'exit.json': 'a5768b2b223ebdb869af8cf8358eccc7a00afb5477e3d023ac3071c0ff9d7168',
    PRIOR / 'diagnostic.json': '1551b84e327d3adc07350e4ec06b8f874608eef4a96983e0f55f29a090eae3cf',
}
NEW_FILES = ('gpu_preflight.py', 'rank_probe.py', 'test_rank_probe.py',
             'PREREG.md', 'APPLICABILITY_THEOREM.md')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def save(name, value):
    with (RUN / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def gpu_dependencies(helper, identities):
    paths = set()
    packages = ('torch', 'nvidia', 'torchgen', 'sympy', 'mpmath', 'networkx',
                'filelock', 'fsspec', 'jinja2', 'markupsafe')
    for folder in (SITE / name for name in packages):
        if not folder.is_dir():
            raise ValueError('required GPU package directory absent: ' + str(folder))
        paths.update(p.resolve() for p in folder.rglob('*') if p.is_file()
                     and (p.suffix in ('.py', '.so') or '.so.' in p.name))
    paths.add(Path('/usr/bin/nvidia-smi').resolve())
    for pattern in ('libcuda.so*', 'libnvidia-*.so*'):
        paths.update(p.resolve() for p in Path('/usr/lib/x86_64-linux-gnu').glob(pattern)
                     if p.is_file())
    if not any(p.name.startswith('libcuda.so') for p in paths):
        raise ValueError('CUDA driver dependency absent')
    for path in sorted(paths):
        helper.bind(identities, path, sha(path))
    return [str(p) for p in sorted(paths)]


def process_status():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmSize:', 'VmRSS:', 'VmHWM:')):
            key, value, unit = line.split()
            if unit != 'kB':
                raise ValueError('unexpected proc memory unit')
            values[key[:-1] + '_bytes'] = int(value) * 1024
    return values


def worker():
    if sys.argv[1:] != ['--worker', '--enabled']:
        raise ValueError('worker needs explicit opt-in')
    limits()
    start = time.monotonic()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    initial = process_status()
    tracemalloc.start()
    record = dict(gpu_computation_completed=False, source_census_completed=False,
                  native_HZ_admitted=False, complete_physical_qualification=False,
                  formal_gain=0, initial_memory=initial, runtime_initialization_included=True)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
            raise ValueError('GPU device binding differs')
        registered = json.loads((RUN / 'preregistered.json').read_text())
        candidate = HERE / 'rank_probe.py'
        if sha(candidate) != registered['source_sha256'][str(candidate)]:
            raise ValueError('candidate changed before worker import')
        probe = load(candidate, 'd017_rank_probe_worker')
        print(json.dumps(dict(event='before_torch_import', memory=initial)), flush=True)
        result = probe.gpu_probe(enabled=True)
        if result.get('correctness') is not True:
            raise ValueError('GPU fixture mismatch')
        if (type(result.get('work_upper_bound')) is not int
                or not 0 <= result['work_upper_bound'] <= 200_000_000
                or type(result.get('retained_entries_upper_bound')) is not int
                or not 0 <= result['retained_entries_upper_bound'] <= 64_000_000):
            raise ValueError('unchanged work or entry cap exceeded')
        telemetry = subprocess.check_output(['/usr/bin/nvidia-smi',
            '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
            text=True, timeout=10)
        own = [line.split(',') for line in telemetry.splitlines()
               if line.split(',')[0].strip() == str(os.getpid())]
        if len(own) != 1:
            raise ValueError('unique own-process CUDA memory telemetry missing')
        context_bytes = int(own[0][1].strip()) * 1024**2
        record.update(gpu_result=result, observed_context_bytes=context_bytes,
                      gpu_computation_completed=True)
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        _, peak = tracemalloc.get_traced_memory()
        metadata = tracemalloc.get_tracemalloc_memory()
        growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0)
        context = record.get('observed_context_bytes', 0)
        observed_ok = (growth + context + RESERVE <= MEMORY_CAP
                       and peak + metadata + context + RESERVE <= MEMORY_CAP)
        record.update(wall_s=time.monotonic() - start, final_memory=process_status(),
                      rss_highwater_growth_bytes=growth, traced_peak_bytes=peak,
                      tracer_metadata_bytes=metadata, summary_reserve_bytes=RESERVE,
                      observed_memory_within_caps=observed_ok,
                      context_telemetry_is_snapshot_not_peak=True)
        if not observed_ok:
            record.setdefault('failure', dict(type='MemoryError', reason='observed full-cost cap exceeded'))
        # Kernel readiness is not complete physical qualification: the process
        # context sample is not an independently captured context peak.
        save('gpu_diagnostic.json', record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return 0 if record['gpu_computation_completed'] and 'failure' not in record else 1


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('supervisor needs explicit opt-in')
    RUN.mkdir(exist_ok=False)
    start = time.monotonic()
    identities, inputs, helper, frozen = {}, {}, None, None
    test_start = None
    record = dict(component_tests_passed=False, gpu_preflight_passed=False,
                  source_census_completed=False, native_HZ_admitted=False,
                  complete_physical_qualification=False, formal_gain=0)
    try:
        limits()
        if not __debug__ or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
            raise ValueError('assertions required')
        if any(sha(path) != digest for path, digest in ANCHORS.items()):
            raise ValueError('frozen authority drift')
        helper = load(OLD, 'd017_inherited_helpers')
        prior = json.loads((PRIOR / 'preregistered.json').read_text())
        done = json.loads((PRIOR / 'exit.json').read_text())
        inventory = json.loads((PRIOR / 'inventory.json').read_text())
        if (done['component_tests_passed'] is not True or done['tests_exit'] != 0
                or done['tests_count'] != 3730 or done['all_stages_passed'] is not False
                or done['source_census_completed'] is not False or not done.get('failure')
                or done['source_drift'] or done['input_drift'] or done['provenance_drift']
                or inventory['count'] != 3730 or inventory['files'] != 164
                or len(prior['tests']) != 164
                or sorted(inventory['nodeids']) != sorted(prior['expected_nodeids'])):
            raise ValueError('inherited population or failed-source status differs')
        identities.update(prior['source_sha256'])
        inputs.update(prior['input_sha256'])
        for path, digest in ANCHORS.items():
            helper.bind(identities, path, digest)
        for name, digest in done['artifacts'].items():
            helper.bind(identities, helper.original_path(PRIOR, name), digest)
        for name in NEW_FILES:
            helper.bind(identities, HERE / name, sha(HERE / name))
        if (Path(sys.executable).resolve() != PYTHON.resolve()
                or sha(sys.executable) != identities[str(PYTHON.resolve())]):
            raise ValueError('inherited interpreter drift')
        selected = helper.select_sources(identities, inputs)
        if selected != prior['selected_sources']:
            raise ValueError('original source population changed')
        if helper.bind_decoder(identities) != prior['decoder_dependency_files']:
            raise ValueError('inherited decoder population changed')
        gpu_files = gpu_dependencies(helper, identities)
        frozen = helper.provenance()
        if frozen != prior['provenance'] or frozen['branch'] != 'redu-hz':
            raise ValueError('production provenance drift')
        if helper.drift(identities) or helper.drift(inputs):
            raise ValueError('pre-run identity drift')
        tests = [*prior['tests'], str(HERE / 'test_rank_probe.py')]
        tree = ast.parse((HERE / 'test_rank_probe.py').read_text())
        names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name.startswith('test_')]
        if len(names) != 5 or len(set(names)) != 5:
            raise ValueError('exact five-test delta required')
        relative = str((HERE / 'test_rank_probe.py').relative_to(ROOT))
        expected = [*inventory['nodeids'], *(relative + '::' + n for n in names)]
        if len(expected) != 3735 or len(set(expected)) != 3735:
            raise ValueError('required 3735-test population differs')
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        (RUN / 'tmp').mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
                   CUDA_CACHE_PATH=str(RUN / 'cuda_cache'), TORCH_HOME=str(RUN / 'torch_home'),
                   XDG_CACHE_HOME=str(RUN / 'xdg_cache'), TRITON_CACHE_DIR=str(RUN / 'triton_cache'),
                   TORCHINDUCTOR_CACHE_DIR=str(RUN / 'inductor_cache'), TMPDIR=str(RUN / 'tmp'))
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=frozen, tests=tests, expected_nodeids=expected,
            required_tests=3735, required_test_files=165, selected_sources=selected,
            gpu_dependency_files=gpu_files, gpu_uuid=GPU, prior_source_qualified=False,
            gpu_module_loading='LAZY', caches_relocated_to_new_run=True,
            prior_failure=done['failure'], inherited_component_tests=3730,
            cpu_affinity=list(os.sched_getaffinity(0)), address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, worker_wall_cap_s=240,
            whole_work_cap=256_000_000, branch_work_cap=200_000_000,
            host_memory_cap_bytes=MEMORY_CAP, retained_entry_cap=64_000_000,
            scope='CPU primitive tests then GPU exact synthetic rank preflight only',
            no_original_model_decode=True, source_census_completed=False, formal_gain=0))
        print(json.dumps(dict(event='frozen_before_import', tests=3735, files=165,
                              gpu_dependencies=len(gpu_files))), flush=True)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short',
                   '-p', 'no:cacheprovider', *tests]
        test_start = time.monotonic()
        with (RUN / 'collection.log').open('x') as stream:
            collected = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        ids = [s for s in (RUN / 'collection.log').read_text().splitlines()
               if s.startswith(('experiments/', 'act/')) and '::' in s]
        if collected.returncode or sorted(ids) != sorted(expected) or len(set(ids)) != 3735:
            raise ValueError('complete collection inventory differs')
        save('inventory.json', dict(nodeids=ids, count=len(ids), files=165))
        remaining = 60 - (time.monotonic() - test_start)
        if remaining <= 0:
            raise TimeoutError('collection exhausted combined budget')
        with (RUN / 'tests.log').open('x') as stream:
            tested = subprocess.run([*command, '--junitxml=' + str(RUN / 'tests.xml')],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=remaining, preexec_fn=limits)
        record.update(test_wall_s=time.monotonic() - test_start,
                      tests_exit=tested.returncode, tests_count=len(ids))
        test_start = None
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [c.get('classname', '').replace('.', '/') + '.py::' + c.get('name', '') for c in cases]
        if (tested.returncode or sorted(actual) != sorted(expected) or record['test_wall_s'] > 60
                or any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))):
            raise ValueError('full component gate failed')
        record['component_tests_passed'] = True
        print(json.dumps(dict(event='full_component_pass', tests=3735,
                              wall_s=record['test_wall_s'])), flush=True)
        if helper.drift(identities) or helper.drift(inputs) or helper.provenance() != frozen:
            raise ValueError('identity drift before GPU worker')
        gpu_env = dict(env, CUDA_VISIBLE_DEVICES=GPU, CUDA_MODULE_LOADING='LAZY')
        with (RUN / 'gpu.log').open('x') as stream:
            result = subprocess.run(
                [sys.executable, '-B', str(HERE / 'gpu_preflight.py'), '--worker', '--enabled'],
                cwd=ROOT, env=gpu_env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=240, preexec_fn=limits)
        record['gpu_exit'] = result.returncode
        report = json.loads((RUN / 'gpu_diagnostic.json').read_text())
        record['gpu_preflight_passed'] = (result.returncode == 0
            and report['gpu_computation_completed'] is True and 'failure' not in report)
        if not record['gpu_preflight_passed']:
            raise ValueError('GPU prerequisite failed; no CPU fallback and no cap relaxation')
    except Exception as exc:
        if test_start is not None:
            record['test_wall_s'] = time.monotonic() - test_start
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        if helper is not None:
            record['source_drift'], record['input_drift'] = helper.drift(identities), helper.drift(inputs)
            try:
                record['provenance_drift'] = frozen is not None and helper.provenance() != frozen
            except Exception as exc:
                record['provenance_drift'] = True
                record['provenance_check_failure'] = str(exc)
            if record['source_drift'] or record['input_drift'] or record['provenance_drift']:
                record['component_tests_passed'] = record['gpu_preflight_passed'] = False
        record['wall_s'] = time.monotonic() - start
        record['artifacts'] = {str(p.relative_to(RUN)): sha(p) for p in RUN.rglob('*') if p.is_file()}
        save('exit.json', record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return 0 if record['component_tests_passed'] and record['gpu_preflight_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(worker() if '--worker' in sys.argv else main())
