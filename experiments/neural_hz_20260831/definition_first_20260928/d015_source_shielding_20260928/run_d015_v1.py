"""Single-use D015 gate. Freeze bytes before imports; never reuse an old RUN."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
DEFINITION = HERE.parent
EXP = DEFINITION.parent
ROOT = EXP.parent.parent
PRIOR = EXP / 'results/d003_reference_semantics_20260928_v1'
RUN = EXP / 'results/d015_source_shielding_20260928_v1'
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
SITE = Path('/data1/Kane/miniconda3/lib/python3.13/site-packages')
AS_CAP = 16 * 1024**3
MEMORY_CAP = 1024**3
WHOLE_WORK_CAP = 256_000_000
BRANCH_WORK_CAP = 200_000_000
TEST_FILES = ('test_shield_kernel_v1.py', 'test_source_packet_v1.py',
              'test_census_worker_v1.py')
NEW_FILES = ('shield_kernel_v1.py', 'source_packet_v1.py', 'census_worker_v1.py',
             'run_d015_v1.py', 'DESIGN.md', 'PREREG.md', *TEST_FILES)
EXPECTED_NEW_TEST_COUNTS = {'test_shield_kernel_v1.py': 18,
    'test_source_packet_v1.py': 12, 'test_census_worker_v1.py': 11}
PRODUCTION = (
    'act/back_end/solver/neural_hz.py', 'act/back_end/solver/solver_hz.py',
    'act/back_end/hybridz_tf/hybridz_tf.py', 'act/back_end/hybridz_tf/tf_mlp.py',
    'act/back_end/hybridz_tf/tf_cnn.py', 'act/back_end/hybridz_tf/exact_linear_op.py',
    'act/back_end/verifier.py', 'act/config/config.py',
    'experiments/neural_hz_20260831/shadow_worker.py',
    'experiments/neural_hz_20260831/shadow_worker_dtype_v2.py',
    'act/pipeline/verification/torch2act.py',
    'act/pipeline/verification/batchnorm_graph.py',
    'act/front_end/vnnlib_loader/onnx_converter.py')
ANCHORS = {
    PRIOR / 'preregistered.json': 'c52bfdfb001447077a20c6337493050260a26d41dbe2e50cadfc58bf231c4c8d',
    PRIOR / 'inventory.json': '70b1e504de6929b33cb7e2b437f55814fc90b495489ad0c59fa0a9c8efc0dbaa',
    PRIOR / 'exit.json': 'f3138274edfd773a799ef5180f7ec6881982e1e4cf417688e371587f1df61ff7',
    DEFINITION / 'd014_guarded_amplitude_20260928/SHA256SUMS':
        'cc3b9715df411d7c94168180d62e7a0ed310d81b1852d8735f535e63de6bc7e4'}
MANIFESTS = {
    'cifar100_2024_universe_v1.json':
        'fa30dafe17cdcafeb08b56da66189795e1623b5556ced7a8247903fef507d948',
    'tinyimagenet_2024_universe_v1.json':
        'a8a0dc7504af2c6b89d099fd5c74f27aa5ae98458c5c151cbb6d0da2ef5c1f59'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def save(name, data):
    with (RUN / name).open('x') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def bind(registry, path, digest):
    name = str(path)
    if name in registry and registry[name] != digest:
        raise ValueError('conflicting frozen identity: ' + name)
    registry[name] = digest


def drift(registry):
    changed = []
    for name, digest in registry.items():
        try:
            if sha(name) != digest:
                changed.append(name)
        except OSError:
            changed.append(name)
    return changed


def provenance():
    digest = hashlib.sha256()
    for name in PRODUCTION:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return dict(branch=subprocess.check_output(
        ['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        candidate_sha256=digest.hexdigest())


def child_limits():
    resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))


def original_path(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('original manifest path must remain relative')
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('original manifest path escaped benchmark root')
    return path


def select_sources(identities, inputs):
    """Read metadata only: lexicographically first original spec per model."""
    selected = []
    for name, expected in sorted(MANIFESTS.items()):
        manifest_path = EXP / 'manifests' / name
        if sha(manifest_path) != expected:
            raise ValueError('source universe manifest drift')
        bind(identities, manifest_path, expected)
        manifest = json.loads(manifest_path.read_text())
        family = manifest['family']
        benchmark = Path(manifest['source_benchmark_root']) / family
        models = {}
        for row in manifest['instances']:
            model_rel = row['model_relative_path']
            digest = row['model_sha256']
            previous = models.setdefault(model_rel, dict(digest=digest, specs={}))
            if previous['digest'] != digest:
                raise ValueError('inconsistent original model identity')
            bind(previous['specs'], row['spec_relative_path'], row['spec_sha256'])
        for model_rel, item in sorted(models.items()):
            spec_rel = min(item['specs'])
            model_path = original_path(benchmark, model_rel)
            spec_path = original_path(benchmark, spec_rel)
            bind(inputs, model_path, item['digest'])
            bind(inputs, spec_path, item['specs'][spec_rel])
            selected.append(dict(family=family, model_relative_path=model_rel,
                spec_relative_path=spec_rel, model_path=str(model_path),
                model_sha256=item['digest'], spec_path=str(spec_path),
                spec_sha256=item['specs'][spec_rel], manifest_path=str(manifest_path),
                manifest_sha256=expected))
    if len(selected) != 3 or len({row['model_path'] for row in selected}) != 3:
        raise ValueError('three-model source population differs')
    return selected


def bind_decoder(identities):
    """Freeze decoder package files without importing or decoding anything."""
    paths = set()
    directories = ['onnx', 'google/protobuf', 'google/_upb', 'ml_dtypes', 'numpy']
    if (SITE / 'numpy.libs').is_dir():
        directories.append('numpy.libs')
    for name in directories:
        directory = SITE / name
        if not directory.is_dir():
            raise ValueError('missing decoder dependency directory: ' + str(directory))
        found = {p for p in directory.rglob('*') if p.is_file()
                 and (p.suffix in ('.py', '.so') or '.so.' in p.name)}
        if not found:
            raise ValueError('empty decoder dependency inventory: ' + name)
        paths.update(found)
    paths.add(SITE / 'typing_extensions.py')
    for path in sorted(paths):
        bind(identities, path, sha(path))
    return [str(path) for path in sorted(paths)]


def diagnostic_gate(result):
    if result.get('source_census_completed') is not True or result.get('memory_gate_passed') is not True:
        raise ValueError('source census or worker memory gate did not pass')
    fields = ('rss_highwater_growth_bytes', 'final_summary_reserve_bytes',
              'traced_peak_bytes', 'tracer_metadata_bytes', 'retained_entries',
              'whole_work_used', 'branch_work_used')
    if any(type(result.get(key)) is not int or result[key] < 0 for key in fields):
        raise ValueError('invalid diagnostic accounting field')
    reserve = result['final_summary_reserve_bytes']
    if (reserve < 65536 or result['rss_highwater_growth_bytes'] + reserve > MEMORY_CAP
        or result['traced_peak_bytes'] + result['tracer_metadata_bytes'] + reserve > MEMORY_CAP
        or result['retained_entries'] > 64_000_000
        or result['whole_work_used'] > WHOLE_WORK_CAP
        or result['branch_work_used'] > BRANCH_WORK_CAP):
        raise ValueError('unchanged memory, entry or work gate failed')
    wall = result.get('wall_s')
    if type(wall) not in (int, float) or not math.isfinite(wall) or not 0 <= wall <= 240:
        raise ValueError('worker wall cap failed')
    if result.get('diagnostic_solver_calls') != 0 or result.get('formal_gain') != 0:
        raise ValueError('census cannot execute a solver or claim formal gain')


def main():
    RUN.mkdir(exist_ok=False)
    start = time.monotonic()
    record = dict(all_stages_passed=False, component_tests_passed=False,
        source_census_completed=False, complete_source_qualification=False,
        native_HZ_admitted=False, formal_gain=0, new_benchmark_solves=0,
        inherited_tests_include_solver_calls=True, diagnostic_solver_calls=0)
    identities, inputs, frozen_provenance = {}, {}, None
    test_start = None
    try:
        child_limits()
        if not __debug__ or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
            raise ValueError('assertions required')
        if EXPECTED_NEW_TEST_COUNTS is None or set(EXPECTED_NEW_TEST_COUNTS) != set(TEST_FILES):
            raise ValueError('root must freeze final per-file D015 test counts before execution')
        if any(sha(path) != digest for path, digest in ANCHORS.items()):
            raise ValueError('D003/D014 authority anchor drift')
        prior = json.loads((PRIOR / 'preregistered.json').read_text())
        done = json.loads((PRIOR / 'exit.json').read_text())
        inventory = json.loads((PRIOR / 'inventory.json').read_text())
        old_tests, old_ids = prior['tests'], inventory['nodeids']
        if (done.get('all_stages_passed') is not True or done.get('semantic_reference_passed') is not True
            or done.get('tests_exit') != 0 or done.get('tests_count') != 3684
            or done.get('source_drift') != [] or done.get('input_drift') != []
            or done.get('provenance_drift') is not False
            or len(old_tests) != 160 or len(set(old_tests)) != 160
            or inventory['count'] != 3684 or len(old_ids) != 3684 or len(set(old_ids)) != 3684
            or sorted(old_ids) != sorted(prior['expected_nodeids'])
            or len(prior['source_sha256']) != 975):
            raise ValueError('qualified complete D003 inheritance required')
        identities.update(prior['source_sha256'])
        for path, digest in ANCHORS.items():
            bind(identities, path, digest)
        if any(name not in identities for name in old_tests):
            raise ValueError('an inherited test file has no frozen source identity')
        for name in NEW_FILES:
            bind(identities, HERE / name, sha(HERE / name))
        executable = str(PYTHON.resolve())
        if Path(sys.executable).resolve() != PYTHON.resolve() or sha(sys.executable) != identities[executable]:
            raise ValueError('inherited Python executable identity differs')
        for name in PRODUCTION:
            if str(ROOT / name) not in identities:
                raise ValueError('missing inherited production identity')
        inputs.update(prior['input_sha256'])
        selected_sources = select_sources(identities, inputs)
        decoder_files = bind_decoder(identities)
        if drift(identities) or drift(inputs):
            raise ValueError('inherited/new source or original input identity drift')
        frozen_provenance = provenance()
        if frozen_provenance != prior['provenance'] or frozen_provenance['branch'] != 'redu-hz':
            raise ValueError('production provenance drift')
        expected_ids, test_paths = list(old_ids), list(old_tests)
        new_counts = {}
        for name in TEST_FILES:
            tree = ast.parse((HERE / name).read_text())
            names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name.startswith('test_')]
            if (not names or len(names) != len(set(names))
                or type(EXPECTED_NEW_TEST_COUNTS[name]) is not int
                or len(names) != EXPECTED_NEW_TEST_COUNTS[name]):
                raise ValueError('explicit new test population differs: ' + name)
            relative = str((HERE / name).relative_to(ROOT))
            expected_ids.extend(relative + '::' + function for function in names)
            test_paths.append(str(HERE / name))
            new_counts[name] = len(names)
        required_tests, required_files = len(expected_ids), 160 + len(TEST_FILES)
        if len(set(expected_ids)) != required_tests:
            raise ValueError('duplicate inherited/new node ID')
        cpu = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpu})
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=frozen_provenance, tests=test_paths, expected_nodeids=expected_ids,
            required_tests=required_tests, required_test_files=required_files,
            inherited_tests=3684, inherited_test_files=160, new_explicit_test_counts=new_counts,
            selected_sources=selected_sources, decoder_dependency_files=decoder_files,
            source_selection='all three distinct models; lexicographically first ORIGINAL spec path per model',
            census_scope='first real-source pilot; direct next-ReLU Conv branches; all channels; corners and center, deduplicated',
            full_400_prevalence_claim=False, actual_cpu_affinity=list(os.sched_getaffinity(0)),
            address_space_bytes=AS_CAP, tests_combined_wall_cap_s=60, diagnostic_wall_cap_s=240,
            whole_work_cap=WHOLE_WORK_CAP, branch_work_cap=BRANCH_WORK_CAP,
            rss_growth_cap_bytes=MEMORY_CAP, traced_peak_plus_metadata_cap_bytes=MEMORY_CAP,
            retained_entries_cap=64_000_000, prior_manifest_bound_not_full_archive_rehashed=True,
            authenticated_scope='all D003 source/input identities, D014 seal, new sources, decoder package files, selected original model/spec bytes',
            decoder_hashing_is_not_protobuf_storage_qualification=True,
            no_archived_HZ_payload_decoding=True, original_model_decode_only=True,
            inherited_tests_include_solver_calls=True, complete_source_qualification=False,
            native_HZ_admitted=False, formal_gain=0))
        print(json.dumps(dict(event='frozen_before_first_project_import', tests=required_tests,
            test_files=required_files, cpu=cpu, scope='component_and_first_source_pilot_only')), flush=True)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short',
                   '-p', 'no:cacheprovider', *test_paths]
        test_start = time.monotonic()
        with (RUN / 'collection.log').open('x') as stream:
            collected = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60, preexec_fn=child_limits)
        ids = [line for line in (RUN / 'collection.log').read_text().splitlines()
               if line.startswith(('experiments/', 'act/')) and '::' in line]
        if (collected.returncode or sorted(ids) != sorted(expected_ids)
            or len(set(ids)) != required_tests
            or len({node.split('::', 1)[0] for node in ids}) != required_files):
            raise ValueError('full inherited/new collection inventory differs')
        save('inventory.json', dict(nodeids=ids, count=len(ids), files=required_files))
        remaining = 60 - (time.monotonic() - test_start)
        if remaining <= 0:
            raise TimeoutError('collection exhausted combined 60 s gate')
        with (RUN / 'tests.log').open('x') as stream:
            tested = subprocess.run([*command, '--junitxml=' + str(RUN / 'tests.xml')],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=remaining, preexec_fn=child_limits)
        record.update(test_wall_s=time.monotonic() - test_start,
                      tests_exit=tested.returncode, tests_count=len(ids))
        test_start = None
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [c.get('classname', '').replace('.', '/') + '.py::' + c.get('name', '') for c in cases]
        if (tested.returncode or sorted(actual) != sorted(expected_ids)
            or record['test_wall_s'] > 60
            or any(c.find(key) is not None for c in cases for key in ('failure', 'error', 'skipped'))):
            raise ValueError('full no-skip/no-failure/60 s test gate failed')
        record['component_tests_passed'] = True
        print(json.dumps(dict(event='all_tests_passed', count=len(ids),
                              wall_s=record['test_wall_s'])), flush=True)
        if drift(identities) or drift(inputs) or provenance() != frozen_provenance:
            raise ValueError('source/input/provenance drift after tests')
        with (RUN / 'diagnostic.log').open('x') as stream:
            worker = subprocess.run([sys.executable, '-B', str(HERE / 'census_worker_v1.py'), '--enabled'],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=240, preexec_fn=child_limits)
        record['diagnostic_exit'] = worker.returncode
        result = json.loads((RUN / 'diagnostic.json').read_text())
        if worker.returncode:
            raise ValueError('bounded source census worker failed')
        diagnostic_gate(result)
        record['source_census_completed'] = True
        record['all_stages_passed'] = True
    except Exception as exc:
        if test_start is not None:
            record['test_wall_s'] = time.monotonic() - test_start
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        record['source_drift'], record['input_drift'] = drift(identities), drift(inputs)
        try:
            record['provenance_drift'] = frozen_provenance is not None and provenance() != frozen_provenance
        except Exception as exc:
            record['provenance_drift'] = True
            record['provenance_check_failure'] = str(exc)
        if record['source_drift'] or record['input_drift'] or record['provenance_drift']:
            record['all_stages_passed'] = False
            record['component_tests_passed'] = False
            record['source_census_completed'] = False
        record['wall_s'] = time.monotonic() - start
        record['artifacts'] = {str(path.relative_to(RUN)): sha(path)
                               for path in RUN.rglob('*') if path.is_file()}
        save('exit.json', record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return 0 if record['all_stages_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
