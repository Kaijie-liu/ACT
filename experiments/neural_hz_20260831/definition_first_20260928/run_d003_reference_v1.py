"""Single-use D003 semantic reference gate; no project import before freezing."""
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent.parent
PRIOR = EXP / 'results/c130_nodewise_branch_20260927_v1'
RUN = EXP / 'results/d003_reference_semantics_20260928_v1'
AS_CAP = 16 * 1024**3
NEW_FILES = ('D003_REFERENCE_DESIGN.md', 'D003_REFERENCE_PREREG.md',
             'd003_domain_v1.py', 'test_d003_domain_v1.py',
             'd003_fixture_worker_v1.py', 'run_d003_reference_v1.py')
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
    PRIOR / 'preregistered.json': '50bc04c4a1c5db36d05b56629ead66f177791f926c077df728261aa1f46ba143',
    PRIOR / 'inventory.json': '8459d7de318c956d50fe33df66dfc7c5aa1e50f612018384500ff947c1b91346',
    PRIOR / 'exit.json': '852f0912305fcee0c13fd4cdb9a1ad32b7e05b37b138c3f38f3b7bbfb596b32b',
    EXP / 'C130_TERMINAL_INTEGRITY_20260927.json':
        'a47e53d30bdd2bbaa4ff31117fa9f9003b2299ce45795e3e36522a42b90bc8d8',
    EXP / 'CHECKPOINT_C130_NODEWISE_BRANCH_20260927_SHA256SUMS':
        'aa1c910c4d27fe1d844772e6f24dadaea9e62a9d999eb3c09a2247f59ff73d0f'}


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


def main():
    RUN.mkdir(exist_ok=False)
    start = time.monotonic()
    record = dict(all_stages_passed=False, semantic_reference_passed=False,
                  complete_source_qualification=False, native_HZ_admitted=False,
                  formal_gain=0, new_benchmark_solves=0,
                  inherited_tests_include_solver_calls=True,
                  d003_diagnostic_solver_calls=0)
    identities, inputs, frozen_provenance = {}, {}, None
    try:
        resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
        if not __debug__ or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
            raise ValueError('assertions required')
        if any(sha(path) != digest for path, digest in ANCHORS.items()):
            raise ValueError('C130 authority anchor drift')
        prior = json.loads((PRIOR / 'preregistered.json').read_text())
        done = json.loads((PRIOR / 'exit.json').read_text())
        if not done['all_stages_passed'] or done['tests_count'] != 3660:
            raise ValueError('unqualified inherited test origin')
        old_tests = prior['tests']
        old_ids = json.loads((PRIOR / 'inventory.json').read_text())['nodeids']
        if len(old_tests) != 159 or len(set(old_tests)) != 159 or len(set(old_ids)) != 3660:
            raise ValueError('inherited test population mismatch')
        identities.update({str(p): h for p, h in ANCHORS.items()})
        for name, digest in prior['source_sha256'].items():
            if name.endswith('.py') or name.endswith('.so'):
                path = str((EXP / name).resolve())
                if path in identities and identities[path] != digest:
                    raise ValueError('conflicting source identity')
                identities[path] = digest
        for name in old_tests:
            if str((EXP / name).resolve()) not in identities:
                raise ValueError('inherited test identity not bound: ' + name)
        for name in NEW_FILES:
            identities[str(HERE / name)] = sha(HERE / name)
        inherited_python = '/data1/Kane/miniconda3/bin/python'
        if (Path(sys.executable).resolve() != Path(inherited_python).resolve()
            or sha(sys.executable) != prior['source_sha256'][inherited_python]):
            raise ValueError('inherited Python executable identity differs')
        identities[str(Path(sys.executable).resolve())] = prior['source_sha256'][inherited_python]
        for name in PRODUCTION:
            identities[str(ROOT / name)] = sha(ROOT / name)
        inputs = dict(prior['input_sha256'])
        if any(sha(p) != h for p, h in identities.items()):
            raise ValueError('inherited executable/source drift')
        if any(sha(p) != h for p, h in inputs.items()):
            raise ValueError('original input identity drift')
        frozen_provenance = provenance()
        if frozen_provenance != prior['provenance']:
            raise ValueError('production provenance drift')
        tree = ast.parse((HERE / 'test_d003_domain_v1.py').read_text())
        names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name.startswith('test_')]
        if len(names) != 24 or len(set(names)) != 24:
            raise ValueError('new explicit test count differs')
        new_path = str((HERE / 'test_d003_domain_v1.py').relative_to(ROOT))
        expected_ids = old_ids + [new_path + '::' + name for name in names]
        test_paths = [str((EXP / name).resolve()) for name in old_tests] + [str(ROOT / new_path)]
        cpu = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpu})
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        save('preregistered.json', dict(source_sha256=identities,
            input_sha256=inputs, provenance=frozen_provenance, tests=test_paths,
            expected_nodeids=expected_ids, required_tests=3684, required_test_files=160,
            actual_cpu_affinity=list(os.sched_getaffinity(0)), address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, diagnostic_wall_cap_s=240,
            rss_growth_cap_bytes=1024**3, traced_peak_plus_metadata_cap_bytes=1024**3,
            retained_entries_cap=64000000, prior_manifest_bound_not_full_archive_rehashed=True,
            authenticated_scope='inherited Python/native dependencies, 3 original byte inputs, production, new sources',
            no_old_payload_decoding=True, inherited_tests_include_solver_calls=True,
            complete_source_qualification=False, native_HZ_admitted=False, formal_gain=0))
        print(json.dumps(dict(event='frozen_before_first_project_import', tests=3684,
                              cpu=cpu, scope='semantic_reference_only')), flush=True)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short',
                   '-p', 'no:cacheprovider', *test_paths]
        clock = time.monotonic()
        with (RUN / 'collection.log').open('x') as stream:
            collected = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60, preexec_fn=child_limits)
        ids = [line for line in (RUN / 'collection.log').read_text().splitlines()
               if line.startswith(('experiments/', 'act/')) and '::' in line]
        if collected.returncode or sorted(ids) != sorted(expected_ids) or len(set(ids)) != 3684:
            raise ValueError('full inherited/new collection inventory differs')
        if len({node.split('::', 1)[0] for node in ids}) != 160:
            raise ValueError('test file population differs')
        save('inventory.json', dict(nodeids=ids, count=len(ids)))
        remaining = 60 - (time.monotonic() - clock)
        if remaining <= 0:
            raise TimeoutError('collection exhausted combined 60s gate')
        with (RUN / 'tests.log').open('x') as stream:
            tested = subprocess.run([*command, '--junitxml=' + str(RUN / 'tests.xml')],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=remaining, preexec_fn=child_limits)
        record.update(test_wall_s=time.monotonic() - clock,
                      tests_exit=tested.returncode, tests_count=len(ids))
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [c.get('classname', '').replace('.', '/') + '.py::' + c.get('name', '') for c in cases]
        if (tested.returncode or sorted(actual) != sorted(expected_ids)
            or record['test_wall_s'] > 60
            or any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))):
            raise ValueError('full no-skip/no-failure/60s test gate failed')
        record['semantic_reference_passed'] = True
        print(json.dumps(dict(event='all_tests_passed', count=len(ids),
                              wall_s=record['test_wall_s'])), flush=True)
        if any(sha(p) != h for p, h in identities.items()):
            raise ValueError('source drift after tests')
        with (RUN / 'diagnostic.log').open('x') as stream:
            worker = subprocess.run([sys.executable, '-B', str(HERE / 'd003_fixture_worker_v1.py')],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=240, preexec_fn=child_limits)
        record['diagnostic_exit'] = worker.returncode
        result = json.loads((RUN / 'diagnostic.json').read_text())
        if worker.returncode or result['prototype_fixture_memory_diagnostic_passed'] is not True:
            raise ValueError('bounded reference fixture diagnostic failed')
        record['all_stages_passed'] = True
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        record['source_drift'] = [p for p, h in identities.items() if not Path(p).is_file() or sha(p) != h]
        record['input_drift'] = [p for p, h in inputs.items() if not Path(p).is_file() or sha(p) != h]
        record['provenance_drift'] = frozen_provenance is not None and provenance() != frozen_provenance
        if record['source_drift'] or record['input_drift'] or record['provenance_drift']:
            record['all_stages_passed'] = False
            record['semantic_reference_passed'] = False
        record['wall_s'] = time.monotonic() - start
        record['artifacts'] = {p.name: sha(p) for p in RUN.iterdir() if p.is_file()}
        save('exit.json', record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return 0 if record['all_stages_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
