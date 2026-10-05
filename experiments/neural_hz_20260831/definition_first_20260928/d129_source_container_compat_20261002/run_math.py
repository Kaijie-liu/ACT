"""Single-use full D129 mathematical and original-source qualification.

All 3913 inherited tests and twenty new tests, then two complete 96-direction
first-CLS source populations. No model CERT, GPU or full replay qualification.
"""
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
RUN = EXP / 'results/d129_source_container_compat_20261002_v1'
PRIOR = EXP / 'results/d120_mixed_consumer_source_20261002_v1'
MATH_PRIOR = EXP / 'results/d128_bounded_attention_source_20261002_v1'
ARCHIVE_PRIOR = EXP / 'results/d090_bound_native_discovery_20261001_v1'
HELPER = HERE.parent / 'd015_batch_binding_20260928_v2/run_v2.py'
GPU = HERE.parent / 'd017_applicability_gpu_20260930/gpu_preflight.py'
FREEZE = HERE / 'freeze.json'
CLOSURE = HERE.parent / 'd090_bound_native_discovery_20261001/project_import_closure.json'
ACT_ROOT = ROOT / 'act'
PLUGIN = ('experiments.neural_hz_20260831.definition_first_20260928.'
          'd129_source_container_compat_20261002.collection_contract')
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
WORKER = HERE / 'source_worker.py'
FILES = ("CONTRACT.md", "PREREG.md", "THEORY.md", "inputs.json", "fast_query.py", "source_binding.py", "source_worker.py", "test_fast_query.py", "test_source_binding.py", "test_source_worker.py", "run_math.py", "collection_contract.py",)
NEW_TESTS = (
    ("test_fast_query.py", ("test_fast_positive_geometry", "test_fast_strong_product_reference", "test_fast_negative_outer_loss", "test_fast_degenerate_shift", "test_fast_bisection_uncertainty", "test_fast_fail_closed", "test_fast_exponential16_certificate", "test_fast_exponential16_fail_closed",)),
    ("test_source_binding.py", ("test_binding_exact_float_payload", "test_binding_layout_and_identity", "test_binding_batchnorm_intervals", "test_binding_shared_patch_templates", "test_binding_all_cls_directions", "test_binding_fail_closed",)),
    ("test_source_worker.py", ("test_bridge_error_rectangle", "test_bridge_original_identity", "test_bridge_signed_composition", "test_bridge_no_root_lower_as_witness", "test_bridge_budget_and_errors", "test_bridge_shared_budget",)),
)
NAMES = tuple(name for _, names in NEW_TESTS for name in names)
INHERITED_SOURCE_ROWS = (320, 640, 640)
ANCHORS = {
    HERE.parent / 'd124_source_phase_fiber_20261002/NATIVE_ATTENTION.md':
        '0e66f1531240ddd9c35638bdc610fc30d45989e7f5d40174d4ff3099351a9360',
    HERE.parent / 'd126_joint_phase_closure_20261002/ARCHIVE_SHA256SUMS':
        '5b94171c79396d5e5f9fa61803f25beeab49ddac8241ced2de5cf5ee75eeeb99',
    MATH_PRIOR / 'preregistered.json':
        'a204a5b5c4ebf991cc3006bf8675c5ca4d348d196ab6a49194830ee6791bec50',
    MATH_PRIOR / 'exit.json':
        'b1415bfe37fbb0bca63577830db5199b16ea51c67546f6b16b8dadf407419f69',
    MATH_PRIOR / 'inventory.json':
        'b941e1555adf32ec87d1a521838b9c92c50568c8e51d3a3b3d3cb6a7dfee814b',
    HERE.parent / 'd125_signed_phase_component_20261002/freeze.json':
        'c83704e08e59b7a315572862ec3797073ed4a903d085ee6a229150424f2f67f4',
    HERE.parent / 'd125_signed_phase_component_20261002/run_math.py':
        '3ee17f05970e0b42a0e8769251e66da161f87ccdfd8f16a8ed0765c41c00ded7',
    HERE.parent / 'd125_signed_phase_component_20261002/collection_contract.py':
        'aa41deb62e9449de2e1aaed440a8a76f620270f8db38ef2571f34708ede18bdd',
    HERE.parent / 'd124_source_phase_fiber_20261002/SIGNED_ENVELOPE.md':
        '1cbcbee430c538e52b3e5eaae59d0334aba505eef07c4c2f0da8eb6272c6a21c',
    HERE.parent / 'd124_source_phase_fiber_20261002/ARCHIVE_SHA256SUMS':
        '6fcf442238e15b998e384601ca3708aeb734d2183881311ace00f89616a1f528',
    HERE.parent / 'd118_shared_curvature_transfer_20261002/THEORY.md':
        '8a7e74f543cd43b1029c4ad48849f8026e897f043a9a4cb23cbde8891362eafb',
    PRIOR / 'preregistered.json': 'f2136f950479ec4bbb2a9368d1783fd2aae43ed2e05aeb390194023b6fef7e7d',
    PRIOR / 'exit.json': 'bb7dccd38dab5119accef6bd89ba3f943c56b26931f82ee74339910e9e1f6b96',
    PRIOR / 'inventory.json': 'adb4f8d24971a9faa26508b53babf3d51ce0bc633a9ba24e9c04363a141d09a4',
    HERE.parent / 'd120_mixed_consumer_source_20261002/freeze.json':
        'f45a4f330b8033de0391a77ad84eabf8c6227f234a30119406f4df0eb877f1c6',
    HERE.parent / 'd120_mixed_consumer_source_20261002/run_math.py':
        '6ee6f6dee4e80b83581c9afdf77a104a2c52cc4f161501646e24eecaebaaaee8',
    HERE.parent / 'd120_mixed_consumer_source_20261002/collection_contract.py':
        'c784eff22a2abe8c1e2251475455383dc26b9f38d105890537ccee5d6d2e2900',
    HERE.parent / 'd090_bound_native_discovery_20261001/freeze.json':
        'fb70bc4f78061b44958141c993abe8b2e8829f2d3e638edeca04960cf4d4a8fd',
    ARCHIVE_PRIOR / 'archive_probe/worker.json':
        'c6c03a2f1f800f7df8173f6b669d0efd8d74ad9078042df8cda4474cc9d16c4f',
    ARCHIVE_PRIOR / 'archive_probe_supervisor/supervisor.json':
        '75c4bc03d6490e6d667a5318d531bffc5d624f0709a9ec62ba66ba08aeb493e8',
    HERE.parent / 'd096_verified_phase_birth_20261001/NEXT_INTEGRATION.md':
        'afe6a37a0d58a024e66dfb8897ee5c16f134a833018eb7817561805a83b2f984',
}
AS_CAP, MEMORY_CAP, RESERVE = 16*1024**3, 1024**3, 65536
CLOSURE_COUNT, CLOSURE_BYTES = 116, 2782307
INHERITED_MODULE = (HERE.parent / 'd112_shared_endpoint_forward_20261002'
                    / 'test_endpoint_forward.py')
INHERITED_RUN = EXP / 'results/d112_shared_endpoint_forward_20261002_v1'
RELOCATED_RUN = RUN / 'inherited_d112_controls'
EVIDENCE_RELOCATION = dict(module_path=str(INHERITED_MODULE),
                           original_run=str(INHERITED_RUN),
                           relocated_run=str(RELOCATED_RUN))


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or linked identity: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path, cap=8*1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= cap:
        raise ValueError('missing, linked or oversized JSON')
    return json.loads(path.read_text())


def merge_identity(identities, path, digest):
    if (type(path) is not str or type(digest) is not str or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('invalid source identity')
    if path in identities and identities[path] != digest:
        raise ValueError('conflicting source identity: ' + path)
    identities[path] = digest


def project_closure(identities, record):
    """Recheck the already inherited D090 closure; admit no additional source.

    Added-file counts in record describe the historical D090 expansion only.
    This version neither recomputes that expansion nor treats it as a new one.
    Authentication reads and hashes files, never imports a project snapshot.
    """
    if (type(record) is not dict or record.get('path') != str(CLOSURE)
            or record.get('sha256') != identities.get(str(CLOSURE))
            or record.get('file_count') != CLOSURE_COUNT
            or record.get('total_bytes') != CLOSURE_BYTES
            or record.get('act_file_count') != CLOSURE_COUNT
            or record.get('act_total_bytes') != CLOSURE_BYTES
            or record.get('outside_act_paths') != []
            or record.get('source_hash_only') is not True
            or sha(CLOSURE) != record['sha256']):
        raise ValueError('inherited closure identity differs')
    closure = read(CLOSURE, RESERVE)
    required = {'schema', 'project_root', 'scope', 'files', 'file_count',
                'total_bytes', 'act_file_count', 'act_total_bytes',
                'outside_act_paths'}
    files = closure.get('files')
    if (set(closure) != required
            or closure.get('schema') != 'd090_project_import_closure_v1'
            or closure.get('project_root') != str(ROOT)
            or closure.get('scope') !=
                'all_existing_act_python_sources_plus_static_external_project_dependencies'
            or type(files) is not dict or list(files) != sorted(files)
            or closure.get('outside_act_paths') != []
            or any(type(closure.get(k)) is not int for k in
                   ('file_count', 'total_bytes', 'act_file_count', 'act_total_bytes'))
            or closure['file_count'] != CLOSURE_COUNT
            or closure['act_file_count'] != CLOSURE_COUNT
            or closure['total_bytes'] != CLOSURE_BYTES
            or closure['act_total_bytes'] != CLOSURE_BYTES
            or len(files) != CLOSURE_COUNT
            or record.get('source_files') != list(files)):
        raise ValueError('inherited project source population differs')
    if ACT_ROOT.is_symlink() or not ACT_ROOT.is_dir():
        raise ValueError('project source root is missing or linked')
    if sorted(str(p) for p in ACT_ROOT.rglob('*.py')) != list(files):
        raise ValueError('current project source population differs')
    total = 0
    for path, item in files.items():
        p = Path(path)
        if (not p.is_absolute() or str(p) != path or p.suffix != '.py'
                or not p.is_relative_to(ACT_ROOT) or p.resolve() != p
                or p.is_symlink() or not p.is_file()
                or type(item) is not dict or set(item) != {'sha256', 'bytes'}
                or type(item['bytes']) is not int
                or not 0 <= item['bytes'] <= CLOSURE_BYTES
                or p.stat().st_size != item['bytes']
                or identities.get(path) != item['sha256']
                or sha(p) != item['sha256']):
            raise ValueError('inherited project source drift: ' + path)
        total += item['bytes']
    if total != CLOSURE_BYTES:
        raise ValueError('project source total bytes differ')
    return record


def save(name, value):
    with (RUN / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))


def host_observations(result, rss0):
    if not tracemalloc.is_tracing():
        raise ValueError('missing host telemetry')
    _, peak = tracemalloc.get_traced_memory()
    metadata = tracemalloc.get_tracemalloc_memory()
    growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0)
    result.update(traced_peak_bytes=peak, tracer_metadata_bytes=metadata,
        rss_highwater_growth_bytes=growth, summary_reserve_bytes=RESERVE,
        host_observations_within_caps=growth+RESERVE <= MEMORY_CAP
            and peak+metadata+RESERVE <= MEMORY_CAP)


def validate_inherited_source_receipt(report, selected, manifest_digest, affinity):
    """Recheck D120 evidence and caps only; never qualify this new candidate."""
    if (type(report) is not dict or report.get('source_census_completed') is not True
            or report.get('source_census_qualified') is not True
            or report.get('memory_gate_passed') is not True or 'failure' in report
            or report.get('manifest_sha256') != manifest_digest
            or report.get('selected_sources') != selected
            or report.get('actual_cpu_affinity') != affinity
            or report.get('address_space_bytes') != AS_CAP
            or report.get('final_summary_reserve_bytes') != RESERVE
            or any(report.get(k) is not False for k in
                   ('native_HZ_admitted', 'gpu_computation_completed',
                    'complete_physical_qualification'))
            or any(type(report.get(k)) is not int or report[k] != 0 for k in
                   ('formal_gain', 'new_benchmark_solves',
                    'diagnostic_solver_calls', 'model_forward_calls'))):
        raise ValueError('complete source qualification receipt differs')
    for key, cap in (('whole_work_used', 256_000_000),
                     ('branch_work_used', 200_000_000),
                     ('evidence_work_used', 40_000_000),
                     ('retained_entries', 64_000_000)):
        if type(report.get(key)) is not int or not 0 <= report[key] <= cap:
            raise ValueError('source resource receipt differs: ' + key)
    for key in ('rss_highwater_growth_bytes', 'traced_peak_bytes', 'tracer_metadata_bytes'):
        if type(report.get(key)) is not int or report[key] < 0:
            raise ValueError('source memory telemetry differs: ' + key)
    if (report['rss_highwater_growth_bytes'] + RESERVE > MEMORY_CAP
            or report['traced_peak_bytes'] + report['tracer_metadata_bytes'] + RESERVE > MEMORY_CAP
            or type(report.get('wall_s')) not in (int, float)
            or not 0 <= report['wall_s'] <= 240):
        raise ValueError('source memory or wall gate failed')
    models = report.get('models')
    if type(models) is not list or len(models) != 3 or len(selected) != 3:
        raise ValueError('complete three-model source population required')
    for index, (model, source, count) in enumerate(zip(models, selected, INHERITED_SOURCE_ROWS)):
        name = 'complete_' + str(index) + '.json'
        if (type(model) is not dict or model.get('model') != source['model_relative_path']
                or model.get('evidence_file') != name
                or type(model.get('summary')) is not dict
                or model['summary'].get('rows') != count
                or model['summary'].get('expected_rows') != count
                or model['summary'].get('canonical_slots') != count * 576
                or model['summary'].get('max_slots') != 576):
            raise ValueError('source model or full row population differs')
        evidence = PRIOR / name
        if (sha(evidence) != model.get('evidence_sha256')
                or type(model.get('evidence_bytes')) is not int
                or evidence.stat().st_size != model['evidence_bytes']):
            raise ValueError('source evidence identity differs')


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    RUN.mkdir(exist_ok=False)
    started, test_started, worker_started = time.monotonic(), None, None
    stage = 'mathematical'
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    identities, inputs = {}, {}
    helper, production, closure_record = None, None, None
    result = dict(schema='d129_source_container_compat_v1',
                  component_tests_passed=False, mathematical_component_gate_passed=False,
                  source_component_qualified=False, worker_launched=False,
                  source_census_completed=False, worker_stage_registered=True,
                  mathematical_stage_only=False,
                  source_census_qualified=False, actual_model_binding_qualified=False,
                  actual_phase_column_binding_verified=False, native_HZ_admitted=False,
                  gpu_computation_completed=False, complete_physical_qualification=False,
                  candidate_physical_gate_evaluated=False,
                  production_snapshot_imported=False,
                  fixed_component_lp_controls_registered=True, solver_rescue_registered=False,
                  new_tests=20, formal_gain=0, new_benchmark_solves=0)
    try:
        limits()
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        sys.dont_write_bytecode = True
        tracemalloc.start()
        if not __debug__ or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
            raise ValueError('assertions required')
        (RUN / 'tmp').mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
            CUDA_LOG_FILE='stderr', CUDA_CACHE_PATH=str(RUN / 'cuda_cache'),
            TORCH_HOME=str(RUN / 'torch_home'), XDG_CACHE_HOME=str(RUN / 'xdg_cache'),
            TRITON_CACHE_DIR=str(RUN / 'triton_cache'),
            TORCHINDUCTOR_CACHE_DIR=str(RUN / 'inductor_cache'), TMPDIR=str(RUN / 'tmp'))
        os.environ.update(env)
        sys.path.insert(0, str(ROOT))
        frozen = read(FREEZE, RESERVE)
        sources = frozen.get('source_sha256')
        if (frozen.get('schema') != 'd129_source_container_compat_v1'
                or frozen.get('worker_stage_registered') is not True
                or frozen.get('mathematical_stage_only') is not False
                or frozen.get('fixed_component_lp_controls_registered') is not True
                or frozen.get('solver_rescue_registered') is not False
                or frozen.get('inherited_component_evidence_relocation') != EVIDENCE_RELOCATION
                or frozen.get('required_tests') != 3933 or frozen.get('required_test_files') != 204
                or frozen.get('new_test_names') != list(NAMES)
                or type(sources) is not dict or set(sources) != {str(HERE / n) for n in FILES}):
            raise ValueError('frozen source and test contract differs')
        for path, digest in sources.items():
            merge_identity(identities, path, digest)
        for path, digest in ANCHORS.items():
            merge_identity(identities, str(path), digest)
        merge_identity(identities, str(FREEZE), sha(FREEZE))
        for path, digest in identities.items():
            if sha(path) != digest:
                raise ValueError('frozen source or historical anchor differs: ' + path)
        prior, done, inventory = (read(PRIOR / n) for n in
                                  ('preregistered.json', 'exit.json', 'inventory.json'))
        if (any(done.get(k) is not True for k in ('component_tests_passed',
                'mathematical_component_gate_passed',
                'all_stages_passed', 'host_observations_within_caps',
                'inventory_validated_before_execution'))
                or any(done.get(k) is not True for k in
                       ('source_component_qualified', 'source_census_qualified',
                        'worker_launched'))
                or done.get('worker_exit') != 0
                or not 0 <= done.get('worker_wall_s', 241) <= 240
                or done.get('supervisor_exit') != 0 or done.get('tests_exit') != 0
                or done.get('tests_count') != 3869 or done.get('test_files') != 194
                or not 0 <= done.get('test_wall_s', 61) <= 60
                or done.get('source_drift') != [] or done.get('input_drift') != []
                or done.get('provenance_drift') is not False or 'failure' in done
                or prior.get('required_tests') != 3869 or prior.get('required_test_files') != 194
                or inventory.get('nodeids') != prior.get('expected_nodeids')
                or inventory.get('count') != 3869 or inventory.get('files') != 194):
            raise ValueError('complete D120 success is not established')
        historical = prior.get('historical_contracts')
        if (type(historical) is not dict or set(historical) != {
                'prior_attempt', 'direct_prior_attempt', 'inherited_mathematical_gate',
                'inherited_native_predicate_gate', 'same_candidate_prior_attempt'}):
            raise ValueError('complete D120 historical contracts are missing')
        for path, digest in prior['source_sha256'].items():
            merge_identity(identities, path, digest)
        inputs.update(prior['input_sha256'])
        for name, digest in done['artifacts'].items():
            path = (PRIOR / name).resolve()
            if not path.is_relative_to(PRIOR.resolve()):
                raise ValueError('historical artifact escapes its directory')
            merge_identity(identities, str(path), digest)
        # D120 remains the source receipt, while the complete latest D125
        # mathematical population is inherited without rerunning its worker.
        math_prior, math_done, math_inventory = (read(MATH_PRIOR / n) for n in
            ('preregistered.json', 'exit.json', 'inventory.json'))
        if (any(math_done.get(k) is not True for k in
                ('component_tests_passed', 'mathematical_component_gate_passed',
                 'host_observations_within_caps', 'inventory_validated_before_execution'))
                or math_done.get('worker_launched') is not True
                or math_done.get('supervisor_exit') != 1
                or math_done.get('all_stages_passed') is not False
                or math_done.get('source_census_qualified') is not False
                or math_done.get('source_diagnostic_failure') != dict(type='AttributeError', reason="'google._upb._message.RepeatedScalarContainer' object has no attribute 'count'")
                or math_done.get('tests_exit') != 0
                or math_done.get('tests_count') != 3913
                or math_done.get('test_files') != 201
                or not 0 <= math_done.get('test_wall_s', 61) <= 60
                or math_done.get('source_drift') != []
                or math_done.get('input_drift') != []
                or math_done.get('provenance_drift') is not False
                or math_done.get('failure') != dict(type='ValueError', stage='source', reason='complete source population gate failed')
                or math_prior.get('required_tests') != 3913
                or math_prior.get('required_test_files') != 201
                or math_inventory.get('count') != 3913
                or math_inventory.get('files') != 201
                or math_inventory.get('nodeids') != math_prior.get('expected_nodeids')
                or math_prior.get('tests', [])[:194] != prior['tests']
                or math_prior.get('expected_nodeids', [])[:3869] != prior['expected_nodeids']
                or len(math_prior.get('input_sha256', {})) != 14
                or any(math_prior['input_sha256'].get(p) != d for p,d in prior['input_sha256'].items())
                or math_prior.get('provenance') != prior['provenance']
                or math_prior.get('project_import_closure') != prior['project_import_closure']
                or math_prior.get('historical_contracts') != historical
                or math_prior.get('selected_sources') != prior['selected_sources']
                or math_prior.get('gpu_dependency_files') != prior['gpu_dependency_files']
                or math_prior.get('decoder_dependency_files') != prior['decoder_dependency_files']):
            raise ValueError('complete D128 mathematical success and preserved source failure not established')
        for path, digest in math_prior['source_sha256'].items():
            merge_identity(identities, path, digest)
        for name, digest in math_done['artifacts'].items():
            path = (MATH_PRIOR / name).resolve()
            if not path.is_relative_to(MATH_PRIOR.resolve()):
                raise ValueError('historical mathematical artifact escapes')
            merge_identity(identities, str(path), digest)
        closure_record = project_closure(identities, prior.get('project_import_closure'))
        failed_archive = prior.get('same_candidate_archive_prior_attempt')
        if (type(failed_archive) is not dict
                or failed_archive.get('archive_probe_qualified') is not False
                or failed_archive.get('archive_loaded') is not False
                or failed_archive.get('path') != str(EXP /
                    'results/d088_native_structure_discovery_20261001_v1/archive_probe')):
            raise ValueError('inherited D088 archive failure record differs')
        old_worker = read(ARCHIVE_PRIOR / 'archive_probe/worker.json')
        old_supervisor = read(ARCHIVE_PRIOR / 'archive_probe_supervisor/supervisor.json')
        if (old_worker.get('archive_probe_qualified') is not False
                or old_worker.get('archive_loaded') is not True
                or old_worker.get('archive_authentication_completed') is not True
                or old_worker.get('archive_census_completed') is not False
                or old_worker.get('memory_gate_passed') is not False
                or old_worker.get('failure', {}).get('type') != 'MemoryError'
                or old_supervisor.get('worker_exit') != 1
                or old_supervisor.get('archive_probe_qualified') is not False
                or old_supervisor.get('worker_receipt', {}).get('sha256') !=
                    ANCHORS[ARCHIVE_PRIOR / 'archive_probe/worker.json']):
            raise ValueError('D090 archive failure identity differs')
        direct_archive = dict(path=str(ARCHIVE_PRIOR / 'archive_probe'),
            worker_sha256=ANCHORS[ARCHIVE_PRIOR / 'archive_probe/worker.json'],
            supervisor_sha256=ANCHORS[ARCHIVE_PRIOR / 'archive_probe_supervisor/supervisor.json'],
            archive_probe_qualified=False, archive_loaded=True,
            archive_census_completed=False, memory_gate_passed=False,
            failure=old_worker['failure'])
        if prior.get('direct_archive_prior_attempt') != direct_archive:
            raise ValueError('D120 inherited direct D090 failure record differs')
        # Preserve the complete prior value, not a replacement success claim.
        direct_archive = prior['direct_archive_prior_attempt']
        if len(inputs) != 9 or len(prior['gpu_dependency_files']) != 4417 or len(prior['decoder_dependency_files']) != 1011:
            raise ValueError('dependency or input population differs')
        for path, digest in {**identities, **inputs}.items():
            if sha(path) != digest:
                raise ValueError('source/input drift: ' + path)
        if Path(sys.executable).resolve() != PYTHON.resolve() or sha(Path(sys.executable).resolve()) != identities[str(PYTHON.resolve())]:
            raise ValueError('interpreter differs')
        if (prior.get('worker_stage_registered') is not True
                or prior.get('source_expected_rows') != list(INHERITED_SOURCE_ROWS)
                or prior.get('source_expected_models') != 3
                or prior.get('source_expected_consumers') != 1600
                or prior.get('source_canonical_slots_per_consumer') != 576
                or type(prior.get('cpu_affinity')) is not list
                or len(prior['cpu_affinity']) != 1):
            raise ValueError('inherited D120 source contract differs')
        old_source = read(PRIOR / 'diagnostic.json', RESERVE)
        old_receipt = dict(path=str(PRIOR / 'diagnostic.json'),
                           sha256=sha(PRIOR / 'diagnostic.json'))
        if done.get('source_worker_receipt') != old_receipt:
            raise ValueError('inherited D120 source receipt differs')
        validate_inherited_source_receipt(old_source, prior['selected_sources'],
            sha(PRIOR / 'preregistered.json'), prior['cpu_affinity'])
        result['inherited_D128_source_failure_preserved'] = math_done['source_diagnostic_failure']
        inherited_source = dict(path=str(PRIOR), exit_sha256=sha(PRIOR / 'exit.json'),
            source_component_qualified=True, source_census_qualified=True,
            source_worker_receipt=old_receipt, qualification_report=old_source,
            resource_contract={key: prior[key] for key in
                ('whole_work_cap', 'branch_work_cap', 'evidence_prepaid_work',
                 'retained_entry_cap', 'rational_bit_cap', 'worker_wall_cap_s',
                 'host_memory_cap_bytes', 'summary_reserve_bytes',
                 'address_space_bytes')},
            reexecuted=False, qualification_transferred_to_candidate=False)
        result['inherited_source_qualification_authenticated'] = True
        helper = load(HELPER, 'd127_readonly_source_helpers')
        gpu = load(GPU, 'd127_readonly_dependency_helpers')
        if (helper.select_sources(identities, inputs) != prior['selected_sources']
                or helper.bind_decoder(identities) != prior['decoder_dependency_files']
                or gpu.gpu_dependencies(helper, identities) != prior['gpu_dependency_files']):
            raise ValueError('dependency inventory drift')
        production = helper.provenance()
        if production != prior['provenance'] or production['branch'] != 'redu-hz':
            raise ValueError('production provenance differs')
        vit_population = read(HERE / 'inputs.json', RESERVE)
        for source in vit_population['selected']:
            for key in ('model', 'spec'):
                merge_identity(inputs, source[key+'_path'], source[key+'_sha256'])
            merge_identity(identities, source['graph_path'], source['graph_sha256'])
        merge_identity(inputs, vit_population['instances_path'], vit_population['instances_sha256'])
        if len(inputs) != 14:
            raise ValueError('complete original input population differs')
        for path, digest in {**identities, **inputs}.items():
            if sha(path) != digest:
                raise ValueError('new input or source identity differs')
        tests = list(math_prior['tests'])
        expected = list(math_prior['expected_nodeids'])
        for filename, names in NEW_TESTS:
            test_path = HERE / filename
            tree = ast.parse(test_path.read_text())
            functions = [node for node in tree.body
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and node.name.startswith('test_')]
            if (len(names) not in (6, 8) or len(functions) != len(names)
                    or [node.name for node in functions] != list(names)
                    or any(not isinstance(node, ast.FunctionDef) or node.decorator_list
                           or node.args.args or node.args.posonlyargs or node.args.kwonlyargs
                           or node.args.vararg or node.args.kwarg for node in functions)):
                raise ValueError('registered plain tests required in each new test file')
            tests.append(str(test_path))
            relative = str(test_path.relative_to(ROOT))
            expected.extend(relative + '::' + name for name in names)
        if (len(tests) != 204 or len(set(tests)) != 204
                or len(expected) != 3933 or len(set(expected)) != 3933
                or tests[:201] != math_prior['tests']
                or expected[:3913] != math_prior['expected_nodeids']):
            raise ValueError('complete inherited population differs')
        if any(path not in identities for path in tests) or helper.drift(identities) or helper.drift(inputs):
            raise ValueError('pre-execution identity drift')
        save('preregistered.json', dict(schema='d129_source_container_compat_v1',
            source_sha256=identities, input_sha256=inputs,
            provenance=production, tests=tests, expected_nodeids=expected,
            required_tests=3933, required_test_files=204, new_test_names=NAMES,
            inherited_tests=3913, inherited_test_files=201,
            inherited_success=dict(path=str(MATH_PRIOR), exit_sha256=sha(MATH_PRIOR / 'exit.json')),
            inherited_source_qualification=inherited_source,
            inherited_predecessor_success=math_prior['inherited_success'],
            inherited_failed_source=dict(path=str(MATH_PRIOR), exit_sha256=sha(MATH_PRIOR/'exit.json'), failure=math_done['source_diagnostic_failure'], qualification_transferred=False),
            inherited_predecessor_chain=math_prior.get('inherited_predecessor_success'),
            historical_contracts=historical,
            same_candidate_archive_prior_attempt=failed_archive,
            direct_archive_prior_attempt=direct_archive,
            project_import_closure=closure_record,
            inherited_test_population_unchanged=True, new_test_files=3,
            inherited_component_evidence_relocation=EVIDENCE_RELOCATION,
            selected_sources=prior['selected_sources'],
            gpu_dependency_files=prior['gpu_dependency_files'],
            decoder_dependency_files=prior['decoder_dependency_files'],
            freeze_sha256=sha(FREEZE), same_process_collection_gate=True,
            single_pytest_process=True, collection_plugin=PLUGIN,
            cpu_affinity=list(os.sched_getaffinity(0)), address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, host_memory_cap_bytes=MEMORY_CAP,
            summary_reserve_bytes=RESERVE, mathematical_stage_only=False,
            whole_work_cap=256_000_000, branch_work_cap=200_000_000,
            evidence_prepaid_work=40_000_000, retained_entry_cap=64_000_000,
            rational_bit_cap=512, worker_wall_cap_s=240,
            vit_source_population=vit_population, source_worker=str(WORKER),
            source_expected_rows=[96,96], source_expected_roots=1152,
            worker_stage_registered=True, single_archive_probe_registered=False,
            inherited_source_population_unchanged=True,
            source_component_qualified=False, source_census_qualified=False,
            source_census_completed=False, actual_model_binding_qualified=False,
            actual_phase_column_binding_verified=False, native_HZ_admitted=False,
            gpu_computation_completed=False, new_benchmark_solves=0,
            fixed_component_lp_controls_registered=True, solver_rescue_registered=False,
            cuda_visible_devices='',
            complete_physical_qualification=False, formal_gain=0))
        manifest_digest = sha(RUN / 'preregistered.json')
        env['NEURAL_HZ_D129_MANIFEST_SHA256'] = manifest_digest
        env['NEURAL_HZ_ACTIVE_COMPONENT_RUN'] = str(RUN)
        print(json.dumps(dict(event='frozen_before_candidate_import', tests=3933, files=204)), flush=True)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider',
                   '-p', PLUGIN, '--junitxml=' + str(RUN / 'tests.xml'), *tests]
        with (RUN / 'tests.log').open('x') as stream:
            test_started = time.monotonic()
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        result.update(tests_exit=process.returncode, test_wall_s=time.monotonic()-test_started,
                      tests_count=3933, test_files=204)
        test_started = None
        checked = read(RUN / 'inventory.json')
        if (checked.get('nodeids') != expected or checked.get('count') != 3933
                or checked.get('files') != 204 or checked.get('manifest_sha256') != manifest_digest
                or checked.get('validated_before_execution') is not True
                or checked.get('inherited_component_evidence_relocation') != EVIDENCE_RELOCATION
                or sha(RUN / 'preregistered.json') != manifest_digest):
            raise ValueError('pre-execution inventory contract differs')
        result['inventory_validated_before_execution'] = True
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [c.get('classname', '').replace('.', '/')+'.py::'+c.get('name', '') for c in cases]
        if (process.returncode != 0 or result['test_wall_s'] > 60 or sorted(actual) != sorted(expected)
                or any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))):
            raise ValueError('complete component test gate failed')
        result['component_tests_passed'] = True
        # Full inherited post-test checks remain mandatory.  This stage has
        # no source worker; the authenticated D120 source result stays historical.
        if (project_closure(identities, closure_record) != closure_record
                or helper.drift(identities) or helper.drift(inputs)
                or helper.provenance() != production
                or sha(RUN / 'preregistered.json') != manifest_digest):
            raise ValueError('post-mathematical identity drift')
        host_observations(result, rss0)
        if not result['host_observations_within_caps']:
            raise MemoryError('supervisor memory gate failed')
        result['mathematical_component_gate_passed'] = True
        stage = 'source'
        env['NEURAL_HZ_SOURCE_MANIFEST'] = str(RUN / 'preregistered.json')
        env['NEURAL_HZ_SOURCE_MANIFEST_SHA256'] = manifest_digest
        print(json.dumps(dict(event='source_worker_start', expected_rows=192, expected_roots=1152)), flush=True)
        with (RUN / 'worker.log').open('x') as stream:
            worker_started = time.monotonic()
            result['worker_launched'] = True
            worker = subprocess.run([sys.executable, '-B', str(WORKER), '--enabled'],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=240, preexec_fn=limits)
        result.update(worker_exit=worker.returncode, worker_wall_s=time.monotonic()-worker_started)
        worker_started = None
        report = read(RUN / 'diagnostic.json', RESERVE)
        result['source_worker_receipt'] = dict(path=str(RUN / 'diagnostic.json'), sha256=sha(RUN / 'diagnostic.json'))
        if 'failure' in report:
            result['source_diagnostic_failure'] = report['failure']
        if (worker.returncode != 0 or result['worker_wall_s'] > 240
                or report.get('source_census_completed') is not True
                or report.get('source_census_qualified') is not True
                or report.get('memory_gate_passed') is not True
                or report.get('manifest_sha256') != manifest_digest
                or report.get('selected_sources') != vit_population['selected']
                or report.get('rows') != 192 or report.get('roots') != 1152
                or len(report.get('models', [])) != 2 or 'failure' in report):
            raise ValueError('complete source population gate failed')
        for key, cap in (('whole_work_used',256_000_000),('branch_work_used',200_000_000),
                         ('evidence_work_used',40_000_000),('retained_entries',64_000_000)):
            if type(report.get(key)) is not int or not 0 <= report[key] <= cap:
                raise ValueError('source resource gate: '+key)
        if (report.get('actual_cpu_affinity') != list(os.sched_getaffinity(0))
                or report.get('address_space_bytes') != AS_CAP
                or not 0 <= report.get('wall_s',241) <= 240
                or report['rss_highwater_growth_bytes']+RESERVE > MEMORY_CAP
                or report['traced_peak_bytes']+report['tracer_metadata_bytes']+RESERVE > MEMORY_CAP
                or any(report.get(k) is not False for k in ('native_HZ_admitted',
                    'actual_model_binding_qualified','actual_phase_column_binding_verified',
                    'gpu_computation_completed','complete_physical_qualification'))
                or any(type(report.get(k)) is not int or report[k] != 0 for k in
                       ('formal_gain','new_benchmark_solves','diagnostic_solver_calls','model_forward_calls'))):
            raise ValueError('source scope or observation gate')
        for index, model in enumerate(report['models']):
            if model['model'] != vit_population['selected'][index]['model'] or model['summary']['rows'] != 96 or model['summary']['roots'] != 576:
                raise ValueError('source per-model population')
            name = 'complete_'+str(index)+'.json'
            if model['evidence_file'] != name or sha(RUN/name) != model['evidence_sha256'] or (RUN/name).stat().st_size != model['evidence_bytes']:
                raise ValueError('source evidence identity')
            evidence = read(RUN/name)
            if len(evidence['rows']) != 96:
                raise ValueError('source full row receipts')
            for j, row in enumerate(evidence['rows']):
                if row['path'] != 'row_'+str(index)+'_'+str(j)+'.json' or sha(RUN/row['path']) != row['sha256'] or (RUN/row['path']).stat().st_size != row['bytes']:
                    raise ValueError('source row identity')
        result['source_census_completed'] = True
        result['source_census_qualified'] = True
        result['source_component_qualified'] = True
    except BaseException as exc:
        if worker_started is not None:
            result['worker_wall_s'] = time.monotonic()-worker_started
        if test_started is not None:
            result['test_wall_s'] = time.monotonic()-test_started
        result['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096],
                                 stage=stage)
    finally:
        try:
            if closure_record is not None and project_closure(identities, closure_record) != closure_record:
                raise ValueError('post-execution project closure differs')
            result['source_drift'] = [p for p, d in identities.items() if sha(p) != d]
            result['input_drift'] = [p for p, d in inputs.items() if sha(p) != d]
            result['provenance_drift'] = production is not None and helper.provenance() != production
            if result['source_drift'] or result['input_drift'] or result['provenance_drift']:
                raise ValueError('post-execution identity drift')
        except BaseException as exc:
            result['mathematical_component_gate_passed'] = False
            result['source_census_qualified'] = False
            result['source_component_qualified'] = False
            result.setdefault('failure', dict(type=type(exc).__name__, reason=str(exc)[:4096]))
        try:
            result['artifacts'] = {str(p.relative_to(RUN)): sha(p) for p in RUN.rglob('*') if p.is_file()}
        except BaseException as exc:
            result['artifact_sealing_failure'] = str(exc)[:4096]
            result['mathematical_component_gate_passed'] = False
            result['source_census_qualified'] = False
            result['source_component_qualified'] = False
            result.setdefault('failure', dict(type=type(exc).__name__, reason='artifact sealing incomplete'))
        try:
            host_observations(result, rss0)
        except BaseException as exc:
            result['host_observations_within_caps'] = False
            result.setdefault('failure', dict(type=type(exc).__name__, reason=str(exc)[:4096]))
        if not result['host_observations_within_caps']:
            result['mathematical_component_gate_passed'] = False
            result['source_census_qualified'] = False
            result['source_component_qualified'] = False
            result.setdefault('failure', dict(type='MemoryError', reason='supervisor memory gate failed'))
        passed = (result['mathematical_component_gate_passed']
                  and result['source_census_qualified'] and 'failure' not in result)
        result.update(all_stages_passed=passed, supervisor_exit=0 if passed else 1,
                      wall_s=time.monotonic()-started,
                      memory_scope='supervisor observations and pytest AS/CPU/time only; D120 historical; current source worker observed separately; no full physical qualification')
        save('exit.json', result)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
