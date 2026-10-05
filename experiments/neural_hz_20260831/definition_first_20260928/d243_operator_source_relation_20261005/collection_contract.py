"""Authenticate the complete D243 population and relocate evidence writers only."""
import hashlib
import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d243_operator_source_relation_20261005_v1'
MANIFEST = RUN / 'preregistered.json'
INVENTORY = RUN / 'inventory.json'
NEW_RECORD_FILES = ('summary.json',)
D112_MODULE = HERE.parent / 'd112_shared_endpoint_forward_20261002/test_endpoint_forward.py'
D112_ORIGINAL = EXP / 'results/d112_shared_endpoint_forward_20261002_v1'
D112_DESTINATION = RUN / 'inherited_d112_controls'
EVIDENCE_RELOCATION = dict(module_path=str(D112_MODULE), original_run=str(D112_ORIGINAL),
                           relocated_run=str(D112_DESTINATION))
D228_MODULE = HERE.parent / 'd228_joint_bank_component_20261005/test_bank.py'
D228_ORIGINAL = EXP / 'results/d228_joint_bank_component_20261005_v1'
D228_DESTINATION = RUN / 'inherited_d228_controls'
D228_FUNCTION = 'test_16_asymmetric_permutations_and_summary'
D228_SOURCE_SHA256 = '12b66890591af9bebfbb44c4d0e6659ba33006141b5e02c38b0092c9920c7f17'
D228_RELOCATION = dict(module_path=str(D228_MODULE), source_sha256=D228_SOURCE_SHA256,
    function_name=D228_FUNCTION, function_firstlineno=571,
    original_run=str(D228_ORIGINAL), relocated_run=str(D228_DESTINATION),
    allowed_filenames=['summary.json'], mechanism='module_local_exact_Path_open_only',
    active_component_run_during_call=str(D228_ORIGINAL), original_function_code_unchanged=True)
# Population, original first line and writer are immutable; only the destination changes.
RECORDERS = (
    ('D207', 'd207_owned_source_phase_20261005', 'test_fiber.py', 24, 44,
     '1c23e38befc50be621e7dacf863f20bddc9ff107b4ef2c3bc3742ee8d6627e02',
     ('source_phase_positive_control.json', 'exact_alias_control.json', 'complete_200_gate_control.json')),
    ('D208', 'd208_multilayer_closure_audit_20261005', 'test_closure.py', 4, 39,
     '4825d15388db02c02ab4918a915baa0501c41849e784b5556a10a949043007e8',
     ('closure_counterexample.json', 'direction_family.json')),
    ('D209', 'd209_owned_slack_closure_20261005', 'test_fiber.py', 16, 35,
     'f150be4bb9e9fdcf4fe4117038ceacebf9410d39085ab86022dacaca0f72af77',
     ('closure_repair.json', 'positive_slack_control.json', 'complete_200_gate_control.json')),
    ('D214', 'd214_parametric_relation_consumption_20261005', 'test_fiber.py', 16, 24,
     '50513b38766fb8d7c7a161b485f9c421a62c7a66f60b5f789dd302d1ba484070',
     ('phase_budget_separation.json', 'multilayer_closure.json', 'complete_bank_cost.json',
      'parametric_family.json')),
    ('D229', 'd229_source_bound_bank_20261005', 'test_binding.py', 8, 48,
     '057bb55396f138fedd83466468e0c56254e9a933c02b91249411893c1b169806',
     ('summary.json',)),
    ('D230', 'd230_source_coupling_counterexample_20261005', 'test_diagnostic.py', 1, 19,
     '5854df7f815262d45be67d9fc1c9e0806c30e8bb92e052ca58ebb341d39215c9',
     ('summary.json',)),
    ('D231', 'd231_phase_compatible_birth_20261005', 'test_birth.py', 12, 52,
     '96240515547221ae4fdd4bb4d2e38faa105a7f66cc66509e9c016b9241ee4dcb',
     ('summary.json',)),
    ('D240', 'd240_phase_supported_component_20261005', 'test_factor.py', 16, 44,
     '0c22812c9123708b291d32b71e3dc761bb75e0aaa46fcaff8df2eeb1d8306829',
     ('summary.json',)),
)
TRUE_FLAGS = ('mathematical_stage_only', 'same_process_collection_gate',
    'single_pytest_process', 'fixed_component_lp_controls_registered',
    'domain_definition_changed', 'new_component_solver_free', 'inherited_test_population_unchanged')
FALSE_FLAGS = ('worker_stage_registered', 'worker_launched', 'source_component_qualified',
    'source_census_completed', 'source_census_qualified', 'actual_model_binding_qualified',
    'actual_phase_column_binding_verified', 'native_HZ_admitted', 'gpu_computation_completed',
    'complete_physical_qualification', 'candidate_physical_gate_evaluated',
    'production_snapshot_imported', 'solver_rescue_registered', 'negative_audit_only',
    'new_set_class', 'new_domain_qualified', 'new_capability_qualified',
    'capability_improvement_claimed', 'pytest_plugin_autoload')
_verified = None
_bindings = []
_d112_binding = None
_d228_binding = None
_d228_call_active = False
_d228_call_finished = False
_d228_redirects = 0


def _reject(reason):
    raise pytest.UsageError('D243 complete population: ' + reason)


def _relocation(record):
    label, directory, filename, count, line, digest, files = record
    return dict(module_path=str(HERE.parent / directory / filename), source_sha256=digest,
        function_name=('_record_file' if label in ('D229', 'D230', 'D231', 'D240') else '_record'), function_firstlineno=line,
        original_run=str(EXP / 'results' / (directory + '_v1')),
        relocated_run=str(RUN / ('inherited_' + label.lower() + '_controls')),
        allowed_filenames=list(files), mechanism='module_local_record_function_only')


def _writer(record):
    label, _, _, _, _, _, files = record
    destination = RUN / ('inherited_' + label.lower() + '_controls')

    def write(name, value):
        if (name not in files or Path(os.environ.get('NEURAL_HZ_ACTIVE_COMPONENT_RUN', '')) != RUN
                or not RUN.is_dir() or RUN.is_symlink()
                or not destination.is_dir() or destination.is_symlink()):
            _reject(label + ' evidence name or current exclusive destination differs')
        with (destination / name).open('x') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    return write


def _population(session):
    if session.testsfailed or session.shouldstop or session.shouldfail:
        _reject('collection error or stop')
    ids = tuple(item.nodeid for item in session.items)
    paths = tuple(str(Path(item.path).absolute()) for item in session.items)
    if len(ids) != 4189 or len(set(ids)) != 4189 or len(set(paths)) != 223:
        _reject('4189 items and 223 original paths required')
    return ids, paths


def _check_bindings():
    if (_d228_binding is None or _d228_call_active
            or _d228_binding[0].Path is not _d228_binding[3]
            or getattr(_d228_binding[0], D228_FUNCTION) is not _d228_binding[1]
            or _d228_binding[1].__code__ is not _d228_binding[2]
            or os.environ.get('NEURAL_HZ_ACTIVE_COMPONENT_RUN') != str(RUN)):
        _reject('D228 function, Path, or active environment was not restored')
    if (_d112_binding is None or _d112_binding.RUN != D112_DESTINATION
            or len(_bindings) != len(RECORDERS)
            or any(getattr(module, original.__name__, None) is not writer
                   for module, original, writer in _bindings)):
        _reject('evidence-only binding changed')


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    global _verified, _d112_binding, _d228_binding
    if _verified is not None:
        _reject('repeated collection')
    digest = os.environ.get('NEURAL_HZ_D243_MANIFEST_SHA256')
    if not isinstance(digest, str) or len(digest) != 64:
        _reject('missing parent identity')
    if MANIFEST.is_symlink() or not MANIFEST.is_file() or not 0 < MANIFEST.stat().st_size <= 8 * 1024**2:
        _reject('invalid manifest')
    raw = MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        _reject('changed manifest')
    manifest = json.loads(raw)
    expected, files = manifest.get('expected_nodeids'), manifest.get('tests')
    relocations = {'inherited_component_evidence_relocation': EVIDENCE_RELOCATION,
                   'inherited_D228_evidence_relocation': D228_RELOCATION}
    relocations.update({'inherited_' + record[0] + '_evidence_relocation': _relocation(record)
                        for record in RECORDERS})
    if (manifest.get('schema') != 'd243_operator_source_relation_v1'
            or manifest.get('required_tests') != 4189 or manifest.get('required_test_files') != 223
            or manifest.get('inherited_tests') != 4169 or manifest.get('inherited_test_files') != 222
            or manifest.get('new_test_files') != 1
            or type(manifest.get('new_test_names')) is not list or len(manifest['new_test_names']) != 20
            or manifest.get('new_evidence_files') != list(NEW_RECORD_FILES)
            or manifest.get('pytest_import_mode') != 'importlib'
            or os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') != '1'
            or any(manifest.get(key) is not True for key in TRUE_FLAGS)
            or any(manifest.get(key) is not False for key in FALSE_FLAGS)
            or manifest.get('failed_D213_reference', {}).get('mathematical_component_gate_passed') is not False
            or manifest.get('failed_D213_reference', {}).get('population_inherited') is not False
            or manifest.get('failed_D213_reference', {}).get('qualification_transferred') is not False
            or manifest.get('formal_gain') != 0 or manifest.get('independent_e0_gain') != 0
            or manifest.get('new_benchmark_solves') != 0
            or any(manifest.get(key) != value for key, value in relocations.items())
            or type(expected) is not list or len(expected) != 4189 or len(set(expected)) != 4189
            or type(files) is not list or len(files) != 223 or len(set(files)) != 223):
        _reject('wrong contract')
    ids, paths = _population(session)
    if ids != tuple(expected) or set(paths) != set(files):
        _reject('population drift')
    if any(str(ROOT / node.split('::', 1)[0]) != path for node, path in zip(ids, paths)):
        _reject('node identity differs from actual source')
    d228_items = [item for item, path in zip(session.items, paths) if path == str(D228_MODULE)]
    if len(d228_items) != 16:
        _reject('missing complete inherited D228 module')
    d228_module = d228_items[0].module
    d228_function = getattr(d228_module, D228_FUNCTION, None)
    if (any(item.module is not d228_module for item in d228_items)
            or getattr(d228_module, '__file__', None) != str(D228_MODULE)
            or getattr(d228_module, 'Path', None) is not Path
            or D228_MODULE.is_symlink() or not D228_MODULE.is_file()
            or hashlib.sha256(D228_MODULE.read_bytes()).hexdigest() != D228_SOURCE_SHA256
            or manifest['source_sha256'].get(str(D228_MODULE)) != D228_SOURCE_SHA256
            or type(d228_function) is not type(_writer)
            or d228_function.__name__ != D228_FUNCTION
            or d228_function.__module__ != d228_module.__name__
            or d228_function.__globals__ is not vars(d228_module)
            or d228_function.__code__.co_filename != str(D228_MODULE)
            or d228_function.__code__.co_firstlineno != 571
            or sum(item.obj is d228_function for item in d228_items) != 1):
        _reject('D228 original function or module-local Path identity differs')
    D228_DESTINATION.mkdir(exist_ok=False)
    if D228_DESTINATION.is_symlink() or D228_DESTINATION.resolve() != D228_DESTINATION:
        _reject('D228 exclusive evidence destination differs')
    _d228_binding = (d228_module, d228_function, d228_function.__code__, Path)
    items = [item for item, path in zip(session.items, paths) if path == str(D112_MODULE)]
    if len(items) != 4:
        _reject('missing complete inherited D112 module')
    module = items[0].module
    if (any(item.module is not module for item in items)
            or getattr(module, '__file__', None) != str(D112_MODULE)
            or getattr(module, 'RUN', None) != D112_ORIGINAL):
        _reject('D112 source or evidence destination differs')
    for record in RECORDERS:
        label, directory, filename, count, line, source_digest, _ = record
        source = HERE.parent / directory / filename
        selected = [item for item, path in zip(session.items, paths) if path == str(source)]
        if len(selected) != count:
            _reject('missing complete inherited ' + label + ' module')
        owner = selected[0].module
        function_name = _relocation(record)['function_name']
        original = getattr(owner, function_name, None)
        if (any(item.module is not owner for item in selected)
                or getattr(owner, '__file__', None) != str(source)
                or source.is_symlink() or not source.is_file()
                or hashlib.sha256(source.read_bytes()).hexdigest() != source_digest
                or manifest['source_sha256'].get(str(source)) != source_digest
                or type(original) is not type(_writer)
                or original.__name__ != function_name or original.__module__ != owner.__name__
                or original.__globals__ is not vars(owner)
                or original.__code__.co_filename != str(source)
                or original.__code__.co_firstlineno != line):
            _reject(label + ' original evidence function identity differs')
        destination = Path(_relocation(record)['relocated_run'])
        destination.mkdir(exist_ok=False)
        writer = _writer(record)
        setattr(owner, function_name, writer)
        _bindings.append((owner, original, writer))
    D112_DESTINATION.mkdir(exist_ok=False)
    module.RUN = D112_DESTINATION
    _d112_binding = module
    with INVENTORY.open('x') as stream:
        json.dump(dict(nodeids=ids, count=4189, files=223, manifest_sha256=digest,
            validated_before_execution=True, **relocations),
            stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    _verified = session, ids, paths, digest


@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    if _verified is None or _verified[0] is not session:
        _reject('no successful collection contract')
    _check_bindings()
    ids, paths = _population(session)
    if (ids != _verified[1] or paths != _verified[2]
            or INVENTORY.is_symlink() or not INVENTORY.is_file()
            or hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != _verified[3]):
        _reject('population or evidence changed before execution')
    return None



class _D228EvidencePath(type(Path())):
    """Only the frozen inline summary open is redirected; no global Path patch."""

    def open(self, mode='r', buffering=-1, encoding=None, errors=None, newline=None):
        global _d228_redirects
        original_summary = D228_ORIGINAL / 'summary.json'
        if Path(self) == original_summary and mode == 'x':
            destination = D228_DESTINATION / 'summary.json'
            if (not _d228_call_active or _d228_redirects != 0
                    or _d228_binding is None or _d228_binding[0].Path is not _D228EvidencePath
                    or os.environ.get('NEURAL_HZ_ACTIVE_COMPONENT_RUN') != str(D228_ORIGINAL)
                    or not RUN.is_dir() or RUN.is_symlink()
                    or not D228_DESTINATION.is_dir() or D228_DESTINATION.is_symlink()
                    or D228_DESTINATION.resolve() != D228_DESTINATION
                    or destination.exists() or destination.is_symlink()):
                _reject('D228 exclusive summary redirection contract differs')
            _d228_redirects += 1
            return destination.open(mode, buffering, encoding, errors, newline)
        if mode not in ('r', 'rb', 'rt'):
            _reject('D228 facade forbids every other write')
        return super().open(mode, buffering, encoding, errors, newline)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    global _d228_call_active, _d228_call_finished
    if str(Path(item.path).absolute()) != str(D228_MODULE) or item.name != D228_FUNCTION:
        yield
        return
    _check_bindings()
    owner, function, original_code, original_path = _d228_binding
    if (item.obj is not function or item.module is not owner
            or _d228_call_finished or _d228_redirects != 0
            or function.__code__ is not original_code):
        _reject('D228 inline-writer call identity or once-only scope differs')
    previous_environment = os.environ.get('NEURAL_HZ_ACTIVE_COMPONENT_RUN')
    try:
        _d228_call_active = True
        owner.Path = _D228EvidencePath
        os.environ['NEURAL_HZ_ACTIVE_COMPONENT_RUN'] = str(D228_ORIGINAL)
        yield
    finally:
        changed = (owner.Path is not _D228EvidencePath
            or os.environ.get('NEURAL_HZ_ACTIVE_COMPONENT_RUN') != str(D228_ORIGINAL)
            or getattr(owner, D228_FUNCTION) is not function
            or function.__code__ is not original_code)
        owner.Path = original_path
        os.environ['NEURAL_HZ_ACTIVE_COMPONENT_RUN'] = previous_environment
        _d228_call_active = False
        _d228_call_finished = True
        if changed:
            _reject('D228 scoped I/O identity changed during its test')
        _check_bindings()


def _check_d228_summary():
    destination = D228_DESTINATION / 'summary.json'
    if (not _d228_call_finished or _d228_redirects != 1
            or D228_DESTINATION.is_symlink() or not D228_DESTINATION.is_dir()
            or {path.name for path in D228_DESTINATION.iterdir()} != {'summary.json'}
            or destination.is_symlink() or not destination.is_file()
            or not 0 < destination.stat().st_size <= 8 * 1024**2):
        _reject('D228 successful population lacks one exclusive relocated summary')
    summary = json.loads(destination.read_text())
    names = list(_d228_binding[0]._NAMES)
    completed = summary.get('completed_tests')
    if (summary.get('schema') != 'd228_joint_bank_component_mathematical_tests_v1'
            or len(names) != 16 or len(set(names)) != 16
            or summary.get('required_tests') != names or summary.get('missing_tests') != []
            or type(completed) is not dict or set(completed) != set(names)
            or summary.get('independent_reference_uses_phase_cells') is not True
            or any(summary.get(key) is not False for key in (
                'candidate_uses_test_reference_or_external_solver', 'actual_model_run',
                'native_qualified', 'gpu_qualified'))
            or summary.get('formal_gain') != 0 or summary.get('independent_e0_gain') != 0):
        _reject('D228 complete relocated summary differs')

@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    if _verified is None:
        return
    _check_bindings()
    if exitstatus == 0:
        _check_d228_summary()
    for record in RECORDERS:
        destination = Path(_relocation(record)['relocated_run'])
        if exitstatus == 0 and (destination.is_symlink()
                or {path.name for path in destination.iterdir()} != set(record[-1])
                or any(path.is_symlink() or not path.is_file() for path in destination.iterdir())):
            _reject('successful inherited ' + record[0] + ' population lacks its complete evidence')
    if exitstatus == 0 and any((RUN / name).is_symlink() or not (RUN / name).is_file()
                               for name in NEW_RECORD_FILES):
        _reject('successful D243 population lacks its complete evidence')
