"""Authenticate one complete D208 collection immediately before its test loop."""
import hashlib
import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d208_multilayer_closure_audit_20261005_v1'
MANIFEST = RUN / 'preregistered.json'
INVENTORY = RUN / 'inventory.json'
INHERITED_MODULE = HERE.parent / 'd112_shared_endpoint_forward_20261002/test_endpoint_forward.py'
INHERITED_RUN = EXP / 'results/d112_shared_endpoint_forward_20261002_v1'
RELOCATED_RUN = RUN / 'inherited_d112_controls'
EVIDENCE_RELOCATION = dict(module_path=str(INHERITED_MODULE), original_run=str(INHERITED_RUN),
                           relocated_run=str(RELOCATED_RUN))

D207_MODULE = HERE.parent / 'd207_owned_source_phase_20261005/test_fiber.py'
D207_RECORD_FILES = ('source_phase_positive_control.json', 'exact_alias_control.json',
                     'complete_200_gate_control.json')
D207_EVIDENCE_RELOCATION = dict(module_path=str(D207_MODULE),
    source_sha256='1c23e38befc50be621e7dacf863f20bddc9ff107b4ef2c3bc3742ee8d6627e02',
    function_name='_record', function_firstlineno=44,
    original_run=str(EXP / 'results/d207_owned_source_phase_20261005_v1'),
    relocated_run=str(RUN / 'inherited_d207_controls'),
    allowed_filenames=list(D207_RECORD_FILES), mechanism='module_local_record_function_only')

_verified = None
_d207_record_binding = None


def _reject(reason):
    raise pytest.UsageError('D208 complete population: ' + reason)


def _d207_record(name, value):
    """Only the frozen D207 evidence writer is relocated; test logic is untouched."""
    if (name not in D207_RECORD_FILES
            or Path(os.environ.get('NEURAL_HZ_ACTIVE_COMPONENT_RUN', '')) != RUN
            or not RUN.is_dir() or RUN.is_symlink()):
        _reject('D207 evidence name or current exclusive destination differs')
    destination = RUN / 'inherited_d207_controls'
    if not destination.is_dir() or destination.is_symlink():
        _reject('D207 evidence destination changed')
    with (destination / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _population(session):
    if session.testsfailed or session.shouldstop or session.shouldfail:
        _reject('collection error or stop')
    ids = tuple(item.nodeid for item in session.items)
    paths = tuple(str(Path(item.path).absolute()) for item in session.items)
    if len(ids) != 4084 or len(set(ids)) != 4084 or len(set(paths)) != 215:
        _reject('4084 items and 215 original paths required')
    return ids, paths


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    global _verified, _d207_record_binding
    if _verified is not None:
        _reject('repeated collection')
    digest = os.environ.get('NEURAL_HZ_D208_MANIFEST_SHA256')
    if not isinstance(digest, str) or len(digest) != 64:
        _reject('missing parent identity')
    if MANIFEST.is_symlink() or not MANIFEST.is_file() or not 0 < MANIFEST.stat().st_size <= 8 * 1024**2:
        _reject('invalid manifest')
    raw = MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        _reject('changed manifest')
    manifest = json.loads(raw)
    expected, files = manifest.get('expected_nodeids'), manifest.get('tests')
    if (manifest.get('schema') != 'd208_multilayer_closure_audit_v1'
            or manifest.get('required_tests') != 4084 or manifest.get('required_test_files') != 215
            or manifest.get('inherited_tests') != 4080 or manifest.get('inherited_test_files') != 214
            or manifest.get('new_test_files') != 1
            or type(manifest.get('new_test_names')) is not list or len(manifest['new_test_names']) != 4
            or manifest.get('inherited_test_population_unchanged') is not True
            or manifest.get('pytest_import_mode') != 'importlib'
            or manifest.get('pytest_plugin_autoload') is not False
            or os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') != '1'
            or manifest.get('mathematical_stage_only') is not True
            or manifest.get('worker_stage_registered') is not False
            or manifest.get('worker_launched') is not False
            or manifest.get('source_component_qualified') is not False
            or manifest.get('source_census_completed') is not False
            or manifest.get('source_census_qualified') is not False
            or manifest.get('actual_model_binding_qualified') is not False
            or manifest.get('actual_phase_column_binding_verified') is not False
            or manifest.get('native_HZ_admitted') is not False
            or manifest.get('gpu_computation_completed') is not False
            or manifest.get('complete_physical_qualification') is not False
            or manifest.get('candidate_physical_gate_evaluated') is not False
            or manifest.get('production_snapshot_imported') is not False
            or manifest.get('same_process_collection_gate') is not True
            or manifest.get('single_pytest_process') is not True
            or manifest.get('fixed_component_lp_controls_registered') is not True
            or manifest.get('solver_rescue_registered') is not False
            or manifest.get('negative_audit_only') is not True
            or manifest.get('domain_definition_changed') is not False
            or manifest.get('new_domain_qualified') is not False
            or manifest.get('new_capability_qualified') is not False
            or manifest.get('capability_improvement_claimed') is not False
            or manifest.get('formal_gain') != 0 or manifest.get('new_benchmark_solves') != 0
            or manifest.get('inherited_component_evidence_relocation') != EVIDENCE_RELOCATION
            or manifest.get('inherited_D207_evidence_relocation') != D207_EVIDENCE_RELOCATION
            or type(expected) is not list or len(expected) != 4084 or len(set(expected)) != 4084
            or type(files) is not list or len(files) != 215 or len(set(files)) != 215):
        _reject('wrong contract')
    ids, paths = _population(session)
    if ids != tuple(expected) or set(paths) != set(files):
        _reject('population drift')
    if any(str(ROOT / i.split('::', 1)[0]) != p for i, p in zip(ids, paths)):
        _reject('node identity differs from actual source')
    inherited_items = [item for item, path in zip(session.items, paths) if path == str(INHERITED_MODULE)]
    if len(inherited_items) != 4:
        _reject('missing inherited D112 test module')
    module = inherited_items[0].module
    if (any(item.module is not module for item in inherited_items)
            or getattr(module, '__file__', None) != str(INHERITED_MODULE)
            or getattr(module, 'RUN', None) != INHERITED_RUN):
        _reject('inherited D112 source or evidence destination differs')

    d207_items = [item for item, path in zip(session.items, paths) if path == str(D207_MODULE)]
    if len(d207_items) != 24:
        _reject('missing complete inherited D207 test module')
    d207_module = d207_items[0].module
    original_record = getattr(d207_module, '_record', None)
    if (any(item.module is not d207_module for item in d207_items)
            or getattr(d207_module, '__file__', None) != str(D207_MODULE)
            or D207_MODULE.is_symlink() or not D207_MODULE.is_file()
            or hashlib.sha256(D207_MODULE.read_bytes()).hexdigest() != D207_EVIDENCE_RELOCATION['source_sha256']
            or manifest['source_sha256'].get(str(D207_MODULE)) != D207_EVIDENCE_RELOCATION['source_sha256']
            or type(original_record) is not type(_d207_record)
            or original_record.__name__ != '_record'
            or original_record.__module__ != d207_module.__name__
            or original_record.__globals__ is not vars(d207_module)
            or original_record.__code__.co_filename != str(D207_MODULE)
            or original_record.__code__.co_firstlineno != 44):
        _reject('D207 original evidence function identity differs')
    (RUN / 'inherited_d207_controls').mkdir(exist_ok=False)
    d207_module._record = _d207_record
    _d207_record_binding = (d207_module, original_record, _d207_record)

    RELOCATED_RUN.mkdir(exist_ok=False)
    module.RUN = RELOCATED_RUN
    with INVENTORY.open('x') as stream:
        json.dump(dict(nodeids=ids, count=4084, files=215, manifest_sha256=digest,
            validated_before_execution=True, inherited_component_evidence_relocation=EVIDENCE_RELOCATION,
            inherited_D207_evidence_relocation=D207_EVIDENCE_RELOCATION),
            stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    _verified = session, ids, paths, digest


@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    if _verified is None or _verified[0] is not session:
        _reject('no successful collection contract')
    if (_d207_record_binding is None
            or _d207_record_binding[0]._record is not _d207_record_binding[2]):
        _reject('D207 evidence-only binding changed before execution')
    ids, paths = _population(session)
    if (ids != _verified[1] or paths != _verified[2]
            or INVENTORY.is_symlink() or not INVENTORY.is_file()
            or hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != _verified[3]):
        _reject('population or evidence changed before execution')
    return None


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    if _verified is None:
        return
    if (_d207_record_binding is None
            or _d207_record_binding[0]._record is not _d207_record_binding[2]):
        _reject('D207 evidence-only binding changed during execution')
    destination = RUN / 'inherited_d207_controls'
    if exitstatus == 0 and (
            destination.is_symlink()
            or {path.name for path in destination.iterdir()} != set(D207_RECORD_FILES)
            or any(path.is_symlink() or not path.is_file() for path in destination.iterdir())):
        _reject('successful inherited D207 population lacks its complete evidence')
