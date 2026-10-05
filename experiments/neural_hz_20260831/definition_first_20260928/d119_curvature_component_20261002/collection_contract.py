"""One authenticated collection and test loop; no standalone collection run."""
import hashlib
import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d119_curvature_component_20261002_v1'
MANIFEST = RUN / 'preregistered.json'
INVENTORY = RUN / 'inventory.json'
INHERITED_MODULE = (HERE.parent / 'd112_shared_endpoint_forward_20261002'
                    / 'test_endpoint_forward.py')
INHERITED_RUN = EXP / 'results/d112_shared_endpoint_forward_20261002_v1'
RELOCATED_RUN = RUN / 'inherited_d112_controls'
EVIDENCE_RELOCATION = dict(module_path=str(INHERITED_MODULE),
                           original_run=str(INHERITED_RUN),
                           relocated_run=str(RELOCATED_RUN))
_verified = None


def _reject(reason):
    raise pytest.UsageError('D119 complete population: ' + reason)


def _population(session):
    if session.testsfailed or session.shouldstop or session.shouldfail:
        _reject('collection error or stop')
    ids = tuple(item.nodeid for item in session.items)
    paths = tuple(str(Path(item.path).absolute()) for item in session.items)
    if len(ids) != 3861 or len(set(ids)) != 3861 or len(set(paths)) != 192:
        _reject('3861 items and 192 original paths required')
    return ids, paths


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    global _verified
    if _verified is not None:
        _reject('repeated collection')
    digest = os.environ.get('NEURAL_HZ_D119_MANIFEST_SHA256')
    if not isinstance(digest, str) or len(digest) != 64:
        _reject('missing parent identity')
    if MANIFEST.is_symlink() or not MANIFEST.is_file() or not 0 < MANIFEST.stat().st_size <= 8*1024**2:
        _reject('invalid manifest')
    raw = MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        _reject('changed manifest')
    manifest = json.loads(raw)
    expected, files = manifest.get('expected_nodeids'), manifest.get('tests')
    if (manifest.get('required_tests') != 3861 or manifest.get('required_test_files') != 192
            or manifest.get('new_test_names') != [
                'test_curvature_canonical_retention',
                'test_curvature_contract_rejections',
                'test_curvature_pair_hull_forward',
                'test_curvature_prefix_support']
            or manifest.get('inherited_test_population_unchanged') is not True
            or manifest.get('inherited_tests') != 3857
            or manifest.get('inherited_test_files') != 191
            or manifest.get('new_test_files') != 1
            or manifest.get('same_process_collection_gate') is not True
            or manifest.get('single_pytest_process') is not True
            or manifest.get('fixed_component_lp_controls_registered') is not True
            or manifest.get('solver_rescue_registered') is not False
            or manifest.get('inherited_component_evidence_relocation') != EVIDENCE_RELOCATION
            or type(expected) is not list or len(expected) != 3861 or len(set(expected)) != 3861
            or type(files) is not list or len(files) != 192 or len(set(files)) != 192):
        _reject('wrong contract')
    ids, paths = _population(session)
    if ids != tuple(expected) or set(paths) != set(files):
        _reject('population drift')
    if any(str(ROOT / i.split('::', 1)[0]) != p for i, p in zip(ids, paths)):
        _reject('node identity differs from actual source')
    # Only the frozen D112 controls' evidence destination is changed. The
    # test functions, assertions, inputs, and budgets remain inherited verbatim.
    inherited_items = [item for item, path in zip(session.items, paths)
                       if path == str(INHERITED_MODULE)]
    if len(inherited_items) != 4:
        _reject('missing inherited D112 test module')
    module = inherited_items[0].module
    if (any(item.module is not module for item in inherited_items)
            or getattr(module, '__file__', None) != str(INHERITED_MODULE)
            or getattr(module, 'RUN', None) != INHERITED_RUN):
        _reject('inherited D112 module or original evidence destination differs')
    RELOCATED_RUN.mkdir(exist_ok=False)
    module.RUN = RELOCATED_RUN
    with INVENTORY.open('x') as stream:
        json.dump(dict(nodeids=ids, count=3861, files=192,
                       manifest_sha256=digest, validated_before_execution=True,
                       inherited_component_evidence_relocation=EVIDENCE_RELOCATION),
                  stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    _verified = session, ids, paths, digest


@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    if _verified is None or _verified[0] is not session:
        _reject('no successful collection contract')
    ids, paths = _population(session)
    if (ids != _verified[1] or paths != _verified[2]
            or INVENTORY.is_symlink() or not INVENTORY.is_file()
            or hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != _verified[3]):
        _reject('population or evidence changed before execution')
    return None
