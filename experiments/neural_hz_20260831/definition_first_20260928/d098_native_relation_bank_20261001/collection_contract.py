"""One authenticated collection and test loop; no standalone collection run."""
import hashlib
import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d098_native_relation_bank_20261001_v1'
MANIFEST = RUN / 'preregistered.json'
INVENTORY = RUN / 'inventory.json'
_verified = None


def _reject(reason):
    raise pytest.UsageError('D098 complete population: ' + reason)


def _population(session):
    if session.testsfailed or session.shouldstop or session.shouldfail:
        _reject('collection error or stop')
    ids = tuple(item.nodeid for item in session.items)
    paths = tuple(str(Path(item.path).absolute()) for item in session.items)
    if len(ids) != 3845 or len(set(ids)) != 3845 or len(set(paths)) != 188:
        _reject('3845 items and 188 original paths required')
    return ids, paths


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    global _verified
    if _verified is not None:
        _reject('repeated collection')
    digest = os.environ.get('NEURAL_HZ_D098_MANIFEST_SHA256')
    if not isinstance(digest, str) or len(digest) != 64:
        _reject('missing parent identity')
    if MANIFEST.is_symlink() or not MANIFEST.is_file() or not 0 < MANIFEST.stat().st_size <= 8*1024**2:
        _reject('invalid manifest')
    raw = MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        _reject('changed manifest')
    manifest = json.loads(raw)
    expected, files = manifest.get('expected_nodeids'), manifest.get('tests')
    if (manifest.get('required_tests') != 3845 or manifest.get('required_test_files') != 188
            or manifest.get('new_test_names') != [
                'test_native_catalog_matches_literal_relations',
                'test_native_bank_survives_actual_merges',
                'test_complete_native_relay_bank',
                'test_native_bank_fail_closed']
            or manifest.get('inherited_test_population_unchanged') is not True
            or manifest.get('inherited_tests') != 3841
            or manifest.get('inherited_test_files') != 187
            or manifest.get('new_test_files') != 1
            or manifest.get('same_process_collection_gate') is not True
            or manifest.get('single_pytest_process') is not True
            or type(expected) is not list or len(expected) != 3845 or len(set(expected)) != 3845
            or type(files) is not list or len(files) != 188 or len(set(files)) != 188):
        _reject('wrong contract')
    ids, paths = _population(session)
    if ids != tuple(expected) or set(paths) != set(files):
        _reject('population drift')
    if any(str(ROOT / i.split('::', 1)[0]) != p for i, p in zip(ids, paths)):
        _reject('node identity differs from actual source')
    with INVENTORY.open('x') as stream:
        json.dump(dict(nodeids=ids, count=3845, files=188,
                       manifest_sha256=digest, validated_before_execution=True),
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
