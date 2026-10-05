"""Frozen same-session collection contract, not a replacement test runner.

Only this isolated D072 process may use the plugin. The parent authenticates
all source files and passes the digest of its exclusive preregistered.json.
Collection, this check, execution and XML writing all share the original 60s.
An invalid population raises immediately, before any test body can run.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d072_single_session_gate_20261001_v1'
MANIFEST = RUN / 'preregistered.json'
INVENTORY = RUN / 'inventory.json'
_verified_session = None
_ordered_nodeids = None
_ordered_files = None
_manifest_digest = None


def _reject(message):
    raise pytest.UsageError('D072 full collection contract: ' + message)


def _population(session):
    if session.testsfailed or session.shouldstop or session.shouldfail:
        _reject('collection is not error-free')
    nodeids = [item.nodeid for item in session.items]
    files = [str(Path(item.path).absolute()) for item in session.items]
    if (len(nodeids) != 3813 or len(set(nodeids)) != 3813
            or len(set(files)) != 180):
        _reject('complete 3813-item/180-file population required')
    return nodeids, files


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    global _verified_session, _ordered_nodeids, _ordered_files, _manifest_digest
    if _verified_session is not None:
        _reject('a second collection is not registered')
    expected_digest = os.environ.get('NEURAL_HZ_D072_MANIFEST_SHA256')
    if type(expected_digest) is not str or len(expected_digest) != 64:
        _reject('parent manifest identity missing')
    if (MANIFEST.is_symlink() or not MANIFEST.is_file()
            or not 0 < MANIFEST.stat().st_size <= 8 * 1024**2):
        _reject('manifest missing, linked or oversized')
    raw = MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        _reject('manifest differs from parent identity')
    manifest = json.loads(raw)
    if (manifest.get('required_tests') != 3813 or manifest.get('required_test_files') != 180
            or manifest.get('same_process_collection_gate') is not True
            or manifest.get('single_pytest_process') is not True):
        _reject('wrong frozen execution contract')
    expected, paths = manifest.get('expected_nodeids'), manifest.get('tests')
    if (type(expected) is not list or len(expected) != 3813 or len(set(expected)) != 3813
            or type(paths) is not list or len(paths) != 180 or len(set(paths)) != 180):
        _reject('invalid manifest population')
    nodeids, files = _population(session)
    if nodeids != expected or set(files) != set(paths):
        _reject('ordered test IDs or original paths differ')
    # Also tie the node ID's path spelling to the actual collected source.
    if any(str(ROOT / nodeid.split('::', 1)[0]) != path
           for nodeid, path in zip(nodeids, files)):
        _reject('node ID and actual original source disagree')
    with INVENTORY.open('x') as stream:
        json.dump(dict(nodeids=nodeids, count=3813, files=180,
                       manifest_sha256=expected_digest,
                       validated_before_execution=True),
                  stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    # Do not mark success before the evidence file has closed successfully.
    _ordered_nodeids, _ordered_files = tuple(nodeids), tuple(files)
    _manifest_digest = expected_digest
    _verified_session = session


@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    if _verified_session is not session or _ordered_nodeids is None:
        _reject('test loop reached without authenticated complete collection')
    nodeids, files = _population(session)
    if tuple(nodeids) != _ordered_nodeids or tuple(files) != _ordered_files:
        _reject('population changed after collection verification')
    if (not INVENTORY.is_file() or INVENTORY.is_symlink()
            or hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != _manifest_digest):
        _reject('collection evidence or manifest changed before execution')
    # Return None so pytest's unchanged standard loop performs every test.
    return None
