"""Read-only static preparation / saved-record audit. Never reruns controls."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ('scoped_proof/pidfd_compat.py', 'scoped_proof/owned_session.py',
                'scoped_proof/owned_bounded.py', 'scoped_proof/io.py',
                'scoped_source/endpoint_batch_observer.py',
                'scripts/pidfd_finite_control.py', 'scripts/audit_pidfd_finite_control.py',
                'scripts/check_endpoint_batch_static.py')


def read(path):
    def unique(items):
        result = {}
        for k, v in items:
            assert k not in result, 'duplicate field'
            result[k] = v
        return result
    raw = Path(path).read_bytes()
    assert len(raw) <= 1024**2
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def static():
    hashes = {}; trees = {}
    for name in SOURCE_FILES:
        raw = (ROOT/name).read_bytes()
        tree = ast.parse(raw, filename=name); compile(tree, name, 'exec')
        hashes[str(ROOT/name)] = hashlib.sha256(raw).hexdigest(); trees[name] = tree
    # Ownership mechanism unchanged: no raw PID/PGID signalling in new path.
    for name in ('scoped_proof/pidfd_compat.py', 'scoped_proof/owned_session.py',
                 'scripts/pidfd_finite_control.py'):
        for n in ast.walk(trees[name]):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                assert n.func.attr not in ('kill', 'killpg', 'syscall')
    owned = trees['scoped_proof/owned_session.py']
    execute = next(n for n in owned.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
    probe = [n.lineno for n in ast.walk(execute) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'probe']
    spawn = [n.lineno for n in ast.walk(execute) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'Popen']
    assert len(probe) == len(spawn) == 1 and probe[0] < spawn[0]
    observer = trees['scoped_source/endpoint_batch_observer.py']
    execute = next(n for n in observer.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
    probe = [n.lineno for n in ast.walk(execute) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'probe']
    admission = [n.lineno for n in ast.walk(execute) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'admission']
    assert len(probe) == len(admission) == 1 and probe[0] < admission[0]
    tree = trees['scripts/pidfd_finite_control.py']
    roster = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'IDS' for t in n.targets))
    assert len(roster) == len(set(roster)) == 8
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assert all(any(isinstance(n, ast.FunctionDef) and n.name == name for n in ast.walk(main))
               for name in roster)
    assert read(ROOT/'docs/hz_endpoint_execution_approvals.json')['batches'] == []
    return dict(status='PASS_STATIC_ONLY', source_hashes=hashes, cases=list(roster),
                controls_executed=0, algorithm_executed=False)


def audit(directory, stdout_path, stderr_path, exit_code):
    freeze_path = ROOT/'docs/hz_pidfd_control_freeze_20261019.json'
    freeze = read(freeze_path)
    freeze_hash = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    root = Path(directory)
    result = read(root/'result.json'); reception = read(root/'reception.json')
    returned = read(stdout_path)
    exported = read(stderr_path)
    assert returned['result'] == result and returned['reception'] == reception
    assert result['schema'] == 'PIDFD_FINITE_CONTROL_V1' and result['caps'] == freeze['caps']
    assert result['freeze_sha256'] == freeze_hash
    for path, expected in freeze['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    assert root == Path(freeze['result_directory'])
    assert reception['result_directory'] == str(root) and reception['result_sha256'] == digest(result)
    started = result['started']
    assert 0 <= result['provisioning_seconds'] and result['full_cold_start_seconds'] is None
    assert started <= result['before_export'] <= reception['received_at'] <= returned['finished_at']
    assert result['seconds_before_export'] == result['before_export']-started
    assert reception['seconds'] == reception['received_at']-started
    assert returned['seconds'] == returned['finished_at']-started < 300
    assert exported['schema'] == 'PIDFD_FINITE_STDOUT_EXPORTED_V1'
    assert returned['finished_at'] <= exported['after_export'] < started+300
    assert exported['seconds'] == exported['after_export']-started
    assert exported['stdout_export_seconds'] == exported['after_export']-returned['finished_at']
    assert reception['export_seconds'] == reception['received_at']-result['before_export']
    cases = result['cases']; ids = [x['id'] for x in cases]
    assert ids == freeze['cases'][:len(ids)]
    assert result['unexecuted'] == freeze['cases'][len(ids):]
    assert all(x['seconds'] >= 0 and x['status'] in ('PASS', 'FAIL') for x in cases)
    assert not any(result[k] for k in ('algorithm_executed', 'model_executed',
                                      'gpu_executed', 'endpoint_batch_authorized'))
    complete = result['status'] == 'PASS'
    if complete:
        assert str(exit_code) == '0' and exported['online_acceptance'] is True
        assert ids == freeze['cases'] and all(x['status'] == 'PASS' for x in cases)
        assert result['online_acceptance'] is reception['online_acceptance'] is returned['online_acceptance'] is True
        assert result['initial_fd_count'] == result['final_fd_count']
        assert result['sampled_batch_rss_bytes'] <= freeze['caps']['rss_bytes']
        sessions = result['sessions']
        assert [x['status'] for x in sessions] == ['COMPLETED', 'ERROR', 'TIMEOUT']
        assert [x['returncode'] for x in sessions] == [0, 7, -9]
        for row in sessions:
            assert row['cleanup_status'] == 'LEADER_REAPED_NO_LIVE_SESSION'
            assert not row['remaining_session']['live']
            assert started <= row['started_at'] <= row['finished_at'] < row['cleanup_deadline'] < started+300
            assert row['run_deadline'] <= row['cleanup_deadline']
            assert row['log_bytes'] == len(base64.b64decode(row['log_base64'], validate=True))
            assert row['pidfd_backend'] == result['backend']
        pid, sid, pgid = map(int, base64.b64decode(sessions[-1]['log_base64']).decode().split())
        assert pgid == pid != sid == sessions[-1]['session']
        events = result['child_events']
        assert [x['event'] for x in events] == ['SPAWN', 'REAP', 'SPAWN', 'REAP']
        assert events[0]['pid'] == events[1]['pid'] and events[1]['returncode'] == -9
        assert events[2]['pid'] == events[3]['pid'] and events[3]['returncode'] == 0
    else:
        assert not returned['online_acceptance'], 'failure cannot be promoted offline'
    return dict(status='PASS_SAVED_COMPATIBILITY_RECORDS' if complete else 'RETAINED_INCOMPLETE',
                freeze_sha256=freeze_hash, result_sha256=digest(result),
                stdout_sha256=hashlib.sha256(Path(stdout_path).read_bytes()).hexdigest(),
                passed=sum(x['status'] == 'PASS' for x in cases), unexecuted=result['unexecuted'],
                seconds=exported['seconds'], process_exit_code=int(exit_code),
                scope='COMPATIBILITY_ONLY_NOT_ALGORITHM_EFFICACY',
                runtime_reexecuted=False, seven_position_batch_authorized=False)


if __name__ == '__main__':
    value = static() if sys.argv[1:] == ['--static'] else audit(*sys.argv[1:])
    print(json.dumps(value, sort_keys=True, indent=2))
