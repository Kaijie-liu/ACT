"""Read-only saved-record audit, not a repeat of either fault or a network proof."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time


def require(ok, why):
    if not ok:
        raise ValueError(why)


def hashed(raw):
    return hashlib.sha256(raw).hexdigest()


def audit(root):
    started = time.monotonic()
    terminal = json.loads((root/'terminal.json').read_bytes())
    freeze = json.loads((root/'freeze.json').read_bytes())
    report = json.loads((root/'worker.json').read_bytes())
    repo = Path(__file__).resolve().parents[1]
    frozen_path = 'docs/hz_pair_factor_faults_freeze_20261019.json'
    require(freeze == json.loads((repo/frozen_path).read_bytes()), 'registered freeze')
    head = terminal['execution_head']
    require(type(head) is str and len(head) == 40 and all(c in '0123456789abcdef' for c in head),
            'execution commit identity')
    committed_freeze = subprocess.check_output(['git', 'show', head+':'+frozen_path], cwd=repo)
    require(json.loads(committed_freeze) == freeze, 'executed freeze identity')
    for name, expected in freeze['files'].items():
        raw = subprocess.check_output(['git', 'show', head+':'+name], cwd=repo)
        require(hashed(raw) == expected, 'executed code identity '+name)
    require(terminal['schema'] == 'PAIR_FACTOR_FAULT_TERMINAL_V1', 'terminal schema')
    require(terminal['status'] == 'PASS' and terminal['returncode'] == 0
            and not terminal['timed_out'] and not terminal['resource_stop']
            and terminal['report_error'] is None, 'completed execution')
    require(0 <= terminal['wall_seconds'] < 300 and not (root/'late.json').exists(), 'deadline')
    require(terminal['peak_group_rss_bytes'] <= 2*2**30, 'sampled RSS')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(actual == set(terminal['files']) | {'terminal.json'}, 'complete artifact inventory')
    for name, ref in terminal['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'relative artifact')
        raw = (root/name).read_bytes()
        require(len(raw) == ref['bytes'] and hashed(raw) == ref['sha256'], 'artifact identity '+name)
    size = sum(r['bytes'] for r in terminal['files'].values())
    require(size == terminal['artifact_bytes_before_terminal'], 'byte accounting')
    total_bytes = size + (root/'terminal.json').stat().st_size
    require(total_bytes <= 134217728, 'artifact limit')
    require(report['schema'] == 'PAIR_FACTOR_FAULT_RESULT_V1' and report['status'] == 'PASS'
            and report['tests'] == len(freeze['tests']) == 2
            and report['failures'] == report['errors'] == 0, 'exactly two passing tests')
    require(not report['unexpected_modules'] and report['real_models'] == report['gpu'] == 0
            and report['full_request_performance_claim'] is False, 'scope')
    rows = report['records']
    require([r['kind'] for r in rows] == ['source', 'factor_fault', 'factor_fault'], 'no added runs')
    require(rows[0]['name'] == freeze['source_name'], 'registered existing source')
    require([r['mode'] for r in rows[1:]] == ['missing', 'permuted'], 'two faults')
    for row in rows:
        require(row['source'] == freeze['source'] and row['forest'] == freeze['forest'], 'R1 identity')
        require(all(type(v) is int and v >= 0 for v in row['work'].values())
                and sum(row['work'].values()) <= 8000000, 'cumulative Work cap')
        require(0 <= row['seconds'] < 300, 'phase cost')
    require(sum(r['seconds'] for r in rows) <= terminal['wall_seconds'], 'phases charged to batch')
    for row in rows[1:]:
        require(row['pair'] == [0, 1] and row['selected_factors'] > 1, 'multi-factor prerequisite')
        a, b = row['factor_ids']
        require(type(a) is int and type(b) is int and 0 <= a < b, 'canonical factor ids')
        pack = lambda x: struct.pack('<I', x)
        before = [pack(1).hex(), pack(2).hex(), (pack(a)+pack(b)).hex()]
        after = ([pack(0).hex(), pack(2).hex(), before[2]] if row['mode'] == 'missing'
                 else [pack(2).hex(), pack(1).hex(), (pack(b)+pack(a)).hex()])
        require(row['before_hex'] == before and row['after_hex'] == after and before != after,
                'actual frozen mutation')
        require(row['error'] == freeze['expected_errors'][row['mode']]
                and row['traversal_ready'] is False and row['check_started'] is True
                and row['nodes_sealed'] is True and row['factor_maps_unsealed'] is True,
                'factor rejection, not unrelated error')
        owner = row['owner']
        require(row['scope_closed'] and row['owner_closed'] and row['all_buffers_released']
                and owner['live_bytes'] == 0
                and owner['allocated_bytes'] == owner['released_bytes']
                and owner['peak_bytes'] <= 134217728, 'complete owned cleanup')
    return dict(status='PASS_TWO_FACTOR_FAULT_RECORDS', execution_head=head, tests=2,
        prior_batch_status='FAIL_PRESERVED', records=rows,
        wall_seconds=terminal['wall_seconds'], peak_group_rss_bytes=terminal['peak_group_rss_bytes'],
        artifact_bytes=total_bytes, audit_seconds=time.monotonic()-started,
        terminal_sha256=hashed((root/'terminal.json').read_bytes()),
        worker_sha256=hashed((root/'worker.json').read_bytes()),
        freeze_sha256=hashed((root/'freeze.json').read_bytes()),
        scope='Saved identities, two expected rejections and costs; no verifier reexecution or network proof.')


if __name__ == '__main__':
    print(json.dumps(audit(Path(sys.argv[1])), sort_keys=True, indent=2))
