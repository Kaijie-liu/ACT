"""Finite stdlib-only lifecycle supervision. No real/native/GPU intake.

The child is trusted not to escape its session. Observed RSS is not a memory
proof. Timely result acceptance is bounded; OS-delayed cleanup can overrun and
is reported, never promoted to a successful request.
"""
from fractions import Fraction as F
import hashlib
import math
import os
from pathlib import Path
import stat
import time
import uuid

from scoped_source.hz_request_store import POLICY, Work, compact, decode, digest, identity, require
from scoped_source.direct_method import method, validate, PROTOCOL, SHARED_PROTOCOL
from scoped_source.direct_lifecycle_package import CODE_FILES, SHARED_CODE, BUNDLE
from scoped_source.direct_lifecycle_execution import EXEC_FAULTS
from scoped_source.wire_package import inventory
from scoped_proof.io import ROOT, PYTHON
from scoped_proof.owned_bounded import execute as run_owned

PROTOCOL_FILE = 'docs/hz_direct_lifecycle_protocol_20261015.md'
SHARED_PROTOCOL_FILE = 'docs/hz_shared_source_protocol_20261015.md'
REPORT_CAP = 2**20
# Candidate and receipt <= 1 MiB each; RLIMIT_FSIZE caps even an abnormal
# worker log at 1 MiB. Progress <= 64 KiB, plus conservative metadata reserve.
META_RESERVE = 3 * REPORT_CAP + 2 * 65536
LOG_CAP = 65536
RSS_LIMIT = 2 * 2**30
COUNTERS = ('row_visits', 'coefficient_visits', 'arithmetic')
FAULTS = EXEC_FAULTS + ('startup_error', 'output_partial', 'wrong_invocation', 'wrong_method',
                       'output_block', 'late_exit', 'receive_expired', 'terminal_expired')
FILES = tuple(sorted(set(CODE_FILES + SHARED_CODE + (
    'scoped_source/shared_build.py','scoped_source/shared_archive.py',SHARED_PROTOCOL_FILE,
    'scoped_source/direct_supervised.py', 'scoped_source/direct_lifecycle_worker.py',
    'scoped_source/direct_lifecycle.py', 'scoped_source/direct_lifecycle_execution.py',
    'scoped_source/direct_source_build.py', 'scoped_source/scaled_references.py',
    'scoped_source/affine_references.py', 'scoped_source/source_dag.py',
    'scoped_source/hz_store_source.py', 'scoped_source/hz_store_write.py', 'scoped_source/hz_store_sparse.py',
    'scoped_source/forest_endpoint_controls.py', 'scoped_proof/io.py', 'scoped_proof/owned_bounded.py',
    PROTOCOL_FILE))))


def bindings():
    result = {}
    for name in FILES:
        p = ROOT / name
        require(not p.is_symlink() and p.is_file() and p.stat().st_size <= 128 * 2**10, 'bounded implementation')
        result[name] = digest(p.read_bytes())
    require(result[PROTOCOL_FILE] == PROTOCOL, 'frozen lifecycle protocol')
    require(result[SHARED_PROTOCOL_FILE] == SHARED_PROTOCOL, 'frozen shared-source protocol')
    return result


def spec(case, arm, profile=0, fault='', *, shared=False):
    require(case in ('half', 'guarded') and fault in FAULTS, 'frozen synthetic case/fault')
    return dict(schema='DIRECT_SUPERVISED_STUB_V1', case=case, method=method(arm, profile,shared=shared), fault=fault)


def validate_spec(value):
    require(type(value) is dict and set(value) == {'schema', 'case', 'method', 'fault'}, 'closed execution spec')
    m = validate(value['method'])
    expected = spec(value['case'], m['production']['kind'], m['producer']['profile'], value['fault'],
                    shared=m['schema']=='SHARED_LIFECYCLE_METHOD_V1')
    require(identity(value) == identity(expected), 'registered execution spec')
    return expected


def fixed_source(root, case):
    """Same registered half/guarded source literals; avoids old native wrappers."""
    from scoped_source.forest_endpoint_controls import capture
    require(case in ('half', 'guarded'), 'fixed source case')
    affine = lambda rows:dict(kind='affine', rows=rows)
    if case == 'half':
        doc = dict(schema='REQUEST_STORE_SOURCE_V1', lower=['-1'], upper=['1'],
                   router=[affine([['0', []], ['1', []], ['0', []]])],
                   experts=[[affine([['0', [[0, '1']]]]), dict(kind='relu'),
                             affine([['2', [[0, '1']]], ['0', [[0, '1/32']]]])] for _ in range(3)])
    else:
        doc = dict(schema='REQUEST_STORE_SOURCE_V1', lower=['-1'], upper=['1'],
                   router=[affine([['0', [[0, '1']]], ['0', [[0, '-1']]], ['0', []]])],
                   experts=[[affine([['1/4', [[0, str(sign)]]]] +
                                    [['0', [[0, str(F(sign*k, 8))]]] for k in range(1, 5)])]
                            for sign in (1, -1)] + [[affine([['1/4', []]] + [['0', []] for _ in range(4)])]])
    return capture(root, doc)


def read_json(path, cap, expected=None):
    p = Path(path)
    require(not p.is_symlink(), 'no symlink receipt')
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'regular receipt')
        raw = stream.read(cap + 1)
    require(len(raw) <= cap and (expected is None or digest(raw) == expected), 'bounded receipt identity')
    value = decode(raw)
    require(compact(value) == raw, 'canonical receipt')
    return value, digest(raw)


def write_json(path, value, cap):
    raw = compact(value)
    require(len(raw) <= cap, 'bounded output')
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return digest(raw)


def census(root, deadline=None):
    """All retained files, including fixture and failed prefixes; no followlinks."""
    total = 0
    result = {}
    hashes = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('storage reception deadline')
        for name in dirs + files:
            p = Path(directory) / name
            require(not p.is_symlink(), 'request symlink')
        for name in files:
            p = Path(directory) / name
            require(stat.S_ISREG(p.stat(follow_symlinks=False).st_mode), 'request regular file')
            size = p.stat().st_size
            result[str(p.relative_to(root))] = size
            total += size
            require(len(result) <= POLICY['members'] and total <= POLICY['total_bytes'], 'whole request byte/member ceiling')
            h = hashlib.sha256()
            fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as stream:
                require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'census regular file')
                count = 0
                while True:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError('storage hashing deadline')
                    raw = stream.read(min(65536, size - count + 1))
                    if not raw:
                        break
                    count += len(raw)
                    require(count <= size, 'census growing file')
                    h.update(raw)
            require(count == size, 'census changed file')
            hashes[str(p.relative_to(root))] = h.hexdigest()
    return dict(bytes=total, files=result, hashes=hashes)


def receive(root, inv, report_hash, work):
    """Trusted worker identity + independent complete aggregation, not another LP proof."""
    work.tick()
    require(bindings() == inv['implementations'], 'unchanged execution/checker implementation')
    p, _ = read_json(root / 'candidate.json', REPORT_CAP, report_hash)
    m = inv['spec']['method']
    require(p['invocation'] == inv['id'] and p['method'] == m and p['deadline'] == inv['run_deadline']
            and p['status'] == 'COMPLETE' and p['initial_work'] == inv['initial_work'], 'complete pending worker binding')
    require(p['isolated'] is True and p['site_disabled'] is True and p['numerical_imports'] == [], 'stdlib worker isolation')
    require(set(p['work']) == set(COUNTERS) and all(type(v) is int and v >= inv['initial_work'][k]
            for k,v in p['work'].items()) and sum(p['work'].values()) <= POLICY['visits'], 'cumulative Work ceiling')
    for k in COUNTERS:
        require(inv['initial_work'][k] + sum(row['work'][k] for row in p['phases']) == p['work'][k], 'phase Work reconciliation')
    require(all(row['complete'] and row['seconds'] >= 0 for row in p['phases'])
            and sum(row['seconds'] for row in p['phases']) <= p['seconds'], 'complete phase cost envelope')
    life = p['snapshots']['lifecycle']
    require(life['setup_calls'] == life['close_calls'] == 1
            and p['snapshots']['owner']['live_bytes'] == 0, 'proposer and checker cleanup')
    artifact = root / 'artifacts'
    refs, total = inventory(artifact / 'package', p['package']['sha256'], work, schema=BUNDLE)
    manifest, _ = read_json(artifact / 'package/manifest.json', 2**20, refs['manifest.json']['sha256'])
    require(manifest['method'] == m and manifest['invocation'] == inv['id'] and
            manifest['source']['sha256'] == p['source'], 'package execution binding')
    for ref in manifest['code']:
        require(ref['sha256'] == inv['implementations'][ref['file'].removeprefix('code/')], 'trusted checker code')
    shared=m['schema']=='SHARED_LIFECYCLE_METHOD_V1'
    if shared:
        require(not (root/'source').exists() and p['snapshots']['source_transfer']['copies']==1,
                'single owned source after transfer')
    source_path=artifact/'package/source/manifest.json' if shared else root/'source/manifest.json'
    source, source_hash = read_json(source_path, POLICY['header_bytes'])
    require(source_hash == p['source'], 'fixture source identity')
    result, _ = read_json(artifact / 'result.json', REPORT_CAP, p['result_ref']['sha256'])
    terminal, _ = read_json(artifact / 'terminal.json', 8192, p['terminal_ref']['sha256'])
    require(result == p['result'] == p['replay']['result'] and terminal['status'] == 'PENDING_RECEPTION'
            and terminal['method'] == m and terminal['invocation'] == inv['id']
            and terminal['result'] == p['result_ref'] and terminal['package'] == p['package'], 'pending terminal/result equality')
    from itertools import combinations
    request = source['declaration']['request']
    expected = [(list(pair), c) for pair in combinations(range(request['experts']), 2)
                for c in range(request['classes']) if c != request['label']]
    require([(r['pair'], r['other']) for r in result['results']] == expected
            and result['required'] == len(expected), 'all pair/property obligations')
    from scoped_source.forest_endpoints import THRESHOLD
    positive = sum(r['lower'] is not None and F(r['lower']) > THRESHOLD for r in result['results'])
    require(all(r['positive'] is (r['lower'] is not None and F(r['lower']) > THRESHOLD)
            for r in result['results']) and result['positive'] == positive, 'exact aggregation sign')
    status = ('CHECKED_POSITIVE_CAPTURED_REAL_SOURCE' if positive == len(expected) else
              'UNKNOWN_MISSING_EVIDENCE' if result['missing_endpoints'] else 'UNKNOWN_NONPOSITIVE')
    require(result['status'] == status and result['checked_endpoints'] + result['missing_endpoints'] == result['required_endpoints']
            and result['native_float_proof'] is False, 'complete result guarantee/counts')
    require(p['replay']['method'] == m and p['replay']['consumer'] == m['replay'], 'actual replay consumer')
    require(p['snapshots']['consumer'] == m['production'], 'actual production consumer')
    require(p['archive']['logical_bytes'] == sum(v for k,v in census(root, work.deadline)['files'].items()
                                               if k.startswith('artifacts/')), 'artifact byte accounting')
    work.tick()
    return dict(result=result, package_sha256=p['package']['sha256'], source=p['source'],
                method=m, invocation=inv['id'], package_bytes=total, worker_work=p['work'],
                snapshots=p['snapshots'], phases=p['phases'], worker_seconds=p['seconds'],
                fixture_seconds=p['fixture_seconds'])


def supervise(root, specification, *, enabled=False, seconds=300., initial_work=None):
    start = time.monotonic()
    require(enabled is True, 'direct supervision is opt-in')
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 300, 'at-most-300s request')
    specification = validate_spec(specification)
    counts = dict.fromkeys(COUNTERS, 0) if initial_work is None else dict(initial_work)
    require(set(counts) == set(COUNTERS) and all(type(v) is int and v >= 0 for v in counts.values())
            and sum(counts.values()) <= POLICY['visits'], 'initial Work budget')
    deadline = start + seconds
    reserve = min(1., seconds / 5)
    run_deadline, cleanup_deadline = deadline - reserve, deadline - reserve / 2
    root = Path(root)
    require(root.is_absolute() and '..' not in root.parts and root.resolve(strict=False) == root
            and root.is_relative_to('/data1/Kane/MOE')
            and root != Path('/data1/Kane/MOE'), 'new project-local request directory')
    require(all(not p.is_symlink() for p in (root, *root.parents)), 'request ancestor symlink')
    root.mkdir(parents=False, exist_ok=False)
    inv = dict(schema='DIRECT_SUPERVISED_INVOCATION_V1', id=identity(dict(nonce=uuid.uuid4().hex)),
               spec=specification, start=start, deadline=deadline, run_deadline=run_deadline,
               cleanup_deadline=cleanup_deadline, initial_work=counts, implementations=bindings())
    inv_hash = write_json(root / 'invocation.json', inv, LOG_CAP)
    child = accepted = report_hash = None
    error = None
    status = 'ERROR'
    counters = None
    try:
        command = [PYTHON, '-I', '-S', '-B', '-c',
                   "import sys;sys.path.insert(0," + repr(str(ROOT)) + ");from scoped_source.direct_lifecycle_worker import main;main()",
                   str(root), inv_hash]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        child = run_owned(command, root / 'worker.log', run_deadline=run_deadline,
                          cleanup_deadline=cleanup_deadline, env=env, rss_limit=RSS_LIMIT)
        if (root / 'worker.log').exists():
            require((root / 'worker.log').stat().st_size <= LOG_CAP, 'bounded worker log')
        else:
            require(child['pid'] is None, 'launched worker log absent')
        if child['status'] != 'COMPLETED' or child['cleanup_status'] != 'LEADER_REAPED_NO_LIVE_GROUP':
            status = 'CLEANUP_INCOMPLETE' if child['cleanup_status'] not in ('LEADER_REAPED_NO_LIVE_GROUP', 'NO_PROCESS') else child['status']
        else:
            pending, report_hash = read_json(root / 'candidate.json', REPORT_CAP)
            counters = pending.get('work')
            if pending.get('status') != 'COMPLETE':
                require(pending.get('status') in ('ERROR', 'TIMEOUT'), 'worker failure status')
                status, error = pending['status'], pending.get('error')
            else:
                work = Work(deadline)
                work.counts.update(pending['work'])
                if specification['fault'] == 'receive_expired':
                    while time.monotonic() < deadline:
                        time.sleep(min(.01, max(0., deadline - time.monotonic())))
                accepted = receive(root, inv, report_hash, work)
                counters = dict(work.counts)
                status = 'COMPLETE'
        storage = census(root, deadline)
        require(storage['bytes'] <= POLICY['total_bytes'] - REPORT_CAP
                and len(storage['files']) < POLICY['members'], 'reserved final terminal bytes/members')
    except BaseException as exc:
        accepted = None
        status = ('CLEANUP_INCOMPLETE' if child is not None and child['cleanup_status'] not in
                  ('LEADER_REAPED_NO_LIVE_GROUP', 'NO_PROCESS') else 'TIMEOUT' if isinstance(exc, TimeoutError) else 'ERROR')
        error = type(exc).__name__ + ': ' + str(exc)[:500]
    # Postmortem cost is retained but can never rescue deadline failure.
    if specification['fault'] == 'terminal_expired':
        while time.monotonic() < deadline:
            time.sleep(min(.01, max(0., deadline - time.monotonic())))
    if time.monotonic() >= deadline:
        status, accepted = ('CLEANUP_INCOMPLETE' if status == 'CLEANUP_INCOMPLETE' else 'TIMEOUT'), None
    receipt = dict(schema='DIRECT_SUPERVISED_RECEIPT_V1', status=status, invocation=inv['id'],
                   invocation_sha256=inv_hash, method=specification['method'], candidate_sha256=report_hash,
                   accepted=accepted, error=error, child=child, work=counters,
                   work_complete=counters is not None and status == 'COMPLETE',
                   seconds_before_publication=time.monotonic() - start, budget=seconds)
    receipt_hash = write_json(root / 'receipt.json', receipt, REPORT_CAP)
    receipt_read, _ = read_json(root / 'receipt.json', REPORT_CAP, receipt_hash)
    require(receipt_read == receipt, 'final receipt equality')
    storage = census(root)
    end = time.monotonic()
    if end >= deadline:
        status, accepted = ('CLEANUP_INCOMPLETE' if status == 'CLEANUP_INCOMPLETE' else 'TIMEOUT'), None
    return dict(status=status, accepted=accepted, receipt_sha256=receipt_hash,
                invocation=inv['id'], seconds=end - start, budget=seconds,
                storage=storage, child=child, work=counters,
                cleanup_after_deadline=max(0., end - deadline), root=str(root))


def audit(root, call):
    """Requires the actual returned call, not a success-looking file alone."""
    root = Path(root)
    require(type(call) is dict and call['root'] == str(root), 'completed call anchor')
    r, _ = read_json(root / 'receipt.json', REPORT_CAP, call['receipt_sha256'])
    inv, _ = read_json(root / 'invocation.json', LOG_CAP, r['invocation_sha256'])
    validate_spec(inv['spec'])
    require(r['invocation'] == inv['id'] == call['invocation'] and r['method'] == inv['spec']['method']
            and r['budget'] == call['budget'] and r['seconds_before_publication'] <= call['seconds'], 'terminal identity/cost')
    storage = census(root)
    require(storage == call['storage'], 'full retained byte accounting')
    states = ('COMPLETE', 'ERROR', 'TIMEOUT', 'RESOURCE_LIMIT', 'CLEANUP_INCOMPLETE')
    require(call['status'] in states and r['status'] in states and r['child'] == call['child']
            and r['work'] == call['work'], 'all terminal states/child/Work binding')
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    require(all(finite(v) for v in (call['seconds'], call['budget'], r['seconds_before_publication'],
                                    inv['start'], inv['deadline'], inv['run_deadline'], inv['cleanup_deadline']))
            and 0 < call['budget'] <= 300 and inv['start'] < inv['run_deadline'] <= inv['cleanup_deadline'] < inv['deadline']
            and abs(inv['deadline'] - inv['start'] - call['budget']) < 1e-7, 'all terminal clocks')
    require(r['status'] == call['status'] or (call['status'] == 'TIMEOUT' and call['seconds'] >= call['budget']),
            'only late final receipt may downgrade status')
    if r['candidate_sha256'] is not None:
        read_json(root / 'candidate.json', REPORT_CAP, r['candidate_sha256'])
    if r['work'] is not None:
        require(set(r['work']) == set(COUNTERS) and all(type(v) is int and v >= inv['initial_work'][k]
                for k,v in r['work'].items()), 'failure/success cumulative Work types')
    child = r['child']
    if child is not None:
        require(all(finite(child[k]) for k in ('seconds', 'execution_seconds', 'cleanup_seconds', 'sampled_peak_rss'))
                and abs(child['seconds'] - child['execution_seconds'] - child['cleanup_seconds']) < 1e-7
                and child['seconds'] <= r['seconds_before_publication'] <= call['seconds']
                and child['run_deadline'] == inv['run_deadline'] and child['cleanup_deadline'] == inv['cleanup_deadline'],
                'whole execution/cleanup cost envelope')
    if call['status'] == 'COMPLETE':
        require(r['status'] == 'COMPLETE' and call['seconds'] < call['budget'] and r['accepted'] == call['accepted']
                and r['child']['status'] == 'COMPLETED'
                and r['child']['cleanup_status'] == 'LEADER_REAPED_NO_LIVE_GROUP'
                and r['child'] == call['child'] and r['work'] == call['work'], 'timely complete execution')
        require(all(type(v) in (int, float) and math.isfinite(v) for v in
                    (call['seconds'], call['budget'], inv['start'], inv['deadline'], inv['run_deadline'], inv['cleanup_deadline']))
                and 0 < call['seconds'] < call['budget'] <= 300
                and inv['start'] < inv['run_deadline'] <= inv['cleanup_deadline'] < inv['deadline']
                and abs(inv['deadline'] - inv['start'] - call['budget']) < 1e-7,
                'whole execution clock contract')
        require(r['child']['run_deadline'] == inv['run_deadline']
                and r['child']['cleanup_deadline'] == inv['cleanup_deadline']
                and not r['child']['remaining_group']['live'], 'supervisor deadline/cleanup binding')
        # Offline structural rereception, explicitly not a new mathematical proof
        # and never used to upgrade an online timeout or failure.
        checked = receive(root, inv, r['candidate_sha256'], Work(time.monotonic() + 60))
        require(checked == r['accepted'], 'independent archived reception equality')
        require(finite(checked['fixture_seconds']) and finite(checked['worker_seconds'])
                and checked['fixture_seconds'] + checked['worker_seconds'] <= child['execution_seconds']
                and child['returncode'] == 0 and child['sampled_peak_rss'] <= RSS_LIMIT
                and finite(child['exit_observed_at']) and child['exit_observed_at'] < inv['run_deadline'],
                'imports/source/worker/cleanup within charged parent clock')
    else:
        require(call['accepted'] is None, 'failure cannot carry accepted proof')
    return dict(status='PASS', execution_status=call['status'], mathematical_recheck=False, bytes=storage['bytes'])
