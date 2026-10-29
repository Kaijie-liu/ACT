"""S1 request execution, shared artifact accounting and finite queue order.

Uses the existing owned-session supervisor. No real evaluation is launched by
this module's controller API until its caller supplies a frozen manifest.
"""
import time
ENTRY = time.monotonic()
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scoped_proof.io import load, sha, save, tick
from scoped_proof.owned_session import execute
import genbab_cpu_ready as backend

CAP = 128*2**20
THREAD_ENV = dict(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
FILES = ('scripts/architecture_s1_execution.py', 'scripts/architecture_s1_inputs.py',
    'act/pipeline/moe/architecture_evaluation.py', 'act/pipeline/moe/architecture_dense.py',
    'act/pipeline/moe/multilayer_training.py', 'act/back_end/moe/multilayer.py',
    'act/back_end/moe/multilayer_external.py', 'act/pipeline/moe/multilayer_study.py',
    'scripts/b2_observed_worker.py', 'scripts/genbab_cpu_ready.py',
    'scripts/multilayer_stage1.py', 'scoped_proof/owned_session.py', 'scoped_proof/io.py')


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_record(path):
    path = Path(path)
    require(not path.is_symlink(), 'record must not be a symlink')
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


class Artifacts:
    """One stage-wide locked byte account; no per-worker reset or allowance.

    Reserve before writing, so interrupted writes leave conservative charges.
    Replacement is only for live telemetry: its old and new buffers both count
    during publication. Failed evidence and all terminal records are immutable.
    """
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.account = self.root/'artifact_account.json'

    @classmethod
    def create(cls, root, *, initial_bytes=0, cap=CAP, terminal_reserve=512*1024):
        root = Path(root); root.mkdir()
        require(0 <= initial_bytes+4096 < cap-terminal_reserve, 'stage artifact space')
        save(root/'artifact_account.json', dict(cap=cap, used=initial_bytes+4096,
            initial_bytes=initial_bytes, terminal_reserve=terminal_reserve,
            accounting='stored bytes plus outstanding reservations; account/lock reserve4096'))
        return cls(root)

    def write(self, path, value, *, replace=False, terminal=False):
        path = Path(path)
        require(path.parent.resolve().is_relative_to(self.root), 'owned artifact directory')
        require(not path.is_symlink(), 'no artifact alias')
        require(not replace or path.name == 'progress.json.gz', 'only live progress is replaceable')
        raw = encoded(value)
        if path.suffix == '.gz':
            raw = gzip.compress(raw, mtime=0)
        temporary = path.with_name(path.name+'.partial')
        with self.account.open('r+') as account:
            fcntl.flock(account, fcntl.LOCK_EX)
            state = json.load(account)
            require(replace or not path.exists(), 'immutable record exists')
            require(not temporary.exists(), 'preserve interrupted publication')
            previous = path.stat().st_size if path.exists() else 0
            limit = state['cap'] if terminal else state['cap']-state['terminal_reserve']
            if state['used']+len(raw) > limit:
                raise BufferError('shared stage artifact cap')
            def update():
                account.seek(0); json.dump(state, account, separators=(',', ':'))
                account.truncate(); account.flush(); os.fsync(account.fileno())
            state['used'] += len(raw); update()
            with temporary.open('xb') as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
            if replace:
                os.replace(temporary, path)
            else:
                os.link(temporary, path); temporary.unlink()
            state['used'] -= previous; update()
        return hashlib.sha256(raw).hexdigest()


def catalog(models, mode):
    """Fixed data-order prefix, both radii, all seeds; rotated model order.

    No accuracy/bound/route based choice. A full request, never a history, is
    the unit of execution. Incomplete items retain their separate denominator.
    """
    require(mode in ('complete', 'incomplete'), 'evaluation queue mode')
    ids = [m['id'] for m in models]
    require(len(ids) == len(set(ids)) and ids, 'unique model roster')
    count = 200 if mode == 'complete' else 1000
    result = []
    for index in range(count):
        for radius in (2, 8):
            shift = (index + (radius == 8)) % len(ids)
            for name in ids[shift:]+ids[:shift]:
                result.append(dict(id=f'{mode}_{index:04d}_e{radius}_{name}',
                    model=name, index=index, radius=radius, mode=mode))
    return result


def audit_value(result, *, family, experts, properties):
    from act.pipeline.moe.multilayer_study import coverage
    expected = math.prod(math.comb(e, 2) for e in experts) if family == 'moe' else 1
    require(result['expected_histories'] == expected, 'full structural catalog')
    require(result['status'] in ('POSITIVE', 'UNKNOWN', 'UNSAFE_REPLAYED', 'TIMEOUT'), 'returned status')
    cov = coverage(result, 'genbab_pruned', expected)
    if family == 'moe':
        from multilayer_stage1 import external_audit
        external_audit(result, experts, properties)
    else:
        require(not result['prefix_bounds'] and not result['exclusions'], 'dense has no route pruning')
        require(len(result['histories']) == result['returned_histories'], 'dense returned catalog')
        raw_spec = [[[[float(j == i) for j in range(properties)]], [0.]] for i in range(properties)]
        for history in result['histories']:
            require(history['history'] == [], 'dense single graph')
            require(encoded(history['outcome']['actual_raw_spec']) == encoded(raw_spec),
                'all dense output properties')
        require(result['verified_histories'] == sum(h['outcome']['status']=='verified'
            for h in result['histories']), 'dense positive count')
    if result['status'] == 'POSITIVE':
        require(cov['proof_complete'] and result['complete'], 'no partial positive')
    if result['status'] == 'TIMEOUT':
        require(not result['complete'] and result['evidence_grade'] == 'NONE', 'timeout is not a certificate')
    return cov


def worker(folder):
    folder = Path(folder); store = Artifacts(folder.parent)
    query = read_record(folder/'request.json.gz'); deadline = query['worker_deadline']
    fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(fd, 1); os.dup2(fd, 2); os.close(fd)
    phases = {}; start = time.monotonic()
    def phase(name):
        nonlocal start
        now = time.monotonic(); phases[name] = now-start; start = now; tick(deadline)
    try:
        backend.limits(deadline)
        spec = load(query['manifest'], query['manifest_sha256'])
        require(all(sha(ROOT/p) == h for p, h in spec['source_files'].items()), 'frozen worker source')
        synthetic = spec.get('synthetic_control', False)
        if synthetic and query.get('case') in ('exception', 'deadline'):
            store.write(folder/'progress.json.gz', dict(phase='synthetic_fault', clock=time.monotonic()), replace=True)
            if query['case'] == 'exception':
                raise RuntimeError('expected synthetic exception')
            while True:
                time.sleep(.05)  # Existing native hard deadline must terminate this child.
        environment = backend.probe()
        require(digest(environment['sources']) == spec['backend_sources_sha256'], 'frozen external engine')
        from act.util.typing_compat import install_typing_override
        install_typing_override()
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float64)
        phase('startup_identity_engine')
        from architecture_s1_inputs import tensors_at, fingerprint
        if synthetic:
            from multilayer_stage0 import fixture, affine
            family = 'moe' if query['case'] == 'tie' else 'dense'
            if family == 'moe':
                model, tensors = fixture('tie')
            else:
                value = {'positive': 2., 'partial': 2., 'zero': 0., 'wrong': -1.}[query['case']]
                model = affine([[0.], [0.]], [value, 0.])
                x = torch.zeros((1, 1), dtype=torch.float64)
                tensors = dict(center=x, lower=x-1, upper=x+1,
                    rows=torch.tensor([[1., -1.]]), thresholds=torch.zeros(1))
            label = 0
        else:
            item = next(m for m in spec['models'] if m['id'] == query['model'])
            family = item['family']
            require(sha(item['checkpoint']) == item['checkpoint_sha256'], 'bound model weights')
            if family == 'dense':
                from act.pipeline.moe.architecture_dense import load_checkpoint
            else:
                from act.pipeline.moe.multilayer_training import load_checkpoint
            model, payload = load_checkpoint(item['checkpoint'])
            require(payload['architecture'] == item['architecture'] and payload['seed'] == item['seed'],
                'bound model architecture and seed')
            model = model.double().eval()
            source = spec['inputs']
            require(sha(source['path']) == source['sha256'], 'same physical input file')
            inputs = torch.load(source['path'], map_location='cpu', weights_only=True)
            inputs_catalog = load(source['catalog'], source['catalog_sha256'])
            sample = inputs_catalog['samples'][query['index']]
            require(sample['index'] == query['index'], 'input catalog order')
            tensors = tensors_at(inputs, query['index'], query['radius'])
            require({k:fingerprint(v) for k,v in tensors.items()} == sample['domains'][str(query['radius'])],
                'all materialized tensors and properties')
            label = sample['label']
            require(int(inputs['labels'][query['index']]) == label, 'physical label')
            del inputs, inputs_catalog
        phase('model_input_load')
        experts = [len(layer.experts) for layer in model.layers] if family == 'moe' else []
        expected = math.prod(math.comb(e, 2) for e in experts) if experts else 1
        from act.pipeline.moe.multilayer_study import coverage
        from act.pipeline.moe.architecture_evaluation import evaluate
        def publish(value):
            tick(deadline)
            store.write(folder/'progress.json.gz', dict(clock=time.monotonic(),
                coverage=coverage(value, 'genbab_pruned', expected)), replace=True)
        with torch.no_grad():
            prediction = int(model(tensors['center']).argmax(1))
        result = evaluate(model, tensors, family=family, mode=query['mode'], deadline=deadline, publish=publish)
        phase('verification')
        cov = audit_value(result, family=family, experts=experts, properties=len(tensors['rows']))
        replayed = False
        if result['status'] == 'UNSAFE_REPLAYED':
            from act.back_end.moe.class_separated_top1 import validate_replay
            point = torch.tensor(result['witness'], dtype=torch.float64)
            replayed = bool(validate_replay(model, point, tensors['lower'], tensors['upper'],
                tensors['rows'], tensors['thresholds']))
            require(replayed, 'full same-model replay')
            if torch.equal(point, tensors['center']):
                result['witness'] = dict(reference='bound input center', sha256=fingerprint(point))
        phase('coverage_and_replay')
        value = dict(result=result, coverage=cov, replayed=replayed, family=family,
            experts=experts, properties=len(tensors['rows']), clean_prediction=prediction,
            label=label, clean_correct=prediction == label, phases=phases,
            backend_sources_sha256=digest(environment['sources']))
        tick(deadline)
        result_sha = store.write(folder/'outcome.json.gz', dict(request_sha256=sha(folder/'request.json.gz'),
            clock=time.monotonic(), value=value))
        if synthetic and query.get('case') == 'partial':
            raise RuntimeError('expected incomplete publication after candidate output')
        tick(deadline)
        store.write(folder/'published.json.gz', dict(sha256=result_sha, clock=time.monotonic()))
        tick(deadline)
        return 0
    except BaseException as exc:
        store.write(folder/'error.json.gz', dict(kind=type(exc).__name__, error=str(exc),
            traceback=traceback.format_exc(), phases=phases, clock=time.monotonic()), terminal=True)
        return 1


def receive(folder, query, session, end):
    """Online acceptance; late/partial evidence never supplies a positive."""
    from b1_observed_batch import session_ok
    folder = Path(folder)
    if not session_ok(session) or not (folder/'published.json.gz').is_file():
        timeout = (session.get('returncode') == 124 or 'TIMEOUT' in session['status']
            or time.monotonic() >= query['worker_deadline'])
        return ('TIMEOUT' if timeout else 'ERROR'), None
    publication = read_record(folder/'published.json.gz')
    require(publication['sha256'] == sha(folder/'outcome.json.gz'), 'published outcome binding')
    outcome = read_record(folder/'outcome.json.gz')
    require(outcome['request_sha256'] == sha(folder/'request.json.gz'), 'request/result identity')
    require(read_record(folder/'request.json.gz') == query, 'receiver request identity')
    if not (outcome['clock'] <= publication['clock'] < query['worker_deadline'] and time.monotonic() < end):
        return 'TIMEOUT', None
    value = outcome['value']; result = value['result']
    spec = load(query['manifest'], query['manifest_sha256'])
    require(value['backend_sources_sha256'] == spec['backend_sources_sha256'], 'received engine identity')
    require(result['evaluation']['mode'] == query['mode']
        and result['evaluation']['family'] == value['family'], 'received evaluation mode')
    if not spec.get('synthetic_control', False):
        item = next(m for m in spec['models'] if m['id'] == query['model'])
        require(value['family'] == item['family'] and value['properties'] == 9, 'received family and properties')
    cov = audit_value(result, family=value['family'], experts=value['experts'], properties=value['properties'])
    require(cov == value['coverage'], 'received complete coverage')
    if result['status'] == 'UNSAFE_REPLAYED':
        require(value['replayed'] is True and 'witness' in result, 'received full-model replay')
    tick(end)
    return result['status'], value


def invoke(root, request, manifest, *, seconds=300, timing_reserve=(12, 10, 5)):
    """Thread-safe per-request launch; caller serializes stage ledger updates."""
    began = time.monotonic(); end = began+seconds
    wt, rt, ct = timing_reserve
    require(0 < ct < rt < wt < seconds, 'nested full-request deadlines')
    require(Path(request['id']).name == request['id'], 'request directory identity')
    store = Artifacts(root); folder = store.root/request['id']; folder.mkdir()
    query = dict(request, manifest=str(Path(manifest).resolve()), manifest_sha256=sha(manifest),
        worker_deadline=end-wt, started_at=began, hard_deadline=end)
    store.write(folder/'request.json.gz', query)
    env = dict(os.environ, **THREAD_ENV)
    session = execute([str(backend.ENVIRONMENT/'bin/python'), '-I', '-B', str(Path(__file__).resolve()),
        'worker', str(folder)], run_deadline=end-rt, cleanup_deadline=end-ct,
        env=env, rss_limit=None, development_policy=backend.POLICY)
    store.write(folder/'session.json.gz', session, terminal=True)
    error = None
    try:
        status, value = receive(folder, query, session, end)
    except Exception as exc:
        status, value, error = 'ERROR', None, dict(kind=type(exc).__name__, error=str(exc))
    partial = (read_record(folder/'progress.json.gz')
        if value is None and (folder/'progress.json.gz').is_file() else None)
    row = dict(id=request['id'], status=status, accepted=value is not None,
        guarantee=None if value is None else value['result']['evidence_grade'],
        coverage=None if value is None else value['coverage'],
        clean_correct=None if value is None else value['clean_correct'], partial=partial,
        session_status=session['status'], returncode=session['returncode'], error=error,
        received_at=time.monotonic(), started_at=began, hard_deadline=end,
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    if time.monotonic() >= end:
        row.update(status='TIMEOUT', accepted=False, guarantee=None)
    store.write(folder/'terminal.json.gz', row, terminal=True)
    row.update(seconds=time.monotonic()-began, terminal_sha256=sha(folder/'terminal.json.gz'))
    if row['seconds'] >= seconds:
        row.update(status='TIMEOUT', accepted=False, guarantee=None, terminal_overrun=True)
    return row


def finite_queue(root, requests, manifest, *, workers, seconds=300, timing_reserve=(12, 10, 5)):
    """One finite queue; parallel workers never write the shared ledger.

    The caller freezes the roster/resources first. Errors stop new admission;
    already launched calls finish within their own budgets and remain recorded.
    This does not retry failures or silently skip them on a later invocation.
    """
    require(type(workers) is int and 1 <= workers <= 8, 'at most eight request workers')
    ids = [r['id'] for r in requests]
    require(ids and len(ids) == len(set(ids)), 'finite unique request roster')
    spec = load(manifest)
    if not spec.get('synthetic_control', False):
        require(ROOT != Path('/data1/Kane/MOE/ACT') and (ROOT/'.frozen_commit').is_file(),
            'real queue requires immutable execution copy')
        require(seconds == 300 and list(timing_reserve) == spec['timing_reserve'], 'frozen request timing')
        require(workers == spec['workers'] and spec['request_catalog_sha256'] == digest(requests),
            'frozen worker count and complete queue')
    store = Artifacts(root); started = time.monotonic(); rows = []; next_index = 0; stopped = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        while pending or (not stopped and next_index < len(requests)):
            while not stopped and len(pending) < workers and next_index < len(requests):
                query = requests[next_index]; next_index += 1
                task = pool.submit(invoke, root, query, manifest, seconds=seconds, timing_reserve=timing_reserve)
                pending[task] = query
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for task in sorted(done, key=lambda f: pending[f]['id']):
                query = pending.pop(task)
                try:
                    row = task.result()
                except Exception as exc:
                    row = dict(id=query['id'], status='ERROR', accepted=False,
                        controller_error=dict(kind=type(exc).__name__, error=str(exc)))
                store.write(store.root/(query['id']+'.ledger.json.gz'), row, terminal=True)
                rows.append(row)
                if row['status'] == 'ERROR' or row.get('terminal_overrun'):
                    stopped = True
    summary = dict(status='PREFIX_STOP' if stopped else 'COMPLETE', requested=len(requests),
        launched=next_index, settled=len(rows), not_executed=len(requests)-next_index,
        states={s:sum(r['status'] == s for r in rows)
            for s in ('POSITIVE', 'UNSAFE_REPLAYED', 'UNKNOWN', 'TIMEOUT', 'ERROR')},
        summed_request_seconds=sum(r.get('seconds', 0) for r in rows),
        missing_request_cost=sum('seconds' not in r for r in rows),
        queue_wall_seconds=time.monotonic()-started, workers=workers,
        request_catalog_sha256=digest(requests), source_manifest_sha256=sha(manifest))
    store.write(store.root/'queue_complete.json.gz', summary, terminal=True)
    return summary


if __name__ == '__main__':
    require(sys.argv[1] == 'worker', 'request workers only; no unfrozen batch CLI')
    raise SystemExit(worker(Path(sys.argv[2])))
