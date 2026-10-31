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
    'act/pipeline/moe/architecture_attack.py',
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


def pgd_catalog(models):
    """Twenty-four thousand auxiliary domains, ten per <=300s batch.

    Twenty seconds per domain for fixed20x2PGD. Shared loading is charged once
    per auxiliary batch, never reported as cold single-request verifier cost.
    """
    ids = [m['id'] for m in models]
    require(ids and len(ids) == len(set(ids)), 'unique auxiliary model roster')
    out = []
    for start in range(0, 1000, 10):
        for radius in (2, 8):
            shift = (start//10+(radius == 8)) % len(ids)
            for model in ids[shift:]+ids[:shift]:
                out.append(dict(id=f'pgd_{start:04d}_e{radius}_{model}', model=model,
                    indices=list(range(start, start+10)), radius=radius, mode='pgd'))
    return out


def bound_input(inputs, inputs_catalog, index, radius):
    from architecture_s1_inputs import tensors_at, fingerprint
    sample = inputs_catalog['samples'][index]
    require(sample['index'] == index, 'input catalog order')
    tensors = tensors_at(inputs, index, radius)
    require({k:fingerprint(v) for k,v in tensors.items()} == sample['domains'][str(radius)],
        'all materialized tensors and properties')
    require(int(inputs['labels'][index]) == sample['label'], 'physical label')
    return tensors, sample['label']


def pgd_value(model, query, spec, store, folder, deadline, torch, *, family,
              tensors=None, label=None, inputs=None, inputs_catalog=None):
    from act.pipeline.moe.architecture_attack import attack
    domains = []
    synthetic = spec.get('synthetic_control', False)
    seed = 0 if synthetic else next(m['seed'] for m in spec['models'] if m['id'] == query['model'])
    for index in query['indices']:
        tick(deadline)
        if not synthetic:
            tensors, label = bound_input(inputs, inputs_catalog, index, query['radius'])
        with torch.no_grad():
            prediction = int(model(tensors['center']).argmax(1))
        began = time.monotonic(); local_deadline = min(deadline-1, began+20)
        outcome = attack(model, tensors, radius=query['radius'],
            seed=20261029+100000*seed+10000*(query['radius'] == 8)+index, deadline=local_deadline)
        received = time.monotonic()
        if received >= local_deadline:
            outcome = dict(status='ATTACK_TIMEOUT', replayed=False, completed_fixed_search=False,
                guarantee='NONE', seconds=received-began, formal_SAFE=False)
        record = dict(index=index, label=label, clean_prediction=prediction, result=outcome,
            started_at=began, received_at=received, deadline=local_deadline)
        name = f'pgd_{index:04d}.json.gz'
        h = store.write(folder/name, record)
        domains.append(dict(index=index, file=name, sha256=h, status=outcome['status']))
        store.write(folder/'progress.json.gz', dict(phase='public_pgd', returned=len(domains),
            expected=len(query['indices']), active_index=index, clock=time.monotonic()), replace=True)
    return dict(result=dict(status='AUXILIARY_COMPLETE', evidence_grade='PUBLIC_PGD_ONLY',
        evaluation=dict(mode='pgd', family=family), formal_SAFE=False), family=family,
        domains=domains, clean_correct=None, coverage=None, replayed=False,
        guarantee='auxiliary attack search, not verification')


def audit_pgd(value, folder, query):
    require(value['result']['status'] == 'AUXILIARY_COMPLETE'
        and value['result']['evidence_grade'] == 'PUBLIC_PGD_ONLY', 'PGD cannot certify')
    require([d['index'] for d in value['domains']] == query['indices'], 'all auxiliary domains')
    for item in value['domains']:
        require(item['file'] == f"pgd_{item['index']:04d}.json.gz", 'auxiliary relative file identity')
        path = Path(folder)/item['file']
        require(sha(path) == item['sha256'], 'auxiliary outcome identity')
        record = read_record(path); result = record['result']
        require(record['index'] == item['index'] and result['status'] == item['status'], 'auxiliary domain identity')
        require(result['status'] in ('UNSAFE_REPLAYED', 'NO_REPLAYED_COUNTEREXAMPLE', 'ATTACK_TIMEOUT'),
            'attack failure not positive')
        require(record['started_at'] < record['deadline'] <= query['worker_deadline'], 'same auxiliary deadline')
        if result['status'] == 'UNSAFE_REPLAYED':
            require(result['replayed'] and 'witness' in result and record['received_at'] < record['deadline'],
                'on-time auxiliary full-model replay')
        if result['status'] == 'NO_REPLAYED_COUNTEREXAMPLE':
            require(result['completed_fixed_search'] and result['iterations'] == 40,
                'reported complete fixed attack')


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
        if query['mode']=='primary':
            from architecture_s2_primary import worker as primary_worker
            return primary_worker(folder,query,spec,store)
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
                value = {'positive': 2., 'partial': 2., 'zero': 0., 'wrong': -1., 'attack': .25}[query['case']]
                model = affine([[1. if query['case'] == 'attack' else 0.], [0.]], [value, 0.])
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
            if spec.get('study') == 'S2A':
                require(query['mode'] in ('complete','incomplete') and query['radius']==2,
                    'S2a fixed secondary method/radius')
                from act.pipeline.moe.architecture_s2_evaluation import load_secondary_input
                tensors,label=load_secondary_input(read_record(source['path']),query['index'])
            else:
                inputs = torch.load(source['path'], map_location='cpu', weights_only=True)
                inputs_catalog = load(source['catalog'], source['catalog_sha256'])
                if query['mode'] != 'pgd':
                    tensors, label = bound_input(inputs, inputs_catalog, query['index'], query['radius'])
                    del inputs, inputs_catalog
        phase('model_input_load')
        if query['mode'] == 'pgd':
            arguments = dict(tensors=tensors, label=label) if synthetic else dict(inputs=inputs, inputs_catalog=inputs_catalog)
            value = pgd_value(model, query, spec, store, folder, deadline, torch, family=family, **arguments)
            audit_pgd(value, folder, query)
            phase('public_pgd_and_reception')
            value.update(phases=phases, backend_sources_sha256=digest(environment['sources']))
            h = store.write(folder/'outcome.json.gz', dict(request_sha256=sha(folder/'request.json.gz'),
                clock=time.monotonic(), value=value))
            tick(deadline)
            store.write(folder/'published.json.gz', dict(sha256=h, clock=time.monotonic()))
            tick(deadline)
            return 0
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
        arguments = {}
        if spec.get('study') == 'S2A' and query['mode']=='complete':
            require(spec['complete_configuration']==backend.CONFIG,'S2a common engine configuration')
            arguments['complete_configuration']=spec['complete_configuration']
        result = evaluate(model, tensors, family=family, mode=query['mode'], deadline=deadline, publish=publish, **arguments)
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
            else:
                from act.pipeline.moe.architecture_attack import encode_point, decode_point
                packed = encode_point(point, tensors['lower'], tensors['upper'])
                require(torch.equal(point, decode_point(packed, tensors['lower'], tensors['upper'])),
                    'lossless original-model witness encoding')
                result['witness'] = packed
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
    if query['mode']=='primary':
        from architecture_s2_primary import receive as receive_primary
        return receive_primary(folder,query,spec,value,end)
    require(value['backend_sources_sha256'] == spec['backend_sources_sha256'], 'received engine identity')
    require(result['evaluation']['mode'] == query['mode']
        and result['evaluation']['family'] == value['family'], 'received evaluation mode')
    if not spec.get('synthetic_control', False):
        item = next(m for m in spec['models'] if m['id'] == query['model'])
        require(value['family'] == item['family'], 'received family')
    if spec.get('study') == 'S2A' and query['mode']=='complete':
        require(result['evaluation'].get('explicit_complete_configuration')==spec['complete_configuration'],
                'received common S2a configuration')
        require(all(row['outcome'].get('branching')==spec['complete_configuration']['bab.branching.method']
                    for row in result['histories']), 'received branch heuristic')
    if query['mode'] == 'pgd':
        audit_pgd(value, folder, query)
        tick(end)
        return result['status'], value
    if not spec.get('synthetic_control', False):
        require(value['properties'] == 9, 'received properties')
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
    python=backend.ACT if request['mode']=='primary' else str(backend.ENVIRONMENT/'bin/python')
    session = execute([python, '-I', '-B', str(Path(__file__).resolve()),
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


def measured_invoke(root, request, manifest, *, seconds, timing_reserve):
    began=time.monotonic()
    try:
        return invoke(root,request,manifest,seconds=seconds,timing_reserve=timing_reserve)
    except Exception as exc:
        # Includes failures before a child/session can be created. These are
        # charged attempts, not zero-time successes or dropped inputs.
        return dict(id=request['id'],status='ERROR',accepted=False,seconds=time.monotonic()-began,
            controller_error=dict(kind=type(exc).__name__,error=str(exc)))


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
                task = pool.submit(measured_invoke, root, query, manifest, seconds=seconds, timing_reserve=timing_reserve)
                pending[task] = query
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for task in sorted(done, key=lambda f: pending[f]['id']):
                query = pending.pop(task)
                try:
                    row = task.result()
                except Exception as exc:
                    row = dict(id=query['id'], status='ERROR', accepted=False,
                        controller_error=dict(kind=type(exc).__name__, error=str(exc)))
                # Terminal and raw evidence already contain the full record.
                # Ledger references them without duplicating per-request data.
                ledger = ({k:row[k] for k in ('id','status','accepted','seconds','terminal_sha256')}
                    if 'terminal_sha256' in row else row)
                if row.get('terminal_overrun'):
                    ledger['terminal_overrun'] = True
                store.write(store.root/(query['id']+'.ledger.json.gz'), ledger, terminal=True)
                rows.append(row)
                if row['status'] == 'ERROR' or row.get('terminal_overrun'):
                    stopped = True
    summary = dict(status='PREFIX_STOP' if stopped else 'COMPLETE', requested=len(requests),
        launched=next_index, settled=len(rows), not_executed=len(requests)-next_index,
        states={s:sum(r['status'] == s for r in rows)
            for s in ('POSITIVE', 'UNSAFE_REPLAYED', 'UNKNOWN', 'TIMEOUT', 'ERROR', 'AUXILIARY_COMPLETE')},
        summed_request_seconds=sum(r.get('seconds', 0) for r in rows),
        missing_request_cost=sum('seconds' not in r for r in rows),
        queue_wall_seconds=time.monotonic()-started, workers=workers,
        request_catalog_sha256=digest(requests), source_manifest_sha256=sha(manifest))
    store.write(store.root/'queue_complete.json.gz', summary, terminal=True)
    return summary


if __name__ == '__main__':
    require(sys.argv[1] == 'worker', 'request workers only; no unfrozen batch CLI')
    raise SystemExit(worker(Path(sys.argv[2])))
