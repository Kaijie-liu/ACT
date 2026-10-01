"""Versioned given-HZ CPU support supervision. No production/GPU admission."""
from fractions import Fraction
import math
import os
from pathlib import Path
import time
import uuid

from scoped_proof.io import ROOT, PYTHON, load, save, sha, tick
from scoped_proof.supervisor import execute
from source_enclosure.format import identity

CONFIG = 'configs/hz_batch_supervision_20261001.json'
PROTOCOL_SHA = '8b3c1312e2f64fd993eddb537be0801877a7ccd012676d5ddd45854c256ffc45'
DONE = 'CHECKED_GIVEN_HZ_SUPPORT_EXECUTION'
PHASES = ('produce', 'check', 'receive')
FILES = (CONFIG, 'scripts/hz_batch_support_supervised.py', 'scripts/hz_batch_support_worker.py',
         'act/back_end/moe/batched_support.py', 'act/back_end/moe/check_batched_support.py',
         'act/back_end/solver/hz_lp_export.py', 'act/back_end/solver/check_hz_lp_export.py',
         'act/back_end/solver/lp_certificate.py', 'act/back_end/solver/solver_hz.py',
         'scoped_source/rowwise_bound.py', 'scoped_source/rowwise_native.py',
         'scoped_proof/io.py', 'scoped_proof/supervisor.py', 'source_enclosure/format.py')
LIMIT = 4 * 2**20


def required_hash(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('mandatory external hash')
    return value


def protocol():
    if sha(ROOT / CONFIG) != PROTOCOL_SHA: raise ValueError('frozen supervision protocol changed')
    cfg = load(ROOT / CONFIG)
    for name, digest in cfg['frozen_dependencies'].items():
        if sha(ROOT / name) != digest: raise ValueError('frozen dependency changed: ' + name)
    return cfg


def sources():
    return {name: sha(ROOT / name) for name in FILES}


def specification(case='guarded_two_sides_positive', control=''):
    cfg = protocol()
    spec = {'schema': 'HZ_BATCH_EXECUTION_SPEC_V1', 'case': case, 'control': control,
            'protocol_sha256': PROTOCOL_SHA, **cfg['cases'][case]}
    validate_spec(spec)
    return spec


def validate_spec(spec):
    cfg = protocol()
    if (spec.get('case') not in cfg['cases'] or spec.get('control') not in ('', *cfg['faults'])
            or spec != {'schema': 'HZ_BATCH_EXECUTION_SPEC_V1', 'case': spec['case'], 'control': spec['control'],
                        'protocol_sha256': PROTOCOL_SHA, **cfg['cases'][spec['case']]}):
        raise ValueError('fixed synthetic batch specification')
    if spec['control'] and spec['case'] != 'guarded_two_sides_positive':
        raise ValueError('faults use the fixed guarded control only')


def bind(root, spec, invocation_sha):
    inv = load(root / 'invocation.json', required_hash(invocation_sha), limit=LIMIT)
    if (inv['spec_sha256'] != identity(spec) or load(root / 'spec.json') != spec
            or inv['sources'] != sources()):
        raise ValueError('caller/implementation identity changed')
    return inv


def receive(root, spec, invocation_sha, payload_sha, stdout_sha, deadline):
    tick(deadline)
    inv = bind(root, spec, invocation_sha)
    payload = load(root / 'payload.json', required_hash(payload_sha), limit=LIMIT)
    checked = load(root / 'check.stdout', required_hash(stdout_sha), limit=LIMIT)
    context = {'invocation': inv['invocation'], 'request_sha256': identity(spec), 'batch_sha256': spec['batch_sha256']}
    if (any(payload.get(k) != v or checked.get(k) != v for k, v in context.items())
            or identity(payload['batch']) != spec['batch_sha256']
            or checked.get('schema') != 'HZ_BATCH_CHECK_OUTPUT_V1'
            or checked.get('payload_sha256') != payload_sha):
        raise ValueError('checker/caller/payload binding')
    result = checked['result']
    if (result['status'] != 'CHECKED_GIVEN_HZ_CONTINUOUS_RELAXATION'
            or result['batch_sha256'] != spec['batch_sha256']
            or result['candidate_sha256'] != identity(payload['candidates'])
            or result['network_or_complete_moe_proof'] is not False
            or result['hard_budget_supervision'] is not False
            or result['n_relaxed_binaries'] != payload['batch']['n_relaxed_binaries']):
        raise ValueError('support guarantee or evidence binding')
    rows = result['results']
    if ([r['id'] for r in rows] != spec['query_ids'] or [r['side'] for r in rows] != spec['sides']):
        raise ValueError('complete caller obligation roster')
    for row in rows:
        required_hash(row['lp_sha256'])
        Fraction(row['bound'])
        if row['bound_kind'] != ('lower' if row['side'] == 'min' else 'upper'):
            raise ValueError('support bound direction')
    accepted = {'schema': 'HZ_BATCH_ACCEPTED_V1', 'status': DONE, **context,
                'payload_sha256': payload_sha, 'checker_stdout_sha256': stdout_sha,
                'required': len(spec['query_ids']), 'result': result,
                'complete_moe_proof': False, 'gpu_execution': False}
    tick(deadline)
    return accepted


def supervise(root, spec, *, expected_request_sha256, budget=300., rss_limit=2*2**30):
    start = time.monotonic()
    if (type(budget) not in (int, float) or not math.isfinite(budget) or not 0 < budget <= 300
            or type(rss_limit) is not int or not 0 < rss_limit <= 2*2**30):
        raise ValueError('bounded execution policy')
    validate_spec(spec)
    if identity(spec) != required_hash(expected_request_sha256): raise ValueError('caller request hash')
    root = Path(root)
    if not root.is_absolute() or not root.resolve().is_relative_to(ROOT.parent / 'baseline_runs'):
        raise ValueError('new project baseline_runs directory required')
    root.mkdir(parents=True, exist_ok=False)
    deadline = start + budget
    work = deadline - min(1., budget / 5)
    token = uuid.uuid4().hex
    save(root / 'spec.json', spec)
    inv_record = save(root / 'invocation.json', {'schema': 'HZ_BATCH_INVOCATION_V1', 'invocation': token,
        'spec_sha256': expected_request_sha256, 'start': start, 'deadline': deadline, 'work_deadline': work,
        'budget': budget, 'rss_limit': rss_limit, 'sources': sources()})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    stages = []
    accepted = None
    status, error = 'ERROR', None
    payload_sha = stdout_sha = None
    try:
        for phase in PHASES:
            bind(root, spec, inv_record['sha256'])
            before = time.monotonic()
            phase_end = work - min(5., budget / 4) if phase == 'produce' else work
            command = [PYTHON, '-B'] + (['-S'] if phase == 'receive' else []) + [
                '-m', 'scripts.hz_batch_support_worker', phase, str(root), '--deadline', str(phase_end),
                '--invocation-sha', inv_record['sha256']]
            if phase != 'produce': command += ['--payload-sha', required_hash(payload_sha)]
            if phase == 'receive': command += ['--stdout-sha', required_hash(stdout_sha)]
            if spec['control'] == 'launch_failure': command[0] = str(root / 'nonexistent-python')
            record = execute(command, root / (phase + '.log'), phase_end, env, rss_limit)
            record.update(phase=phase, executable=command[0], start_seconds=before-start, end_seconds=time.monotonic()-start)
            stages.append(record)  # Charge execution even if its output is absent.
            try:
                if record['status'] == 'COMPLETED' and phase in ('produce', 'check'):
                    path = root / ('payload.json' if phase == 'produce' else 'check.stdout')
                    if path.is_symlink() or path.stat().st_size > LIMIT: raise ValueError('output file admission')
                    digest = sha(path)  # Parent-owned anchor, never the worker's reported SHA.
                    if phase == 'produce': payload_sha = digest
                    else: stdout_sha = digest
                    record['output_sha256'] = digest
            except Exception as exc:
                record['output_reception_error'] = repr(exc)
                raise
            finally:
                record['end_seconds'] = time.monotonic() - start
                if start + record['end_seconds'] >= phase_end:
                    record['worker_status'] = record['status']
                    record['status'] = 'TIMEOUT'
                save(root / (phase + '_stage.json'), record)
            if record['status'] != 'COMPLETED':
                status = record['status']
                break
        else:
            accepted = receive(root, spec, inv_record['sha256'], payload_sha, stdout_sha, work)
            if load(root / 'accepted.json', limit=LIMIT) != accepted: raise ValueError('receiver file disagreement')
            status = DONE
    except Exception as exc:
        error, status, accepted = repr(exc), 'ERROR', None
    status, accepted, charged = terminal_observation(status, accepted, start, work)
    terminal = {'schema': 'HZ_BATCH_TERMINAL_V1', 'invocation': token, 'invocation_sha256': inv_record['sha256'],
                'request_sha256': expected_request_sha256, 'payload_sha256': payload_sha,
                'checker_stdout_sha256': stdout_sha, 'status_before_publication': status,
                'stages': stages, 'accepted': accepted, 'error': error,
                'required': len(spec['query_ids']), 'seconds_before_publication': charged,
                'stage_seconds': sum(r['seconds'] for r in stages),
                'overhead_before_publication_seconds': charged - sum(r['seconds'] for r in stages)}
    terminal_record = save(root / 'terminal.json', terminal)
    receipt_record = save(root / 'receipt.json', {'schema': 'HZ_BATCH_RECEIPT_V1', 'invocation': token,
        'terminal_sha256': terminal_record['sha256'], 'status': status, 'budget': budget,
        'seconds_before_receipt': time.monotonic()-start, 'complete_moe_proof': False})
    elapsed = time.monotonic() - start
    finish = save(root / 'finish.json', {'schema': 'HZ_BATCH_FINISH_V1', 'invocation': token,
        'receipt_sha256': receipt_record['sha256'], 'status': status if elapsed < budget else 'TIMEOUT',
        'seconds_before_finish_marker': elapsed})
    return finish_observation(root,status,budget,start,finish['sha256'],token)


def terminal_observation(status, accepted, start, work):
    """Use one sample for the terminal's acceptance and charged pre-publication cost."""
    charged = time.monotonic() - start
    if start + charged >= work: status, accepted = 'TIMEOUT', None
    return status, accepted, charged


def finish_observation(root, status, budget, start, finish_sha, token):
    """The *same* final sample determines status and reported elapsed time."""
    elapsed = time.monotonic() - start
    if elapsed >= budget:
        status = 'TIMEOUT'
        save(root / 'publication_timeout.json', {'status': status, 'seconds': elapsed})
        elapsed = time.monotonic() - start  # Already TIMEOUT; charge marker publication too.
    return {'status': status, 'seconds': elapsed, 'root': str(root), 'invocation': token,
            'finish_sha256': finish_sha, 'complete_support_execution': status == DONE, 'complete_moe_proof': False}


def audit(root, completed_call=None, *, recheck=False):
    """Recompute record accounting; optional exact recheck, never a new proposal."""
    root = Path(root)
    spec = load(root / 'spec.json')
    validate_spec(spec)
    finish = load(root / 'finish.json', None if completed_call is None else required_hash(completed_call.get('finish_sha256')))
    receipt = load(root / 'receipt.json', required_hash(finish['receipt_sha256']))
    terminal = load(root / 'terminal.json', required_hash(receipt['terminal_sha256']))
    inv = bind(root, spec, terminal['invocation_sha256'])
    if [v.get('schema') for v in (inv, terminal, receipt, finish)] != [
            'HZ_BATCH_INVOCATION_V1', 'HZ_BATCH_TERMINAL_V1', 'HZ_BATCH_RECEIPT_V1', 'HZ_BATCH_FINISH_V1']:
        raise ValueError('terminal schema')
    if (any(v['invocation'] != inv['invocation'] for v in (terminal, receipt, finish))
            or terminal['request_sha256'] != identity(spec) or terminal['required'] != len(spec['query_ids'])
            or receipt['budget'] != inv['budget'] or receipt['complete_moe_proof'] is not False):
        raise ValueError('terminal identity/denominator')
    def finite(v): return type(v) in (int, float) and math.isfinite(v)
    if (not all(finite(inv[k]) for k in ('start','deadline','work_deadline','budget'))
            or not 0 < inv['budget'] <= 300 or type(inv['rss_limit']) is not int
            or not 0 < inv['rss_limit'] <= 2*2**30
            or abs(inv['deadline']-inv['start']-inv['budget']) > 1e-7
            or abs(inv['work_deadline']-(inv['deadline']-min(1.,inv['budget']/5))) > 1e-7):
        raise ValueError('budget contract')
    stages = terminal['stages']
    if [s['phase'] for s in stages] != list(PHASES[:len(stages)]): raise ValueError('stage prefix')
    end = 0.
    for i, s in enumerate(stages):
        if load(root / (s['phase']+'_stage.json')) != s: raise ValueError('stage artifact')
        expected_executable = str(root/'nonexistent-python') if spec['control']=='launch_failure' else PYTHON
        if s.get('executable') != expected_executable: raise ValueError('stage executable binding')
        phase_end = inv['work_deadline'] - min(5.,inv['budget']/4) if s['phase']=='produce' else inv['work_deadline']
        if (not all(finite(s[k]) for k in ('seconds','start_seconds','end_seconds'))
                or not 0 <= end <= s['start_seconds'] <= s['end_seconds']
                or not 0 <= s['seconds'] <= s['end_seconds']-s['start_seconds']+1e-8
                or s.get('deadline_monotonic', phase_end) != phase_end):
            raise ValueError('stage deadline/chronology/cost')
        if s['status'] not in ('COMPLETED','ERROR','TIMEOUT','RESOURCE_LIMIT'):
            raise ValueError('stage status')
        if s['status'] != 'COMPLETED' and i != len(stages)-1: raise ValueError('continued failed stage')
        if s['pid'] is None:
            if not ((s['status']=='TIMEOUT' and s['seconds']==0) or
                    (s['status']=='ERROR' and s.get('cleanup_included') is True and s.get('error'))):
                raise ValueError('unlaunched stage')
            if s['returncode'] is not None or s['sampled_peak_rss'] != 0: raise ValueError('unlaunched cost')
        elif (type(s['pid']) is not int or s['pid'] <= 0 or s.get('cleanup_included') is not True
                or type(s['returncode']) is not int or 'deadline_monotonic' not in s
                or s['status']=='COMPLETED' and s['returncode']!=0):
            raise ValueError('process exit/cleanup')
        if (type(s['sampled_peak_rss']) is not int or s['sampled_peak_rss'] < 0
                or s['status']=='RESOURCE_LIMIT' and s['sampled_peak_rss'] <= inv['rss_limit']
                or s['status']=='COMPLETED' and s['sampled_peak_rss'] > inv['rss_limit']):
            raise ValueError('resource cost')
        end = s['end_seconds']
    costs = [terminal['stage_seconds'], terminal['overhead_before_publication_seconds'],
             terminal['seconds_before_publication'], receipt['seconds_before_receipt'], finish['seconds_before_finish_marker']]
    if (not all(finite(v) and v >= 0 for v in costs)
            or abs(sum(s['seconds'] for s in stages)-costs[0]) > 1e-8
            or abs(costs[0]+costs[1]-costs[2]) > 1e-8 or not end <= costs[2] <= costs[3] <= costs[4]):
        raise ValueError('complete cost accounting')
    status = terminal['status_before_publication']
    if status not in (DONE,'ERROR','TIMEOUT','RESOURCE_LIMIT') or receipt['status'] != status:
        raise ValueError('terminal status')
    if status == DONE:
        if terminal['error'] is not None: raise ValueError('successful terminal contains exception')
        if len(stages)!=3 or any(s['status']!='COMPLETED' or s.get('output_reception_error') for s in stages):
            raise ValueError('incomplete execution accepted')
        if inv['start']+costs[2] >= inv['work_deadline']: raise ValueError('late reception accepted')
        for s in stages:
            if inv['start']+s['end_seconds'] >= s['deadline_monotonic']: raise ValueError('late worker accepted')
        if (stages[0].get('output_sha256') != terminal['payload_sha256'] or
                stages[1].get('output_sha256') != terminal['checker_stdout_sha256']):
            raise ValueError('parent output anchors')
        accepted = receive(root, spec, terminal['invocation_sha256'], terminal['payload_sha256'],
                           terminal['checker_stdout_sha256'], time.monotonic()+300)
        if accepted != terminal['accepted'] or accepted != load(root/'accepted.json'): raise ValueError('accepted evidence changed')
    elif terminal['accepted'] is not None: raise ValueError('failed run claimed reception')
    if finish['status'] != (status if costs[4] < inv['budget'] else 'TIMEOUT'):
        raise ValueError('publication status')
    marker = None
    if (root/'publication_timeout.json').exists():
        marker = load(root/'publication_timeout.json')
        if marker['status'] != 'TIMEOUT' or not finite(marker['seconds']) or marker['seconds'] < inv['budget']:
            raise ValueError('publication timeout evidence')
    final_status = 'TIMEOUT' if marker is not None else finish['status']
    observed = completed_call is not None
    if observed:
        call = completed_call
        if (call['status'] != final_status or call['invocation'] != inv['invocation'] or call['root'] != str(root)
                or not finite(call['seconds']) or call['seconds'] < costs[4]
                or call['complete_support_execution'] != (call['status']==DONE) or call['complete_moe_proof'] is not False
                or call['seconds'] >= inv['budget'] and call['status'] != 'TIMEOUT'
                or marker is not None and call['seconds'] < marker['seconds']
                or call['status']==DONE and call['seconds'] >= inv['budget']):
            raise ValueError('actual completed API observation')
    exact = None
    if recheck and status == DONE:
        from act.back_end.moe.check_batched_support import check_batch
        payload = load(root/'payload.json', terminal['payload_sha256'])
        exact = check_batch(payload['batch'], payload['candidates'], expected_batch_sha256=spec['batch_sha256'],
                            deadline=time.monotonic()+300)
        if exact != terminal['accepted']['result']: raise ValueError('independent bound recheck')
    return {'status': 'PASS', 'execution_status': final_status, 'observed_completed_api': observed,
            'complete_support_execution': observed and final_status==DONE,
            'complete_moe_proof': False, 'math_rechecked': exact is not None,
            'required': len(spec['query_ids']), 'stage_seconds': costs[0],
            'overhead_before_publication_seconds': costs[1], 'seconds_before_finish_marker': costs[4],
            'sampled_peak_rss': max((s['sampled_peak_rss'] for s in stages), default=0),
            'new_native_solves': 0, 'gpu_execution': False}
