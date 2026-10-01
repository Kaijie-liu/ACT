"""Versioned device support supervision; current frozen roster is CPU/stub only."""
from fractions import Fraction
import math
import os
from pathlib import Path
import time
import uuid

from scoped_proof.io import ROOT, PYTHON, load, save, sha, tick
from scoped_proof.supervisor import execute
from scripts import hz_device_admission as admission
from source_enclosure.format import identity

CONFIG = 'configs/hz_device_supervision_20261001.json'
PROTOCOL_SHA = '81009d4874e0f6b18ae700878cea6e5638494d9b312085e2755d3790eaffc131'
DONE = 'CHECKED_GIVEN_HZ_DEVICE_SUPPORT_EXECUTION'
PHASES = ('admit', 'produce', 'check', 'receive')
FILES = (CONFIG, 'scripts/hz_device_admission.py', 'act/back_end/moe/batched_support_device.py',
         'docs/hz_device_candidates_20261001_r3.json',
         'scripts/hz_batch_support_worker.py', 'scripts/hz_batch_support_supervised.py',
         'configs/hz_batch_supervision_20261001.json', 'scripts/hz_device_supervised.py', 'scripts/hz_device_worker.py',
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


def specification(case='guarded_two_sides_positive', control='', device='cpu', gpu_uuid=None):
    cfg=protocol()
    if control in ('admission_busy_stub','admission_error_stub'):
        device, gpu_uuid='cuda:0','GPU-f491c2c6-a093-590a-6b8a-13b5f76aadcc'
    spec={'schema':'HZ_DEVICE_EXECUTION_SPEC_V1','case':case,'control':control,
          'device':device,'gpu_uuid':gpu_uuid,'protocol_sha256':PROTOCOL_SHA,**cfg['cases'][case]}
    validate_spec(spec)
    return spec


def validate_spec(spec):
    cfg=protocol()
    if (spec.get('case') not in cfg['cases'] or spec.get('control') not in ('',*cfg['faults'])
            or spec.get('device') not in ('cpu','cuda:0')
            or (spec['device']=='cpu' and spec.get('gpu_uuid') is not None)
            or (spec['device']=='cuda:0' and not admission.valid_uuid(spec.get('gpu_uuid')))
            or spec != {'schema':'HZ_DEVICE_EXECUTION_SPEC_V1','case':spec['case'],'control':spec['control'],
                        'device':spec['device'],'gpu_uuid':spec['gpu_uuid'],
                        'protocol_sha256':PROTOCOL_SHA,**cfg['cases'][spec['case']]}):
        raise ValueError('fixed synthetic device specification')
    if spec['control'] and spec['case']!='guarded_two_sides_positive': raise ValueError('fault source')
    gpu_fault=spec['control'] in ('admission_busy_stub','admission_error_stub')
    if spec['control'] and (spec['device']=='cuda:0')!=gpu_fault: raise ValueError('fault device')
    if spec['device']=='cuda:0' and not gpu_fault:
        raise ValueError('physical CUDA requires separate frozen execution and cleanup admission')


def phase_deadline(inv, phase):
    if phase=='admit': return min(inv['work_deadline'],inv['start']+3)
    if phase=='produce': return inv['work_deadline']-min(5.,inv['budget']/4)
    return inv['work_deadline']


def device_context(spec, inv):
    if spec['device']=='cpu': return None
    return {'schema':'HZ_DEVICE_EXECUTION_CONTEXT_V1','batch_sha256':spec['batch_sha256'],
            'gpu_uuid':spec['gpu_uuid'],'deadline':phase_deadline(inv,'produce'),
            'allocator_limit_bytes':1073741824,'invocation':inv['invocation']}


def bound_admission(root,spec,inv,digest):
    record=load(root/'admission.json',required_hash(digest),limit=LIMIT)
    admission.validate(record,inv['invocation'],spec,phase_deadline(inv,'admit'))
    if any(o['start']<inv['start'] for o in record['observations']): raise ValueError('stale admission source')
    return record


def metadata(root,spec,inv,payload,admission_sha):
    gate=bound_admission(root,spec,inv,admission_sha)
    if gate['status'] not in ('CPU_NO_CUDA','READY'): raise ValueError('resource gate not passed')
    c=payload['candidates']; context=device_context(spec,inv)
    if (c.get('device')!=spec['device'] or c.get('dtype')!='float64' or c.get('iterations')!=128
            or c.get('algorithm')!='projected_dual_subgradient_multiobjective_v1'
            or c.get('hard_budget_supervision') is not False
            or c.get('execution_context_sha256')!=(None if context is None else identity(context))
            or payload.get('admission_sha256')!=admission_sha):
        raise ValueError('candidate device/context binding')
    if spec['device']=='cpu' and (c.get('hardware') is not None or c.get('allocator_memory') is not None):
        raise ValueError('CPU cannot claim GPU')
    if context is not None:
        if c.get('hardware',{}).get('gpu_uuid')!=spec['gpu_uuid']: raise ValueError('observed hardware identity')
        fresh=load(root/'pre_cuda_observation.json',required_hash(payload.get('pre_cuda_sha256')),limit=LIMIT)
        rebuilt=admission.parse(fresh['gpu_text'],fresh['process_text'],spec['gpu_uuid'])
        if any(fresh.get(k)!=v for k,v in rebuilt.items()) or not rebuilt['ready']:
            raise ValueError('last-moment resource gate')
        if not gate['observations'][-1]['end']<=fresh['start']<=fresh['end']<phase_deadline(inv,'produce'):
            raise ValueError('last-moment admission chronology')
    costs=c['cost_seconds']
    if (set(costs)!={'validation','host_tensors','device_initialization_sync','transfer_and_setup_sync',
                    'optimization_sync','readback_sync','exact_candidate_evaluation','other','total'}
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in costs.values())
            or abs(sum(v for k,v in costs.items() if k!='total')-costs['total'])>1e-8):
        raise ValueError('candidate component cost')
    if c.get('cost_scope')!='proposal API only; caller must charge HZ preparation, independent checking, publication and cleanup':
        raise ValueError('proposal cost scope')


def bind(root, spec, invocation_sha):
    inv = load(root / 'invocation.json', required_hash(invocation_sha), limit=LIMIT)
    if (inv['spec_sha256'] != identity(spec) or load(root / 'spec.json') != spec
            or inv['sources'] != sources()):
        raise ValueError('caller/implementation identity changed')
    if inv.get('cuda_context') != device_context(spec,inv): raise ValueError('parent device context binding')
    return inv


def receive(root, spec, invocation_sha, payload_sha, stdout_sha, deadline, admission_sha):
    tick(deadline)
    inv = bind(root, spec, invocation_sha)
    payload = load(root / 'payload.json', required_hash(payload_sha), limit=LIMIT)
    checked = load(root / 'check.stdout', required_hash(stdout_sha), limit=LIMIT)
    metadata(root,spec,inv,payload,admission_sha)
    context = {'invocation': inv['invocation'], 'request_sha256': identity(spec), 'batch_sha256': spec['batch_sha256']}
    if (any(payload.get(k) != v or checked.get(k) != v for k, v in context.items())
            or identity(payload['batch']) != spec['batch_sha256']
            or checked.get('schema') != 'HZ_DEVICE_CHECK_OUTPUT_V1'
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
    accepted = {'schema': 'HZ_DEVICE_ACCEPTED_V1', 'status': DONE, **context,
                'payload_sha256': payload_sha, 'checker_stdout_sha256': stdout_sha,
                'required': len(spec['query_ids']), 'result': result,
                'complete_moe_proof': False, 'gpu_execution': spec['device']=='cuda:0'}
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
    inv_data = {'schema': 'HZ_DEVICE_INVOCATION_V1', 'invocation': token,
        'spec_sha256': expected_request_sha256, 'start': start, 'deadline': deadline, 'work_deadline': work,
        'budget': budget, 'rss_limit': rss_limit, 'sources': sources()}
    inv_data['cuda_context']=device_context(spec,inv_data)
    inv_record=save(root/'invocation.json',inv_data)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    stages = []
    accepted = None
    status, error = 'ERROR', None
    payload_sha = stdout_sha = admission_sha = None
    try:
        for phase in PHASES:
            bind(root, spec, inv_record['sha256'])
            before = time.monotonic()
            phase_end = phase_deadline(inv_data,phase)
            env['CUDA_VISIBLE_DEVICES']=spec['gpu_uuid'] if phase=='produce' and spec['device']=='cuda:0' else ''
            command = [PYTHON, '-B'] + (['-S'] if phase == 'receive' else []) + [
                '-m', 'scripts.hz_device_worker', phase, str(root), '--deadline', str(phase_end),
                '--invocation-sha', inv_record['sha256']]
            if phase in ('check','receive'): command += ['--payload-sha', required_hash(payload_sha)]
            if phase != 'admit': command += ['--admission-sha', required_hash(admission_sha)]
            if phase == 'receive': command += ['--stdout-sha', required_hash(stdout_sha)]
            if spec['control'] == 'launch_failure': command[0] = str(root / 'nonexistent-python')
            record = execute(command, root / (phase + '.log'), phase_end, env, rss_limit)
            record.update(phase=phase, executable=command[0], cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],
                          start_seconds=before-start, end_seconds=time.monotonic()-start)
            stages.append(record)  # Charge execution even if its output is absent.
            gate = None
            try:
                if record['status'] == 'COMPLETED' and phase in ('admit', 'produce', 'check'):
                    path = root / ({'admit':'admission.json','produce':'payload.json','check':'check.stdout'}[phase])
                    if path.is_symlink() or path.stat().st_size > LIMIT: raise ValueError('output file admission')
                    digest = sha(path)  # Parent-owned anchor, never the worker's reported SHA.
                    if phase == 'admit': admission_sha = digest
                    elif phase == 'produce': payload_sha = digest
                    else: stdout_sha = digest
                    record['output_sha256'] = digest
                    if phase=='admit': gate=bound_admission(root,spec,inv_data,admission_sha)
            except Exception as exc:
                record['output_reception_error'] = repr(exc)
                raise
            finally:
                record['end_seconds'] = time.monotonic() - start
                if start + record['end_seconds'] >= phase_end:
                    record['worker_status'] = record['status']
                    record['status'] = 'TIMEOUT'
                save(root / (phase + '_stage.json'), record)
            if record['status']=='COMPLETED' and phase=='admit':
                if gate['status']=='RESOURCE_UNAVAILABLE': status='RESOURCE_UNAVAILABLE'; break
            if record['status'] != 'COMPLETED':
                status = record['status']
                break
        else:
            accepted = receive(root, spec, inv_record['sha256'], payload_sha, stdout_sha, work, admission_sha)
            if load(root / 'accepted.json', limit=LIMIT) != accepted: raise ValueError('receiver file disagreement')
            status = DONE
    except Exception as exc:
        error, status, accepted = repr(exc), 'ERROR', None
    status, accepted, charged = terminal_observation(status, accepted, start, work)
    terminal = {'schema': 'HZ_DEVICE_TERMINAL_V1', 'invocation': token, 'invocation_sha256': inv_record['sha256'],
                'request_sha256': expected_request_sha256, 'payload_sha256': payload_sha,
                'checker_stdout_sha256': stdout_sha, 'admission_sha256':admission_sha, 'status_before_publication': status,
                'stages': stages, 'accepted': accepted, 'error': error,
                'required': len(spec['query_ids']), 'seconds_before_publication': charged,
                'stage_seconds': sum(r['seconds'] for r in stages),
                'overhead_before_publication_seconds': charged - sum(r['seconds'] for r in stages)}
    terminal_record = save(root / 'terminal.json', terminal)
    receipt_record = save(root / 'receipt.json', {'schema': 'HZ_DEVICE_RECEIPT_V1', 'invocation': token,
        'terminal_sha256': terminal_record['sha256'], 'status': status, 'budget': budget,
        'seconds_before_receipt': time.monotonic()-start, 'complete_moe_proof': False})
    elapsed = time.monotonic() - start
    finish = save(root / 'finish.json', {'schema': 'HZ_DEVICE_FINISH_V1', 'invocation': token,
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
            'HZ_DEVICE_INVOCATION_V1', 'HZ_DEVICE_TERMINAL_V1', 'HZ_DEVICE_RECEIPT_V1', 'HZ_DEVICE_FINISH_V1']:
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
        visible=spec['gpu_uuid'] if s['phase']=='produce' and spec['device']=='cuda:0' else ''
        if s.get('cuda_visible_devices')!=visible: raise ValueError('stage CUDA isolation')
        phase_end = phase_deadline(inv,s['phase'])
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
        # Failed calls must retain their anchored evidence too; failure is not
        # permission to remove the successfully published prefix.
        output={'admit':('admission.json','admission_sha256'),
                'produce':('payload.json','payload_sha256'),
                'check':('check.stdout','checker_stdout_sha256')}.get(s['phase'])
        if (output and (s['status']=='COMPLETED' or s.get('worker_status')=='COMPLETED')
                and not s.get('output_reception_error')):
            required_hash(s.get('output_sha256'))
            required_hash(terminal.get(output[1]))
        if output and s.get('output_sha256') is not None:
            name,key=output
            if required_hash(s['output_sha256'])!=terminal.get(key): raise ValueError('failed-prefix output anchor')
            if not (spec['control']=='rewrite_stdout' and s['phase']=='check'):
                load(root/name,s['output_sha256'],limit=LIMIT)
            elif sha(root/name)==s['output_sha256']:
                raise ValueError('controlled output rewrite absent')
            if s['phase']=='admit': bound_admission(root,spec,inv,s['output_sha256'])
        elif output and terminal.get(output[1]) is not None:
            raise ValueError('terminal claimed unanchored prefix')
    costs = [terminal['stage_seconds'], terminal['overhead_before_publication_seconds'],
             terminal['seconds_before_publication'], receipt['seconds_before_receipt'], finish['seconds_before_finish_marker']]
    if (not all(finite(v) and v >= 0 for v in costs)
            or abs(sum(s['seconds'] for s in stages)-costs[0]) > 1e-8
            or abs(costs[0]+costs[1]-costs[2]) > 1e-8 or not end <= costs[2] <= costs[3] <= costs[4]):
        raise ValueError('complete cost accounting')
    status = terminal['status_before_publication']
    if status not in (DONE,'ERROR','TIMEOUT','RESOURCE_LIMIT','RESOURCE_UNAVAILABLE') or receipt['status'] != status:
        raise ValueError('terminal status')
    if status == DONE:
        if terminal['error'] is not None: raise ValueError('successful terminal contains exception')
        if len(stages)!=4 or any(s['status']!='COMPLETED' or s.get('output_reception_error') for s in stages):
            raise ValueError('incomplete execution accepted')
        if inv['start']+costs[2] >= inv['work_deadline']: raise ValueError('late reception accepted')
        for s in stages:
            if inv['start']+s['end_seconds'] >= s['deadline_monotonic']: raise ValueError('late worker accepted')
        if (stages[0].get('output_sha256') != terminal['admission_sha256'] or
                stages[1].get('output_sha256') != terminal['payload_sha256'] or
                stages[2].get('output_sha256') != terminal['checker_stdout_sha256']):
            raise ValueError('parent output anchors')
        accepted = receive(root, spec, terminal['invocation_sha256'], terminal['payload_sha256'],
                           terminal['checker_stdout_sha256'], time.monotonic()+300, terminal['admission_sha256'])
        if accepted != terminal['accepted'] or accepted != load(root/'accepted.json'): raise ValueError('accepted evidence changed')
        p=load(root/'payload.json',terminal['payload_sha256'])
        if p['candidates']['cost_seconds']['total']>stages[1]['seconds']:
            raise ValueError('proposal cost exceeds producer execution')
    elif terminal['accepted'] is not None: raise ValueError('failed run claimed reception')
    if status=='RESOURCE_UNAVAILABLE':
        if len(stages)!=1 or stages[0]['status']!='COMPLETED' or terminal['payload_sha256'] is not None:
            raise ValueError('resource refusal must not start producer')
        gate=bound_admission(root,spec,inv,terminal['admission_sha256'])
        if gate['status']!='RESOURCE_UNAVAILABLE' or stages[0].get('output_sha256')!=terminal['admission_sha256']:
            raise ValueError('resource refusal anchor')
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
            'new_native_solves': 0, 'gpu_execution': False,
            'cuda_requested': spec['device']=='cuda:0', 'physical_cuda_admitted': False}
