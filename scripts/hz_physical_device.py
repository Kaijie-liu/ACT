"""Policy for tiny CPU/CUDA execution; physical calls need an execution freeze."""
from fractions import Fraction
import time

from scoped_proof.io import ROOT, load, sha
from source_enclosure.format import identity
from scripts import hz_device_lifecycle as old
from scripts import hz_device_admission as admission
from scripts import hz_device_release as release
from scripts.hz_propagation_supervised import finite, required_hash

CONFIG='configs/hz_physical_device_design_20261001.json'
CONFIG_SHA='003e7dd82790d7597c5c154e1d27c4a5b80c92b34f98269360c7a8bb038968ce'
DONE='CHECKED_GIVEN_HZ_DEVICE_EXECUTION'
WORKER='scripts.hz_physical_worker'
LIMIT=4*2**20
FILES=tuple(dict.fromkeys((*old.FILES,CONFIG,'scoped_proof/device_lifecycle.py',
    'scripts/hz_physical_device.py','scripts/hz_physical_worker.py',
    'scripts/test_hz_physical_device.py','scripts/run_hz_physical_device.py')))


def protocol():
    cfg=load(ROOT/CONFIG,CONFIG_SHA); old.protocol()
    for name,digest in cfg['frozen_dependencies'].items():
        if sha(ROOT/name)!=digest: raise ValueError('frozen source changed: '+name)
    return cfg


def sources(): return {n:sha(ROOT/n) for n in FILES}


def spec(case='guarded_two_sides_positive',device='cpu',control=''):
    cfg=protocol()
    if (case not in cfg['cases'] or device not in ('cpu','cuda:0') or control not in ('',*cfg['cpu_controls'])
            or control and case!=cfg['cases'][0]): raise ValueError('fixed physical/control roster')
    if control and device!=('cuda:0' if control in ('admission_busy_stub','pre_cuda_busy_stub') else 'cpu'):
        raise ValueError('control device binding')
    return {'schema':'HZ_PHYSICAL_SPEC_V1','case':case,'device':device,'control':control,
            'gpu_uuid':cfg['gpu_uuid'] if device=='cuda:0' else None,
            'design_sha256':CONFIG_SHA,**old.protocol()[1]['cases'][case]}


def validate_spec(s,permit):
    if s!=spec(s['case'],s['device'],s['control']): raise ValueError('request identity')
    if s['control']:
        if permit is not None: raise ValueError('control must not use physical permit')
        return
    if type(permit) is not dict or set(permit)!={'path','sha256'}: raise ValueError('separate physical execution freeze required')
    from pathlib import Path
    path=Path(permit['path'])
    if not path.is_absolute() or path.resolve().parent!=ROOT/'configs': raise ValueError('committed freeze location')
    frozen=load(path,required_hash(permit['sha256']),LIMIT)
    if (frozen['schema']!='HZ_PHYSICAL_EXECUTION_FREEZE_V1' or frozen['design_sha256']!=CONFIG_SHA
            or frozen['sources']!=sources() or frozen['roster']!=physical_roster()): raise ValueError('physical execution source/roster')
    controls_receipt(frozen)


def controls_receipt(frozen):
    from pathlib import Path
    from scripts.test_hz_physical_device import TEST_NAMES
    root=Path(frozen['controls_root'])
    summary=load(root/'summary.json',required_hash(frozen['controls_summary_sha256']),LIMIT)
    if (summary['status']!='PASS' or summary['tests']!=8 or summary['calls']!=6 or summary['names']!=sorted(TEST_NAMES)
            or any(summary[k] for k in ('failures','errors','skipped','expected_failures','unexpected_successes','physical_cuda_calls'))
            or sha(root/'tests.log')!=summary['tests_sha256'] or sha(root/'calls.json')!=summary['calls_sha256']):
        raise ValueError('physical controls not complete')
    bindings=load(root/'implementation.json',required_hash(summary['implementation_sha256']),LIMIT)
    if bindings!=sources(): raise ValueError('different controlled implementation')
    report=load(frozen['controls_audit_path'],required_hash(frozen['controls_audit_sha256']),LIMIT)
    if (report['status']!='PASS' or report['summary_sha256']!=frozen['controls_summary_sha256']
            or report['implementation_sha256']!=summary['implementation_sha256'] or report['tests']!=8
            or set(report['calls'])!=set(protocol()['cpu_controls']) or report['physical_cuda_calls']!=0
            or any(r['status']!='PASS' for r in report['calls'].values())):
        raise ValueError('controlled independent audit binding')


def physical_roster():
    return [{'case':case,'device':d} for i,case in enumerate(protocol()['cases'])
            for d in (('cpu','cuda:0') if i%2==0 else ('cuda:0','cpu'))]


def budget(s): return protocol()['cpu_controls'][s['control']][0] if s['control'] else 30


def cutoff(inv,phase,begin):
    if phase=='admit': return min(inv['work'],inv['start']+3)
    if phase=='produce': return inv['work']-min(7.,inv['budget']/2)
    if phase=='release': return min(inv['work'],begin+min(3.,inv['budget']/3))
    return inv['work']


def visible(inv,phase):
    return inv['spec']['gpu_uuid'] if phase=='produce' and inv['spec']['device']=='cuda:0' and not inv['spec']['control'] else ''


def context(inv,deadline):
    if inv['spec']['device']=='cpu': return None
    return {'schema':'HZ_DEVICE_EXECUTION_CONTEXT_V1','batch_sha256':inv['spec']['batch_sha256'],
            'gpu_uuid':inv['spec']['gpu_uuid'],'deadline':deadline,'allocator_limit_bytes':2**30,
            'invocation':inv['invocation']}


def admission_record(root,inv,stage):
    r=load(root/'admit.json',required_hash(stage['output_sha256']),LIMIT)
    p=load(root/'admit_plan.json',stage['plan_sha256']); s=dict(inv['spec'])
    simulated=s['control'] in ('admission_busy_stub','pre_cuda_busy_stub')
    # Reuse the frozen observer's validator with its registered simulated flag.
    s['control']='admission_busy_stub' if simulated and r['status']=='RESOURCE_UNAVAILABLE' else ''
    copy=dict(r); copy['simulated']=bool(s['control'])
    admission.validate(copy,inv['invocation'],s,p['run_deadline'])
    if r['simulated'] is not simulated: raise ValueError('physical/stub admission identity')
    if any(o['start']<p['begin'] for o in r['observations']): raise ValueError('stale admission')
    return r


def admitted(root,inv,stage): return admission_record(root,inv,stage)['status'] in ('READY','CPU_NO_CUDA')


def produced(root,inv,stage):
    r=load(root/'produce.json',stage['output_sha256'],LIMIT)
    if r['invocation']!=inv['invocation'] or r['spec_sha256']!=identity(inv['spec']): raise ValueError('producer identity')
    if r['status']=='RESOURCE_UNAVAILABLE':
        pre=load(root/'pre_cuda.json',r['pre_cuda_sha256'],LIMIT)
        validate_pre(root,inv,pre)
        if pre['ready']: raise ValueError('refusal without busy pre-CUDA observation')
        return False
    if r['status']!='CANDIDATES': raise ValueError('producer status')
    return True


def validate_pre(root,inv,record):
    p=load(root/'produce_plan.json'); row=record['observation']
    if (record['invocation']!=inv['invocation'] or record['simulated'] is not bool(inv['spec']['control'])
            or record['ready']!=row['ready']): raise ValueError('pre-CUDA identity')
    rebuilt=admission.parse(row['gpu_text'],row['process_text'],inv['spec']['gpu_uuid'])
    if any(row.get(k)!=v for k,v in rebuilt.items()): raise ValueError('pre-CUDA raw observation')
    for k in ('start','end'): finite(row[k])
    if not p['begin']<=row['start']<=row['end']<p['run_deadline']: raise ValueError('pre-CUDA deadline')


def payload(root,inv,digest):
    p=load(root/'produce.json',required_hash(digest),LIMIT); c=p['candidates']; s=inv['spec']
    if (p['status']!='CANDIDATES' or p['invocation']!=inv['invocation'] or p['spec_sha256']!=identity(s)
            or identity(p['batch'])!=s['batch_sha256'] or c['batch_sha256']!=s['batch_sha256']
            or c['device']!=s['device'] or c['dtype']!='float64' or c['iterations']!=128
            or c['algorithm']!='projected_dual_subgradient_multiobjective_v1' or c['hard_budget_supervision'] is not False):
        raise ValueError('candidate source/algorithm/device')
    stage=load(root/'produce_stage.json'); launch=load(root/'produce_plan.json',stage['plan_sha256'])
    ctx=context(inv,launch['run_deadline'])
    if p['context']!=ctx or c['execution_context_sha256']!=(None if ctx is None else identity(ctx)):
        raise ValueError('bound device context')
    if s['device']=='cpu':
        if (c['hardware'] is not None or c['allocator_memory'] is not None or p['pre_cuda_sha256'] is not None
                or p.get('cuda_intent_sha256') is not None or (root/'cuda_intent.json').exists()):
            raise ValueError('unexpected CUDA CPU metadata')
    else:
        pre=load(root/'pre_cuda.json',required_hash(p['pre_cuda_sha256']),LIMIT); validate_pre(root,inv,pre)
        if not pre['ready'] or pre['simulated'] or c['hardware']['gpu_uuid']!=s['gpu_uuid']:
            raise ValueError('CUDA lacks fresh real admission/hardware identity')
        finite(c['hardware']['total_memory'],minimum=2**30)
        if type(c['hardware'].get('name')) is not str or not c['hardware']['name']: raise ValueError('hardware name')
        for n in ('peak_allocated','peak_reserved'):
            if type(c['allocator_memory'][n]) is not int or not 0<=c['allocator_memory'][n]<=2**30:
                raise ValueError('allocator envelope')
        if c['allocator_memory']['peak_allocated']>c['allocator_memory']['peak_reserved']: raise ValueError('allocator ordering')
        intent=load(root/'cuda_intent.json',required_hash(p.get('cuda_intent_sha256')),LIMIT)
        if intent!={'invocation':inv['invocation'],'pre_cuda_sha256':p['pre_cuda_sha256'],
                    'producer_pid':stage['pid'],'not_completion_evidence':True}: raise ValueError('CUDA intent identity')
    env=p['environment']
    if (set(env)!={'torch','cuda_build','cpu_threads'} or type(env['torch']) is not str or not env['torch']
            or type(env['cpu_threads']) is not int or env['cpu_threads']!=1
            or (s['device']=='cuda:0' and (type(env['cuda_build']) is not str or not env['cuda_build']))):
        raise ValueError('producer environment')
    costs=c['cost_seconds']; required={'validation','host_tensors','device_initialization_sync','transfer_and_setup_sync',
        'optimization_sync','readback_sync','exact_candidate_evaluation','other','total'}
    if set(costs)!=required: raise ValueError('candidate cost keys')
    for v in costs.values(): finite(v)
    if (abs(costs['total']-sum(v for k,v in costs.items() if k!='total'))>1e-8
            or costs['total']>stage['seconds'] or stage['output_sha256']!=digest
            or c['cost_scope']!='proposal API only; caller must charge HZ preparation, independent checking, publication and cleanup'):
        raise ValueError('candidate nested cost/anchor')
    return p


def check(root,inv,digest,deadline):
    p=payload(root,inv,digest)
    from act.back_end.moe.check_batched_support import check_batch
    r=check_batch(p['batch'],p['candidates'],expected_batch_sha256=inv['spec']['batch_sha256'],deadline=deadline)
    return {'invocation':inv['invocation'],'payload_sha256':digest,'result':r}


def receive(root,inv,inputs):
    p=payload(root,inv,inputs['produce']); checked=load(root/'check.json',required_hash(inputs['check']),LIMIT)
    r=checked['result']; s=inv['spec']
    if (checked['invocation']!=inv['invocation'] or checked['payload_sha256']!=inputs['produce']
            or r['status']!='CHECKED_GIVEN_HZ_CONTINUOUS_RELAXATION' or r['batch_sha256']!=s['batch_sha256']
            or r['candidate_sha256']!=identity(p['candidates']) or [x['id'] for x in r['results']]!=s['query_ids']
            or [x['side'] for x in r['results']]!=s['sides'] or r['network_or_complete_moe_proof'] is not False
            or r['hard_budget_supervision'] is not False): raise ValueError('complete exact output reception')
    for x in r['results']:
        Fraction(x['bound']); required_hash(x['lp_sha256'])
        if x['bound_kind']!=('lower' if x['side']=='min' else 'upper'): raise ValueError('bound direction')
    return {'status':DONE,'invocation':inv['invocation'],'inputs':inputs,'checked':r,'device':s['device'],
            'complete_moe_proof':False}


def release_gate(root,inv,records):
    p,r=records['produce'],records['release']
    if not old.cleanup_confirmed(p) or not old.cleanup_confirmed(r) or r['status']!='COMPLETED':
        return {'status':'DEVICE_RELEASE_UNCONFIRMED','release_confirmed':False}
    value=load(root/'release.json',required_hash(r['output_sha256']),LIMIT)
    psha=sha(root/'produce_stage.json')
    if inv['spec']['device']=='cpu':
        if value!={'status':'CPU_NO_CUDA_TO_RELEASE','invocation':inv['invocation'],'producer_sha256':psha}:
            raise ValueError('CPU release binding')
        return {'status':'CPU_NO_CUDA_TO_RELEASE','release_confirmed':True,'next_admission_established':False}
    plan=load(root/'release_plan.json',r['plan_sha256'])
    result=release.validate(value,invocation=inv['invocation'],gpu_uuid=inv['spec']['gpu_uuid'],
        producer_sha=psha,producer_pid=p['pid'],after=inv['start']+p['end_seconds'],
        deadline=plan['run_deadline'],simulated=bool(inv['spec']['control']))
    if not result['release_confirmed']: result['status']='DEVICE_RELEASE_UNCONFIRMED'
    return result


def decision(root,inv,stages):
    r={s['phase']:s for s in stages}
    if any(not old.cleanup_confirmed(s) for s in stages): return 'CLEANUP_INCOMPLETE',None,None
    if r['admit']['status']!='COMPLETED': return r['admit']['status'],None,None
    if not admitted(root,inv,r['admit']): return 'RESOURCE_UNAVAILABLE',None,None
    if 'produce' not in r: return 'ERROR',None,None
    if 'release' not in r: return 'DEVICE_RELEASE_UNCONFIRMED',None,None
    release_result=release_gate(root,inv,r)
    if not release_result['release_confirmed']: return 'DEVICE_RELEASE_UNCONFIRMED',release_result,None
    if r['produce']['status']!='COMPLETED': return r['produce']['status'],release_result,None
    if not produced(root,inv,r['produce']): return 'RESOURCE_UNAVAILABLE',release_result,None
    for p in ('check','receive'):
        if p not in r: return 'ERROR',release_result,None
        if r[p]['status']!='COMPLETED': return r[p]['status'],release_result,None
    accepted=receive(root,inv,{p:r[p]['output_sha256'] for p in ('produce','release','check')})
    if accepted!=load(root/'receive.json',r['receive']['output_sha256']): raise ValueError('receiver bytes')
    return DONE,release_result,accepted
