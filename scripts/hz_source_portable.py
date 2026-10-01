"""Trusted publisher/launcher for OFFLINE finite source proof relocation."""
import os
import base64
import math
from pathlib import Path
import shutil
import time

from scoped_proof.io import ROOT,PYTHON,load,save,sha,tick
from scoped_proof.owned_bounded import execute
from scoped_source import hz_portable_verify as v

CONFIG='configs/hz_source_portable_20261001.json'
CONFIG_SHA='6fdc90a78a81fa24600fa7fb1f97de258d6ad0ad62abf0d1fd678e9e454fb36a'
BOOTSTRAP='scoped_source/hz_portable_verify.py'
FILES=(*v.CODE,BOOTSTRAP,'scripts/hz_source_portable.py','scripts/test_hz_source_portable.py',
       'scripts/run_hz_source_portable.py','scoped_proof/io.py','scoped_proof/owned_bounded.py',
       CONFIG,'docs/hz_source_portable_design_20261001.md','docs/hz_source_supervision_20261001_r4.json')


def protocol(): return load(ROOT/CONFIG,CONFIG_SHA)
def sources(): return {n:sha(ROOT/n) for n in FILES}


def code_bytes():
    cfg=protocol(); old=Path(cfg['reference_root'])
    report=load(ROOT/cfg['reference_report'],cfg['reference_report_sha256'])
    summary=load(old/'summary.json',report['summary_sha256'])
    frozen=load(old/'implementation.json',summary['implementation_sha256'])
    for n in v.CODE:
        if sha(ROOT/n)!=frozen[n]: raise ValueError('frozen mathematical leaf changed: '+n)
    return {**{'code/'+n:(ROOT/n).read_bytes() for n in v.CODE},
            **{'code/'+n:b'' for n in v.INITS},'verify.py':(ROOT/BOOTSTRAP).read_bytes()}


def descriptors(bodies): return {n:{'bytes':len(b),'sha256':v.digest(b)} for n,b in sorted(bodies.items())}


def trusted_code_sha(): return v.identity(descriptors(code_bytes()))


def reference(name):
    cfg=protocol(); case=cfg['cases'][name]; old=Path(cfg['reference_root'])/name
    report=load(ROOT/cfg['reference_report'],cfg['reference_report_sha256'])
    payload=load(old/'produce.json',case['produce_sha256'])
    source=load(old/'prefix_source.json',case['source_sha256'])
    if v.identity(source)!=case['source_sha256'] or v.identity(payload['package'])!=case['package_sha256']:
        raise ValueError('registered source/package identity')
    return source,payload['package'],report['calls'][name]['obligations']


def publish(root,source,package,expected,deadline):
    tick(deadline); root=Path(root); root.mkdir(parents=True,exist_ok=False)
    if v.identity(source)!=expected['source_sha256'] or v.identity(package)!=expected['package_sha256']:
        raise ValueError('external publication identity')
    bodies=code_bytes()
    for n,obj in [('source.json',source),('package.json',package)]:
        bodies[n]=__import__('json').dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    refs=descriptors(bodies)
    for n,b in bodies.items():
        tick(deadline); p=root/n; p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f: f.write(b)
    m={'schema':'HZ_SOURCE_PORTABLE_V1','source_sha256':expected['source_sha256'],
       'package_sha256':expected['package_sha256'],
       'code_sha256':v.identity({n:refs[n] for n in sorted(v.CODE_NAMES)}),'files':refs}
    anchor={'manifest_sha256':save(root/'manifest.json',m)['sha256'],
            **{k:m[k] for k in ('source_sha256','package_sha256','code_sha256')}}
    v.envelope(root,anchor,deadline); tick(deadline); return anchor


def preflight(root,anchors,deadline):
    # Independent caller anchor, before executing verify.py, not its self-check.
    if anchors['code_sha256']!=trusted_code_sha(): raise ValueError('caller checker identity')
    return v.envelope(root,anchors,deadline)


def command(root,anchors,deadline,*,bodies=None):
    if bodies is None: _,bodies,_=preflight(root,anchors,time.monotonic()+30)
    launcher=('import sys,base64; path=sys.argv.pop(1); raw=base64.b64decode(sys.argv.pop(1)); '
              'exec(compile(raw,path,"exec"),{"__name__":"__main__","__file__":path})')
    cmd=['/usr/bin/env','--chdir='+str(root),PYTHON,'-B','-I','-S','-c',launcher,
         str(root/'verify.py'),base64.b64encode(bodies['verify.py']).decode()]
    for n in ('manifest','source','package','code'): cmd += ['--'+n+'-sha',anchors[n+'_sha256']]
    return cmd+['--deadline',str(deadline)]


def launch(root,anchors,log,deadline):
    _,bodies,_=preflight(root,anchors,deadline)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    return run_child(command(root,anchors,deadline-.5,bodies=bodies),root,anchors,log,deadline,env)


def run_child(cmd,root,anchors,log,deadline,env):
    begin=time.monotonic()
    stage=execute(cmd,log,run_deadline=deadline-1,cleanup_deadline=deadline-.5,
                  env=env,rss_limit=protocol()['rss_limit'])
    stage['receipt']={'root':str(root),'log':str(log),'anchors':anchors,'deadline':deadline,
                      'command_sha256':v.identity(cmd),'start':begin,
                      'stdout_sha256':sha(log) if Path(log).exists() else None}
    stage['receipt']['end']=time.monotonic()
    return stage


def validate_stage(root,anchors,stage,log,expected_deadline=None,success=True):
    r=stage['receipt']; end=r['deadline']
    if (r['root']!=str(root) or r['log']!=str(log) or r['anchors']!=anchors
            or expected_deadline is not None and end!=expected_deadline
            or r['stdout_sha256']!=sha(log)):
        raise ValueError('invocation/stdout binding')
    times=[r[k] for k in ('start','end','deadline')]+[stage[k] for k in (
        'seconds','execution_seconds','cleanup_seconds','run_deadline','cleanup_deadline','exit_observed_at')]
    if any(type(t) not in (int,float) or not math.isfinite(t) or t<0 for t in times):
        raise ValueError('finite nonnegative stage clocks')
    if (stage['run_deadline']!=end-1 or stage['cleanup_deadline']!=end-.5
            or not r['start']<=stage['exit_observed_at']<stage['run_deadline']
            or not stage['exit_observed_at']<=r['end']<stage['cleanup_deadline']
            or abs(stage['seconds']-stage['execution_seconds']-stage['cleanup_seconds'])>1e-8
            or stage['seconds']>r['end']-r['start']):
        raise ValueError('stage deadline/cost conservation')
    if (type(stage.get('pid')) is not int or stage['pid']<=0 or type(stage.get('returncode')) is not int
            or stage.get('remaining_group',{}).get('live')!=[] or stage.get('cleanup_unconfirmed_stub')
            or stage.get('cleanup_status')!='LEADER_REAPED_NO_LIVE_GROUP'
            or stage.get('descendant_on_leader_exit') is not False or stage.get('error') is not None
            or stage.get('status')!=('COMPLETED' if success else 'ERROR')
            or (stage['returncode']==0)!=success):
        raise ValueError('checker did not complete with confirmed cleanup')
    v.hash_value(r['command_sha256']); return r


def receive(root,anchors,stage,log,deadline,expected_result,expected_deadline=None):
    tick(deadline)
    r=validate_stage(root,anchors,stage,log,expected_deadline)
    result=load(log,r['stdout_sha256'],limit=v.MEMBER_LIMIT)
    if (result['schema']!='HZ_SOURCE_PORTABLE_CHECK_V1' or result['anchors']!=anchors
            or result['result']!=expected_result or result['isolated'] is not True
            or result['site_disabled'] is not True or result['offline_recheck_only'] is not True
            or result['deployed_float_SAFE'] is not False or result['numerical_modules_loaded']!=[]):
        raise ValueError('returned result binding/scope')
    _,bodies,total=preflight(root,anchors,deadline)
    required={p.replace('/','.'):'code/'+p+'/__init__.py' for p in v.PACKAGES}
    required.update({p[:-3].replace('/','.'):'code/'+p for p in v.CODE})
    imported={n:{'path':p,'sha256':v.digest(bodies[p])} for n,p in required.items()}
    if (result['loaded_modules']!=imported or result['bundle_bytes']!=total
            or result['bundle_reads']!=sorted(v.MEMBERS|{'manifest.json'})):
        raise ValueError('actual module/read closure')
    tick(deadline); return result


def return_observed(path,observed,deadline):
    save(path,observed)
    tick(deadline)
    return observed


def run_case(root,name):
    begin=time.monotonic(); deadline=begin+protocol()['normal_budget_seconds']; root=Path(root)
    root.mkdir(parents=True,exist_ok=False); costs={}; status='ERROR'; error=None; result=None; stage=None
    def measured(key,fn):
        t=time.monotonic()
        try: return fn()
        finally: costs[key]=time.monotonic()-t
    try:
        source,package,expected=measured('archive_read',lambda:reference(name))
        anchors=measured('publication',lambda:publish(root/'original',source,package,protocol()['cases'][name],deadline))
        measured('relocation',lambda:shutil.copytree(root/'original',root/'relocated'))
        tick(deadline)
        stage=measured('preflight_and_child',lambda:launch(root/'relocated',anchors,root/'checker.log',deadline))
        save(root/'stage.json',stage)
        result=measured('receipt',lambda:receive(root/'relocated',anchors,stage,root/'checker.log',deadline,expected,deadline))
        status='CHECKED_OFFLINE_RELOCATION'
    except Exception as exc:
        error=repr(exc)
        if stage is not None and stage['status']=='CLEANUP_INCOMPLETE': status='CLEANUP_INCOMPLETE'
        elif isinstance(exc,TimeoutError) or time.monotonic()>=deadline: status='TIMEOUT'
        elif stage is not None: status=stage['status'] if stage['status']!='COMPLETED' else 'ERROR'
    now=time.monotonic()
    terminal={'case':name,'status':status,'error':error,'start':begin,'deadline':deadline,
              'seconds_before_publication':now-begin,'cost_seconds':costs,'result':result,
              'result_file_sha256':sha(root/'checker.log') if (root/'checker.log').exists() else None,
              'stage_sha256':sha(root/'stage.json') if (root/'stage.json').exists() else None,
              'offline_recheck_only':True,'generation_cost_not_in_this_measurement':True}
    save(root/'terminal.json',terminal)
    end=time.monotonic()
    observed={'status':status if end<deadline or status=='CLEANUP_INCOMPLETE' else 'TIMEOUT','start':begin,'end':end,'seconds':end-begin,
              'terminal_sha256':sha(root/'terminal.json'),'terminal_publication_seconds':end-now}
    # This final observation includes hashing and publication above. A durable
    # earlier record cannot turn a late API return into acceptance.
    return return_observed(root/'observed.json',observed,deadline)
