"""Owned outer watchdog and immutable terminals for a supplied-LP diagnostic."""
import math
from pathlib import Path
import subprocess
import time
from single_check_portable.execution import ROOT,ACT,read,save_new,left
from portable_proof.runtime import digest
from evidence_cohort.run import environment,wait_owned

STAGES=('load','propose','package','check')


def phase_state(owner,end,limit):
    if owner['killed'] or owner['return_code']==3 or end>=limit:return 'TIMEOUT'
    return 'COMPLETED' if owner['return_code']==0 else 'ERROR'


def drive(root):
    root=Path(root);p=read(root/'plan.json');start=p['started'];stages=[];error=None
    status='ERROR';complete=False
    try:
        for name in STAGES:
            limit=218 if name in ('load','propose') else 298
            begin=time.monotonic()-start
            if begin>=limit:raise TimeoutError('no phase budget')
            save_new(root/(name+'_entered.json'),{'phase':name,'seconds':begin})
            if name=='check':
                pack=read(root/'packing.json')
                cmd=[ACT,'-I','-S',str(root/'portable/verify.py'),str(root/'portable/bundle.json'),
                     '--bundle-sha256',pack['bundle_sha256'],'--statement-sha256',pack['statement_sha256'],
                     '--timeout-seconds',repr(max(.000001,start+298-time.monotonic()))]
            else:cmd=[ACT,'-m','lp_diagnostic.worker',name,str(root)]
            with (root/(name+'.log')).open('xb') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=environment(),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                owner=wait_owned(proc,start+limit)
            end=time.monotonic()-start
            row={'name':name,'start_seconds':begin,'end_seconds':end,'elapsed_seconds':end-begin,
                 'limit_seconds':limit,'process':owner,'state':phase_state(owner,end,limit)}
            save_new(root/(name+'_stage.json'),row);stages.append(row)
            if row['state']!='COMPLETED':status=row['state'];break
        else:
            out=read(root/'check.log')
            if (out['status']!='CHECKED_LP_DIAGNOSTIC' or out['statement_sha256']!=p['spec']['statement_sha256'] or
                out['lp_sha256']!=p['spec']['statement']['lp_sha256'] or not out['isolated'] or
                not out['site_disabled'] or out['solver_or_model_imported'] or out['network_SAFE'] or out['network_UNSAFE']):
                raise ValueError('isolated checker result binding')
            status='CHECKED_LP_DIAGNOSTIC';complete=True
    except TimeoutError:status='TIMEOUT'
    except Exception as exc:error=repr(exc);status='ERROR'
    if time.monotonic()-start>=298:status='TIMEOUT';complete=False
    # Only driver-owned files. Partial files remain in timeout inventories.
    inventory={str(f.relative_to(root)):digest(f.read_bytes()) for f in root.rglob('*')
               if f.is_file() and f.name not in ('plan.json','driver.log')}
    save_new(root/'candidate.json',{'status':status,'complete_independent_check':complete,
        'stages':stages,'error':error,'artifact_sha256':inventory,
        'plan_sha256':digest((root/'plan.json').read_bytes()),'wall_seconds':time.monotonic()-start})


def review_candidate(root):
    root=Path(root);p=read(root/'plan.json');c=read(root/'candidate.json')
    if c['plan_sha256']!=digest((root/'plan.json').read_bytes()):raise ValueError('plan drift')
    for name,sha in c['artifact_sha256'].items():
        f=(root/name).resolve()
        if not f.is_relative_to(root.resolve()) or digest(f.read_bytes())!=sha:raise ValueError('artifact drift')
    previous=0
    for i,row in enumerate(c['stages']):
        if i>=4 or row['name']!=STAGES[i]:raise ValueError('phase order')
        limit=218 if i<2 else 298
        if (row['limit_seconds']!=limit or
            not previous<=row['start_seconds']<=row['end_seconds']<=c['wall_seconds'] or
            abs(row['end_seconds']-row['start_seconds']-row['elapsed_seconds'])>1e-8 or
            row['state']!=phase_state(row['process'],row['end_seconds'],limit) or
            row!=read(root/(row['name']+'_stage.json')) or
            read(root/(row['name']+'_entered.json'))!={'phase':row['name'],'seconds':row['start_seconds']}):
            raise ValueError('phase clock/identity')
        previous=row['end_seconds']
    required={n+'_stage.json' for n in STAGES[:len(c['stages'])]}
    required|={n+'_entered.json' for n in STAGES[:len(c['stages'])]}
    if len(c['stages'])==4 and all(r['state']=='COMPLETED' for r in c['stages']):
        required|={'prepared.json','packing.json','check.log','portable/verify.py','portable/bundle.json',
                   'proposal/bundle.json','proposal/native.json','proposal/diagnostic.json','proposal/terminal.json'}
    if not required<=set(c['artifact_sha256']):raise ValueError('missing artifact binding')
    expected='ERROR';complete=False
    if c['wall_seconds']>=298:expected='TIMEOUT'
    elif c['error']:expected='ERROR'
    elif len(c['stages'])==4 and all(r['state']=='COMPLETED' for r in c['stages']):
        pack=read(root/'packing.json');o=read(root/'check.log');t=read(root/'proposal/terminal.json')
        if (pack['original_started']!=p['started'] or pack['upstream_precheck_retained'] is not True or
            pack['checker_sha256']!=p['checker_sha256'] or
            digest((root/'portable/verify.py').read_bytes())!=pack['checker_sha256'] or
            digest((root/'portable/bundle.json').read_bytes())!=pack['bundle_sha256'] or
            pack['source_bundle_sha256']!=digest((root/'proposal/bundle.json').read_bytes()) or
            pack['bundle_sha256']!=pack['source_bundle_sha256'] or
            pack['statement_sha256']!=p['spec']['statement_sha256'] or
            t['deadline_monotonic']!=p['started']+218 or t['native_calls']!=1 or
            t['status']!='COMPLETED_DIAGNOSTIC' or
            o['status']!='CHECKED_LP_DIAGNOSTIC' or o['statement_sha256']!=pack['statement_sha256'] or
            o['lp_sha256']!=p['spec']['statement']['lp_sha256'] or
            not o['isolated'] or not o['site_disabled'] or o['solver_or_model_imported'] or
            o['network_SAFE'] or o['network_UNSAFE']):raise ValueError('proof/clock binding')
        native=read(root/'proposal/native.json')
        if (native['lp_sha256']!=p['spec']['statement']['lp_sha256'] or
            native['statement_sha256']!=p['spec']['statement_sha256'] or
            not 0<native['granted_seconds']<=60 or native['native_seconds']<0 or
            native['native_seconds']!=t['native_seconds'] or
            t['elapsed_seconds']<native['native_seconds']):raise ValueError('native accounting')
        # Compare component and isolated conclusions; neither native metadata nor
        # a JSON audit replaces the actual standalone rational check.
        local=read(root/'proposal/diagnostic.json')
        if any(o[k]!=v for k,v in local.items()):raise ValueError('checker differential')
        expected='CHECKED_LP_DIAGNOSTIC';complete=True
    elif c['stages'] and c['stages'][-1]['state']!='COMPLETED':expected=c['stages'][-1]['state']
    elif c['status']=='TIMEOUT' and len(c['stages'])<4:
        next_limit=218 if len(c['stages'])<2 else 298
        if c['wall_seconds']<next_limit:raise ValueError('unexplained timeout')
        expected='TIMEOUT'
    if (c['status'],c['complete_independent_check'])!=(expected,complete):raise ValueError('candidate acceptance')
    return c


def supervise(spec,destination,*,started):
    if not math.isfinite(started) or started>time.monotonic():raise ValueError('original start required')
    root=Path(destination).resolve()
    if not root.is_relative_to(Path('/data1/Kane/MOE')):raise ValueError('outside workspace')
    root.mkdir(exist_ok=False)
    save_new(root/'plan.json',{'spec':spec,'started':started,'total_seconds':300,'work_seconds':298,
        'proposal_deadline_seconds':218,'checker_sha256':digest((ROOT/'lp_sandwich/check.py').read_bytes())})
    owner=None;c=None;error=None;status='TIMEOUT'
    try:
        left(started)
        with (root/'driver.log').open('xb') as log:
            proc=subprocess.Popen([ACT,'-m','lp_diagnostic.flow',str(root)],cwd=ROOT,env=environment(),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            owner=wait_owned(proc,started+298)
        exited=time.monotonic()-started
        if owner['killed'] or exited>=298:status='TIMEOUT'
        elif owner['return_code']!=0:status='ERROR'
        else:c=review_candidate(root);status=c['status']
    except TimeoutError:exited=time.monotonic()-started
    except Exception as exc:exited=time.monotonic()-started;error=repr(exc);status='ERROR'
    observed=time.monotonic()-started
    if observed>=300:status='TIMEOUT'
    v={'schema':'SUPERVISED_LP_DIAGNOSTIC_V1','status':status,
       'complete_independent_check':status=='CHECKED_LP_DIAGNOSTIC' and bool(c and c['complete_independent_check']),
       'outer_process':owner,'driver_exit_seconds':exited,'observed_seconds':observed,'error':error,
       'plan_sha256':digest((root/'plan.json').read_bytes()),
       'candidate_sha256':digest((root/'candidate.json').read_bytes()) if c else None,
       'network_SAFE':False,'network_UNSAFE':False,'budget_seconds':300}
    save_new(root/'outer.json',v)
    save_new(root/'publication.json',{'outer_sha256':digest((root/'outer.json').read_bytes()),
        'observed_seconds':time.monotonic()-started})
    if time.monotonic()-started>=300:
        save_new(root/'publication_timeout.json',{'status':'TIMEOUT','complete_independent_check':False})
        v.update(status='TIMEOUT',complete_independent_check=False)
    return v


def audit(root):
    root=Path(root);v=read(root/'outer.json');p=read(root/'plan.json');pub=read(root/'publication.json')
    if (v['plan_sha256']!=digest((root/'plan.json').read_bytes()) or
        pub['outer_sha256']!=digest((root/'outer.json').read_bytes()) or
        (p['total_seconds'],p['work_seconds'],p['proposal_deadline_seconds'],v['budget_seconds'])!=(300,298,218,300) or
        not 0<=v['driver_exit_seconds']<=v['observed_seconds']<=pub['observed_seconds'] or
        v['network_SAFE'] or v['network_UNSAFE']):raise ValueError('outer clock/identity')
    owner=v['outer_process'];status='TIMEOUT';complete=False
    if owner and not owner['killed'] and v['driver_exit_seconds']<298 and v['observed_seconds']<300:
        if owner['return_code']!=0 or v['error']:status='ERROR'
        else:
            if v['candidate_sha256']!=digest((root/'candidate.json').read_bytes()):raise ValueError('candidate drift')
            c=review_candidate(root);status,complete=c['status'],c['complete_independent_check']
    elif owner is None and v['error'] and v['observed_seconds']<298:status='ERROR'
    if (v['status'],v['complete_independent_check'])!=(status,complete):raise ValueError('outer acceptance')
    if pub['observed_seconds']>=300 or (root/'publication_timeout.json').exists():
        return {**v,'status':'TIMEOUT','complete_independent_check':False}
    return v


def costs(root):
    root=Path(root);v=audit(root);parts={};unreadable=[]
    def saved(p):
        if not p.exists():return None
        try:return read(p)
        except (ValueError,UnicodeError):
            if v['status']!='TIMEOUT':raise
            unreadable.append(str(p.relative_to(root)));return None
    for name in STAGES:
        row=saved(root/(name+'_stage.json'));event=saved(root/(name+'_entered.json'))
        if row:parts[name]={'seconds':row['elapsed_seconds'],'state':row['state'],'censored':row['state']=='TIMEOUT'}
        elif event:parts[name]={'seconds':None,'state':'INTERRUPTED','censored':True,
            'observed_window_seconds':max(0,v['driver_exit_seconds']-event['seconds'])}
        else:parts[name]={'seconds':None,'state':'NOT_COMPLETED_OR_NOT_REACHED','censored':None}
    whole=read(root/'publication.json')['observed_seconds'];observed=sum(p['seconds'] for p in parts.values() if p['seconds'] is not None)
    if observed>whole+.01:raise ValueError('double-counted phase cost')
    native=saved(root/'proposal/native.json');component=saved(root/'proposal/terminal.json')
    return {'whole_diagnostic_seconds':whole,'phases':parts,'observed_phase_sum_seconds':observed,
        'residual_clock_seconds':whole-observed,'native_seconds':None if native is None else native['native_seconds'],
        'component_seconds':None if component is None else component['elapsed_seconds'],
        'native_calls':None if component is None else component['native_calls'],
        'unreadable_interrupted_records':unreadable,
        'identity':'whole = disjoint completed phase windows + residual; native/component nested in propose, never added again',
        'scope':'supplied-LP diagnostic only; historic network propagation excluded, no end-to-end MoE cost claim',
        'endpoint':'publication observation; late publication invalidates acceptance'}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);drive(p.parse_args().root)
