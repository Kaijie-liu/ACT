"""Saved-only four-call full upstream audit, no production/solver execution."""
import argparse
import copy
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scoped_proof.io import load,save,sha
from source_enclosure.format import identity


def review():
    from readonly_upstream.audit import review as review_call,cost_check
    from source_cost_supervised.audit import finite
    start=time.monotonic(); path=ROOT/'configs/backend_controls/readonly_upstream_r1.json'
    cfg=load(path);root=Path(cfg['output'])
    gate=load(ROOT/'docs/readonly_upstream_controls_20260925_r1.json',cfg['gate_sha256'])
    if (gate['status']!='PASS' or gate['tests']!=264 or gate['sources']!=cfg['sources']
            or cfg['budget_seconds']!=300 or cfg['threads']!=2 or cfg['rss_limit']!=8*2**30
            or cfg['reserve_seconds']!=2 or cfg['no_retry_or_expansion'] is not True
            or cfg['parent_audit_sha256']!=sha(ROOT/'docs/readonly_source_audit_20260925_r1.json')):
        raise ValueError('frozen controls/resources')
    for name,digest in cfg['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('source changed: '+name)
    recipes=load(ROOT/'configs/backend_controls/source_cost_supervised_r1.json')['calls']
    expected=[]
    for i,spec in enumerate(recipes):
        for enabled in ((False,True) if i==0 else (True,False)):
            name=spec['id']+('_readonly' if enabled else '_direct')
            expected.append({'id':name,'fixture_id':spec['id'],'spec':dict(spec,id=name),
                'method':{'schema':'READONLY_UPSTREAM_R1','readonly':enabled}})
    launch=load(root/'launch.json');execution=load(root/'execution.json')
    if (cfg['calls']!=expected or len(expected)!=4 or launch['config']!=cfg or launch['config_sha256']!=sha(path)
            or execution['config_sha256']!=sha(path) or execution['no_retries'] is not True
            or execution['real_requests']!=0 or execution['native_solver_queries']!=0
            or [c['id'] for c in expected]!=[r['id'] for r in execution['rows']]):raise ValueError('roster/launch')
    rows=[];mutations=0
    for call,row in zip(cfg['calls'],execution['rows']):
        if load(root/(call['id']+'_terminal.json'))!=row:raise ValueError('batch terminal')
        if not row['launched']:
            if row['status']!='NOT_STARTED_RESOURCE' or row['seconds'] is not None:raise ValueError('admission')
            rows.append({'call':call,'status':row['status']});continue
        folder=root/call['id'];inv=load(folder/'invocation.json')
        if (load(folder/'spec.json',inv['spec_sha256'])!=call['spec'] or load(folder/'method.json')!=call['method']
                or inv['budget_seconds']!=300 or inv['rss_limit']!=8*2**30
                or not finite(row['caller_seconds']) or row['caller_seconds']<row['returned']['seconds_including_terminal']
                or row['status']!=row['returned']['status']):raise ValueError('call/cost')
        result=review_call(folder,row['returned'],recheck=True,expected=identity(call['method']))
        cost=load(folder/'cost.json');terminal=load(folder/'terminal.json')
        for key in ('seconds_before_ledger','overhead_seconds','budget_seconds','work_deadline_monotonic'):
            bad=copy.deepcopy(cost);bad[key]=-1
            try:cost_check(inv,bad,terminal,row['returned'])
            except ValueError:mutations+=1
            else:raise AssertionError('cost mutation accepted')
        item={'call':call,'status':row['status'],'audit':result,'cost':cost,
              'returned':row['returned'],'caller_seconds':row['caller_seconds']}
        if row['status']=='COMPLETED':
            report=load(folder/'profile/report.json');totals={}
            for op in report['operations']:
                v=totals.setdefault(op['phase']+':'+op['role'],{'seconds':0.,'calls':0})
                v['seconds']+=op['seconds'];v['calls']+=1
            item.update(profile=report,components=totals,parser=load(folder/'profile/parser_stats.json')['stats'])
        rows.append(item)
    pairs=[]
    for name in ('repair_small','repair_medium'):
        arms={r['call']['method']['readonly']:r for r in rows if r['call']['fixture_id']==name}
        complete=all(arms[b]['status']=='COMPLETED' for b in (False,True))
        entry={'fixture':name,'both_completed':complete}
        if complete:
            a,b=arms[False],arms[True]
            if a['profile']['files']!=b['profile']['files'] or a['profile']['source_check']!=b['profile']['source_check']:
                raise ValueError('newly generated upstream bytes/verdict differential')
            entry.update(source_and_construction_bytes_identical=True,source_conclusion_identical=True,
                readonly_minus_direct_whole_seconds=b['returned']['seconds_including_terminal']-a['returned']['seconds_including_terminal'],
                phase_differences={y['name']:y['seconds']-x['seconds'] for x,y in zip(a['profile']['phases'],b['profile']['phases'])})
        pairs.append(entry)
    if sum(r.get('caller_seconds',0) for r in rows)>execution['batch_seconds_before_summary']:raise ValueError('batch cost')
    files={str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file()}
    return {'audit':'PASS','issues':0,'config_sha256':sha(path),'audit_source_sha256':sha(Path(__file__)),
        'launch_head':launch['head'],'sources_rechecked':len(cfg['sources']),'calls':4,
        'completed':sum(r['status']=='COMPLETED' for r in rows),'rows':rows,'paired_differences':pairs,
        'cost_mutations_rejected':mutations,'files':files,'real_requests':0,'native_solver_queries':0,
        'new_real_certificates':0,'complete_output_positive_proof':False,
        'batch_seconds_before_summary':execution['batch_seconds_before_summary'],
        'separate_audit_seconds':time.monotonic()-start}


if __name__=='__main__':
    if not sys.flags.no_site:raise ValueError('python -S required')
    def forbid(event,args):
        if event=='import' and (args[0].split('.')[0] in ('torch','numpy','scipy','act','highspy') or
            args[0] in ('readonly_upstream.worker','readonly_source.check','readonly_source.cache','readonly_source.view',
                'parsed_source_reuse.check','parsed_source_reuse.cache','source_cost_supervised.worker',
                'source_cost_controls.profile','residual_proof.build','checked_route_frontier.build',
                'shared_route_residual.propose','source_construction_lab.fixtures','full_source.obligations')):
            raise ImportError('saved-only audit: '+args[0])
        if event.startswith(('subprocess.','socket.')) or event in ('os.system','os.fork','os.exec'):raise PermissionError('saved-only')
    sys.addaudithook(forbid)
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path);a=p.parse_args();result=review()
    if a.report:save(a.report,result)
    print({k:result[k] for k in ('audit','issues','calls','completed','sources_rechecked','cost_mutations_rejected','separate_audit_seconds')})
