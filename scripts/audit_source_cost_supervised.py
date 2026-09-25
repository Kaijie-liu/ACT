"""Independent saved-only archive of the frozen two-call repair diagnosis.

This script neither launches workers nor calls source producers or solvers.
Rerunning it only audits saved files; an output report cannot be overwritten.
"""
import argparse
import copy
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoped_proof.io import load, save, sha


def review():
    from source_cost_supervised.audit import review as review_call, cost_check, finite
    began = time.monotonic()
    config_path = ROOT/'configs/backend_controls/source_cost_supervised_r1.json'
    cfg = load(config_path); root = Path(cfg['output'])
    gate = load(ROOT/'docs/source_cost_supervised_controls_20260925_r1.json', cfg['gate_sha256'])
    if gate['status'] != 'PASS' or gate['tests'] != 180 or gate['sources'] != cfg['sources']:
        raise ValueError('controls binding')
    for name, digest in cfg['sources'].items():
        if sha(ROOT/name) != digest: raise ValueError('source drift: '+name)
    launch = load(root/'launch.json'); execution = load(root/'execution.json')
    if (launch['config'] != cfg or launch['config_sha256'] != sha(config_path)
            or execution['config_sha256'] != sha(config_path) or execution['no_retries'] is not True
            or execution['real_requests'] != 0 or execution['native_solver_queries'] != 0
            or [r['id'] for r in execution['rows']] != ['repair_small','repair_medium']
            or [c['id'] for c in cfg['calls']] != ['repair_small','repair_medium']
            or cfg['budget_seconds'] != 300 or cfg['rss_limit'] != 8*2**30):
        raise ValueError('frozen denominator/config')
    results = []; rejected = 0
    for spec, row in zip(cfg['calls'], execution['rows']):
        if load(root/(spec['id']+'_terminal.json')) != row: raise ValueError('batch terminal binding')
        if not row['launched']:
            if row['status']!='NOT_STARTED_RESOURCE' or row['seconds'] is not None: raise ValueError('unlaunched status')
            results.append({'id':spec['id'],'status':row['status']}); continue
        folder=root/spec['id']; inv=load(folder/'invocation.json')
        if (load(folder/'spec.json',inv['spec_sha256'])!=spec or inv['budget_seconds']!=300
                or inv['rss_limit']!=cfg['rss_limit']): raise ValueError('call identity/resource')
        if (row['status']!=row['returned']['status'] or not finite(row['caller_seconds'])
                or row['caller_seconds'] < row['returned']['seconds_including_terminal']):
            raise ValueError('enclosing call cost')
        audited=review_call(folder,row['returned'],recheck=True)
        terminal=load(folder/'terminal.json');cost=load(folder/'cost.json')
        for key in ('seconds_before_ledger','overhead_seconds','budget_seconds','work_deadline_monotonic'):
            bad=copy.deepcopy(cost);bad[key]=-1
            try:cost_check(inv,bad,terminal,row['returned'])
            except ValueError:rejected+=1
            else:raise AssertionError('cost corruption accepted')
        entry={'id':spec['id'],'fixture':spec['fixture'],'status':row['status'],'audit':audited,
               'returned':row['returned'],'caller_seconds':row['caller_seconds'],'cost':cost}
        if row['status']=='COMPLETED':
            report=load(folder/'profile/report.json'); totals={}
            for op in report['operations']:
                k=op['phase']+':'+op['role']
                v=totals.setdefault(k,{'seconds':0.,'calls':0})
                v['seconds']+=op['seconds'];v['calls']+=1
            entry.update(profile=report,components=totals)
        results.append(entry)
    if sum(r.get('caller_seconds',0) for r in results)>execution['batch_seconds_before_summary']:
        raise ValueError('batch cost too small')
    files={str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size}
           for p in sorted(root.rglob('*')) if p.is_file()}
    return {'audit':'PASS','issues':0,'config_sha256':sha(config_path),
            'audit_source_sha256':sha(Path(__file__)),'launch_head':launch['head'],
            'sources_rechecked':len(cfg['sources']),'calls':2,
            'completed_profiles':sum(r['status']=='COMPLETED' for r in results),
            'cost_mutations_rejected':rejected,'results':results,'files':files,
            'real_requests':0,'native_solver_queries':0,'new_real_certificates':0,
            'complete_output_positive_proof':False,'batch_seconds_before_summary':execution['batch_seconds_before_summary'],
            'separate_audit_seconds':time.monotonic()-began}


if __name__=='__main__':
    if not sys.flags.no_site:raise ValueError('python -S required')
    def forbid(event,args):
        if event=='import' and (args[0].split('.')[0] in ('torch','numpy','scipy','act','highspy')
                or args[0] in ('source_cost_supervised.worker','source_cost_controls.profile',
                              'residual_proof.build','checked_route_frontier.build',
                              'shared_route_residual.propose','source_construction_lab.fixtures',
                              'full_source.obligations')):
            raise ImportError('saved-only audit: '+args[0])
        if event.startswith(('subprocess.','socket.')) or event in ('os.system','os.fork','os.exec'):
            raise PermissionError('saved-only audit')
    sys.addaudithook(forbid)
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path);a=p.parse_args()
    result=review()
    if a.report:save(a.report,result)
    print({k:result[k] for k in ('audit','issues','completed_profiles','sources_rechecked','cost_mutations_rejected','separate_audit_seconds')})
