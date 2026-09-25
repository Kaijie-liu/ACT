"""Saved-only independent original-checker replay and finite cost audit."""
import argparse
import copy
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scoped_proof.io import load,save,sha
from source_enclosure.format import identity


def review():
    from readonly_source.execution import receive
    from source_cost_supervised.audit import cost_check,finite
    from residual_proof.check import check as original_check
    began=time.monotonic(); path=ROOT/'configs/backend_controls/readonly_source_r1.json'
    cfg=load(path); root=Path(cfg['output'])
    gate=load(ROOT/'docs/readonly_source_controls_20260925_r1.json',cfg['gate_sha256'])
    if (gate['status']!='PASS' or gate['tests']!=240 or gate['sources']!=cfg['sources']
            or cfg['budget_seconds']!=300 or cfg['threads']!=2 or cfg['reserve_seconds']!=2
            or cfg['rss_limit']!=8*2**30 or not cfg['no_retry_or_expansion']
            or sha(ROOT/'docs/parsed_source_reuse_audit_20260925_r1.json')!=cfg['parent_audit_sha256']):
        raise ValueError('controls/freeze identity')
    for name,digest in cfg['sources'].items():
        if sha(ROOT/name)!=digest: raise ValueError('frozen source changed: '+name)
    launch=load(root/'launch.json'); execution=load(root/'execution.json')
    modes=('none','copy','readonly')
    expected=[(name,repeat,mode) for name in ('repair_small','repair_medium') for repeat in range(3)
              for mode in modes[repeat:]+modes[:repeat]]
    actual=[(c['fixture_id'],c['repeat'],c['method']['representation']) for c in cfg['calls']]
    if (actual!=expected or launch['config']!=cfg or launch['config_sha256']!=sha(path)
            or execution['config_sha256']!=sha(path) or execution['no_retries'] is not True
            or execution['real_requests']!=0 or execution['native_solver_queries']!=0
            or [c['id'] for c in cfg['calls']]!=[r['id'] for r in execution['rows']]):
        raise ValueError('finite roster/identity')
    source_checks={}
    for call in cfg['calls']:
        key=call['fixture_id']; method=call['method']
        if key in source_checks: continue
        for name in ('source','construction'):
            r=method[name]
            if Path(r['path']).stat().st_size!=r['bytes'] or sha(r['path'])!=r['sha256']: raise ValueError('source bytes')
        doc=load(method['source']['path']); bundle=load(method['construction']['path']); t=time.monotonic()
        result=original_check(doc,bundle,invocation=bundle['invocation'],expected_source_sha256=call['spec']['source_sha256'],deadline=time.monotonic()+300)
        if identity(result)!=method['expected_result_sha256']: raise ValueError('original no-cache differential')
        source_checks[key]={'result':result,'separate_seconds':time.monotonic()-t}
    rows=[]; mutations=0
    for call,row in zip(cfg['calls'],execution['rows']):
        if load(root/(call['id']+'_terminal.json'))!=row: raise ValueError('batch terminal drift')
        if not row['launched']:
            if row['status']!='NOT_STARTED_RESOURCE' or row['seconds'] is not None: raise ValueError('admission')
            rows.append({'call':call,'status':row['status']}); continue
        folder=root/call['id']; returned=row['returned']; inv=load(folder/'invocation.json')
        terminal=load(folder/'terminal.json',returned['terminal_sha256']); cost=load(folder/'cost.json',terminal['cost_sha256'])
        if (load(folder/'spec.json',inv['spec_sha256'])!=call['spec'] or load(folder/'method.json')!=call['method']
                or inv['budget_seconds']!=300 or inv['rss_limit']!=8*2**30
                or not finite(row['caller_seconds']) or row['caller_seconds']<returned['seconds_including_terminal']
                or row['status']!=returned['status']): raise ValueError('call identity/cost')
        cost_check(inv,cost,terminal,returned)
        for key in ('overhead_seconds','seconds_before_ledger','budget_seconds'):
            bad=copy.deepcopy(cost); bad[key]=-1
            try: cost_check(inv,bad,terminal,returned)
            except ValueError: mutations+=1
            else: raise AssertionError('cost mutation accepted')
        item={'call':call,'status':row['status'],'returned':returned,'caller_seconds':row['caller_seconds'],'cost':cost}
        if row['status']=='COMPLETED':
            checked=receive(folder,identity(call['method']),deadline=time.monotonic()+300)
            if checked!=load(folder/'received.json',cost['received_sha256']): raise ValueError('receipt drift')
            candidate=load(folder/'candidate.json'); stats=load(folder/'parse_stats.json')['stats']
            if candidate['seconds_before_candidate']>cost['stages'][0]['seconds']: raise ValueError('worker cost')
            if candidate['result']!=source_checks[call['fixture_id']]['result']: raise ValueError('source verdict drift')
            item.update(candidate=candidate,stats=stats)
        rows.append(item)
    groups=[]; pairs=[]
    for name in ('repair_small','repair_medium'):
        for mode in modes:
            xs=[r for r in rows if r['call']['fixture_id']==name and r['call']['method']['representation']==mode and r['status']=='COMPLETED']
            group={'fixture':name,'representation':mode,'completed':len(xs),'required':3}
            for label,get in [('check_seconds',lambda r:r['candidate']['check_seconds']),
                              ('whole_seconds',lambda r:r['returned']['seconds_including_terminal']),
                              ('parse_total_seconds',lambda r:r['stats']['parser']['seconds']['total'])]:
                values=[get(r) for r in xs]
                group[label]={'median':statistics.median(values),'min':min(values),'max':max(values)} if values else None
            group['counts']=[{k:r['stats']['parser'][k] for k in ('lookups','hits','parses','peak_entries','peak_payload_bytes','peak_cells')} for r in xs]
            group['median_parser_components']={k:statistics.median(r['stats']['parser']['seconds'][k] for r in xs)
                for k in ('snapshot','identity_lookup','reference_parse','freeze','copy','retention')} if xs else None
            groups.append(group)
        for repeat in range(3):
            xs={r['call']['method']['representation']:r for r in rows if r['call']['fixture_id']==name and r['call']['repeat']==repeat}
            complete=all(xs[m]['status']=='COMPLETED' for m in modes)
            if complete:
                for k in ('lookups','hits','parses','peak_entries','peak_payload_bytes','peak_cells'):
                    if xs['copy']['stats']['parser'][k]!=xs['readonly']['stats']['parser'][k]: raise ValueError('cache policy/count drift')
            for baseline in ('none','copy'):
                pairs.append({'fixture':name,'repeat':repeat,'baseline':baseline,'complete':complete,
                    'readonly_minus_baseline_check_seconds':xs['readonly']['candidate']['check_seconds']-xs[baseline]['candidate']['check_seconds'] if complete else None,
                    'readonly_minus_baseline_whole_seconds':xs['readonly']['returned']['seconds_including_terminal']-xs[baseline]['returned']['seconds_including_terminal'] if complete else None})
    files={str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file()}
    return {'audit':'PASS','issues':0,'config_sha256':sha(path),'audit_source_sha256':sha(Path(__file__)),
        'launch_head':launch['head'],'sources_rechecked':len(cfg['sources']),'calls':18,
        'completed':sum(r['status']=='COMPLETED' for r in rows),'source_rechecks':source_checks,'rows':rows,
        'groups':groups,'paired_differences':pairs,'cost_mutations_rejected':mutations,'files':files,
        'real_requests':0,'native_solver_queries':0,'new_real_certificates':0,
        'separate_audit_seconds':time.monotonic()-began}


if __name__=='__main__':
    if not sys.flags.no_site: raise ValueError('python -S required')
    def forbid(event,args):
        if event=='import' and (args[0].split('.')[0] in ('torch','numpy','scipy','act','highspy') or
                args[0] in ('residual_proof.build','checked_route_frontier.build','shared_route_residual.propose',
                            'source_construction_lab.fixtures','parsed_source_reuse.check','parsed_source_reuse.cache',
                            'readonly_source.check','readonly_source.cache','readonly_source.view')):
            raise ImportError('saved audit forbids producer/solver/new representation: '+args[0])
        if event.startswith(('subprocess.','socket.')) or event in ('os.system','os.fork','os.exec'): raise PermissionError('saved-only')
    sys.addaudithook(forbid)
    p=argparse.ArgumentParser(); p.add_argument('--report',type=Path); a=p.parse_args(); result=review()
    if a.report: save(a.report,result)
    print({k:result[k] for k in ('audit','issues','calls','completed','sources_rechecked','cost_mutations_rejected','separate_audit_seconds')})
