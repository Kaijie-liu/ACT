"""Saved-only full-cost/source/route/bound audit. No model or solver import."""
import argparse
import json
import math
from pathlib import Path
import time

from frontier_proof.contract import (accepted_routes, file_record, output_inputs,
                                    phases, policy, route_inputs)
from frontier_proof.supervisor import accept
from scoped_proof.evidence import POSITIVE, bind_source, roster
from scoped_proof.io import load, sha
from source_enclosure.format import identity


def audit(root, *, recheck=True):
    root = Path(root); started = time.monotonic()
    inv = load(root/'invocation.json'); spec = load(root/'spec.json', inv['spec_file_sha256'])
    mode = policy(spec)['mode']; order = phases(spec)
    cost = load(root/'cost.json'); terminal = load(root/'terminal.json',cost['terminal_sha256'])
    receipt = load(root/'receipt.json',cost['receipt_sha256']); token = inv['invocation']
    scope = spec['scope']; required = len(roster(scope)); budget = inv['budget_seconds']
    if (any(obj['invocation'] != token for obj in (cost,terminal,receipt)) or
            receipt['terminal_sha256'] != cost['terminal_sha256'] or
            any(obj['request_sha256'] != identity(scope) for obj in (cost,terminal)) or
            not 0 < budget <= 300 or cost['budget_seconds'] != budget or terminal['budget_seconds'] != budget or
            terminal['required'] != required):
        raise ValueError('identity/budget/denominator')
    if (abs(inv['deadline_monotonic']-inv['started_monotonic']-budget)>1e-8 or
            abs(inv['deadline_monotonic']-inv['work_deadline_monotonic']-min(2.,budget/10))>1e-8):
        raise ValueError('absolute deadline/publication reserve')
    stages = terminal['stages']; names = [r['phase'] for r in stages]
    if names != list(order[:len(names)]): raise ValueError('phase coverage/order')
    previous = 0.
    for row in stages:
        if load(root/(row['phase']+'_stage.json')) != row: raise ValueError('stage receipt changed')
        a,b,s = (row[k] for k in ('start_seconds','end_seconds','seconds'))
        if not all(math.isfinite(v) for v in (a,b,s)) or not previous<=a<=b or not 0<=s<=b-a+1e-6:
            raise ValueError('stage cost')
        target = (inv['started_monotonic']+a+inv['work_deadline_monotonic'])/2 if row['phase']=='propose' else inv['work_deadline_monotonic']
        if (abs(row['deadline_monotonic']-target)>1e-8 or row['cleanup_included'] is not True or
                row['status'] == 'COMPLETED' and inv['started_monotonic']+b >= target):
            raise ValueError('deadline renewed/late phase')
        previous = b
    values = [cost[k] for k in ('end_to_end_seconds','stage_seconds','overhead_seconds')]
    if (not all(math.isfinite(v) and v>=0 for v in values) or
            abs(values[1]-sum(r['seconds'] for r in stages))>1e-8 or abs(values[0]-values[1]-values[2])>1e-8 or
            values[0]<previous or values[0]<terminal['seconds_before_publication'] or values[0]<receipt['seconds_before_receipt']):
        raise ValueError('full cost does not close')
    overrun = (root/'publication_timeout.json').exists()
    status = 'TIMEOUT' if overrun else cost['status']
    if (status not in (POSITIVE,'NOT_CLOSED','TIMEOUT','ERROR','RESOURCE_LIMIT') or
            receipt['status'] != terminal['status_before_publication'] or
            not overrun and cost['complete_output_positive_proof'] != (status==POSITIVE) or
            values[0]>=budget and status!='TIMEOUT'):
        raise ValueError('terminal status inconsistency')
    by_stage = {r['phase']:r for r in stages}
    frontier_result, result = None, None
    bundle = None
    doc = None
    if (root/'source.json').exists():
        doc = load(root/'source.json'); bind_source(doc,scope)
    if mode=='checked_frontier' and (root/'router_prefix.json').exists():
        doc, pre, route_rows, route_context = route_inputs(root,time.monotonic()+300,require_complete=False)
        if by_stage.get('route_propose',{}).get('status')=='COMPLETED' and route_context['completion'] is None:
            raise ValueError('completed route phase without receipt')
        if recheck:
            from checked_route_frontier.check import check_frontier
            frontier_result = check_frontier(doc,pre,route_rows,expected_source_sha256=bind_source(doc,scope),deadline=time.monotonic()+300)
        if (root/'route_check.json').exists():
            _,_,_,accepted = accepted_routes(root,time.monotonic()+300)
            if recheck and accepted['frontier'] != frontier_result:
                raise ValueError('route proof changed')
        if by_stage.get('route_check',{}).get('status')=='COMPLETED' and not (root/'route_check.json').exists():
            raise ValueError('missing route check')
    if by_stage.get('route_propose',{}).get('status')=='COMPLETED' and not (root/'router_prefix.json').exists():
        raise ValueError('missing completed router prefix')
    if (root/'construction.json').exists():
        bundle = load(root/'construction.json')
        if mode=='checked_frontier':
            d,p,r,accepted = accepted_routes(root,time.monotonic()+300)
            if bundle['prefix']!=p or bundle['route_candidates']!=r or bundle['frontier']!=accepted['frontier']:
                raise ValueError('source/route bundle differs')
            record = {'invocation':token,'request_sha256':identity(scope),
                      'construction':file_record(root/'construction.json'),'route_check':file_record(root/'route_check.json')}
            if (root/'construction_receipt.json').exists():
                if load(root/'construction_receipt.json') != record: raise ValueError('construction receipt')
            elif by_stage.get('construct',{}).get('status')=='COMPLETED':
                raise ValueError('missing completed construction receipt')
        if (root/'source_check.json').exists() and recheck:
            if mode=='checked_frontier':
                from checked_route_frontier.check import check
            else:
                from scoped_source.check import check
            source = check(doc,bundle,expected_source_sha256=bind_source(doc,scope),deadline=time.monotonic()+300)
            if source != load(root/'source_check.json'): raise ValueError('source check differs')
    if by_stage.get('source_check',{}).get('status')=='COMPLETED' and not (root/'source_check.json').exists():
        raise ValueError('missing source receipt')
    if recheck and (root/'evidence_check.json').exists():
        bundle = load(root/'construction.json')
        if mode=='checked_frontier':
            from checked_route_frontier.evidence import aggregate
            candidates, complete = output_inputs(root,scope,bundle,token,time.monotonic()+300)
        else:
            from scoped_proof.evidence import aggregate
            candidates={int(p.stem):load(p) for p in sorted((root/'candidates').glob('*.json'))}
            complete=(root/'proposal_complete.json').exists()
            if complete:
                m=load(root/'proposal_complete.json')
                expected={f'{i:04d}.json' for i in range(required)}
                if (m['invocation']!=token or m['request_sha256']!=identity(scope) or m['required']!=required or
                        set(m['files'])!=expected or any(file_record(root/'candidates'/n)!=v for n,v in m['files'].items())):
                    raise ValueError('exhaustive completion receipt')
        result=aggregate(scope,doc,bundle,candidates,invocation=token,proposal_complete=complete,deadline=time.monotonic()+300)
        if result!=load(root/'evidence_check.json'): raise ValueError('fresh output proof differs')
    if status==POSITIVE:
        if names!=list(order) or any(r['status']!='COMPLETED' for r in stages):
            raise ValueError('positive after partial/failed phases')
        candidate=accept(root,scope,token,mode)
        if candidate['status']!=POSITIVE or recheck and (result is None or result['status']!=POSITIVE):
            raise ValueError('positive proof absent')
    return {'audit':'PASS','issues':0,'mode':mode,'effective_status':status,'required':required,
        'end_to_end_seconds':values[0],'stage_seconds':{r['phase']:r['seconds'] for r in stages},
        'route_bounds_received':None if frontier_result is None else len(frontier_result['bounds']),
        'checked_excluded_pairs':None if frontier_result is None else frontier_result['excluded_pairs'],
        'checked_bounds':None if result is None else result['checked_bounds'],
        'discharged_by_exclusion':None if result is None else result.get('discharged_by_exclusion',0),
        'published_expert_traces':None if bundle is None else len(bundle['experts'] if mode=='checked_frontier' else bundle['networks'][1:]),
        'published_pairs':None if bundle is None else len(bundle['pairs']),
        'construction_independently_rechecked':recheck and (root/'source_check.json').exists(),
        'sampled_peak_rss':cost['sampled_peak_rss'],
        'ledger':{'cost_sha256':sha(root/'cost.json'),'terminal_sha256':cost['terminal_sha256'],'receipt_sha256':cost['receipt_sha256']},
        'new_solves':0,'separate_audit_seconds':time.monotonic()-started,'complete_output_positive_proof':status==POSITIVE}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    print(json.dumps(audit(a.root.resolve())))
