"""Independent receipt and cost validation; no producer or new checker import."""
import argparse
from collections import Counter
from pathlib import Path
import time

from scoped_proof.io import load,save,sha,tick
from source_cost_supervised import audit as original
from readonly_upstream.execution import validate_method

POLICY='READONLY_EXACT_SOURCE_VIEW_R1'
cost_check=original.cost_check
journal_check=original.journal_check


def receive(root,inv,spec,*,deadline,expected=None):
    root=Path(root); expected=sha(root/'method.json') if expected is None else expected
    method=load(root/'method.json',expected,limit=65536); validate_method(method)
    out=original.receive(root,inv,spec,deadline=deadline)
    candidate=load(root/'candidate.json'); report=load(root/'profile/report.json')
    binding=report['upstream_method']; record=binding['parser_stats_record']
    original.file_check(root/'profile/parser_stats.json',record,deadline)
    data=load(root/'profile/parser_stats.json',record['sha256'],limit=2**20)
    for value in (binding,data,candidate):
        if (value['invocation']!=inv['invocation'] or value['spec_sha256']!=inv['spec_sha256']
                or value['method_sha256']!=expected): raise ValueError('upstream mode/invocation binding')
    if binding['schema']!='READONLY_UPSTREAM_REPORT_R1' or binding['readonly'] is not method['readonly']:
        raise ValueError('upstream representation')
    stats=data['stats']; p=stats['parser']; end=stats['after_close']; seconds=p['seconds']
    check_seconds=next(x['seconds'] for x in report['phases'] if x['name']=='source_check')
    if (stats['schema']!='SOURCE_PARSE_REUSE_CHECK_R1' or stats['status']!='COMPLETED'
            or stats['complete_output_positive_proof'] is not False or stats['native_solver_queries']!=0
            or p['policy']!=POLICY or end['policy']!=POLICY or p['enabled'] is not method['readonly']
            or p['scope']!=[inv['invocation'],spec['source_sha256']] or p['results_or_bounds_cached'] is not False
            or p['failures']!=0 or p['lookups']!=p['hits']+p['parses']
            or (not method['readonly'] and (p['hits'] or p['live_entries']))
            or not end['closed'] or end['live_entries'] or end['live_cells'] or end['live_payload_bytes']
            or not all(original.finite(v) for v in seconds.values()) or not original.finite(p['other_seconds'])
            or abs(sum(v for k,v in seconds.items() if k!='total')+p['other_seconds']-seconds['total'])>1e-8
            or not original.finite(stats['seconds_before_stats'])
            or not seconds['total']<=stats['seconds_before_stats']<=check_seconds):
        raise ValueError('all parser checks/cost/closure required')
    frontier=report['source_check']['frontier']; n=len(frontier['needed_experts']); k=frontier['retained_pairs']
    expected_counts={('prefix','router_propagation'):1,('construct','expert_propagation'):n,
        **{('construct',s):k for s in ('factor_join','guards','projection','weighted_lp')},
        ('source_check','route_recheck'):1,('source_check','expert_check'):n,
        **{('source_check',s):k for s in ('join_check','guard_check','projection_check','weighted_lp_check')}}
    actual=Counter((v['phase'],v['role']) for v in report['operations'])
    if actual!=Counter({key:v for key,v in expected_counts.items() if v}): raise ValueError('upstream component inventory')
    traces=original.journal_check(root/'journal.jsonl',inv,complete=True,deadline=deadline)
    if [v['name'] for v in traces['finished'] if v['kind']=='worker']!=['generate','profile']:
        raise ValueError('upstream generation/profile missing')
    # Parser costs are nested within predicate costs, never added twice.
    tick(deadline); return out


def review(root,returned,*,recheck=False,expected=None):
    root=Path(root); start=time.monotonic()
    out=original.review(root,returned,recheck=recheck)
    if returned['status']=='COMPLETED':
        inv=load(root/'invocation.json'); spec=load(root/'spec.json',inv['spec_sha256'])
        receipt=receive(root,inv,spec,deadline=time.monotonic()+300,expected=expected)
        if receipt!=out['receipt']: raise ValueError('receipt differential')
    out['separate_audit_seconds']=time.monotonic()-start
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('--method-sha256',required=True)
    a=p.parse_args(); inv=load(a.root/'invocation.json',limit=65536)
    spec=load(a.root/'spec.json',inv['spec_sha256'],limit=65536)
    out=receive(a.root,inv,spec,deadline=inv['work_deadline_monotonic'],expected=a.method_sha256)
    save(a.root/'received.json',out); tick(inv['work_deadline_monotonic'])
