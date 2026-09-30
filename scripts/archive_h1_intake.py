"""Read-only object-intake audit/derivation. No new LP queries or model loading."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scoped_proof.io import load, save, sha
from scoped_source.sparse_supervised import audit
from scoped_source.sparse_check import check
from scoped_source.sparse_ir import index


def derive(root):
    bindings=load(root/'implementation.json')
    for name,digest in bindings.items():
        if sha(root/'implementation'/name)!=digest or sha(ROOT/name)!=digest:
            raise ValueError('use the bound implementation, not changed source')
    frozen=load(root/'pre_execution_spec.json'); records=[]
    for name in ('dependency','full','mutate_model','capture_exception','capture_delay'):
        p=root/name; observed=load(p/'caller_observation.json'); checked=audit(p,observed)
        terminal=load(p/'terminal.json')
        row={'case':name,'status':observed['status'],'seconds':observed['seconds'],
            'audit':checked,'terminal_sha256':sha(p/'terminal.json'),
            'caller_sha256':sha(p/'caller_observation.json'),
            'stage_seconds':{s['phase']:s['seconds'] for s in terminal['stages']},
            'publication_and_overhead_seconds':observed['seconds']-terminal['stage_seconds']}
        if name in ('dependency','full'):
            doc=load(p/'bundle/source.json'); proof=load(p/'bundle/proof.json')
            result=check(doc,proof,expected_source_sha256=frozen['source_sha256'],deadline=time.monotonic()+30)
            if result!=load(p/'accepted.json')['result']: raise ValueError('mathematical recheck mismatch')
            events=[__import__('json').loads(line) for line in (p/'produce_events.jsonl').read_text().splitlines()]
            row.update(required=result['required'],positive=result['positive'],
                checked_lp_bounds=result['lp_bounds_checked'],
                missing=sum(r['lower_bound'] is None for r in result['obligations']),
                bounds=result['obligations'],source_blocks=result['source_blocks_checked'],
                source_nodes=result['source_nodes'],duty_union_nodes=result['duty_union_nodes'],
                lp_rows=result['lp_rows_reconstructed'],bundle_bytes=load(p/'built.json')['bundle_bytes'],
                producer_events={v['operation']:v['seconds'] for v in events if v['event']=='EXIT'},
                proposal_errors=[{'pair':r['pair'],'competitor':r['competitor'],'error':r['proposal_error']}
                    for r in proof['obligations'] if r['proposal_error'] is not None])
        records.append(row)
    full=root/'full'; doc=load(full/'bundle/source.json'); proof=load(full/'bundle/proof.json')
    _,nodes,outputs=index(doc,frozen['source_sha256'],lambda:None)
    endpoints={k for v in outputs.values() for k in v}
    dependency=load(root/'dependency/bundle/proof.json')
    actual={v for duty in dependency['obligations'] for v in duty['variables']}
    hidden=set(nodes)-endpoints
    if not hidden<=actual: raise ValueError('unexpected missing dense hidden dependencies')
    ranges={i:proof['bank'][key]['bounds'] for i,key in enumerate(outputs['router'])}
    conflicts=[]
    from itertools import combinations
    for pair in combinations(range(doc['request']['experts']),2):
        witnesses=[{'selected':i,'outside':j,'strict_gap':str(F(ranges[j][0])-F(ranges[i][1]))}
            for i in pair for j in ranges if j not in pair and F(ranges[j][0])>F(ranges[i][1])]
        if witnesses: conflicts.append({'pair':list(pair),'witnesses':witnesses})
    by={r['case']:r for r in records}
    return {'schema':'H1_MODEL_OBJECT_INTAKE_CONTROL_ARCHIVE_V1','root':str(root),
        'spec':frozen,'implementation':bindings,'cases':records,
        'counts':dict(Counter(r['status'] for r in records)),
        'same_observed_bounds':by['dependency']['bounds']==by['full']['bounds'],
        'hidden_and_input_nodes':len(hidden),'dependency_retains_all_hidden_and_input':True,
        'router_ranges_from_rechecked_full_source':{str(k):v for k,v in ranges.items()},
        'offline_interval_guard_conflicts':conflicts,
        'conflicts_used_by_online_acceptance':False,
        'proposal_native_exit_reason_recorded':False,
        'real_requests_started':0,'new_solves_during_archive':0,'performance_claim':False,
        'source_and_output_rechecked_with_standard_library':True,
        'native_float_proof':False,'route_change_witness_checked':False,
        'prior_attempts':[{'root':'/data1/Kane/MOE/baseline_runs/h1_intake_controls_20260930_r1',
            'tests':13,'failures':2,'reason':'Test expectations incorrectly required complete candidate bounds. Fixed seed/label/matrices unchanged; now check complete roster and fail-closed partial evidence. No acceptance gate changed.'},
            {'root':'/data1/Kane/MOE/baseline_runs/h1_intake_controls_20260930_r2',
             'tests':13,'failures':0,'reason':'Development pass; subsequent read-only review found tensor-instance serialization overrides. Final controls reject these too.'}]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--output',type=Path);g.add_argument('--check',type=Path)
    a=p.parse_args(); report=derive(a.root)
    if a.output: save(a.output,report)
    elif load(a.check)!=report: raise ValueError('archive mismatch')
    print({'status':'PASS','counts':report['counts'],'same_observed_bounds':report['same_observed_bounds'],
           'retained_hidden_and_input':report['hidden_and_input_nodes'],'new_solves':0})
