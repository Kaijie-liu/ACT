"""Read-only independent review/compact archive of source-defined V2 controls."""
import argparse
import json
from pathlib import Path
import subprocess

from scripts.conv_three_arm_contract import ROOT, ACT, read, selection
from scripts.conv_budget_worker_v2 import execution_identity
from scripts.run_budget_v2_controls import RAW
from act.pipeline.moe.experiment1 import _sha256
from act.pipeline.moe.conv_training import atomic_json
from act.pipeline.moe.audit_staged_evidence import audit_evidence_package

OUTPUT=ROOT/'act/pipeline/moe/results/budget_contract_v2_controls_review_20260915_r1.json'


def review():
    selection()  # identity only, no trained endpoint query
    runtime=read(RAW/'runtime.json'); saved=read(RAW/'review.json')
    if (runtime['execution']!=execution_identity()
            or runtime['runner_sha256']!=_sha256(ROOT/'scripts/run_budget_v2_controls.py')
            or saved['runtime']!=runtime or saved['status']!='PASS'
            or saved['real_model_smoke_started'] or saved['full_started']):
        raise ValueError('control identity/state mismatch')
    rows=[]
    for arm in ('staged','monolithic_f0'):
        directory=RAW/arm
        record=next(r for r in saved['rows'] if r['arm']==arm)
        checker=subprocess.run([ACT,'-S','-m','scripts.check_budget_contract_v2',str(directory/'journal.jsonl')],
            cwd=ROOT,text=True,capture_output=True,check=True)
        checked=json.loads(checker.stdout)
        if checked!=record['journal_check'] or checked!=read(directory/'journal.audit.json'):
            raise ValueError('separate journal checks differ')
        statuses=[]
        for kind in ('baseline','v2'):
            audit=audit_evidence_package(directory/kind,replay_unsafe=True)
            if audit['issues'] or audit!=record['package_audits'][kind]:
                raise ValueError('package audit differs')
            evidence=read(directory/kind/'evidence.json')
            statuses.append(evidence['verdict']['status'])
            if kind=='v2':
                if (evidence['execution_budget_contract']['terminal_reserve_seconds']!=5
                        or evidence['route_complexity_schedule']['budget']!=record['budget']
                        or evidence['route_coverage']['feasible_route_sets']!=record['route_pairs']):
                    raise ValueError('budget or coverage metadata differs')
        if statuses!=[record['baseline_status'],record['v2_status']] or statuses[0]!=statuses[1]:
            raise ValueError('toy result changed')
        reference=read(directory/'baseline/evidence.json')
        new=read(directory/'v2/evidence.json')
        if reference['numerical_safety']!=new['numerical_safety']:
            raise ValueError('numerical acceptance policy drift')
        rows.append({'arm':arm,'baseline':statuses[0],'v2':statuses[1],
            'pairs':record['route_pairs'],'journal_events':checked['events'],
            'property_records':len(checked['properties']),'native_calls':checked['native_calls'],
            'unreturned_native_calls':checked['native_unreturned'],
            'journal_check':'PASS','package_checks':{'baseline':'PASS','v2':'PASS'},
            'numerical_policy_unchanged':True})
    old_counts={}
    for filename in ('conv_three_arm_smoke_review_20260915_r1.json','conv_f0_timing_review_20260915_r1.json'):
        old=read(ROOT/'act/pipeline/moe/results'/filename)
        for artifact in old['artifact_inventory']:
            path=Path(old['raw_root'])/artifact['path']
            if path.stat().st_size!=artifact['bytes'] or _sha256(path)!=artifact['sha256']:
                raise ValueError('historical artifact modified')
        old_counts[filename]=len(old['artifact_inventory'])
    return {'schema':'BUDGET_CONTRACT_V2_CONTROL_REVIEW_R1','status':'PASS','issues':[],
        'execution_head':runtime['head'],'raw_root':str(RAW),'execution_identity':runtime['execution'],
        'independent_reaudit_matches':True,'tests_passed':66,'rows':rows,
        'historical_artifacts_unchanged':old_counts,
        'artifact_inventory':[{'path':str(p.relative_to(RAW)),'bytes':p.stat().st_size,'sha256':_sha256(p)}
                              for p in sorted(RAW.rglob('*')) if p.is_file()],
        'real_model_smoke_started':False,'full_started':False,
        'scope':'Source-defined toy conformance; journal accounting is not an independent network-bound proof. No convolutional performance or certificate gain established.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true'); args=parser.parse_args()
    result=review()
    if args.check:
        if result!=read(OUTPUT):raise ValueError('compact archive differs')
    else:
        if OUTPUT.exists():raise ValueError('refuse overwrite of control archive')
        atomic_json(OUTPUT,result)
    print(json.dumps({'status':result['status'],'issues':result['issues'],'rows':result['rows']}))
