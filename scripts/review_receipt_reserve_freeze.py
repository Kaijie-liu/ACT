"""Independent saved-only scope/identity review. Standard library, no runner.

This checks two protocols, not any positive network proof or performance claim.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'configs/backend_controls/metamoe_receipt_reserve_r1.json'
SCOPE=ROOT/'configs/backend_controls/source_output_closure_scope_r1.json'
REPORT=ROOT/'docs/metamoe_receipt_reserve_freeze_review_20260923_r1.json'


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def require(v,why):
    if not v:raise ValueError(why)


def review():
    began=time.monotonic();cfg=read(CFG)
    parent_path=ROOT/'configs/backend_controls/metamoe_las_followup_r1.json';parent=read(parent_path)
    gate_path=ROOT/'docs/metamoe_receipt_reserve_controls_20260923_r1.json';gate=read(gate_path)
    require(cfg['parent_sha256']==sha(parent_path) and cfg['controls_sha256']==sha(gate_path),'parent/control identity')
    require(gate['status']=='PASS' and gate['controls_passed'] is True and gate['tests']==55 and
        gate['real_requests_executed']==0 and all(r['returncode']==0 and r['tests']>0 for r in gate['rows']),'controls')
    worker='act/back_end/solver/native_feasibility_worker.py'
    require(all(cfg['files'][k]==h for k,h in parent['files'].items() if k!=worker),'unexpected old source rebinding')
    require(cfg['explicit_worker_rebinding']=={'path':worker,'old_sha256':parent['files'][worker],
        'new_sha256':sha(ROOT/worker)},'worker rebinding')
    for name,h in cfg['files'].items():require(sha(ROOT/name)==h,'frozen file drift: '+name)
    for name,h in gate['source_sha256'].items():require(cfg['files'][name]==h,'untested execution source')
    for name,h in gate['regression_sha256'].items():require(sha(ROOT/name)==h,'control changed')
    changed={k for k in set(parent)|set(cfg) if parent.get(k)!=cfg.get(k)}
    allowed={'protocol','output_root','requests','roster','arms','files','execution_commit','variants',
        'parent_sha256','controls_sha256','explicit_worker_rebinding','scope','single_factor','cost',
        'automatic_followup','numerical_guarantees_equated','backend_las_repair','repair_control_sha256','author_scope'}
    require(changed<=allowed,'unregistered configuration change')
    require(cfg['seconds']==300 and cfg['margin']==1e-7 and cfg['epsilon']==2/255 and
        cfg['group_rss_limit_bytes']==8*2**30 and cfg['act_options']==parent['act_options'],'resource/gate drift')
    selected=[]
    for dataset in ('CIFAR10','MNIST'):
        selected.extend([r for r in parent['requests'] if r['dataset']==dataset][:2])
    require(cfg['requests']==selected and [r['id'] for r in selected]==['cifar10_1','cifar10_2','mnist_1','mnist_3'],'selection drift')
    variants=['full_native','receipt_reserve']
    roster=[[r['id'],v] for i,r in enumerate(selected) for v in (variants if i%2==0 else variants[::-1])]
    require(cfg['roster']==roster and cfg['variants']==dict(zip(variants,[1.,.8])) and cfg['arms']==['act'],'comparison factor/order')
    require(cfg['automatic_followup'] is False and not Path(cfg['output_root']).exists(),'unexpected real launch')
    for name,h in gate['source_sha256'].items():
        blob=subprocess.check_output(['git','show',cfg['execution_commit']+':'+name],cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest()==h,'uncommitted tested source')

    # Independently reconstruct the second protocol's complete finite roster.
    s=read(SCOPE)
    require(s['status']=='PROTOCOL_ONLY_IMPLEMENTATION_GATE_PENDING' and s['execution_authorized'] is False
        and s['real_requests_executed']==0 and s['historical_bounds_or_facts_allowed'] is False,'scope authority')
    selection=read(ROOT/'act/pipeline/moe/configs/schedule_confirmation_100_selection_r1.json')
    sample=selection['samples'][0]
    require(s['model']==selection['models']['seed0'] and s['sample']=={k:sample[k] for k in
        ('sample_rank','dataset_index','label','center')} and sample['dataset_index']==4088,'same-object selection')
    pairs=[[i,j] for i in range(8) for j in range(i+1,8)]
    obligations=[{'pair':p,'label':7,'competitor':k} for p in pairs for k in range(10) if k!=7]
    require(s['pairs']==pairs and s['output_obligations']==obligations and s['required_output_obligations']==252
        and s['no_pair_exclusion'] is True,'route/property coverage')
    require(s['gate_range']==['0','1'] and s['requested_domain']['radius']=='2/255' and
        s['requested_domain']['old_materialized_endpoints_are_not_authoritative'] is True and
        Fraction(s['margin'])==Fraction.from_float(1e-7),'new source scope')
    for name,h in s['files'].items():require(sha(ROOT/name)==h,'source scope binding')
    require(sha(s['stored_request']['path'])==s['stored_request']['sha256'] and
        sha(s['model']['checkpoint'])==s['model']['checkpoint_sha256'],'stored object binding')
    return {'audit':'PASS','issues':0,'config_sha256':sha(CFG),'source_scope_sha256':sha(SCOPE),
        'implementation_commit':cfg['execution_commit'],'control_tests':55,
        'frozen_requests':4,'frozen_calls':8,'per_request_seconds':300,
        'real_calls_executed_or_queued':0,'new_output_root_absent':True,
        'changed_config_fields':sorted(changed),'all_old_file_bindings_unchanged_except':worker,
        'planned_source_pairs':28,'planned_source_output_obligations':252,
        'source_adapter_implemented_or_proof_claimed':False,
        'reviewer_sha256':sha(__file__),'review_seconds':time.monotonic()-began,
        'scope':'separate saved-only implementation by same researcher, not third-party technical review'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    record=review()
    if a.check:
        old=read(REPORT)
        require({k:v for k,v in old.items() if k!='review_seconds'}==
                {k:v for k,v in record.items() if k!='review_seconds'},'saved review drift')
        print('PASS, 0 issues; two scopes checked; real calls 0; no new certificate')
    else:
        with REPORT.open('x') as f:json.dump(record,f,indent=2,sort_keys=True,allow_nan=False)
        print('PASS, 0 issues; freeze only, no launch')
