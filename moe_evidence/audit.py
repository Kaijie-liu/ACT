"""Independent terminal, identity and conditional-proof review (not network reproof)."""
import json
import math
from pathlib import Path
import subprocess
from portable_proof.runtime import digest
from scripts.optional_evidence_dev_contract import ACT,read
from moe_evidence.execution import PHASES,LEVELS,accept
from moe_evidence.schema import validate_request


def audit_request(directory,request,arm):
    root=Path(directory);t=read(root/'terminal.json');r=request['evidence_request'];rid=validate_request(r)
    if (t['arm']!=arm or t['dataset_index']!=request['sample']['dataset_index'] or read(root/'request.json')!=request
            or t['request_sha256']!=digest((root/'request.json').read_bytes()) or t['budget_seconds']!=300
            or t['evidence_level']!=LEVELS[arm] or t['production_gate_changed'] or t['deployed_float_SAFE']
            or not math.isfinite(t['wall_seconds']) or t['wall_seconds']<0):raise ValueError('terminal identity/policy differs')
    actual={str(p.relative_to(root)):digest(p.read_bytes()) for p in root.rglob('*') if p.is_file() and p!=root/'terminal.json'}
    if actual!=t['artifact_sha256']:raise ValueError('partial/complete artifact inventory differs')
    phases=t['stages'];names=list(phases)
    # JSON keys are sorted by the writer; reconstruct the registered phase order.
    ordered={k:phases[k] for k in PHASES[arm] if k in phases}
    if set(names)!=set(PHASES[arm][:len(phases)]):raise ValueError('missing/reordered phase inventory')
    previous=0
    for name,v in ordered.items():
        if (not all(math.isfinite(v[k]) for k in ('start_seconds','elapsed_seconds','allowed_seconds'))
            or v['start_seconds']<previous or v['elapsed_seconds']<0 or v['allowed_seconds']<=0
            or v['start_seconds']+v['allowed_seconds']>298+.01
            or v['start_seconds']+v['elapsed_seconds']>t['wall_seconds']+.01):raise ValueError('invalid phase clock')
        if v['state']=='COMPLETED' and v['return_code']!=0:raise ValueError('failed child called complete')
        previous=v['start_seconds']+v['elapsed_seconds']
    if t['outer_timeout']!=any(v['state']=='OUTER_TIMEOUT' for v in phases.values()):raise ValueError('hidden timeout')
    result_file={'matched':'package/manifest.json','crown':'external.json','evidence':'check.log'}[arm]
    candidate,complete=accept(arm,ordered,lambda:read(root/result_file),t['wall_seconds'])
    if t['error']:
        if t['status']!='ERROR' and t['wall_seconds']<=300:raise ValueError('exception promoted')
    elif t['status']!=candidate or t['complete_independent_check']!=complete:raise ValueError('terminal promotion mismatch')
    detail={'status':'PASS','issues':[],'terminal':t['status'],'pairs':None,'facts':None,'replayed':False,
            'conditional_check':False,'complete_package':False}
    config=read(request['config']['path'])
    expected={'model_state':r['model_state'],'checkpoint':{'path':request['subject']['checkpoint'],
        'sha256':request['subject']['checkpoint_sha256']},**{k:r[k] for k in ('center','lower','upper')},
        'property':{'kind':'TOP1_ROBUST','clean_prediction':r['clean_prediction'],'classes':r['classes']},
        'epsilon':r['epsilon'],'config_sha256':digest(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode())}
    journal=root/'budget_journal.jsonl'
    if journal.exists():
        from scripts.check_budget_contract_v2 import check
        identity={'conditional_evidence_request':rid} if arm=='evidence' else {'request_sha256':t['request_sha256']}
        # Evidence capture intentionally stops before floating F0, with no WORK_COMPLETE.
        detail['journal']=check(journal,identity=identity,killed=arm=='evidence' or t['status'] in ('ERROR','TIMEOUT'))
    if arm=='matched':
        evidence=None
        if candidate not in ('ERROR','TIMEOUT'):
            from act.pipeline.moe.audit_staged_evidence import audit_evidence_package
            check=audit_evidence_package(root/'package',replay_unsafe=True)
            if check['status']!='PASS':raise ValueError(str(check))
            evidence=read(root/'package/evidence.json')
            if evidence['identity']!=expected or evidence['numerical_safety']!=config['numerical_safety']:
                raise ValueError('production package scope differs')
            detail.update(complete_package=True,replayed=t['status']=='UNSAFE',pairs=evidence['route_coverage']['feasible_route_sets'])
        if (root/'common_facts.json').exists():
            from act.pipeline.moe.common_fact_snapshot import check_snapshot
            snap=read(root/'common_facts.json');check_snapshot(snap,expected_identity=expected,expected_config=config,evidence=evidence)
            p=snap['payload']
            detail['facts']={'pairs':p['feasible_route_sets'],'branches':[
                {'expert':b['candidate'],'interval':b['proof_output_bounds']} for b in p['branches']]}
    elif arm=='evidence':
        if (root/'common_facts.json').exists():
            snap=read(root/'common_facts.json')
            if snap['request_id']!=rid or snap['identity']!={k:r[k] for k in ('model_state','center','lower','upper')}:
                raise ValueError('evidence source facts scope differs')
            detail['facts']={k:snap[k] for k in ('pairs','branches')}
        if complete:
            from moe_evidence.checker import check_manifest
            from moe_evidence.storage import loader
            result=check_manifest(read(root/'manifest.json'),r,loader(root))
            if result!=read(root/'independent.json') or result!=read(root/'check.log')['result']:
                raise ValueError('fresh request aggregation differs')
            info=read(root/'packing.json')
            run=subprocess.run([ACT,'-I','-S',str(root/'portable/verify.py'),'--bundle-hash',info['bundle_sha256'],
                '--statement-hash',info['statement_sha256']],cwd=root,capture_output=True,text=True,check=True,timeout=300)
            if json.loads(run.stdout)['result']!=result:raise ValueError('fresh portable check differs')
            detail.update(conditional_check=True,complete_package=True,pairs=result['route_pairs'])
    elif candidate not in ('ERROR','TIMEOUT'):
        # The frozen external arm is specifically E4/C10; not the generic proof API.
        if r['experts']!=4 or r['classes']!=10:raise ValueError('external audit scope E4/C10')
        from scripts.audit_conv_three_arm import external_result
        row={'job_id':root.name,'status':t['status'],'return_code':phases['crown']['return_code'],
            'outer_timeout':False,'package':None,**{key:digest((root/name).read_bytes()) for name,key in
                (('routes.json','routes_sha256'),('external.json','external_sha256')) if (root/name).exists()}}
        ext=external_result(root.parent,row,request)
        detail.update(complete_package=ext['complete'],replayed=ext['replayed'],pairs=ext['pairs'])
    return detail
