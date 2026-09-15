"""Fresh terminal/package/proof audit, never retroactive budget acceptance."""
import argparse
from pathlib import Path
import subprocess
import time

from portable_proof.runtime import digest
from scripts.optional_evidence_dev_contract import ACT, FREEZE, OUTPUT, ROOT, read, save, verify_freeze


def audit():
    started=time.monotonic()
    frozen=verify_freeze();runtime=read(OUTPUT/'runtime.json')
    if runtime['freeze_sha256']!=digest(FREEZE.read_bytes()):raise ValueError('freeze changed')
    rows=runtime['rows']
    if [r['arm'] for r in rows]!=frozen['arms'][:len(rows)] or runtime['unattempted']!=frozen['arms'][len(rows):]:
        raise ValueError('roster incomplete/reordered')
    details=[]
    for row in rows:
        d=OUTPUT/row['arm']
        if read(d/'terminal.json')!=row or read(d/'job.json')!=frozen['job']:
            raise ValueError('terminal/request identity mismatch')
        for name,sha in row['artifact_sha256'].items():
            p=d/name
            if not p.resolve().is_relative_to(d) or digest(p.read_bytes())!=sha:
                raise ValueError('artifact identity mismatch')
        if row['wall_seconds']>300 and row['status']!='TIMEOUT':raise ValueError('late success')
        if row['budget_seconds']!=300 or row['production_gate_changed'] or row['deployed_float_SAFE']:
            raise ValueError('acceptance contract changed')
        last=0.
        for stage in sorted(row['stages'].values(), key=lambda item:item['start_seconds']):
            if stage['start_seconds']<last or abs(stage['allowed_seconds']-(298-stage['start_seconds']))>.05:
                raise ValueError('stage clock reset/overlap')
            last=stage['start_seconds']+stage['elapsed_seconds']
        detail={'arm':row['arm'],'status':row['status'],'wall_seconds':row['wall_seconds'],
                'stages':row['stages'],'evidence_level':row['evidence_level']}
        if row['arm']=='production_matched_v2' and (d/'package').exists():
            from act.pipeline.moe.audit_staged_evidence import audit_evidence_package
            checked=audit_evidence_package(d/'package',replay_unsafe=True)
            if checked['status']!='PASS':raise ValueError('production package audit failed')
            evidence=read(d/'package/evidence.json');req=frozen['job']['parent_request']
            identity=evidence['identity']
            if identity['model_state']!=req['subject']['model_state'] or any(identity[k]!=req['sample'][k] for k in ('center','lower','upper')):
                raise ValueError('production model/input identity differs')
            detail['package_audit']=checked
        if row['arm']=='optional_checked_evidence':
            grants=read(d/'evidence_grants.json') if (d/'evidence_grants.json').exists() else []
            if len(grants)>frozen['job']['protocol']['max_lp_queries']:raise ValueError('extra LP query')
            for g in grants:
                if not 0<g['grant_seconds']<=min(60,220-g['entered_seconds'])+.05:
                    raise ValueError('LP grant outside common budget')
            detail['proposal_calls']=len(grants)
            if row['complete_independent_check']:
                if row['stages']['check']['state']!='COMPLETED':raise ValueError('unchecked positive')
                info=read(d/'packing.json');meta=read(d/'portable/bundle.json')
                from scripts.check_conv_request_sign_lp import expected_scope
                if meta['statement']['request']!=expected_scope(frozen['job']) or meta['statement']['routes']['feasible']!=[[1,2]]:
                    raise ValueError('proof about different input/model/pair')
                t=time.monotonic()
                process=subprocess.run([ACT,'-I','-S',str(d/'portable/verify.py'),
                    '--bundle-hash',info['bundle_sha256'],'--statement-hash',info['statement_sha256']],
                    cwd=d,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=150)
                if process.returncode:raise ValueError('fresh portable recheck failed: '+process.stderr[-1000:])
                import json
                fresh=json.loads(process.stdout)['result'];original=read(d/'check.log')['result']
                if fresh!=original:raise ValueError('fresh proof differs')
                if row['status']=='CHECKED_CONDITIONAL' and fresh['positive_obligations']!=9:
                    raise ValueError('incomplete positive')
                detail.update(result=fresh,portable_bytes=info['bundle_bytes'],
                              archival_recheck_seconds=time.monotonic()-t)
            elif row['status']=='CHECKED_CONDITIONAL':raise ValueError('missing independent check')
        details.append(detail)
    return {'schema':'OPTIONAL_EVIDENCE_DEV_V1_REVIEW','status':'PASS','issues':[],
            'execution_head':runtime['execution_head'],'classification':runtime['classification'],
            'execution_state':runtime['state'],'completed':len(rows),'planned':2,
            'details':details,'audit_seconds':time.monotonic()-started,
            'production_gate_changed':False,'speedup_claim':False,'new_generalization_claim':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):raise ValueError('new output required')
    result=audit();save(args.output,result)
    print(result['status'],[(d['arm'],d['status'],d['wall_seconds']) for d in result['details']])
