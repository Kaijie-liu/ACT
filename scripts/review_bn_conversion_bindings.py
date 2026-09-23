"""Saved-identity BN impact ledger, not re-solving or all-domain proof.

Scope: JSON configs with a DIRECT files mapping for torch2act.py. Inclusion
in a source bundle is exposure, not proof that conversion executed or that
every positive result used an affected graph. Other binding schemas remain
outside this ledger's completeness claim.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write

ROOT=Path(__file__).resolve().parents[1]
SOURCE='act/pipeline/verification/torch2act.py'
OLD_HEAD='22b4b2f60fdbde32bcae0cfda871ccaab3e196a6'


def classify(digest,old,new):
    return ('PRE_REPAIR_SOURCE_EXPOSURE' if digest==old else
            'REPAIRED_SOURCE_BOUND_NOT_DOMAIN_PROOF' if digest==new else 'UNREVIEWED_CONVERTER_VERSION')


def review():
    old=hashlib.sha256(subprocess.check_output(['git','show',OLD_HEAD+':'+SOURCE],cwd=ROOT)).hexdigest()
    new=sha256(ROOT/SOURCE);rows=[];inventory={}
    for p in sorted((ROOT/'configs').rglob('*.json')):
        inventory[str(p.relative_to(ROOT))]=sha256(p)
        try:c=json.loads(p.read_text())
        except (ValueError,UnicodeError):continue
        files=c.get('files',{}) if isinstance(c,dict) else {}
        if not isinstance(files,dict):continue
        bindings={k:v for k,v in files.items() if k.endswith(SOURCE)}
        if not bindings:continue
        if len(set(bindings.values()))!=1:raise ValueError('conflicting converter identities: '+str(p))
        digest=next(iter(bindings.values()));root=Path(c.get('output_root','/nonexistent'))
        evidence=[]
        # Only complete worker-level records, never count internal solver
        # statuses as original-model outcomes. Missing records remain visible.
        if root.is_dir() and root.is_relative_to('/data1/Kane/MOE'):
            candidates=[root/'result.json',*sorted(root.glob('*/result.json')),*sorted(root.glob('*/*/result.json'))]
            for f in candidates:
                if not f.is_file():continue
                try:r=json.loads(f.read_text())
                except ValueError:continue
                if not isinstance(r,dict) or not ('arm' in r and 'request_id' in r):continue
                evidence.append({'file':str(f),'sha256':sha256(f),'request_id':r['request_id'],
                    'arm':r['arm'],'status':r.get('status'),'grade':r.get('evidence_grade'),
                    'exposure_applies_to_this_backend':r['arm']=='act'})
        rows.append({'config':str(p.relative_to(ROOT)),'config_sha256':sha256(p),'converter_sha256':digest,
            'classification':classify(digest,old,new),'protocol':c.get('protocol'),
            'output_root':str(root),'output_exists':root.is_dir(),'worker_records':evidence,
            'meaning':'binding exposure only; original author backend and original full-model replay do not execute ACT conversion'})
    anchors=[ROOT/'docs'/p for p in ('metamoe_bn_edge_finding_20260923_r1.md',
        'metamoe_assignment_layers_archive_20260923_r1.json','metamoe_bn_corrected_archive_20260923_r1.json')]
    return {'status':'SAVED_BINDING_REVIEW_COMPLETE','source':SOURCE,'old_git_object':OLD_HEAD,
        'old_sha256':old,'repaired_sha256':new,'rows':rows,'config_inventory':inventory,
        'counts':dict(Counter(r['classification'] for r in rows)),
        'direct_point_evidence':{str(p.relative_to(ROOT)):sha256(p) for p in anchors},
        'new_solver_calls':0,'new_model_forwards':0,'historical_files_modified':False,
        'limits':['Direct config.files schema only; not every historical manifest format.',
                  'Source binding is not executed-graph proof. Known MNIST0 expert mismatch is independently localized.',
                  'Do not promote old affected-graph conclusions to source claims; preserve old UNKNOWN/failed results.',
                  'Repaired BN controls establish finite point conformance, not all-domain lowering.',
                  'No automatic retraction/endorsement of unrelated BN-free/other-converter results.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    v=review();write(a.output,v);print(v['status'],v['counts'])
