"""Saved-only full evidence audit plus construction mode/provenance binding."""
import argparse
import json
from pathlib import Path
import time
from scoped_proof.audit import audit as original_audit
from scoped_proof.io import load, sha
from scoped_parse_proof.contract import policy, validate_receipt


def audit(root, *, recheck=True):
    root=Path(root); start=time.monotonic()
    inv=load(root/'invocation.json'); spec=load(root/'spec.json',inv['spec_file_sha256'])
    p=policy(spec); result=original_audit(root,recheck=recheck)
    terminal=load(root/'terminal.json'); stages={s['phase']:s for s in terminal['stages']}
    construction_complete='construct' in stages and stages['construct']['status']=='COMPLETED'
    accepted=None
    if construction_complete or (root/'construction_receipt.json').exists():
        accepted=validate_receipt(root)
    if (root/'construction_acceptance.json').exists():
        if load(root/'construction_acceptance.json') != accepted: raise ValueError('construction acceptance drift')
    if 'source_check' in stages and stages['source_check']['status']=='COMPLETED':
        if accepted is None or not (root/'construction_acceptance.json').exists():
            raise ValueError('successful check missing construction receipt acceptance')
    result.update(construction_mode=p['mode'], construction_receipt_checked=accepted is not None,
        adapter_checker_cache=False, spec_sha256=inv['spec_file_sha256'],
        separate_audit_seconds=time.monotonic()-start)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    print(json.dumps(audit(a.root)))
