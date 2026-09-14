"""Recheck all supplied real-request LP evidence and archive compact identities."""
import argparse
import json
from pathlib import Path
from fractions import Fraction
from act.pipeline.moe.check_request_lp import check_directory
from act.pipeline.moe.request_lp_control import frozen_request
from act.pipeline.moe.paired_followup import save
from act.pipeline.moe.experiment1 import PROJECT_ROOT, _sha256
from act.back_end.solver.lp_certificate import identity
from act.back_end.solver.check_hz_lp_export import check_export


def build():
    root=PROJECT_ROOT/'data/moe/results/request_lp_20260914_r1'
    manifest=json.loads((root/'manifest.json').read_text()); launch=json.loads((root/'launch.json').read_text())
    request=frozen_request()
    if launch['request']!=request or manifest['request']!=request: raise ValueError('request identity drift')
    checked=check_directory(root,expected_request_id=identity(request))
    if checked!=json.loads((root/'check.json').read_text()): raise ValueError('aggregation mismatch')
    proofs={}
    # Independently check ALL exports, including membership rows that did not
    # establish a positive fact, not merely those selected by aggregation.
    def read(ref):
        p=(root/ref['file']).resolve()
        if not p.is_relative_to(root) or _sha256(p)!=ref['sha256']: raise ValueError('proof ref mismatch')
        return json.loads(p.read_text())
    for key,entry in manifest['proofs'].items():
        export=read(entry['export']);certificate=read(entry['certificate']) if entry['certificate'] else None
        result=check_export(export,certificate,expected_source_sha256=entry['hz_sha256'])
        bound=result['bound']['checked_lower_bound'] if certificate else None
        if bound is not None and bound!=entry['checked_lower_bound']: raise ValueError('recorded bound mismatch')
        proofs[key]={'kind':entry['kind'],'scope':entry['scope'],'property_index':entry['property_index'],
                     'status':entry['status'],'checked_lower_bound':bound,'n_relaxed_binaries':export['n_relaxed_binaries'],
                     'factors':len(export['lp']['c'])}
    return {'classification':'SEPARATE_OBSERVED_REQUEST_LP_CONTROL_NOT_PERFORMANCE',
            'launch':launch,'terminal':json.loads((root/'terminal.json').read_text()),'check':checked,
            'routes':manifest['routes'],'obligations':manifest['obligations'],'proofs':proofs,
            'checked_lp_count':sum(p['checked_lower_bound'] is not None for p in proofs.values()),
            'positive_lp_count':sum(p['checked_lower_bound'] is not None and Fraction(p['checked_lower_bound'])>Fraction(1e-7) for p in proofs.values()),
            'raw_root':str(root),'raw_hashes':{p.name:_sha256(p) for p in sorted(root.iterdir()) if p.is_file()},
            'audit':'ALL_STORED_HZ_EXPORTS_AND_AVAILABLE_DUALS_RECHECKED_NO_SOLVE'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=build();save(args.output,result)
    print(json.dumps({'check':result['check'],'checked_lp_count':result['checked_lp_count'],'positive_lp_count':result['positive_lp_count']},indent=2))
