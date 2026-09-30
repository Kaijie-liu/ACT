"""Solver-free full-inventory differential of old H2 controls and opt-in kernels."""
import argparse
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.check_h2_factored import derive as parent_audit
from scoped_source.endpoint_source_check import reconstruct, reconstruct_mc
from scoped_source.factored_io import load, referenced
from scoped_source.sparse_check import check_bound as old_check
from scoped_source.rowwise_bound import check_bound
from scoped_source.rowwise_native import validate_native
from source_enclosure.format import identity

CONFIG='configs/h2_rowwise_controls_20260930.json'
PROTOCOL_SHA='e64c885cda177b238db08088e2079ea5db1245de5f4744eacbac1c5105f73872'
FILES=(CONFIG,'scoped_source/rowwise_bound.py','scoped_source/rowwise_native.py',
       'scoped_source/rowwise_tests.py','scripts/check_h2_rowwise.py')


def legacy_native():
    """Unchanged stdlib reference kernels, without ACT's package initializer."""
    def load_module(name,path):
        spec=importlib.util.spec_from_file_location(name,ROOT/path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
    base=load_module('old_lp_kernel','act/back_end/solver/lp_certificate.py')
    with patch.dict(sys.modules,{'act.back_end.solver.lp_certificate':base}):
        return load_module('old_sparse_kernel','act/back_end/solver/sparse_lp_certificate.py')


def derive():
    raw=(ROOT/CONFIG).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PROTOCOL_SHA: raise ValueError('frozen rowwise protocol')
    cfg=json.loads(raw); path=ROOT/cfg['parent_archive']
    archive_raw=path.read_bytes()
    if hashlib.sha256(archive_raw).hexdigest()!=cfg['parent_archive_sha256']:
        raise ValueError('fixed parent archive identity')
    parent=json.loads(archive_raw); root=Path(cfg['parent_root'])
    # Checks all sources/members, lowering, guards, objectives, reuse and old results.
    if parent_audit(root)!=parent: raise ValueError('old full-inventory audit changed')
    reference=legacy_native(); results=[]; lp_count=cert_count=fact_count=0
    for control in parent['controls']:
        deadline=time.monotonic()+cfg['per_control_deadline_seconds']
        def tick():
            if time.monotonic()>=deadline: raise TimeoutError('bounded archive differential')
        name,mode=control['case'],control['mode']; tree=root/(name+'-'+mode)
        doc=load(root,name+'-source.json'); _, request, _=reconstruct(doc,identity(doc),tick)
        manifest=load(tree,'manifest.json')
        if (identity(manifest)!=control['proof_manifest_sha256'] or
                manifest['source_manifest_sha256']!=control['source_manifest_sha256'] or manifest['mode']!=mode):
            raise ValueError('re-read parent manifest identity')
        by_pair={tuple(r['pair']):referenced(tree,r) for r in manifest['pairs']}
        for duty in request['duties']:
            tick(); pair=tuple(duty['pair']); j=duty['competitor']
            record=next(v for v in by_pair[pair]['duties'] if v['competitor']==j)
            targets=[]
            if mode=='endpoints':
                for end in record['endpoints']:
                    t=F(end['weight'])
                    lp={**duty['base'],'c':[str(t*F(a)+(1-t)*F(b)) for a,b in zip(duty['a']['c'],duty['b']['c'])],
                        'offset':str(t*F(duty['a']['offset'])+(1-t)*F(duty['b']['offset']))}
                    targets.append((end['weight'],lp,end['lp_sha256'],end['certificate']))
            else:
                targets.append((None,reconstruct_mc(duty),record['lp_sha256'],record['certificate']))
            for weight,lp,lp_sha,cert in targets:
                if identity(lp)!=lp_sha: raise ValueError('original target LP identity')
                native=validate_native(lp,deadline=deadline); lp_count+=1
                row={'case':name,'mode':mode,'pair':list(pair),'competitor':j,'weight':weight,
                     'lp_sha256':lp_sha,'native_validation':native}
                if mode=='mccormick' and record['origin']=='SOURCE_BOX_REUSE':
                    if cert['kind']!='SOURCE_BOX_FACT': raise ValueError('fact/dual distinction')
                    fact_count+=1; row.update(kind='SOURCE_BOX_FACT_CHECKED_BY_PARENT',certificate=cert)
                else:
                    if cert is None: raise ValueError('unexpected missing archived certificate')
                    old=old_check(lp,cert); value,residual=reference.evaluate(lp,cert)
                    current=check_bound(lp,cert,deadline=deadline)
                    if (old['checked_lower_bound']!=current['checked_lower_bound'] or
                            str(value)!=current['checked_lower_bound'] or list(map(str,residual))!=current['residual']):
                        raise ValueError('exact lower bound/residual differential')
                    if (native['rows_checked'],native['entries_checked'])!=(current['rows_checked'],current['entries_checked']):
                        raise ValueError('complete matrix traversal disagreement')
                    cert_count+=1; row.update(kind='DUAL_BOUND',certificate_sha256=identity(cert),
                        checked_lower_bound=current['checked_lower_bound'],residual_sha256=identity(current['residual']),
                        old_and_new_bound_equal=True,old_and_new_residual_equal=True,
                        zero_dual_rows_checked=current['zero_dual_rows_checked'])
                results.append(row); tick()
    if (len(parent['controls']),lp_count,cert_count,fact_count)!=(
            cfg['expected_packages'],cfg['expected_target_lps'],cfg['expected_dual_certificates'],cfg['expected_box_facts']):
        raise ValueError('fixed full-inventory counts')
    bounds=[F(r['checked_lower_bound']) for r in results if r['kind']=='DUAL_BOUND']
    return {'schema':'H2_ROWWISE_ARCHIVE_DIFFERENTIAL_V1','status':'PASS',
        'implementation_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES},
        'protocol_sha256':PROTOCOL_SHA,'parent_archive_sha256':cfg['parent_archive_sha256'],'parent_root':str(root),
        'packages':len(parent['controls']),'target_lps':lp_count,'dual_certificates':cert_count,'box_facts':fact_count,
        'positive_bounds':sum(v>0 for v in bounds),'negative_bounds':sum(v<0 for v in bounds),
        'zero_bounds':sum(v==0 for v in bounds),'records':results,'new_solves':0,'real_requests_started':0,
        'new_certificate_gain':False,'production_integration':False,'performance_claim':False,'hard_supervision':False}


def main():
    p=argparse.ArgumentParser(description=__doc__); g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--output',type=Path); g.add_argument('--check',type=Path); a=p.parse_args(); result=derive()
    if a.output:
        if not a.output.resolve().is_relative_to(Path('/data1/Kane/MOE')): raise ValueError('output outside project')
        with a.output.open('x') as f: json.dump(result,f,sort_keys=True,indent=2); f.write('\n')
    elif json.loads(a.check.read_text())!=result: raise ValueError('rowwise archive changed')
    print(json.dumps({k:result[k] for k in ('status','target_lps','dual_certificates','box_facts','positive_bounds','negative_bounds','new_solves')}))


if __name__=='__main__': main()
