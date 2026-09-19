"""Fresh read-only reconstruction of inventory and analytic evidence; no import."""
import argparse
from fractions import Fraction
from pathlib import Path
import time
from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest
from native_import_audit.inventory import run as inventory
from native_import_audit.run import sources
from native_import_audit.probe import runtime
from sparse_supervised.study import verify


def review(receipt, output):
    begin=time.monotonic();v=read(receipt);verify()
    if v['status']!='PASS' or v['issues'] or v['sources']!=sources() or v['runtime']!=runtime():
        raise ValueError('analysis identity')
    if inventory(v['runtime']['defaults'])!=v['inventory']:
        raise ValueError('read-only inventory mismatch')
    root=Path(v['artifact_root'])
    actual={str(p.relative_to(root)):digest(p.read_bytes()) for p in sorted(root.rglob('*')) if p.is_file()}
    if actual!=v['artifact_sha256']:raise ValueError('analytic artifact drift')
    threshold=Fraction(v['runtime']['defaults']['small_matrix_value'])
    for case in v['analytic_cases']:
        folder=root/case['case']
        if read(folder/'result.json')!=case or (folder/'native.log').read_text()!=case['log']:
            raise ValueError('saved case binding')
        expected=case['intended'];actual=case['readback']
        if any(expected[k]!=actual[k] for k in ('cost','offset','lower','upper','sense')):
            raise ValueError('nonmatrix field changed')
        if len(expected['rows'])!=len(actual['rows']):raise ValueError('row count drift')
        removed=0
        for before,after in zip(expected['rows'],actual['rows']):
            if (before['lower'],before['upper'])!=(after['lower'],after['upper']):raise ValueError('row bound drift')
            retained=[[j,x] for j,x in before['entries'] if abs(Fraction(x))>threshold]
            if after['entries']!=retained:raise ValueError('not precisely tiny-entry removal')
            removed+=len(before['entries'])-len(retained)
        if case['status']!=('HighsStatus.kWarning' if removed else 'HighsStatus.kOk'):
            raise ValueError('warning status mismatch')
        if case['optimization_calls'] or case['real_native_imports']:raise ValueError('scope violation')
    for field,name in (('analytic_semantic_witness','observed_min_A'),('signed_semantic_witness','observed_signed_A')):
        witness=v[field];point=list(map(Fraction,witness['point']))
        case=next(c for c in v['analytic_cases'] if c['case']==name)
        for label,side in (('original','intended'),('imported','readback')):
            row=case[side]['rows'][0]
            lhs=sum((Fraction(x)*point[j] for j,x in row['entries']),Fraction(0))
            box=all(Fraction(l)<=x<=Fraction(u) for l,x,u in zip(case[side]['lower'],point,case[side]['upper']))
            if str(lhs)!=witness[label+'_lhs'] or (box and lhs<=Fraction(row['upper']))!=witness[label+'_feasible']:
                raise ValueError('semantic witness arithmetic')
    result={'status':'PASS','issues':[],'receipt_sha256':digest(receipt.read_bytes()),
            'analytic_cases':len(v['analytic_cases']),'artifact_files':len(v['artifact_sha256']),
            'small_matrix_count':v['inventory']['small_matrix_count'],'exact_semantic_witnesses':2,
            'sources':sources(),'seconds':time.monotonic()-begin,'new_native_imports':0,'optimization_calls':0,
            'scope':'saved readback and original inventory reconstruction, not native import replay or network proof'}
    save_new(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();v=review(a.receipt,a.output);print({k:v[k] for k in ('status','issues','analytic_cases','artifact_files','small_matrix_count')})
