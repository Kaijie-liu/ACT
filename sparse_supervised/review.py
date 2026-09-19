"""Fresh-process structural/cost review and moved isolated checks, no new solve."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from single_check_portable.execution import ROOT, ACT, read, save_new
from portable_proof.runtime import digest
from sparse_supervised.controls import old
from sparse_supervised.flow import sources, audit, costs


def review(controls, destination):
    begin = time.monotonic()
    old()
    c = read(controls)
    if c['status'] != 'PASS' or c['sources'] != sources():
        raise ValueError('current controls required')
    base = Path(c['control_artifact_root'])
    actual = {str(p.relative_to(base)): digest(p.read_bytes()) for p in sorted(base.rglob('*')) if p.is_file()}
    if actual != c['artifact_sha256']:
        raise ValueError('retained control artifacts changed')
    moved = Path(tempfile.mkdtemp(prefix='sparse_supervised_review_', dir=ROOT / 'data/moe/results'))
    records = []
    for entry in c['terminals']:
        root = Path(entry['root'])
        if audit(root) != entry['terminal'] or costs(root) != entry['costs']:
            raise ValueError('terminal or complete cost reconstruction')
        out = None
        if entry['terminal']['complete_independent_check']:
            folder = moved / root.name
            shutil.copytree(root / 'portable', folder)
            pack = read(root / 'packing.json')
            result = subprocess.run([ACT, '-I', '-S', str(folder / 'verify.py'), str(folder / 'bundle.json'),
                '--bundle-sha256', pack['bundle_sha256'], '--statement-sha256', pack['statement_sha256'],
                '--timeout-seconds', '30'], cwd=folder, capture_output=True, text=True, check=True, timeout=35)
            out = json.loads(result.stdout)
            expected = read(root / 'check.log')
            if {k:v for k,v in out.items() if k!='seconds'} != {k:v for k,v in expected.items() if k!='seconds'}:
                raise ValueError('moved original-LP check mismatch')
            save_new(folder / 'recheck.json', out)
        records.append({'root':str(root),'status':entry['terminal']['status'], 'costs':entry['costs'],
                        'moved_check':out})
    old()
    if c['sources'] != sources():
        raise ValueError('review source drift')
    result = {'status':'PASS','issues':[], 'controls_sha256':digest(controls.read_bytes()),
              'sources':sources(),'artifact_files':len(actual),'reviewed_terminals':len(records),
              'moved_isolated_checks':sum(r['moved_check'] is not None for r in records),
              'records':records,'seconds':time.monotonic()-begin,
              'moved_artifact_root':str(moved),
              'moved_artifact_sha256':{str(p.relative_to(moved)):digest(p.read_bytes())
                                       for p in sorted(moved.rglob('*')) if p.is_file()},
              'real_solver_calls':0,'new_basis_reconstructions':0,
              'scope':'structural/clock audit plus standalone original-LP arithmetic; not network proof'}
    save_new(destination,result)
    return {k:result[k] for k in ('status','issues','artifact_files','reviewed_terminals','moved_isolated_checks')}


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    print(json.dumps(review(a.controls,a.output),indent=2))
