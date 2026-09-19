"""Isolated diagnostic phases; all use the original request clock."""
import argparse
from pathlib import Path
import shutil
import sys
import time
from single_check_portable.execution import ROOT,read,save_new
from portable_proof.runtime import digest
from lp_sandwich.check import identity,rational,validate_statement,deadline_tick


def execute(phase,root):
    root=Path(root);plan=read(root/'plan.json');spec=plan['spec'];start=plan['started']
    limit=start+(218 if phase in ('load','propose') else 298);tick=deadline_tick(limit);tick()
    if phase=='load':
        ref=spec['export'];path=Path(ref['path']);raw=path.read_bytes();tick()
        if digest(raw)!=ref['sha256'] or ref['sha256']!=spec['statement']['export_sha256']:
            raise ValueError('frozen export bytes changed')
        from lp_sandwich.check import strict_json
        ex=strict_json(raw);s=spec['statement'];lp=ex['lp']
        if (identity(ex['source'])!=ex['source_sha256'] or ex['source_sha256']!=s['source_sha256'] or
            list(map(rational,ex['q']))!=list(map(rational,s['property']['q'])) or
            rational(ex['offset'])!=rational(s['property']['constant'])):
            raise ValueError('source/property binding')
        validate_statement(s,lp,spec['statement_sha256'],tick)
        save_new(root/'prepared.json',{'lp':lp,'statement':s});tick()
    elif phase=='propose':
        from lp_sandwich.propose import propose_to
        v=read(root/'prepared.json');tick()
        t=propose_to(v['lp'],v['statement'],root/'proposal',deadline=limit)
        if t['status']=='TIMEOUT':raise TimeoutError('proposal reserve cutoff')
        if t['status']!='COMPLETED_DIAGNOSTIC':raise ValueError('proposal error: '+str(t['error']))
        tick()
    elif phase=='package':
        bundle_path=root/'proposal/bundle.json';b=read(bundle_path)
        if identity(b['statement'])!=spec['statement_sha256'] or b['statement']!=spec['statement']:
            raise ValueError('bundle obligation drift')
        native=read(root/'proposal/native.json');terminal=read(root/'proposal/terminal.json')
        if (native['statement_sha256']!=spec['statement_sha256'] or native['lp_sha256']!=spec['statement']['lp_sha256'] or
            terminal['native_calls']!=1 or terminal['deadline_monotonic']!=start+218):
            raise ValueError('native count/clock/identity')
        checker=ROOT/'lp_sandwich/check.py'
        if digest(checker.read_bytes())!=plan['checker_sha256']:raise ValueError('checker code drift')
        out=root/'portable';out.mkdir(exist_ok=False)
        shutil.copyfile(checker,out/'verify.py');shutil.copyfile(bundle_path,out/'bundle.json');tick()
        save_new(root/'packing.json',{'bundle_sha256':digest((out/'bundle.json').read_bytes()),
            'checker_sha256':digest((out/'verify.py').read_bytes()),'statement_sha256':spec['statement_sha256'],
            'lp_sha256':spec['statement']['lp_sha256'],'source_bundle_sha256':digest(bundle_path.read_bytes()),
            'original_started':start,'upstream_precheck_retained':True});tick()
    else:raise ValueError('unregistered worker phase')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('load','propose','package'));p.add_argument('root',type=Path)
    a=p.parse_args()
    try:execute(a.phase,a.root)
    except TimeoutError:sys.exit(3)
