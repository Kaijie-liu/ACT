"""Relocate the private closeout artifact without executing any model.

Run from its unpacked root after installing its hash-locked fresh environments.
This changes locations only; frozen numerical sources/objects remain identical.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def write(path,value):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2)+'\n')


def configure(base):
    base=base.resolve()
    for name in ('smoke','results','manifest.json','locations.json'):
        if (base/name).exists():raise ValueError('use an unused unpacked artifact directory: '+name)
    spec=json.loads((base/'template_manifest.json').read_text())
    for m in spec['models']:
        p=base/'assets'/(m['id']+'.pt')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=m['checkpoint_sha256']:raise ValueError('weight identity')
        m['checkpoint']=str(p)
    inputs=base/'assets/inputs.json.gz'
    if hashlib.sha256(inputs.read_bytes()).hexdigest()!=spec['inputs']['sha256']:raise ValueError('input identity')
    spec['inputs']['path']=str(inputs)
    location=dict(schema='S2A_RELOCATION_V1',primary_python=str(base/'env-primary/bin/python'),
        engine_environment=str(base/'env-engine'),author_source=str(base/'author-source'),
        compat_source=str(base/'engine-source'),install_receipts=str(base/'assets/historic_install_receipts'))
    write(base/'locations.json',location)
    h=hashlib.sha256((base/'locations.json').read_bytes()).hexdigest()
    spec['reproduction']['locations_sha256']=h
    write(base/'manifest.draft.json',spec)
    code="import importlib.metadata as m,json,sys;print(json.dumps(dict(python=sys.version,prefix=sys.prefix,base_prefix=sys.base_prefix,packages=sorted([(d.metadata['Name'],d.version) for d in m.distributions()]))))"
    inventory={}
    for role in ('primary','engine'):
        python=base/f'env-{role}/bin/python'
        inventory[role]=json.loads(subprocess.check_output([str(python),'-I','-B','-c',code],text=True))
    write(base/'environment_inventory.json',inventory)
    env=dict(os.environ,ACT_REPRODUCTION_LOCATIONS=str(base/'locations.json'),
        ACT_REPRODUCTION_LOCATIONS_SHA256=h,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run([str(base/'env-primary/bin/python'),'-I','-B',
        str(base/'execution/scripts/architecture_s2_reproduce.py'),'freeze',str(base)],env=env,check=True)
    print('Configured only; no model calls. Use run_reproduction.py smoke, then run.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);a=p.parse_args();configure(a.base)
