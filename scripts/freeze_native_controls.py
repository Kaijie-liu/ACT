"""Prepare the user-approved finite CPU control roster, using stdlib only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

from scoped_source.hz_request_store import POLICY, compact, digest, identity, require
from scoped_source.native_protocol import PYTHON, SITE, CONTROL_FAULTS, closure
from scoped_source.native_method import ARMS, method
from scoped_source.hz_request_store import Work

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT/'docs/hz_native_controls_freeze_20261018.json'
REGISTRY = ROOT/'docs/hz_native_execution_approvals.json'
CONTROL_CODE = ('scripts/freeze_native_controls.py','scripts/run_native_controls.py',
    'scripts/native_control_components.py','scripts/audit_native_controls.py')


def file_record(path):
    h = hashlib.sha256(); n = 0
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(65536),b''):
            n += len(b); h.update(b)
    return dict(file=str(path),bytes=n,sha256=h.hexdigest())


def main():
    p=argparse.ArgumentParser(); p.add_argument('directory'); a=p.parse_args()
    root=Path(a.directory)
    require(root.is_absolute() and root.resolve()==root and root.is_relative_to('/data1/Kane/MOE')
            and not root.exists() and not FREEZE.exists(), 'new one-shot freeze locations')
    require(json.loads(REGISTRY.read_bytes())['profiles']==[], 'no previous active native registrations')
    root.mkdir(); start=time.monotonic(); work=Work(start+300)
    sources={}
    for case in ('mixed','clipped','guarded'):
        d=root/('preview-'+case); d.mkdir()
        # Compute the source identity from the existing deterministic declaration
        # generator; no network, Torch, propagation or candidate evaluation.
        if case=='mixed':
            from scoped_source.source_dag_controls import mixed
            path,h=mixed(d/'generated')
        elif case=='clipped':
            from scoped_source.affine_reference_controls import deep
            path,h=deep(d/'generated',center=(.875,-.75),radius='1/4')
        else:
            from scoped_source.direct_supervised import fixed_source
            path=d/'generated'; h=fixed_source(path,'guarded')
        work.tick(); sources[case]=dict(kind='fixed_control',source=h,case=case)
    site=Path(SITE); torch=site/'torch'
    version=next(n.value.value for n in ast.parse((torch/'version.py').read_bytes()).body
                 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__version__' for t in n.targets))
    primary=[Path(PYTHON),torch/'__init__.py',torch/'version.py',
             *sorted(torch.glob('_C.*.so')),torch/'lib/libtorch_cpu.so',torch/'lib/libc10.so']
    env=dict(python=PYTHON,site=SITE,torch_version=version,files=[file_record(x) for x in primary])
    code={n:digest((ROOT/n).read_bytes()) for n in closure(ROOT)}
    scripts={n:digest((ROOT/n).read_bytes()) for n in CONTROL_CODE}
    records=[]
    for i,case in enumerate(sources):
        for arm in (ARMS if i%2==0 else tuple(reversed(ARMS))):
            profile=dict(schema='NATIVE_SHARED_EXECUTION_V1',method=method(arm),source=sources[case],
                policy=dict(POLICY),seconds=300,as_bytes=8*2**30,file_bytes=128*2**20,
                rss_bytes=2*2**30,implementation=code,environment=env)
            records.append(dict(id=f'normal-{case}-{arm}',kind='normal',profile=profile,sha256=identity(profile)))
    # Both arms cover exactly the same fault roster on the same fixed source.
    for fault in CONTROL_FAULTS:
        for arm in ARMS:
            base=next(r['profile'] for r in records if r['kind']=='normal'
                      and r['profile']['source']['case']=='mixed' and r['profile']['method']['producer']['representation']==arm)
            profile=dict(base,control=fault)
            records.append(dict(id=f'fault-{fault}-{arm}',kind='fault',profile=profile,sha256=identity(profile)))
    doc=dict(schema='NATIVE_FINITE_CONTROLS_FREEZE_V1',date='2026-10-18',base_head='3397193195bc73342aadce9f3056cbe57416feb7',
        authorization='User explicitly approved 8 GiB AS and finite CPU controls in thread; no real models or GPU.',
        records=records,control_code=scripts,source_previews=sources,preparation_seconds=time.monotonic()-start,
        protocol_sha256=digest((ROOT/'docs/hz_native_controls_protocol_20261018.md').read_bytes()),
        normal_slots=6,fault_slots=20,component_batches=1,component_case_cap=32,
        per_request_seconds=300,cumulative_seconds=8100,admission_attempts=1,retries=0,
        sequence='six normal positions, component batch, twenty fault positions; stop at first unexpected outcome',
        rss_measurement='sampled managed process group PLUS calling parent; not hard aggregate RAM',
        component_sources=['mixed','clipped','guarded','equation-only deep(center=(1,2^-1000),radius=1/4)',
                           'equation-only existing zero-radius source'],
        native_float_tolerance=dict(rtol=1e-12,atol=1e-12,certificate_acceptance_unchanged=True),
        physical_latency_guaranteed=False,real_model_runs=0,gpu_runs=0,full_topology_admitted=False)
    work.tick()
    with FREEZE.open('xb') as f: f.write(compact(doc))
    # Generated permission artifact, not a hidden change to limits in POLICY.
    REGISTRY.write_bytes(compact(dict(schema='NATIVE_EXECUTION_APPROVAL_REGISTRY_V1',
                                    profiles=sorted(r['sha256'] for r in records))))
    print(compact(dict(freeze=str(FREEZE),sha256=digest(compact(doc)),profiles=len(records),
                       preparation_seconds=doc['preparation_seconds'])).decode())


if __name__=='__main__': main()
