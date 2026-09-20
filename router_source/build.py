"""Build a local-only extension of the immutable input98 proof artifact."""
import argparse
import json
from pathlib import Path
import shutil
import time

from router_source.capture import ROOT,capture,sha
from router_source.checker import compact,digest
from router_source.propose import propose

OLD=Path('/data1/Kane/MOE/portable_conv98_20260915_v1/relocated')
BUNDLE='8f35a4ba23b51bdcc829535a47880e5f6158a6fbaaf119b4f7744e0c2278606b'
STATEMENT='7c31f551137b33257e178c40eeea55bf4b94e3438ae00ddb3ed16e2808e56b00'
JOB=ROOT/'data/moe/results/conv_pre_f0_rational_20260915_r2/rank24_monolithic/job.json'
JOB_HASH='bdf31dacf5cc30b6ddc2feb2e04b57f4af9cce54308143a296c76a23e7945058'


def save(path,obj):
    with path.open('xb') as f:f.write(compact(obj))


def build(root):
    start=time.monotonic();bundle=root/'relocated'
    if sha(OLD/'bundle.json')!=BUNDLE or sha(JOB)!=JOB_HASH:raise ValueError('frozen sources changed')
    meta=json.loads((OLD/'bundle.json').read_bytes());statement=meta['statement']
    if digest(compact(statement))!=STATEMENT:raise ValueError('old proof statement changed')
    # Identity validate the whole transported dependency inventory before capture.
    for name,h in meta['files'].items():
        p=(OLD/name).resolve()
        if not p.is_relative_to(OLD.resolve()) or sha(p)!=h:raise ValueError('old bundle dependency changed')
    begin=time.monotonic()
    doc=capture(json.loads(JOB.read_bytes()),statement)
    captured=time.monotonic()-begin
    begin=time.monotonic();proof=propose(doc);proposed=time.monotonic()-begin
    begin=time.monotonic();shutil.copytree(OLD,bundle)
    save(bundle/'router_source.json',doc);save(bundle/'router_proof.json',proof)
    shutil.copyfile(ROOT/'router_source/checker.py',bundle/'router_check.py')
    shutil.copyfile(ROOT/'router_source/verify.py',bundle/'verify_with_router.py')
    files={name:sha(bundle/name) for name in ('router_source.json','router_proof.json','router_check.py','verify_with_router.py')}
    extension={'schema':'SOURCE_ROUTER_EXTENSION_V1','files':files,
        'source_identity':digest(compact(doc)),'old_bundle_sha256':BUNDLE,'old_statement_sha256':STATEMENT}
    save(bundle/'extension.json',extension)
    save(root/'generation.json',{'source_capture_seconds':captured,'proposal_seconds':proposed,
        'copy_and_serialization_seconds':time.monotonic()-begin,
        'whole_generation_seconds_before_publication':time.monotonic()-start,
        'extension_sha256':sha(bundle/'extension.json'),
        'new_bytes':sum((bundle/name).stat().st_size for name in files)+(bundle/'extension.json').stat().st_size,
        'bundle_bytes':sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file()),
        'new_solver_calls':0,'network_forward_calls':0,'new_HZ_propagations':0,
        'source_scope':'Frozen stored proof + checkpoint parameter/input-byte capture; old HZ/proposal costs excluded.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    a=p.parse_args();build(a.root.resolve())
