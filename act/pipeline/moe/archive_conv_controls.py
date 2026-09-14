"""Read-only control identity review; writes a compact derived archive only."""
import argparse
import json
from pathlib import Path
from act.pipeline.moe.conv_training import atomic_json,sha,CONFIG
from act.pipeline.moe.conv_training_supervisor import check_gate


def archive(first,second,smoke,output):
    if output.exists():raise FileExistsError(output)
    passed=check_gate(second/'summary.json')
    failed=json.loads((first/'summary.json').read_text())
    for name,digest in failed['hashes'].items():
        if sha(first/name)!=digest:raise ValueError('failed attempt identity drift')
    if failed['status']!='FAILED' or failed['results']['act']['status']!='CONFORMANCE_PASS':
        raise ValueError('first attempt history differs')
    if failed['results']['act']['model_state']!=passed['results']['act']['model_state']:
        raise ValueError('model changed in repair')
    if failed['results']['act']['inputs']!=passed['results']['act']['inputs']:
        raise ValueError('input changed in repair')
    control=json.loads((smoke/'summary.json').read_text())
    if (control['status']!='PASS' or control['device']!='cuda' or control['config_sha256']!=sha(CONFIG)
        or control['checkpoint_replay_max_error']!=0 or not control['optimizer_continuation_equal']
        or control['metrics']['router_max_update']<=0 or control['metrics']['samples']!=256):
        raise ValueError('training smoke acceptance failed')
    result=dict(status='PASS',issues=[],fullshape=passed,failed_attempt=failed,cuda_smoke=control,
        sources={str(p):sha(p) for p in (first/'summary.json',second/'summary.json',smoke/'summary.json')},
        scope='artifact identity and conformance acceptance review, not independent proof of HZ propagation or CROWN bounds')
    atomic_json(output,result)
    print(json.dumps({'status':'PASS','formal_SAFE':False,'output':str(output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--failed',type=Path,required=True)
    p.add_argument('--passed',type=Path,required=True);p.add_argument('--smoke',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    archive(a.failed.resolve(),a.passed.resolve(),a.smoke.resolve(),a.output.resolve())
