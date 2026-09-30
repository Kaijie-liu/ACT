"""Recheck the one frozen full-size source preparation; never output SAFE."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scoped_proof.io import load,save,sha
from scoped_source.capacity_prep_audit import audit
from scoped_source.capacity_intake import PROTOCOL_SHA
from scripts.archive_h2_capacity_preparation import derive as controls

EXECUTION_HEAD='63a11faf93b202acdce00888b7e3c36923d949b1'
CONTROL_ROOT=Path('/data1/Kane/MOE/baseline_runs/h2_capacity_preparation_controls_20261001_r3')
CALL_ROOT=Path('/data1/Kane/MOE/baseline_runs/h2_capacity_preparation_full_20261001_r1')


def derive():
    gate=controls(CONTROL_ROOT)
    if gate!=load(ROOT/'docs/h2_capacity_preparation_controls_20261001_r3.json'):
        raise ValueError('frozen control gate drift')
    call=load(CALL_ROOT/'caller_observation.json'); inv=load(CALL_ROOT/'invocation.json')
    spec=load(CALL_ROOT/'spec.json')
    if (spec!={'case':'full_size','control':'','recipe_sha256':PROTOCOL_SHA} or
            inv['budget']!=300 or inv['rss_limit']!=2**31):
        raise ValueError('unique frozen preparation contract')
    checked=audit(CALL_ROOT,call,expected_sources=load(CONTROL_ROOT/'implementation.json'))
    events={}
    for phase in ('prepare','receive'):
        path=CALL_ROOT/(phase+'_events.jsonl')
        events[phase]={'sha256':sha(path),'records':[json.loads(line) for line in path.read_text().splitlines()]} if path.exists() else None
    result={'schema':'H2_FULL_SIZE_IDENTITY_ARCHIVE_V1','execution_head':EXECUTION_HEAD,
        'recipe_sha256':PROTOCOL_SHA,'control_archive_sha256':sha(ROOT/'docs/h2_capacity_preparation_controls_20261001_r3.json'),
        'caller_observation_sha256':sha(CALL_ROOT/'caller_observation.json'),'call':call,
        'audit':checked,'events':events,'required_pairs_later':28,'required_properties_later':252,
        'output_lp_calls':0,'checked_output_bounds':0,'complete_output_proof':False,
        'trained_model':False,'full_path_capacity_admitted':False,'real_model_admitted':False,
        'next_rule':'bind these identities in a separate timed protocol; recreate and recapture in each arm; no free prepared source'}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True); g.add_argument('--output',type=Path); g.add_argument('--check',type=Path)
    args=p.parse_args(); start=time.monotonic(); result=derive()
    if args.output:
        if not args.output.resolve().is_relative_to(Path('/data1/Kane/MOE')): raise ValueError('output outside workspace')
        save(args.output,result)
    elif load(args.check)!=result: raise ValueError('preparation archive differs')
    print({'audit_status':'PASS','preparation_status':result['audit']['status'],
           'audit_seconds':time.monotonic()-start,'api_seconds':result['call']['seconds'],
           'source':result['audit']['source_identity'],'full_path_capacity_admitted':False})
