"""Orchestration identity only; frozen ACT algorithms remain untouched."""
import json
from pathlib import Path

from act.pipeline.moe.freeze_conv_three_arm import ROOT, SELECTION, PROTOCOL, provenance, jobs
from act.pipeline.moe.experiment1 import _sha256

ACT = '/data1/Kane/miniconda3/envs/act-py312/bin/python'
SELECTION_HASH = 'f0b44898893e53226000a1cbdb91c05fb989a432022f8ffaa197aa9fb488ef09'
PROTOCOL_HASH = '62032c82ef46625ec80c09e3c55a7b986fdfe51d39b4c316cb1a192b462503f0'
ARMS = ('adaptive', 'monolithic', 'crown')
WRAPPERS = ('conv_three_arm_contract.py', 'run_conv_three_arm.py',
            'audit_conv_three_arm.py', 'test_conv_three_arm.py')
DEFAULT_ROOT = ROOT/'data/moe/results/conv_three_arm_smoke_20260915_r1'


def read(path):
    return json.loads(Path(path).read_text())


def wrapper_hashes():
    return {f'scripts/{name}': _sha256(ROOT/'scripts'/name) for name in WRAPPERS}


def selection():
    if _sha256(SELECTION) != SELECTION_HASH or _sha256(PROTOCOL) != PROTOCOL_HASH:
        raise ValueError('frozen protocol/selection drift')
    value = read(SELECTION)
    cfg, identities = provenance()
    if value['protocol'] != cfg or value['identities'] != identities:
        raise ValueError('frozen algorithm/config/model/environment drift')
    if value['smoke_jobs'] != jobs(value['smoke_samples']) or len(value['smoke_jobs']) != 6:
        raise ValueError('smoke schedule drift')
    for sample in value['smoke_samples']:
        record = value['materialized_inputs'][str(sample['dataset_index'])]
        if _sha256(Path(record['path'])) != record['sha256']:
            raise ValueError('materialized input drift')
    return value


def request_for(value, job, head):
    sample = value['smoke_samples'][job['rank']]
    if (not any(all(job.get(k)==v for k,v in registered.items()) for registered in value['smoke_jobs'])
            or sample['dataset_index'] != job['dataset_index']):
        raise ValueError('job not in frozen smoke')
    return {'protocol': value['protocol']['protocol'], 'method': job['method'],
            'epsilon': 2/255, 'topology': {'num_experts': 4, 'top_k': 2, 'classes': 10},
            'subject': value['subject'], 'sample': sample,
            'tensors': value['materialized_inputs'][str(sample['dataset_index'])],
            'config': value['identities']['method_configs'].get(job['method']), 'head': head}


def gate(rows, complete):
    # Conformance, never positive counts or effect-size selection.
    return (len(rows) == 6 and all(r['status'] != 'ERROR' for r in rows)
            and all(complete.get(a, 0) >= 1 for a in ARMS))
