"""Frozen one-observed-request development comparison; no new cohort claim."""
import json
from pathlib import Path
import subprocess
from portable_proof.runtime import digest, original_bytes

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / 'docs/optional_evidence_dev_v1_freeze.json'
OUTPUT = ROOT / 'data/moe/results/optional_evidence_dev_20260915_v1'
ACT = '/data1/Kane/miniconda3/envs/act-py312/bin/python'
FILES = ('scripts/optional_evidence_budget.py', 'scripts/optional_evidence_dev_contract.py',
         'scripts/optional_evidence_dev_worker.py', 'scripts/run_optional_evidence_dev.py',
         'scripts/review_optional_evidence_dev.py', 'scripts/test_optional_evidence_budget.py',
         'scripts/build_portable_conv_proof.py', 'portable_proof/runtime.py',
         'portable_proof/launcher.py', 'docs/optional_evidence_dev_v1.md')


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    # Atomic publication; old run directories are never reused.
    path = Path(path)
    tmp = path.with_name(path.name + '.pending')
    with tmp.open('xb') as f:
        f.write(original_bytes(value)); f.flush()
        import os
        os.fsync(f.fileno())
    tmp.replace(path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def make_freeze():
    from scripts.conv_pre_f0_r2_contract import verify_freeze
    old = verify_freeze()
    sources = {**old['sources'], **{p: digest((ROOT / p).read_bytes()) for p in FILES}}
    return {'schema': 'OPTIONAL_EVIDENCE_DEV_V1', 'classification': 'POSTSELECTED_POSITIVE_CONTROL_NOT_CONFIRMATORY',
            'arms': ['production_matched_v2', 'optional_checked_evidence'], 'total_seconds': 300,
            'proposal_cap_seconds': 60, 'proposal_terminal_check_reserve_seconds': 80,
            'outer_terminal_reserve_seconds': 2, 'dataset_index': 98,
            'job': {**old['job'], 'protocol': old['protocol']}, 'sources': sources,
            'input16_or_ACT_only_queries': False, 'retry': False, 'production_gate_changed': False}


def verify_freeze():
    value = read(FREEZE)
    if value['schema'] != 'OPTIONAL_EVIDENCE_DEV_V1':
        raise ValueError('wrong protocol')
    for name, sha in value['sources'].items():
        if digest((ROOT / name).read_bytes()) != sha:
            raise ValueError('source drift: ' + name)
    return value


def validate_job(directory):
    freeze = verify_freeze()
    d = Path(directory).resolve()
    if d.parent != OUTPUT or d.name not in freeze['arms']:
        raise ValueError('unregistered request directory')
    job = read(d / 'job.json')
    if job != freeze['job']:
        raise ValueError('request identity changed')
    req = job['parent_request']
    if digest(Path(req['config']['path']).read_bytes()) != req['config']['sha256']:
        raise ValueError('configuration changed')
    return job
