"""Freeze ONLY a complete old-input smoke. Does not run models or solvers."""
import json
from pathlib import Path
import subprocess
import time
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from metamoe_csr_paired_r4 import ROOT, PARENT, OLD, GATE, OUTPUT, PROTOCOL, ROSTER, ADDED, contract


def build_config():
    start = time.monotonic()
    parent = json.loads(PARENT.read_text())
    cfg = json.loads(PARENT.read_text())
    for name, digest in parent['files'].items():
        if sha256(name) != digest:
            raise ValueError('no source rebind allowed after R4: '+name)
    from audit_metamoe_csr_r4 import audit
    review = audit(PARENT)  # Saved-only recheck, no propagation/inference/solve.
    saved = json.loads(GATE.read_text())
    strip_clock = lambda d: {k: v for k, v in d.items() if k != 'separate_review_seconds'}
    if strip_clock(review) != strip_clock(saved) or review['passed'] is not True:
        raise ValueError('representation review does not match archive')
    cfg.update(protocol=PROTOCOL, seconds=300, output_root=OUTPUT,
        requests=json.loads(OLD.read_text())['requests'], arms=['act', 'author'],
        roster=[list(x) for x in ROSTER], automatic_followup=False,
        scope='FOUR old-input full-model calls only; freeze is not execution or new-cohort approval',
        selection='EXACT existing CIFAR10_0/MNIST_0 materialized tensors, including CIFAR clean error',
        cost='300s/request including startup/imports/validation/route/guard/solve/write/cleanup; separate postflight/audit disclosed',
        failure='ERROR/SOURCE_CHANGED stops remaining calls; timeout/refusal retained; no retry or resume')
    cfg['prerequisites'] = {str(p.relative_to(ROOT)): sha256(p) for p in (PARENT, OLD, GATE)}
    cfg['files'].update(cfg['prerequisites'])
    cfg['files'].update({name: sha256(name) for name in ADDED})
    cfg['parent_config_sha256'] = sha256(PARENT)
    cfg['explicit_source_rebinding'] = {}
    cfg['execution_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    cfg['preparation_seconds'] = time.monotonic()-start
    contract(cfg)
    return cfg


if __name__ == '__main__':
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'feat/moe-route-verification':
        raise ValueError('wrong branch')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('commit reviewed implementation before freeze')
    if Path(OUTPUT).exists():
        raise ValueError('output root already exists; preserve it')
    cfg = build_config()
    dest = ROOT/'configs/recent_moe/metamoe_csr_paired_smoke_r4.json'
    write(dest, cfg)
    print('FROZEN_NOT_EXECUTED', dest, sha256(dest))
