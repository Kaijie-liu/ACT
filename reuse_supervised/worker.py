"""Original-clock isolated worker; unchanged capture and optional upstream reuse."""
import argparse
from pathlib import Path
import sys
from scripts.optional_evidence_budget import EvidenceBudget, EvidenceBudgetExpired


def execute(stage, root, budget, arm):
    if arm not in ('reuse_off','reuse_on'): raise ValueError('unregistered arm')
    if stage == 'propose':
        from moe_evidence.worker import validate_transport
        from scripts.optional_evidence_dev_contract import read
        from upstream_reuse.proposal import run
        request, _ = validate_transport(read(root/'request.json'))
        run(root, request, budget, source_enabled=arm=='reuse_on', matrix_enabled=arm=='reuse_on')
    elif stage == 'capture':
        from moe_evidence.worker import execute as original
        original(stage, root, budget)
    else: raise ValueError('unregistered phase')
    budget.remaining(2)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('capture','propose'));p.add_argument('root',type=Path)
    p.add_argument('--started',required=True,type=float);p.add_argument('--arm',required=True,choices=('reuse_off','reuse_on'))
    a=p.parse_args()
    try:execute(a.stage,a.root,EvidenceBudget(a.started),a.arm)
    except EvidenceBudgetExpired:sys.exit(3)
