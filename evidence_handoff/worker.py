"""New optional phase worker; old cohort worker and proof semantics stay frozen."""
import argparse
from pathlib import Path
import sys

from scripts.optional_evidence_budget import EvidenceBudget, EvidenceBudgetExpired


def execute(stage, root, budget):
    if stage == 'propose':
        from moe_evidence.worker import validate_transport
        from scripts.optional_evidence_dev_contract import read
        from evidence_handoff.proposal import propose_with_handoff
        validate_transport(read(root/'request.json'))
        propose_with_handoff(root, budget)
    else:
        from moe_evidence.worker import execute as original
        original(stage, root, budget)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('capture', 'propose', 'precheck', 'package'))
    p.add_argument('directory', type=Path); p.add_argument('--started', type=float, required=True)
    a = p.parse_args()
    try:
        execute(a.stage, a.directory, EvidenceBudget(a.started))
    except EvidenceBudgetExpired:
        sys.exit(3)
