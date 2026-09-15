"""Control-tested evidence-only driver, not registered for a new real cohort.

Caller must supply a whole-driver watchdog (298s, same outer start) and treat
this output as a candidate. There is deliberately no cohort launch command.
"""
import argparse
import os
from pathlib import Path
import time

from scripts.optional_evidence_dev_contract import ACT, read, save
from scripts.optional_evidence_budget import EvidenceBudget, EvidenceBudgetExpired, terminal_status
from portable_proof.runtime import digest
from moe_evidence.execution import PHASES, LEVELS, phase, accept


def run(root, control, started):
    budget = EvidenceBudget(started); stages = {}; complete = False; error = None
    req = read(root/'request.json'); verdict = 'UNKNOWN_MISSING_EVIDENCE'
    try:
        for name in PHASES['evidence']:
            save(control/'active.json', {'phase': name, 'entered_seconds': time.monotonic()-started})
            if name == 'check':
                packing = read(root/'packing.json')
                cmd = [ACT, '-I', '-S', str(root/'portable/verify.py'),
                       '--bundle-hash', packing['bundle_sha256'], '--statement-hash', packing['statement_sha256']]
            else:
                cmd = [ACT, '-m', 'evidence_handoff.worker', name, str(root), '--started', repr(started)]
            stages[name] = phase(cmd, root, name, budget, os.environ.copy())
            save(root/'stage_progress.json', stages)
            if stages[name]['state'] != 'COMPLETED':
                break
        verdict, complete = accept('evidence', stages, lambda: read(root/'check.log'), time.monotonic()-started)
    except EvidenceBudgetExpired:
        verdict = 'TIMEOUT'
    except Exception as exc:
        verdict = 'ERROR'; error = repr(exc)
    save(control/'active.json', {'phase': 'terminal_inventory', 'entered_seconds': time.monotonic()-started})
    inventory = {str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
    elapsed = time.monotonic()-started
    terminal = {'schema': 'EVIDENCE_HANDOFF_EXECUTION_V1', 'arm': 'evidence',
                'dataset_index': req['sample']['dataset_index'], 'request_sha256': digest((root/'request.json').read_bytes()),
                'stages': stages, 'artifact_sha256': inventory, 'error': error,
                'complete_independent_check': complete, 'evidence_level': LEVELS['evidence'],
                'production_gate_changed': False, 'deployed_float_SAFE': False, 'budget_seconds': 300,
                'wall_seconds': elapsed, 'status': terminal_status(verdict, elapsed, 300, complete),
                'outer_timeout': any(v['state'] == 'OUTER_TIMEOUT' for v in stages.values())}
    save(control/'candidate.json', terminal)
    return terminal


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path); p.add_argument('control', type=Path)
    p.add_argument('--started', type=float, required=True); a = p.parse_args()
    run(a.root, a.control, a.started)
