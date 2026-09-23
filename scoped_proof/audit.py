"""Saved-only terminal/cost audit; optional fresh exact proof check, no solver."""
import argparse
import json
import math
from pathlib import Path
import time
from source_enclosure.format import identity
from scoped_proof.io import load, sha
from scoped_proof.evidence import POSITIVE, aggregate, roster
from scoped_proof.supervisor import PHASES, accept


def audit(root, *, recheck=True):
    root = Path(root); start = time.monotonic()
    invocation = load(root/'invocation.json'); spec = load(root/'spec.json', invocation['spec_file_sha256'])
    cost = load(root/'cost.json'); terminal = load(root/'terminal.json', cost['terminal_sha256'])
    receipt = load(root/'receipt.json', cost['receipt_sha256'])
    token = invocation['invocation']; scope = spec['scope']; required = len(roster(scope))
    for obj in (cost, terminal, receipt):
        if obj['invocation'] != token: raise ValueError('terminal invocation mismatch')
    if receipt['terminal_sha256'] != cost['terminal_sha256']: raise ValueError('receipt terminal identity')
    if cost['request_sha256'] != identity(scope) or terminal['request_sha256'] != identity(scope):
        raise ValueError('request identity mismatch')
    budget = invocation['budget_seconds']
    if not 0 < budget <= 300 or cost['budget_seconds'] != budget or terminal['budget_seconds'] != budget:
        raise ValueError('budget mismatch')
    if terminal['required'] != required: raise ValueError('denominator mismatch')
    stages = terminal['stages']; names = [r['phase'] for r in stages]
    if names != list(PHASES[:len(names)]): raise ValueError('stage omission/reordering')
    previous = 0.
    for stage in stages:
        if load(root/(stage['phase']+'_stage.json')) != stage: raise ValueError('stage ledger changed')
        a, b, s = (stage[k] for k in ('start_seconds','end_seconds','seconds'))
        if not all(math.isfinite(v) for v in (a,b,s)) or not previous <= a <= b or not 0 <= s <= b-a+1e-6:
            raise ValueError('invalid stage timing')
        previous = b
    values = [cost[k] for k in ('end_to_end_seconds','stage_seconds','overhead_seconds')]
    if (not all(math.isfinite(v) and v >= 0 for v in values) or
        abs(values[1]-sum(r['seconds'] for r in stages)) > 1e-8 or
        abs(values[0]-values[1]-values[2]) > 1e-8 or values[0] < previous or
        values[0] < terminal['seconds_before_publication'] or values[0] < receipt['seconds_before_receipt']):
        raise ValueError('incomplete/invalid end-to-end cost')
    status = 'TIMEOUT' if (root/'publication_timeout.json').exists() else cost['status']
    if status not in (POSITIVE,'NOT_CLOSED','TIMEOUT','ERROR','RESOURCE_LIMIT'):
        raise ValueError('unregistered terminal state')
    if receipt['status'] != terminal['status_before_publication']:
        raise ValueError('terminal/receipt status disagreement')
    if not (root/'publication_timeout.json').exists() and cost['complete_output_positive_proof'] != (status == POSITIVE):
        raise ValueError('terminal positive flag disagreement')
    if values[0] >= budget and status != 'TIMEOUT': raise ValueError('late acceptance')
    result = None
    if status == POSITIVE:
        if names != list(PHASES) or any(s['status'] != 'COMPLETED' for s in stages):
            raise ValueError('positive after partial/failed execution')
        candidate = accept(root, scope, token)
        if candidate['status'] != POSITIVE or not cost['complete_output_positive_proof']:
            raise ValueError('positive missing complete exact receipt')
    if recheck and (root/'evidence_check.json').exists():
        doc, bundle = load(root/'source.json'), load(root/'construction.json')
        candidates = {int(p.stem): load(p) for p in sorted((root/'candidates').glob('*.json'))}
        result = aggregate(scope, doc, bundle, candidates, invocation=token,
            proposal_complete=(root/'proposal_complete.json').exists(), deadline=time.monotonic()+300)
        if result != load(root/'evidence_check.json'): raise ValueError('saved proof recheck differs')
    if status == POSITIVE and recheck and (result is None or result['status'] != POSITIVE):
        raise ValueError('positive independent recheck missing')
    return {'audit': 'PASS', 'issues': 0, 'effective_status': status, 'required': required,
        'end_to_end_seconds': values[0], 'ledger': {'cost_sha256': sha(root/'cost.json'),
        'terminal_sha256': cost['terminal_sha256'], 'receipt_sha256': cost['receipt_sha256']},
        'checked_bounds': None if result is None else result['checked_bounds'],
        'separate_audit_seconds': time.monotonic()-start, 'new_solves': 0,
        'complete_output_positive_proof': status == POSITIVE}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path); args = parser.parse_args()
    print(json.dumps(audit(args.root)))
