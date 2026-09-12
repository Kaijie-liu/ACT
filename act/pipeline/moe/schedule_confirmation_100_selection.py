"""Separately authorized 100 NEW endpoints; never append to the completed thirty.

This preparation performs clean forwards only. It reuses deployment-parity
selection mechanics, not prior route complexity, bounds or verification answers.
"""
import argparse
import json
from act.pipeline.moe import schedule_confirmation_selection as selector
from act.pipeline.moe.experiment1 import PROJECT_ROOT, _sha256
from act.pipeline.moe.paired_followup import save

BASE = PROJECT_ROOT/'act/pipeline/moe'
OUTPUT = BASE/'configs/schedule_confirmation_100_selection_r1.json'
INVENTORY = PROJECT_ROOT/'data/moe/results/schedule_confirmation_100_selection_20260912_r1/excluded_artifacts.json'
PREVIOUS_SELECTION = BASE/'configs/schedule_confirmation_selection_r2.json'
PREVIOUS_REVIEW = BASE/'results/schedule_confirmation_review_20260912_r2.json'
CONFIG = BASE/'configs/schedule_confirmation_100_r1.json'
REVIEW = BASE/'results/schedule_confirmation_100_selection_review_20260912_r1.json'
COUNT = 100
START = 4000


def prior_identity():
    review = json.loads(PREVIOUS_REVIEW.read_text())
    if review['full']['audit']['status'] != 'PASS' or review['full']['audit']['rows'] != 270:
        raise ValueError('previous thirty-input completion must be archived')
    return {'selection': str(PREVIOUS_SELECTION), 'selection_sha256': _sha256(PREVIOUS_SELECTION),
            'review': str(PREVIOUS_REVIEW), 'review_sha256': _sha256(PREVIOUS_REVIEW),
            'relationship': 'SEPARATE_COHORT_NO_POOLING_NO_RETUNING'}


def validate_prior(selection):
    if selection.get('previous_completed_experiment') != prior_identity():
        raise ValueError('prior completion identity drift or missing separation')
    previous = json.loads(PREVIOUS_SELECTION.read_text())
    old = {s['dataset_index'] for s in previous['samples']}
    chosen = {s['dataset_index'] for s in selection['samples']}
    if not old.issubset(selection['excluded_indices']) or old.intersection(chosen):
        raise ValueError('hundred-input cohort overlaps or fails to exclude prior thirty')
    if selection['models'] != previous['models'] or selection['request'] != previous['request']:
        raise ValueError('model or radius change is not authorized')
    if 'supersedes_selection' in selection:
        raise ValueError('new experiment must not supersede the thirty')


def generate(records):
    value = selector.generate(records, count=COUNT, start=START,
                              inventory_path=INVENTORY, repair=False)
    value['previous_completed_experiment'] = prior_identity()
    validate_prior(value)
    return value


def freeze():
    if OUTPUT.exists() or INVENTORY.parent.exists():
        raise ValueError('no overwrite or re-selection')
    # Include ALL earlier selection manifests, even failed/retired ones. Terminal
    # ledgers include matched and legacy deaths without packages as exclusions.
    records, union = selector.inventory(ignore_selection_paths={OUTPUT})
    prior_indices = {s['dataset_index'] for s in json.loads(PREVIOUS_SELECTION.read_text())['samples']}
    if not prior_indices.issubset(union): raise ValueError('prior thirty absent from inventory')
    INVENTORY.parent.mkdir()
    save(INVENTORY, records)
    value = generate(records)
    save(OUTPUT, value)
    return {'selection': str(OUTPUT), 'count': COUNT, 'excluded_count': len(union),
            'sources': len(records), 'indices': [s['dataset_index'] for s in value['samples']]}


def audit_selection():
    value = json.loads(OUTPUT.read_text())
    validate_prior(value)
    records = selector.verify_exclusions(value, count=COUNT, start=START)
    if value != generate(records): raise ValueError('clean-only reconstruction mismatch')
    return {'status': 'PASS', 'issues': [], 'selection_sha256': _sha256(OUTPUT),
            'sample_count': COUNT, 'excluded_count': len(value['excluded_indices']),
            'index_range': [value['samples'][0]['dataset_index'], value['samples'][-1]['dataset_index']],
            'prior_thirty_overlap': 0,
            'scope': 'Separate-process CPU/float64 batch-one clean-only reconstruction; no new verification endpoint.'}


def register():
    if REVIEW.exists() or CONFIG.exists(): raise ValueError('no overwrite of registration')
    checked = audit_selection()
    save(REVIEW, checked)
    config = json.loads((BASE/'configs/schedule_confirmation_r2.json').read_text())
    config.update(classification='FROZEN_100_NEW_INPUT_ROUTE_COMPLEXITY_CONFIRMATION', sample_count=COUNT,
                  selection=str(OUTPUT), selection_sha256=_sha256(OUTPUT),
                  selection_audit=str(REVIEW), selection_audit_sha256=_sha256(REVIEW),
                  smoke_output=str(PROJECT_ROOT/'data/moe/results/schedule_confirmation_100_smoke_20260912_r1'),
                  output=str(PROJECT_ROOT/'data/moe/results/schedule_confirmation_100_full_20260912_r1'))
    save(CONFIG, config)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    print(json.dumps(freeze() if args.freeze else register() if args.register else audit_selection()))


if __name__ == '__main__': main()
