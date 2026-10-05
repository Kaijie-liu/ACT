"""Same numerical/live transaction, conservative whole-reference proof."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831 import c5_relu36_live_worker_v1 as prior
from experiments.neural_hz_20260831.c5_reference_lower_bound_v1 import lower_bound_roots


def main():
    failed = prior.EXPERIMENT / 'evidence/c5_relu36_live_20260905_v1.json'
    if prior.qualifier._sha256(failed) != 'c00a8daed2725fdacc935d522304bc0e8182f70fac7ad4b419c0afb1e9fd10c8':
        raise ValueError('prior closed boundary evidence drift')
    old_output, old_expand = prior.OUTPUT, prior.qualifier.expanded_roots
    old_write = prior.qualifier._atomic_exclusive_json
    output = prior.EXPERIMENT / 'evidence/c5_relu36_live_20260905_v2.json'

    def annotated_write(path, payload):
        if path == output:
            payload = dict(payload, schema='c5_relu36_live_v2',
                wrapper_sha256=prior.qualifier._sha256(Path(__file__)),
                physical_proof_kind='whole_candidate_vs_verified_reference_subset_v1',
                complete_expanded_reference_materialized=False,
                prior_aggregate_reference_rejection_preserved=True)
            for boundary in payload.get('boundaries', {}).values():
                boundary['baseline_lower_bound'] = boundary.pop('baseline')
                boundary['inequality'] = 'complete_candidate < measured_reference_subset <= complete_reference'
        return old_write(path, payload)

    prior.OUTPUT = output
    prior.qualifier.expanded_roots = lower_bound_roots
    prior.qualifier._atomic_exclusive_json = annotated_write
    try:
        prior.main()
    finally:
        prior.OUTPUT, prior.qualifier.expanded_roots = old_output, old_expand
        prior.qualifier._atomic_exclusive_json = old_write


if __name__ == '__main__':
    main()
