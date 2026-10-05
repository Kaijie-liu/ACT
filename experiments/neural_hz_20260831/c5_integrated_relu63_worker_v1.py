"""Advance the unchanged opt-in runtime only after the ReLU36 proof."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831 import c5_integrated_prefix_worker_v2 as prior


def main():
    evidence = ROOT / 'experiments/neural_hz_20260831/evidence/c5_relu36_live_20260905_v2.json'
    if prior._sha256(evidence) != 'b45d017ac76a55fdc535d411f4c1ee9aa93496b0d603d18da1b286dc02d7e2c3':
        raise ValueError('ReLU36 boundary qualification drift')
    proof = json.loads(evidence.read_text())
    if not all(proof.get(key) for key in ('numeric_equivalence_passed', 'live_registered_gates_passed', 'input_roots_unchanged')):
        raise ValueError('ReLU36 boundary qualification incomplete')
    if not all(item['passed'] for item in proof['boundaries'].values()):
        raise ValueError('ReLU36 physical boundary not proved')
    prior.main()


if __name__ == '__main__':
    main()
