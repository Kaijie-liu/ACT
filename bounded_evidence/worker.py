"""SYNTHETIC-control adapter: change only construction publication in old work.

Not wired into real execute/run. All original construction, source checking,
native proposals, exact checking, receipts and all-duty aggregation stay intact.
"""
import argparse
from pathlib import Path

from bounded_evidence.stream import save as streaming_save
from scoped_proof.io import load, save

POLICY = {'schema': 'BOUNDED_EVIDENCE_PUBLICATION_V1', 'artifact': 'construction.json',
          'block_bytes': 65536, 'controls_only': True}


def work(root, deadline):
    from residual_proof import worker
    inv = load(root / 'invocation.json')
    spec = load(root / 'spec.json', inv['spec_file_sha256'])
    if spec.get('evidence_publication') != POLICY:
        raise ValueError('explicit bound publication-control policy required')
    original = worker.save

    def publish(path, value):
        if Path(path) != root / 'construction.json':
            return original(path, value)
        stats = {'request_sha256': inv['request_sha256'], 'invocation': inv['invocation']}
        try:
            result = streaming_save(path, value, deadline=deadline, metrics=stats)
            stats['record'] = result
            return result
        finally:
            # The unchanged supervisor also captures killed workers without this
            # diagnostic file. It is NEVER used as a proof/acceptance receipt.
            save(root / 'construction_serialization.json', stats)

    worker.save = publish
    try:
        worker.work('construct', root, deadline)
    finally:
        worker.save = original


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--deadline', type=float, required=True)
    args = p.parse_args()
    work(args.root.resolve(), args.deadline)
