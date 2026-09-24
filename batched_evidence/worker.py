"""Synthetic control adapter; preserves every original checker and receipt."""
import argparse
from pathlib import Path
from scoped_proof.io import load, save
from batched_evidence.stream import save as streaming_save

POLICY = {'schema': 'BATCHED_EVIDENCE_PUBLICATION_V1', 'artifact': 'construction.json',
          'block_bytes': 65536, 'batch_bytes': 16384, 'batch_items': 128, 'controls_only': True}


def work(root, deadline):
    from residual_proof import worker
    inv = load(root / 'invocation.json')
    spec = load(root / 'spec.json', inv['spec_file_sha256'])
    if spec.get('evidence_publication') != POLICY: raise ValueError('bound control policy required')
    original = worker.save
    def publish(path, value):
        if Path(path) != root / 'construction.json': return original(path, value)
        stats = {'request_sha256': inv['request_sha256'], 'invocation': inv['invocation']}
        try:
            result = streaming_save(path, value, deadline=deadline, metrics=stats)
            stats['record'] = result
            return result
        finally:
            save(root / 'construction_serialization.json', stats)
    worker.save = publish
    try: worker.work('construct', root, deadline)
    finally: worker.save = original


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    p.add_argument('--deadline', type=float, required=True); a = p.parse_args()
    work(a.root.resolve(), a.deadline)
