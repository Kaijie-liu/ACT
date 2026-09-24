"""Synthetic construction or independent check worker, one inherited deadline."""
import argparse
from pathlib import Path
import sys
import time
from scoped_proof.io import load, save, sha, tick, Events
from source_enclosure.format import identity


def work(phase, root, deadline):
    plan = load(root / 'plan.json')
    events = Events(root, phase, plan['started_monotonic'])
    tick(deadline)
    if phase == 'build':
        from source_construction_lab.fixtures import document
        from source_construction_lab.build import construct
        source = events.call('generate_synthetic_source', lambda: document(**plan['fixture']))
        source_record = events.call('serialize_source', lambda: save(root / 'source.json', source))
        def emit(row):
            events.emit(row['event'], **{k: v for k, v in row.items() if k != 'event'})
        bundle, report = construct(source, expected_source_sha256=identity(source),
            deadline=deadline, mode=plan['mode'], emit=emit)
        bundle_record = events.call('serialize_construction', lambda: save(root / 'construction.json', bundle))
        tick(deadline)
        save(root / 'build_receipt.json', {'mode': plan['mode'], 'source': source_record,
            'construction': bundle_record, 'report': report, 'not_a_proof_verdict': True})
    elif phase == 'check':
        from scoped_source.check import check
        receipt = load(root / 'build_receipt.json')
        doc, bundle = events.call('read_source_and_construction', lambda: (
            load(root / 'source.json', receipt['source']['sha256']),
            load(root / 'construction.json', receipt['construction']['sha256'])))
        result = events.call('independent_uncached_check', lambda: check(doc, bundle,
            expected_source_sha256=identity(doc), deadline=deadline))
        if any(m.split('.')[0] in ('act', 'torch', 'numpy', 'scipy', 'source_construction_lab.cache',
                                   'source_construction_lab.build') for m in sys.modules):
            raise ValueError('unexpected numerical/model runtime')
        if 'source_construction_lab.build' in sys.modules or 'source_construction_lab.cache' in sys.modules:
            raise ValueError('independent worker imported construction adapter')
        tick(deadline)
        save(root / 'check_receipt.json', {'source_sha256': receipt['source']['sha256'],
            'construction_sha256': receipt['construction']['sha256'], 'result': result,
            'no_producer_imported': True, 'no_model_or_solver_imported': True})
    else:
        raise ValueError('unknown worker phase')
    tick(deadline)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=('build', 'check'))
    p.add_argument('root', type=Path); p.add_argument('--deadline', type=float, required=True)
    a = p.parse_args()
    if not sys.flags.no_site: p.error('synthetic worker requires python -S')
    work(a.phase, a.root, a.deadline)
