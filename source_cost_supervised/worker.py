"""Charge imports, source generation, V2 checks, persistence and receipts.

Instrumentation only wraps unchanged functions. No output solver or bound
candidate is called. Partial journals are diagnostic evidence, never proofs.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time
from unittest.mock import patch

from scoped_proof.io import load, save, sha, tick, Events
from source_enclosure.format import identity

ROLES = [('check', 'route_check', 'route_recheck'),
         ('build', 'network', 'expert_propagation'), ('build', 'join', 'factor_join'),
         ('build', 'guards', 'guards'), ('build', 'project', 'projection'),
         ('build', 'output_lp', 'weighted_lp'), ('check', 'check_network', 'expert_check'),
         ('check', 'check_join', 'join_check'), ('check', 'check_guards', 'guard_check'),
         ('check', 'check_projection', 'projection_check'), ('check', 'check_outputs', 'weighted_lp_check')]


class Journal:
    def __init__(self, root, invocation):
        self.path = Path(root) / 'journal.jsonl'
        self.binding = {k: invocation[k] for k in ('invocation', 'spec_sha256')}
        self.index = 0

    def emit(self, event, kind, name):
        row = dict(self.binding, index=self.index, event=event, kind=kind,
                   name=name, monotonic=time.monotonic())
        # Tiny append records; a killed trailing write is retained, not repaired.
        with self.path.open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n'); f.flush()
        self.index += 1

    def call(self, kind, name, fn):
        self.emit('ENTER', kind, name)
        try:
            result = fn()
        except BaseException:
            self.emit('ERROR', kind, name); raise
        self.emit('EXIT', kind, name)
        return result


def run(root):
    root = Path(root)
    inv = load(root / 'invocation.json', limit=65536)
    spec = load(root / 'spec.json', inv['spec_sha256'], limit=65536)
    deadline = inv['work_deadline_monotonic']; tick(deadline)
    from source_cost_supervised.audit import validate_spec
    validate_spec(spec)
    journal = Journal(root, inv)
    from source_construction_lab.fixtures import document
    doc = journal.call('worker', 'generate', lambda: document(**spec['fixture']))
    if identity(doc) != spec['source_sha256']: raise ValueError('generated source identity')
    from source_cost_controls import profile
    from residual_proof import build, check

    class ObservedEvents(Events):
        def emit(self, event, **fields):
            journal.emit({'ENTER': 'ENTER', 'EXIT': 'EXIT', 'EXIT_ERROR': 'ERROR'}[event],
                         'phase', fields['operation'])
            super().emit(event, **fields)

    def instrument(module, name, role):
        original = getattr(module, name)
        def wrapped(*args, **kwargs):
            return journal.call('component', role, lambda: original(*args, **kwargs))
        return patch.object(module, name, wrapped)

    with ExitStack() as stack:
        stack.enter_context(patch.object(profile, 'Events', ObservedEvents))
        for mod, name, role in ROLES:
            stack.enter_context(instrument({'build': build, 'check': check}[mod], name, role))
        journal.call('worker', 'profile', lambda: profile.run(doc, root / 'profile', deadline=deadline))
    tick(deadline)
    records = {}
    for name in ('source.json', 'construction.json', 'report.json'):
        path = root / 'profile' / name
        records[name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        tick(deadline)
    records['journal.jsonl'] = {'sha256': sha(journal.path), 'bytes': journal.path.stat().st_size}
    save(root / 'candidate.json', {'schema': 'SOURCE_COST_CANDIDATE_V1',
        'invocation': inv['invocation'], 'spec_sha256': inv['spec_sha256'],
        'source_sha256': spec['source_sha256'], 'records': records,
        'complete_output_positive_proof': False, 'native_solver_calls': 0})
    tick(deadline)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    run(p.parse_args().root)
