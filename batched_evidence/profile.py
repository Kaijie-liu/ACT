"""Separate, unchanged synthetic construction/source-check cost decomposition.

No encoder comparison, model, checkpoint or native LP solve. Wrappers ONLY time
calls to existing construction/check functions; saved source is freshly checked
by the archive in another process. Instrumentation overhead remains charged.
"""
import argparse
from contextlib import ExitStack
from pathlib import Path
import time
from unittest.mock import patch

from scoped_proof.io import Events, save
from source_enclosure.format import identity
from source_construction_lab.fixtures import document
from bounded_evidence.stream import save as publish

FIXTURES = {
    'small': {'experts': 4, 'classes': 3, 'width': 4, 'depth': 1, 'seed': 724},
    'medium': {'experts': 8, 'classes': 10, 'width': 8, 'depth': 2, 'seed': 724},
}


def run(root, fixture, deadline):
    from checked_route_frontier.build import prefix
    from shared_route_residual.propose import propose
    from residual_proof import build, check
    started = time.monotonic(); events = Events(root, 'profile', started)
    records = []; phases = []; current = None
    def timed(module, name, role):
        original = getattr(module, name)
        def wrapped(*args, **kwargs):
            begin = time.monotonic(); status = 'ERROR'
            try:
                result = original(*args, **kwargs); status = 'COMPLETED'; return result
            finally:
                records.append({'phase': current, 'role': role, 'status': status,
                                'seconds': time.monotonic() - begin})
        return patch.object(module, name, wrapped)
    def phase(name, fn):
        nonlocal current
        current = name; begin = time.monotonic()
        value = events.call(name, fn)
        seconds = time.monotonic() - begin
        measured = sum(r['seconds'] for r in records if r['phase'] == name)
        if measured > seconds + 1e-8: raise ValueError('overlapping component accounting')
        phases.append({'name': name, 'seconds': seconds, 'component_seconds': measured,
                       'other_seconds': seconds - measured})
        return value
    doc = phase('generate_declared_source', lambda: document(**FIXTURES[fixture]))
    digest = phase('source_identity', lambda: identity(doc))
    pre = phase('router_prefix', lambda: prefix(doc, expected_source_sha256=digest, deadline=deadline))
    rows = [phase('shared_route_proposal', lambda: propose(doc, pre, invocation='profile', deadline=deadline))]
    with ExitStack() as stack:
        for module, name, role in [
            (check, 'route_check', 'route_recheck'), (build, 'network', 'expert_propagation'),
            (build, 'join', 'shared_factor_join'), (build, 'guards', 'guard_construction'),
            (build, 'project', 'property_projection'), (build, 'output_lp', 'weighted_lp_construction'),
            (check, 'check_network', 'expert_source_check'), (check, 'check_join', 'join_check'),
            (check, 'check_guards', 'guard_check'), (check, 'check_projection', 'projection_check'),
            (check, 'check_outputs', 'weighted_lp_check')]:
            stack.enter_context(timed(module, name, role))
        bundle = phase('retained_construction', lambda: build.finish(doc, pre, rows, mode='shared',
            invocation='profile', expected_source_sha256=digest, deadline=deadline))
        records_files = phase('unchanged_R1_stream_publication', lambda: {
            'source': publish(root / 'source.json', doc, deadline=deadline),
            'construction': publish(root / 'construction.json', bundle, deadline=deadline)})
        result = phase('independent_algorithm_source_check', lambda: check.check(doc, bundle,
            invocation='profile', expected_source_sha256=digest, deadline=deadline))
    phase('check_report_publication', lambda: save(root / 'source_check.json', result))
    save(root / 'profile.json', {'fixture': fixture, 'parameters': FIXTURES[fixture],
        'files': records_files, 'phases': phases, 'operations': records,
        'worker_before_report_seconds': time.monotonic() - started, 'source_sha256': digest,
        'original_duties': result['original_output_obligations'],
        'retained_duties': result['output_obligations'], 'excluded_duties': result['excluded_output_obligations'],
        'source_check_status': result['status'], 'output_bounds_generated': 0,
        'complete_output_positive_proof': False, 'real_requests': 0, 'native_solver_calls': 0,
        'scope': 'instrumented unchanged synthetic construction/check; not real4099 attribution'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    p.add_argument('--fixture', choices=FIXTURES, required=True)
    p.add_argument('--deadline', type=float, required=True); a = p.parse_args()
    run(a.root, a.fixture, a.deadline)
