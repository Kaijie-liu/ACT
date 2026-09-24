"""V2 shared-certificate reception and unchanged construction/check timings.

Accept a declared synthetic source explicitly. No real-model loader, native
solver, sample chooser or automatic retry entrypoint. R1's frozen list-wrapping
bug remains preserved; the checker is not modified to accept that bad type.
"""
from contextlib import ExitStack
from pathlib import Path
import time
from unittest.mock import patch

from bounded_evidence.stream import save as publish
from scoped_proof.io import Events, save
from scoped_source.graph import clock
from source_enclosure.format import identity


def run(doc, root, *, deadline):
    from checked_route_frontier.build import prefix
    from shared_route_residual.propose import propose
    from shared_route_residual.format import binding
    from residual_proof import build, check
    tick = clock(deadline); root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); events = Events(root, 'source_cost_v2', start)
    phase_name = None; operations = []; phases = []
    def phase(name, fn):
        nonlocal phase_name
        phase_name = name; began = time.monotonic(); tick()
        value = events.call(name, fn); tick()
        seconds = time.monotonic() - began
        parts = sum(v['seconds'] for v in operations if v['phase'] == name)
        if parts > seconds + 1e-8: raise ValueError('overlapping timing')
        phases.append({'name': name, 'seconds': seconds, 'component_seconds': parts,
                       'other_seconds': seconds - parts})
        return value
    def instrument(module, name, role):
        original = getattr(module, name)
        def wrapped(*args, **kwargs):
            before = time.monotonic(); state = 'ERROR'
            try:
                value = original(*args, **kwargs); state = 'COMPLETED'; return value
            finally:
                operations.append({'phase': phase_name, 'role': role, 'status': state,
                                   'seconds': time.monotonic() - before})
        return patch.object(module, name, wrapped)
    digest = phase('identity', lambda: identity(doc))
    pre = phase('prefix', lambda: prefix(doc, expected_source_sha256=digest, deadline=deadline))
    # Correct boundary: shared mode takes ONE certificate dict, not [dict].
    certificate = phase('proposal', lambda: propose(doc, pre, invocation='profile_v2', deadline=deadline))
    if (type(certificate) is not dict or certificate.get('binding') != binding(doc, pre, 'profile_v2')):
        raise ValueError('shared certificate must be a bound dictionary')
    with ExitStack() as stack:
        for module, name, role in [
            (check, 'route_check', 'route_recheck'), (build, 'network', 'expert_propagation'),
            (build, 'join', 'factor_join'), (build, 'guards', 'guards'),
            (build, 'project', 'projection'), (build, 'output_lp', 'weighted_lp'),
            (check, 'check_network', 'expert_check'), (check, 'check_join', 'join_check'),
            (check, 'check_guards', 'guard_check'), (check, 'check_projection', 'projection_check'),
            (check, 'check_outputs', 'weighted_lp_check')]:
            stack.enter_context(instrument(module, name, role))
        bundle = phase('construct', lambda: build.finish(doc, pre, certificate, mode='shared',
            invocation='profile_v2', expected_source_sha256=digest, deadline=deadline))
        files = phase('publish_R1', lambda: {'source': publish(root / 'source.json', doc, deadline=deadline),
            'construction': publish(root / 'construction.json', bundle, deadline=deadline)})
        checked = phase('source_check', lambda: check.check(doc, bundle, invocation='profile_v2',
            expected_source_sha256=digest, deadline=deadline))
    report = {'schema': 'SOURCE_COST_DIAGNOSTIC_V2', 'source_sha256': digest, 'files': files,
              'phases': phases, 'operations': operations, 'source_check': checked,
              'complete_output_positive_proof': False, 'native_solver_calls': 0,
              'seconds_before_report': time.monotonic() - start}
    save(root / 'report.json', report); tick()
    return report
