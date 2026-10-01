"""Owned finite CPU workers. Every fixture is created inside charged execution."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from scoped_proof.io import Events, PYTHON, load, save, tick
from scripts import hz_propagation_supervised as api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=api.PHASES)
    parser.add_argument('root', type=Path)
    parser.add_argument('--invocation-sha', required=True)
    parser.add_argument('--deadline', required=True, type=float)
    parser.add_argument('--payload-sha')
    parser.add_argument('--checker-sha')
    args = parser.parse_args()
    root, phase, end = args.root, args.phase, args.deadline
    spec = load(root/'spec.json')
    inv = api.bind(root, spec, args.invocation_sha)
    if end != api.run_deadline(inv, phase) or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('worker deadline/device changed')
    events = Events(root, phase, inv['start'])
    events.emit('WORKER_READY', phase=phase)

    def fault(name, *, delay=False):
        if spec['control'] == name:
            events.emit('FAULT_REACHED', name=name)
            if delay:
                time.sleep(max(0., inv['deadline']-time.monotonic())+1)
            return True
        return False

    tick(end)
    if phase == 'produce':
        fault('produce_delay', delay=True)
        events.emit('ENTER', operation='cold_ACT_imports')
        from unittest.mock import patch
        import torch
        from act.util.device_manager import initialize_device
        from act.back_end.moe import checked_propagation as propagation
        from act.back_end.moe.test_checked_propagation import fixture, config
        initialize_device('cpu', 'float64'); torch.set_num_threads(1)
        events.emit('EXIT', operation='cold_ACT_imports')
        tick(end)
        data = events.call('fixture_construction', lambda: fixture(
            n=2 if spec['case']=='two_layers' else 1, two_layers=spec['case']=='two_layers'))
        net, hz, router, bounds = data
        base = propagation.CheckedGuardedHybridzTF

        class Observed(base):
            def __init__(self, *a, **kw):
                events.call('guarded_constructor', lambda: super(Observed, self).__init__(*a, **kw))

            def apply(self, L, *a, **kw):
                if L.kind.upper() == 'DENSE' and fault('propagate_exception'):
                    raise RuntimeError('frozen propagation exception')
                return super().apply(L, *a, **kw)

            def _native_fallback(self, *a, **kw):
                fault('native_wait_stub', delay=True)
                return super()._native_fallback(*a, **kw)

            def _propagate_sparse_hz(self, L, *a, **kw):
                if L.kind.upper() != 'RELU':
                    return super()._propagate_sparse_hz(L, *a, **kw)
                try:
                    return events.call('ReLU_'+str(L.id),
                        lambda: super(Observed, self)._propagate_sparse_hz(L, *a, **kw))
                finally:
                    # A prefix may survive failed later layers, never replace them.
                    save(root/('prefix_layer'+str(L.id)+'.json'),
                         {'invocation': inv['invocation'], 'scope_sha256': self.scope_sha256,
                          'events': self.events, 'complete_propagation': False})

        with patch.object(propagation, 'CheckedGuardedHybridzTF', Observed), \
             patch('act.back_end.hybridz_tf.tf_mlp._guarded_support_query',
                   side_effect=AssertionError('native solver forbidden')), \
             patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')):
            result = events.call('actual_analyzer_and_support', lambda: propagation.propagate_checked_guarded(
                net, input_hz=hz, router_hz=router, route=(0,1), expert=0,
                request='frozen-synthetic-integration', entry_bounds=bounds,
                config=config(), deadline=end,
                options=propagation.CheckedSupportOptions(enabled=spec['case']!='disabled'),
                retain_guard=spec['case']!='guard_discarded'))
        if fault('partial_package'):
            result['events'][0]['package']['accepted']['results'].pop()
        if fault('wrong_scope'):
            result['scope']['expert'] = 1
        payload = {'schema': 'HZ_PROPAGATION_PAYLOAD_V1', 'invocation': inv['invocation'],
                   'spec_sha256': inv['spec_sha256'], 'propagation': result}
        events.emit('ENTER', operation='payload_serialization')
        fault('serialization_delay', delay=True)
        save(root/'payload.json', payload)
        events.emit('EXIT', operation='payload_serialization')
        if fault('descendant'):
            child = subprocess.Popen([PYTHON, '-S', '-c', 'import time; time.sleep(60)'])
            events.emit('DESCENDANT_STARTED', pid=child.pid, pgid=os.getpgrp())
            # Parent supervisor must detect this live same-group child after exit.
    elif phase == 'check':
        fault('check_delay', delay=True)
        if fault('check_exception'):
            raise RuntimeError('frozen checker exception')
        payload = load(root/'payload.json', api.required_hash(args.payload_sha), api.LIMIT)
        summary = events.call('independent_exact_check', lambda: api.exact_check(payload, spec, inv, end))
        if fault('missing_check'):
            return
        output = {'schema': 'HZ_PROPAGATION_CHECK_V1', 'invocation': inv['invocation'],
                  'spec_sha256': inv['spec_sha256'], 'payload_sha256': args.payload_sha,
                  'summary': summary, 'complete_moe_proof': False}
        events.call('check_serialization', lambda: save(root/'check.json', output))
    else:
        fault('receive_delay', delay=True)
        accepted = events.call('receive_binding', lambda: api.receive(
            root, spec, args.invocation_sha, args.payload_sha, args.checker_sha, end))
        events.call('accepted_serialization', lambda: save(root/'accepted.json', accepted))
    tick(end)
    events.emit('WORKER_COMPLETE', phase=phase)


if __name__ == '__main__':
    main()
