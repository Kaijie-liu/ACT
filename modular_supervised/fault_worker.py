"""Analytic test harness only. Production supervise/drive never dispatches here."""
import argparse
from pathlib import Path
import time
from unittest.mock import patch
from single_check_portable.execution import save_new
from modular_supervised.worker import execute


def run(mode, root):
    if mode == 'driver_stall':
        save_new(root/'fault_entered.json', {'synthetic':True,'mode':mode})
        time.sleep(60)
        return
    if mode.startswith('drive_'):
        from modular_supervised import flow
        from single_check_portable.execution import ACT
        fault = mode.removeprefix('drive_')
        phase = {'native_stall': 'capture', 'construct_stall': 'construct', 'map_exception': 'map',
                 'readback_stall': 'capture', 'readback_exception': 'capture',
                 'check_stall': 'check', 'prime_schedule_stall':'construct',
                 'finite_field_stall':'construct','CRT_stall':'construct','reconstruction_stall':'construct','exact_residual_stall':'construct',
                 'arithmetic_exception':'construct','arithmetic_limit':'construct','modular_exhausted':'construct',
                 'serialization_stall':'construct','serialization_exception':'construct'}[fault]
        original = flow.owned_stage
        def injected(folder, name, command, started, limit):
            if name == phase:
                command = [ACT, '-m', 'modular_supervised.fault_worker', fault, str(folder)]
            return original(folder, name, command, started, limit)
        with patch.object(flow, 'owned_stage', injected):
            flow.drive(root)
        return
    def stall(*args, **kwargs):
        save_new(root / 'fault_entered.json', {'synthetic': True, 'mode': mode})
        target = root / ('native/capture.json' if mode == 'native_stall' else 'construction.json')
        with target.open('xb') as f:
            f.write(b'{"interrupted":')
        time.sleep(60)
    if mode in ('prime_schedule_stall','finite_field_stall','CRT_stall','reconstruction_stall','exact_residual_stall','arithmetic_exception'):
        from modular_supervised.journal import TimedBudget
        original = TimedBudget.where
        target = 'finite_field' if mode == 'arithmetic_exception' else mode.removesuffix('_stall')
        def entered(self, phase, operation, **fields):
            original(self, phase, operation, **fields)
            if phase == target:
                save_new(root/'fault_entered.json', {'synthetic':True,'mode':mode})
                if mode == 'arithmetic_exception':raise OSError('synthetic arithmetic exception')
                time.sleep(60)
        with patch.object(TimedBudget,'where',entered):execute('construct',root)
    elif mode == 'modular_exhausted':
        from modular_basis.engine import Unresolved
        def exhausted(system,b,stats):
            raise Unresolved('synthetic bounded schedule exhaustion; no feasible point')
        with patch('modular_supervised.construction.eliminate',exhausted):execute('construct',root)
    elif mode == 'arithmetic_limit':
        def limit(system,b,stats):
            b.fail('integer bits',4097,4096)
        with patch('modular_supervised.construction.eliminate',limit):execute('construct',root)
    elif mode in ('serialization_stall','serialization_exception'):
        from modular_supervised import worker
        original=worker.save_new
        def interrupted(path,value):
            if path.name=='construction.json':
                save_new(root/'fault_entered.json', {'synthetic':True,'mode':mode})
                path.write_bytes(b'{"partial":')
                if mode.endswith('exception'):raise OSError('synthetic serialization exception')
                time.sleep(60)
            return original(path,value)
        with patch.object(worker,'save_new',interrupted):execute('construct',root)
    elif mode == 'native_stall':
        import highspy
        with patch.object(highspy.Highs, 'run', stall):
            execute('capture', root)
    elif mode == 'construct_stall':
        with patch('modular_supervised.construction.eliminate', stall):
            execute('construct', root)
    elif mode in ('readback_stall', 'readback_exception'):
        from fidelity_supervised import native
        original = native.model_snapshot
        calls = 0
        def after_return(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                save_new(root / 'fault_entered.json', {'synthetic': True, 'mode': mode})
                if mode == 'readback_exception':
                    raise OSError('analytic post-native readback failure')
                time.sleep(60)
            return original(*args, **kwargs)
        with patch.object(native, 'model_snapshot', after_return):
            execute('capture', root)
    elif mode == 'map_exception':
        def fail(*args, **kwargs):
            raise OSError('analytic post-capture mapping exception')
        with patch('fidelity_supervised.native.map_capture', fail):
            execute('map', root)
    elif mode == 'check_stall':
        save_new(root / 'fault_entered.json', {'synthetic': True, 'mode': mode})
        print('{"partial":', flush=True)
        time.sleep(60)
    else:
        raise ValueError('unregistered analytic fault')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    run(args.mode, args.root)
