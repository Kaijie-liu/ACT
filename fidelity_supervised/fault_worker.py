"""Analytic test harness only. Production supervise/drive never dispatches here."""
import argparse
from pathlib import Path
import time
from unittest.mock import patch
from single_check_portable.execution import save_new
from fidelity_supervised.worker import execute


def run(mode, root):
    if mode.startswith('drive_'):
        from fidelity_supervised import flow
        from single_check_portable.execution import ACT
        fault = mode.removeprefix('drive_')
        phase = {'native_stall': 'capture', 'construct_stall': 'construct', 'map_exception': 'map',
                 'readback_stall': 'capture', 'readback_exception': 'capture',
                 'check_stall': 'check'}[fault]
        original = flow.owned_stage
        def injected(folder, name, command, started, limit):
            if name == phase:
                command = [ACT, '-m', 'fidelity_supervised.fault_worker', fault, str(folder)]
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
    if mode == 'native_stall':
        import highspy
        with patch.object(highspy.Highs, 'run', stall):
            execute('capture', root)
    elif mode == 'construct_stall':
        with patch('sparse_basis.engine.eliminate', stall):
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
