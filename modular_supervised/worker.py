"""Separate processes, one original clock; no phase may reset its budget."""
import argparse
from pathlib import Path
import shutil
import sys
import time
from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest
from lp_sandwich.check import identity, validate_statement, deadline_tick, strict_json
from sparse_basis.engine import Limit, POLICY as NATIVE_POLICY
from modular_basis.engine import POLICY


def execute(phase, root):
    from modular_supervised.flow import limits, verify_sources
    root = Path(root)
    plan = read(root / 'plan.json')
    limit = plan['started'] + limits(phase)
    tick = deadline_tick(limit)
    tick()
    verify_sources(plan)
    spec = plan['spec']
    if phase == 'load':
        from modular_supervised.inputs import load
        v = load(spec, tick)
        save_new(root / 'prepared.json', v)
    elif phase == 'capture':
        from fidelity_supervised.native import capture
        v = read(root / 'prepared.json')
        capture(v['lp'], v['statement'], root / 'native', deadline=limit)
    elif phase == 'map':
        from fidelity_supervised.native import map_capture
        v = read(root / 'prepared.json')
        r = read(root / 'native/capture.json')
        m = map_capture(v['lp'], v['statement'], r, identity(r), deadline=limit)
        save_new(root / 'mapping.json', m)
    elif phase == 'construct':
        from modular_supervised.construction import propose
        v = read(root / 'prepared.json')
        m = read(root / 'mapping.json')
        if m['status'] != 'MAPPED_HINT_ONLY':
            raise ValueError('unmapped basis cannot enter constructor')
        result = propose(v['lp'], v['statement'], m['candidate'], m['hint'],
                         spec['statement_sha256'], identity(m['hint']), deadline=limit, root=root,
                         original_started=plan['started'])
        # Both result and candidate serialization are charged inside construct.
        writing=time.monotonic()-plan['started']
        save_new(root/'construction_write_entered.json',{'start_seconds':writing,'started':plan['started']})
        save_new(root / 'construction.json', result)
        if result['status'] == 'CANDIDATE_ONLY':
            save_new(root / 'bundle.json', result['bundle'])
        end=time.monotonic()-plan['started']
        save_new(root/'construction_write.json',{'started':plan['started'],'start_seconds':writing,
                 'end_seconds':end,'seconds':end-writing,
                 'construction_sha256':digest((root/'construction.json').read_bytes())})
        if result['status'] == 'TIMEOUT':
            raise TimeoutError('constructor exhausted original proposal deadline')
        if result['status'] == 'ERROR':
            raise ValueError('constructor error: ' + str(result['error']))
    elif phase == 'package':
        v = read(root / 'prepared.json')
        b = read(root / 'bundle.json')
        r = read(root / 'construction.json')
        if (b != r['bundle'] or b['lp'] != v['lp'] or b['statement'] != v['statement']
                or r['status'] != 'CANDIDATE_ONLY' or r['deadline_monotonic'] != plan['started'] + 218):
            raise ValueError('candidate identity/clock drift')
        out = root / 'portable'
        out.mkdir(exist_ok=False)
        shutil.copyfile(ROOT / 'lp_sandwich/check.py', out / 'verify.py')
        shutil.copyfile(root / 'bundle.json', out / 'bundle.json')
        save_new(root / 'packing.json', {
            'original_started': plan['started'],
            'bundle_sha256': digest((out / 'bundle.json').read_bytes()),
            'checker_sha256': digest((out / 'verify.py').read_bytes()),
            'statement_sha256': spec['statement_sha256']})
    else:
        raise ValueError('unregistered phase')
    tick()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('load', 'capture', 'map', 'construct', 'package'))
    p.add_argument('root', type=Path)
    a = p.parse_args()
    try:
        execute(a.phase, a.root)
    except TimeoutError:
        sys.exit(3)
    except Limit as exc:
        from modular_supervised.flow import limits
        p = read(a.root / 'plan.json')
        save_new(a.root / (a.phase + '_limit.json'), {'phase': a.phase, 'error': str(exc),
                 'deadline_monotonic': p['started'] + limits(a.phase), 'policy': dict(POLICY if a.phase=='construct' else NATIVE_POLICY)})
        sys.exit(4)
