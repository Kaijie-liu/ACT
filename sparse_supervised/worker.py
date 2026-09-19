"""Separate processes, one original clock; no phase may reset its budget."""
import argparse
from pathlib import Path
import shutil
import sys
from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest
from lp_sandwich.check import identity, validate_statement, deadline_tick, strict_json
from sparse_basis.engine import Limit, POLICY


def execute(phase, root):
    from sparse_supervised.flow import limits, verify_sources
    root = Path(root)
    plan = read(root / 'plan.json')
    limit = plan['started'] + limits(phase)
    tick = deadline_tick(limit)
    tick()
    verify_sources(plan)
    spec = plan['spec']
    if phase == 'load':
        from sparse_supervised.inputs import load
        v = load(spec, tick)
        save_new(root / 'prepared.json', v)
    elif phase == 'capture':
        from sparse_basis.native import capture
        v = read(root / 'prepared.json')
        capture(v['lp'], v['statement'], root / 'native', deadline=limit)
    elif phase == 'map':
        from sparse_basis.native import map_capture
        v = read(root / 'prepared.json')
        r = read(root / 'native/capture.json')
        m = map_capture(v['lp'], v['statement'], r, identity(r), deadline=limit)
        save_new(root / 'mapping.json', m)
    elif phase == 'construct':
        from sparse_basis.engine import propose
        v = read(root / 'prepared.json')
        m = read(root / 'mapping.json')
        if m['status'] != 'MAPPED_HINT_ONLY':
            raise ValueError('unmapped basis cannot enter constructor')
        result = propose(v['lp'], v['statement'], m['candidate'], m['hint'],
                         spec['statement_sha256'], identity(m['hint']), deadline=limit)
        # Preserve candidate-only output, including unsuccessful attempts.
        save_new(root / 'construction.json', result)
        if result['status'] == 'TIMEOUT':
            raise TimeoutError('constructor exhausted original proposal deadline')
        if result['status'] == 'ERROR':
            raise ValueError('constructor error: ' + str(result['error']))
        if result['status'] == 'CANDIDATE_ONLY':
            save_new(root / 'bundle.json', result['bundle'])
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
        from sparse_supervised.flow import limits
        p = read(a.root / 'plan.json')
        save_new(a.root / (a.phase + '_limit.json'), {'phase': a.phase, 'error': str(exc),
                 'deadline_monotonic': p['started'] + limits(a.phase), 'policy': dict(POLICY)})
        sys.exit(4)
