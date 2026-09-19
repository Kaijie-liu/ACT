"""Owned capture→map→construct→pack→check supervision, never a network verdict."""
import math
from pathlib import Path
import subprocess
import time
from single_check_portable.execution import ROOT, ACT, read, save_new
from portable_proof.runtime import digest
from evidence_cohort.run import environment, wait_owned
from lp_diagnostic.flow import phase_state

STAGES = ('load', 'capture', 'map', 'construct', 'package', 'check')
UNRESOLVED = ('UNSUPPORTED_MAPPING', 'UNRESOLVED_SINGULAR_BASIS', 'LIMIT')


def limits(phase):
    if phase not in STAGES:
        raise ValueError('unregistered phase')
    return 218 if phase in STAGES[:4] else 298


def sources():
    paths = [p for folder in ('basis_supervised', 'native_basis', 'exact_basis', 'exact_primal')
             for p in (ROOT / folder).glob('*.py')]
    paths += [ROOT / p for p in ('lp_sandwich/check.py', 'lp_diagnostic/flow.py',
                                'single_check_portable/execution.py', 'evidence_cohort/run.py',
                                'portable_proof/runtime.py')]
    return {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sorted(paths)}


def verify_sources(plan):
    if plan['sources'] != sources():
        raise ValueError('execution source drift')


def inventory(root):
    return {str(f.relative_to(root)): digest(f.read_bytes()) for f in sorted(root.rglob('*'))
            if f.is_file() and str(f.relative_to(root)) not in
            ('plan.json', 'driver.log', 'candidate.json', 'outer.json', 'publication.json', 'publication_timeout.json')}


def owned_stage(root, name, command, started, limit):
    """Generic owned process boundary, also exercised with synthetic fault workers."""
    begin = time.monotonic() - started
    if begin >= limit:
        raise TimeoutError('no phase budget')
    save_new(root / (name + '_entered.json'), {'phase': name, 'seconds': begin})
    with (root / (name + '.log')).open('xb') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        owner = wait_owned(proc, started + limit)
    end = time.monotonic() - started
    row = {'name': name, 'start_seconds': begin, 'end_seconds': end,
           'elapsed_seconds': end - begin, 'limit_seconds': limit, 'process': owner,
           'state': phase_state(owner, end, limit)}
    save_new(root / (name + '_stage.json'), row)
    return row


def semantic_stop(root, phase):
    if phase == 'map':
        status = read(root / 'mapping.json')['status']
        if status == 'UNSUPPORTED_MAPPING':
            return status
        if status != 'MAPPED_HINT_ONLY':
            raise ValueError('unknown mapping outcome')
    if phase == 'construct':
        status = read(root / 'construction.json')['status']
        if status in UNRESOLVED[1:]:
            return status
        if status != 'CANDIDATE_ONLY':
            raise ValueError('unknown constructor outcome')
    return None


def drive(root):
    root = Path(root)
    p = read(root / 'plan.json')
    start = p['started']
    stages, error, status = [], None, 'ERROR'
    try:
        verify_sources(p)
        for name in STAGES:
            if name == 'check':
                pack = read(root / 'packing.json')
                cmd = [ACT, '-I', '-S', str(root / 'portable/verify.py'), str(root / 'portable/bundle.json'),
                       '--bundle-sha256', pack['bundle_sha256'],
                       '--statement-sha256', pack['statement_sha256'],
                       '--timeout-seconds', repr(max(.000001, start + 298 - time.monotonic()))]
            else:
                cmd = [ACT, '-m', 'basis_supervised.worker', name, str(root)]
            row = owned_stage(root, name, cmd, start, limits(name))
            stages.append(row)
            if row['state'] != 'COMPLETED':
                status = row['state']
                break
            stop = semantic_stop(root, name)
            if stop:
                status = stop
                break
        else:
            status = 'CHECKED_LP_DIAGNOSTIC'
    except TimeoutError:
        status = 'TIMEOUT'
    except Exception as exc:
        status, error = 'ERROR', repr(exc)
    if time.monotonic() - start >= 298:
        status = 'TIMEOUT'
    save_new(root / 'candidate.json', {
        'status': status, 'complete_independent_check': status == 'CHECKED_LP_DIAGNOSTIC',
        'stages': stages, 'error': error, 'artifact_sha256': inventory(root),
        'plan_sha256': digest((root / 'plan.json').read_bytes()), 'wall_seconds': time.monotonic() - start})


def review_candidate(root):
    """Structural/clock audit. The isolated checker, not this audit, proves LP bounds."""
    from lp_sandwich.check import identity
    from native_basis.adapter import map_capture
    root = Path(root)
    p, c = read(root / 'plan.json'), read(root / 'candidate.json')
    if c['plan_sha256'] != digest((root / 'plan.json').read_bytes()) or c['artifact_sha256'] != inventory(root):
        raise ValueError('plan/artifact drift or omission')
    previous = 0
    for i, row in enumerate(c['stages']):
        if i >= len(STAGES) or row['name'] != STAGES[i]:
            raise ValueError('phase order')
        name, limit = STAGES[i], limits(STAGES[i])
        if (row['limit_seconds'] != limit or
                not previous <= row['start_seconds'] <= row['end_seconds'] <= c['wall_seconds'] or
                abs(row['end_seconds'] - row['start_seconds'] - row['elapsed_seconds']) > 1e-8 or
                row['state'] != phase_state(row['process'], row['end_seconds'], limit) or
                row != read(root / (name + '_stage.json')) or
                read(root / (name + '_entered.json')) != {'phase': name, 'seconds': row['start_seconds']}):
            raise ValueError('phase clock/identity')
        if i and c['stages'][i - 1]['state'] != 'COMPLETED':
            raise ValueError('execution after failed phase')
        previous = row['end_seconds']
    names = [r['name'] for r in c['stages'] if r['state'] == 'COMPLETED']
    spec = p['spec']
    if 'load' in names:
        v = read(root / 'prepared.json')
        if v['statement'] != spec['statement'] or identity(v['statement']) != spec['statement_sha256'] or identity(v['lp']) != spec['statement']['lp_sha256']:
            raise ValueError('prepared identity')
    if 'capture' in names:
        r = read(root / 'native/capture.json')
        if (r['deadline_monotonic'] != p['started'] + 218 or r['native_calls'] != 1 or
                not 0 < r['options']['time_limit'] <= 10 or
                not 0 <= r['native_seconds'] <= r['seconds'] <= c['stages'][1]['elapsed_seconds'] or
                read(root / 'native/input.json') != {**v, 'submitted': r['submitted']} or
                read(root / 'native/submission.json') != {
                    'expected': r['submitted'], 'readback': r['readback_before'], 'options': r['options']}):
            raise ValueError('native count/cost/clock')
        mapped = map_capture(v['lp'], v['statement'], r, identity(r))
    if 'map' in names and read(root / 'mapping.json') != mapped:
        raise ValueError('mapping identity')
    if 'construct' in names:
        construction = read(root / 'construction.json')
        if (construction['deadline_monotonic'] != p['started'] + 218 or construction['solver_calls'] != 0 or
                construction['attempts'] != 1 or construction['statement_sha256'] != spec['statement_sha256'] or
                construction['hint_sha256'] != identity(mapped['hint']) or construction['feasibility_certified'] or
                construction['network_SAFE'] or construction['network_UNSAFE'] or
                not 0 <= construction['seconds'] <= c['stages'][3]['elapsed_seconds']):
            raise ValueError('construction provenance/clock')
    expected = 'ERROR'
    if c['wall_seconds'] >= 298:
        expected = 'TIMEOUT'
    elif c['error']:
        expected = 'ERROR'
    elif c['stages'] and c['stages'][-1]['state'] != 'COMPLETED':
        expected = c['stages'][-1]['state']
    elif names and semantic_stop(root, names[-1]):
        expected = semantic_stop(root, names[-1])
    elif len(names) == len(STAGES):
        pack = read(root / 'packing.json')
        b = read(root / 'portable/bundle.json')
        out = read(root / 'check.log')
        if (pack['original_started'] != p['started'] or pack['statement_sha256'] != spec['statement_sha256'] or
                pack['checker_sha256'] != p['sources']['lp_sandwich/check.py'] or
                digest((root / 'portable/verify.py').read_bytes()) != pack['checker_sha256'] or
                digest((root / 'portable/bundle.json').read_bytes()) != pack['bundle_sha256'] or
                digest((root / 'bundle.json').read_bytes()) != pack['bundle_sha256'] or
                b != construction['bundle'] or b['lp'] != v['lp'] or b['statement'] != v['statement'] or
                out['status'] != 'CHECKED_LP_DIAGNOSTIC' or out['statement_sha256'] != spec['statement_sha256'] or
                out['lp_sha256'] != spec['statement']['lp_sha256'] or not out['isolated'] or
                not out['site_disabled'] or out['solver_or_model_imported'] or out['network_SAFE'] or out['network_UNSAFE'] or
                not 0 <= out['seconds'] <= c['stages'][5]['elapsed_seconds']):
            raise ValueError('isolated proof binding')
        expected = 'CHECKED_LP_DIAGNOSTIC'
    elif c['status'] == 'TIMEOUT' and len(c['stages']) < len(STAGES):
        if c['wall_seconds'] < limits(STAGES[len(c['stages'])]):
            raise ValueError('unexplained timeout')
        expected = 'TIMEOUT'
    if (c['status'], c['complete_independent_check']) != (expected, expected == 'CHECKED_LP_DIAGNOSTIC'):
        raise ValueError('candidate acceptance')
    # Unsupported mapping must stop before any reconstruction; no fake missing proof.
    if 'map' in names and mapped['status'] == 'UNSUPPORTED_MAPPING' and len(c['stages']) != 3:
        raise ValueError('continued unsupported mapping')
    return c


def supervise(spec, destination, *, started):
    if not math.isfinite(started) or started > time.monotonic():
        raise ValueError('original start required')
    root = Path(destination).resolve()
    if not root.is_relative_to(Path('/data1/Kane/MOE')):
        raise ValueError('outside workspace')
    root.mkdir(exist_ok=False)
    save_new(root / 'plan.json', {'schema': 'BASIS_SUPERVISION_PLAN_V1', 'spec': spec, 'started': started,
                                 'total_seconds': 300, 'work_seconds': 298, 'proposal_seconds': 218,
                                 'sources': sources()})
    owner, c, error, status = None, None, None, 'TIMEOUT'
    try:
        if time.monotonic() >= started + 298:
            raise TimeoutError('no remaining work budget')
        with (root / 'driver.log').open('xb') as log:
            proc = subprocess.Popen([ACT, '-m', 'basis_supervised.flow', str(root)], cwd=ROOT,
                                    env=environment(), stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            owner = wait_owned(proc, started + 298)
        exited = time.monotonic() - started
        if owner['killed'] or exited >= 298:
            status = 'TIMEOUT'
        elif owner['return_code'] != 0:
            status = 'ERROR'
        else:
            c = review_candidate(root)
            status = c['status']
    except TimeoutError:
        exited = time.monotonic() - started
    except Exception as exc:
        exited, error, status = time.monotonic() - started, repr(exc), 'ERROR'
    observed = time.monotonic() - started
    if observed >= 300:
        status = 'TIMEOUT'
    terminal = {'schema': 'SUPERVISED_BASIS_V1', 'status': status,
                'complete_independent_check': status == 'CHECKED_LP_DIAGNOSTIC',
                'outer_process': owner, 'driver_exit_seconds': exited, 'observed_seconds': observed,
                'error': error, 'plan_sha256': digest((root / 'plan.json').read_bytes()),
                'candidate_sha256': digest((root / 'candidate.json').read_bytes()) if c else None,
                'artifact_sha256': inventory(root), 'budget_seconds': 300,
                'network_SAFE': False, 'network_UNSAFE': False}
    save_new(root / 'outer.json', terminal)
    save_new(root / 'publication.json', {'outer_sha256': digest((root / 'outer.json').read_bytes()),
                                        'observed_seconds': time.monotonic() - started})
    if time.monotonic() >= started + 300:
        save_new(root / 'publication_timeout.json', {'status': 'TIMEOUT', 'complete_independent_check': False})
        terminal.update(status='TIMEOUT', complete_independent_check=False)
    return terminal


def audit(root):
    root = Path(root)
    p, v, pub = read(root / 'plan.json'), read(root / 'outer.json'), read(root / 'publication.json')
    if (v['plan_sha256'] != digest((root / 'plan.json').read_bytes()) or
            pub['outer_sha256'] != digest((root / 'outer.json').read_bytes()) or
            (p['total_seconds'], p['work_seconds'], p['proposal_seconds'], v['budget_seconds']) != (300, 298, 218, 300) or
            not 0 <= v['driver_exit_seconds'] <= v['observed_seconds'] <= pub['observed_seconds'] or
            v['artifact_sha256'] != inventory(root) or v['network_SAFE'] or v['network_UNSAFE']):
        raise ValueError('outer clock/identity/artifact drift')
    owner, status = v['outer_process'], 'TIMEOUT'
    if owner and not owner['killed'] and v['driver_exit_seconds'] < 298 and v['observed_seconds'] < 300:
        if owner['return_code'] != 0 or v['error']:
            status = 'ERROR'
        else:
            if v['candidate_sha256'] != digest((root / 'candidate.json').read_bytes()):
                raise ValueError('candidate drift')
            status = review_candidate(root)['status']
    elif owner is None and v['error'] and v['observed_seconds'] < 298:
        status = 'ERROR'
    if (v['status'], v['complete_independent_check']) != (status, status == 'CHECKED_LP_DIAGNOSTIC'):
        raise ValueError('outer acceptance')
    if pub['observed_seconds'] >= 300 or (root / 'publication_timeout.json').exists():
        return {**v, 'status': 'TIMEOUT', 'complete_independent_check': False}
    return v


def costs(root):
    root = Path(root)
    terminal = audit(root)
    parts, unreadable = {}, []
    def saved(path):
        if not path.exists():
            return None
        try:
            return read(path)
        except (ValueError, UnicodeError):
            if terminal['status'] not in ('TIMEOUT', 'ERROR'):
                raise
            unreadable.append(str(path.relative_to(root)))
            return None
    for name in STAGES:
        row, event = saved(root / (name + '_stage.json')), saved(root / (name + '_entered.json'))
        if row:
            parts[name] = {'seconds': row['elapsed_seconds'], 'state': row['state'], 'censored': row['state'] == 'TIMEOUT'}
        elif event:
            parts[name] = {'seconds': None, 'state': 'INTERRUPTED', 'censored': True,
                           'observed_window_seconds': max(0, terminal['driver_exit_seconds'] - event['seconds'])}
        else:
            parts[name] = {'seconds': None, 'state': 'NOT_REACHED_OR_UNRECORDED', 'censored': None}
    whole = read(root / 'publication.json')['observed_seconds']
    observed = sum(v['seconds'] for v in parts.values() if v['seconds'] is not None)
    if observed > whole + 1e-8:
        raise ValueError('overlapping phase costs')
    native, constructed = saved(root / 'native/capture.json'), saved(root / 'construction.json')
    return {'whole_supplied_LP_seconds': whole, 'phases': parts, 'observed_phase_sum_seconds': observed,
            'residual_seconds': whole - observed,
            'native_seconds': native['native_seconds'] if native else None,
            'native_calls': native['native_calls'] if native else None,
            'construction_seconds': constructed['seconds'] if constructed else None,
            'unreadable_partial_records': unreadable,
            'accounting': 'whole = disjoint recorded phase windows + residual; native/construction nested, not additive',
            'scope': 'supplied LP only; no historical network propagation/range/F0 cost or end-to-end MoE claim',
            'endpoint': 'publication observation; cleanup overruns remain charged and invalidate acceptance'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    drive(parser.parse_args().root)
