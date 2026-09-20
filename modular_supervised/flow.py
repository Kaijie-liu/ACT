"""Owned capture→map→construct→pack→check supervision, never a network verdict."""
import math
from pathlib import Path
import subprocess
import time
from single_check_portable.execution import ROOT, ACT, read, save_new
from portable_proof.runtime import digest
from evidence_cohort.run import environment, wait_owned
from lp_diagnostic.flow import phase_state as prior_phase_state
from modular_basis.engine import POLICY
from sparse_basis.engine import POLICY as NATIVE_POLICY
from modular_supervised.journal import JOURNAL_POLICY
from fidelity_supervised.native import OPTIONS as NATIVE_OPTIONS

STAGES = ('load', 'capture', 'map', 'construct', 'package', 'check')
UNRESOLVED = ('UNSUPPORTED_MAPPING', 'UNRESOLVED_MODULAR_RECONSTRUCTION', 'LIMIT')


def phase_state(owner, elapsed, limit):
    if not owner['killed'] and owner['return_code'] == 4 and elapsed < limit:
        return 'LIMIT'
    return prior_phase_state(owner, elapsed, limit)


def limits(phase):
    if phase not in STAGES:
        raise ValueError('unregistered phase')
    return 218 if phase in STAGES[:4] else 298


def sources():
    paths = [p for folder in ('modular_supervised', 'modular_basis', 'sparse_basis', 'exact_primal')
             for p in (ROOT / folder).glob('*.py')]
    paths += [ROOT / p for p in ('docs/modular_supervised_v1.md', 'fidelity_supervised/native.py', 'docs/modular_basis_controls_attempt002.json', 'docs/modular_basis_v1_review_r2.json', 'lp_sandwich/check.py', 'lp_diagnostic/flow.py',
                                'lp_diagnostic/study.py', 'docs/lp_diagnostic_v1_freeze.json',
                                'docs/sparse_basis_controls_attempt004.json', 'docs/native_import_analysis_attempt002.json',
                                'single_check_portable/execution.py', 'evidence_cohort/run.py',
                                'portable_proof/runtime.py')]
    return {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sorted(paths)}


def verify_sources(plan):
    if (plan['sources'] != sources() or plan['schema'] != 'MODULAR_BASIS_SUPERVISION_PLAN_V1' or
            plan['component_policy'] != POLICY or plan['native_policy'] != NATIVE_POLICY or plan['journal_policy'] != JOURNAL_POLICY or plan['native_options'] != NATIVE_OPTIONS or
            (plan['total_seconds'], plan['work_seconds'], plan['proposal_seconds']) != (300, 298, 218)):
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
                cmd = [ACT, '-m', 'modular_supervised.worker', name, str(root)]
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


def review_candidate(root, *, deadline=None):
    """Structural/clock audit. The isolated checker, not this audit, proves LP bounds."""
    from lp_sandwich.check import identity
    from fidelity_supervised.native import map_capture
    from modular_basis.engine import POLICY
    deadline = time.monotonic() + 300 if deadline is None else deadline
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
        if row['state'] == 'LIMIT':
            bound = read(root / (name + '_limit.json'))
            if bound['phase'] != name or bound['deadline_monotonic'] != p['started'] + limit or bound['policy'] != (POLICY if name == 'construct' else NATIVE_POLICY):
                raise ValueError('component limit binding')
        previous = row['end_seconds']
    names = [r['name'] for r in c['stages'] if r['state'] == 'COMPLETED']
    spec = p['spec']
    if 'load' in names:
        v = read(root / 'prepared.json')
        from modular_supervised.inputs import load
        from lp_sandwich.check import deadline_tick
        if load(spec, deadline_tick(deadline)) != v:
            raise ValueError('original supplied LP drift')
        if v['statement'] != spec['statement'] or identity(v['statement']) != spec['statement_sha256'] or identity(v['lp']) != spec['statement']['lp_sha256']:
            raise ValueError('prepared identity')
    if 'capture' in names:
        r = read(root / 'native/capture.json')
        raw = read(root / 'native/raw_native.json')
        imported = read(root / 'native/import.json')
        imp_status = read(root / 'native/import_status.json')
        from fidelity_supervised.native import preflight
        from lp_sandwich.check import deadline_tick
        if (read(root/'native/preflight.json')!=preflight(r['submitted'],deadline_tick(deadline)) or
                imported['status']!='HighsStatus.kOk' or imported['readback']!=r['submitted'] or
                imported['submitted_sha256']!=identity(r['submitted']) or imported['options']!=NATIVE_OPTIONS or
                not 0<=imported['seconds']<=r['seconds'] or
                imp_status!={'status':'HighsStatus.kOk','options':NATIVE_OPTIONS,
                             'submitted_sha256':identity(r['submitted']), 'deadline_monotonic':p['started']+218}):
            raise ValueError('import fidelity/protocol identity')
        if (r['schema'] != 'FIDELITY_NATIVE_BASIS_CAPTURE_V2' or r['policy'] != NATIVE_POLICY or
                raw != {k: r[k] for k in RAW_KEYS} or set(raw) != set(RAW_KEYS) or
                r['deadline_monotonic'] != p['started'] + 218 or r['native_calls'] != 1 or
                not 0 < r['options']['time_limit'] <= 10 or
                not 0 <= r['native_seconds'] <= r['seconds'] <= c['stages'][1]['elapsed_seconds'] or
                read(root / 'native/input.json') != {**v, 'submitted': r['submitted']} or
                read(root / 'native/submission.json') != {
                    'expected': r['submitted'], 'readback': r['readback_before'], 'options': r['options']}):
            raise ValueError('native count/cost/clock')
        mapped = map_capture(v['lp'], v['statement'], r, identity(r), deadline=deadline)
    if 'map' in names and read(root / 'mapping.json') != mapped:
        raise ValueError('mapping identity')
    if 'construct' in names:
        construction = read(root / 'construction.json')
        if (construction['schema'] != 'TIMED_MODULAR_BASIS_PROPOSAL_V1' or construction['policy'] != POLICY or
                construction['deadline_monotonic'] != p['started'] + 218 or construction['solver_calls'] != 0 or
                construction['attempts'] != 1 or construction['statement_sha256'] != spec['statement_sha256'] or
                construction['hint_sha256'] != identity(mapped['hint']) or construction['feasibility_certified'] or
                construction['network_SAFE'] or construction['network_UNSAFE'] or
                not 0 <= construction['seconds'] <= c['stages'][3]['elapsed_seconds']):
            raise ValueError('construction provenance/clock')
        from modular_supervised.journal import journal_costs
        journal_costs(root, allow_partial=False, construction=construction)
        write=read(root/'construction_write.json')
        entered=read(root/'construction_write_entered.json')
        stage=c['stages'][3]
        if (write['started']!=p['started'] or write['construction_sha256']!=digest((root/'construction.json').read_bytes()) or
                entered!={'started':p['started'],'start_seconds':write['start_seconds']} or
                not stage['start_seconds']<=construction['journal']['component_started']-p['started'] or
                construction['journal']['component_started']-p['started']+construction['seconds']>write['start_seconds']+1e-7 or
                not stage['start_seconds']<=write['start_seconds']<=write['end_seconds']<=stage['end_seconds'] or
                abs(write['seconds']-write['end_seconds']+write['start_seconds'])>1e-8):
            raise ValueError('construction serialization clock/binding')
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
    save_new(root / 'plan.json', {'schema': 'MODULAR_BASIS_SUPERVISION_PLAN_V1', 'spec': spec, 'started': started,
                                 'total_seconds': 300, 'work_seconds': 298, 'proposal_seconds': 218,
                                 'component_policy': dict(POLICY), 'native_policy':dict(NATIVE_POLICY), 'journal_policy':dict(JOURNAL_POLICY), 'native_options':dict(NATIVE_OPTIONS), 'sources': sources()})
    owner, c, error, status = None, None, None, 'TIMEOUT'
    try:
        if time.monotonic() >= started + 298:
            raise TimeoutError('no remaining work budget')
        with (root / 'driver.log').open('xb') as log:
            proc = subprocess.Popen([ACT, '-m', 'modular_supervised.flow', str(root)], cwd=ROOT,
                                    env=environment(), stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            owner = wait_owned(proc, started + 298)
        exited = time.monotonic() - started
        if owner['killed'] or exited >= 298:
            status = 'TIMEOUT'
        elif owner['return_code'] != 0:
            status = 'ERROR'
        else:
            c = review_candidate(root, deadline=started + 298)
            status = c['status']
    except TimeoutError:
        exited = time.monotonic() - started
    except Exception as exc:
        exited, error, status = time.monotonic() - started, repr(exc), 'ERROR'
    observed = time.monotonic() - started
    if observed >= 300:
        status = 'TIMEOUT'
    terminal = {'schema': 'SUPERVISED_MODULAR_BASIS_V1', 'status': status,
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
    verify_sources(p)
    if v['schema'] != 'SUPERVISED_MODULAR_BASIS_V1':
        raise ValueError('terminal version mismatch')
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
    from modular_supervised.journal import journal_costs
    if (root/'arithmetic_events').exists():
        progress=journal_costs(root,allow_partial=True)
        if progress.get('last_observed_request_seconds',0)>v['driver_exit_seconds']+1e-8:
            raise ValueError('arithmetic progress after driver exit')
    return v


RAW_KEYS = ('run_status', 'model_status', 'basis_valid', 'value_valid', 'column_status', 'row_status',
            'column_values', 'row_values', 'basic_variables_status', 'basic_variables', 'native_objective',
            'native_seconds', 'lp_sha256', 'statement_sha256')


def raw_cost(root, raw, window):
    """Only recorded native return counts as known. This certifies no native value."""
    from lp_sandwich.check import identity
    from fidelity_supervised.native import OPTIONS
    v = read(root / 'prepared.json')
    inp, sub = read(root / 'native/input.json'), read(root / 'native/submission.json')
    if (set(raw) != set(RAW_KEYS) or
            (raw['lp_sha256'], raw['statement_sha256']) != (identity(v['lp']), identity(v['statement'])) or
            inp != {**v, 'submitted': sub['expected']} or sub['readback'] != sub['expected'] or
            set(sub['options']) != set(OPTIONS) | {'time_limit'} or
            any(sub['options'][k] != value for k, value in OPTIONS.items()) or
            not 0 < sub['options']['time_limit'] <= 10 or
            isinstance(raw['native_seconds'], bool) or not math.isfinite(raw['native_seconds']) or
            not 0 <= raw['native_seconds'] <= window):
        raise ValueError('partial native provenance or cost')
    return raw['native_seconds']


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
    raw = saved(root / 'native/raw_native.json')
    native_seconds, native_calls, evidence, invalid = None, None, None, []
    if raw is not None:
        try:
            part = parts['capture']
            window = part['seconds'] if part['seconds'] is not None else part.get('observed_window_seconds')
            if window is None:
                raise ValueError('no observed capture window')
            native_seconds = raw_cost(root, raw, window)
            native_calls, evidence = 1, 'native/raw_native.json'
            if native is not None and any(native[k] != raw[k] for k in RAW_KEYS):
                raise ValueError('raw/complete capture mismatch')
        except (ValueError, KeyError, TypeError, OSError) as exc:
            if terminal['status'] not in ('TIMEOUT', 'ERROR'):
                raise
            invalid.append(repr(exc))
            native_seconds, native_calls, evidence = None, None, None
    construction_seconds = None
    if constructed is not None:
        try:
            construction_seconds = constructed['seconds']
            part = parts['construct']
            window = part['seconds'] if part['seconds'] is not None else part.get('observed_window_seconds')
            if (window is None or isinstance(construction_seconds, bool) or
                    not math.isfinite(construction_seconds) or not 0 <= construction_seconds <= window):
                raise ValueError('partial construction cost')
        except (ValueError, KeyError, TypeError) as exc:
            if terminal['status'] not in ('TIMEOUT', 'ERROR'):
                raise
            invalid.append(repr(exc))
            construction_seconds = None
    from modular_supervised.journal import journal_costs
    arithmetic=journal_costs(root, allow_partial=terminal['status'] in ('TIMEOUT','ERROR'), construction=constructed)
    writing=saved(root/'construction_write.json')
    entered=saved(root/'construction_write_entered.json')
    if writing is not None:
        p=read(root/'plan.json')
        if (writing['started']!=p['started'] or writing['construction_sha256']!=digest((root/'construction.json').read_bytes()) or
                not 0<=writing['start_seconds']<=writing['end_seconds']<=whole or
                abs(writing['seconds']-writing['end_seconds']+writing['start_seconds'])>1e-8):
            raise ValueError('partial serialization accounting')
    serialization={'seconds':None if writing is None else writing['seconds'],
                   'censored':writing is None and entered is not None,
                   'observed_window_seconds':None if entered is None or writing is not None else
                       max(0,terminal['driver_exit_seconds']-entered['start_seconds'])}
    return {'whole_supplied_LP_seconds': whole, 'phases': parts, 'observed_phase_sum_seconds': observed,
            'residual_seconds': whole - observed,
            'native_seconds': native_seconds, 'native_calls': native_calls, 'native_cost_evidence': evidence,
            'invalid_partial_cost_records': invalid,
            'construction_seconds': construction_seconds,
            'arithmetic_progress':arithmetic, 'construction_serialization':serialization,
            'unreadable_partial_records': unreadable,
            'accounting': 'whole = disjoint recorded phase windows + residual; native/construction nested, not additive',
            'scope': 'supplied LP only; no historical network propagation/range/F0 cost or end-to-end MoE claim',
            'endpoint': 'publication observation; cleanup overruns remain charged and invalidate acceptance'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    drive(parser.parse_args().root)
