"""Finite actual HybridZ propagation under one CPU request budget.

Not a whole-MoE proof, portable checker or physical GPU admission protocol.
"""
from fractions import Fraction
import math
import os
from pathlib import Path
import time
import uuid

from scoped_proof.io import ROOT, PYTHON, load, save, sha, tick
from scoped_proof.owned_bounded import execute
from source_enclosure.format import identity
from scripts.run_hz_checked_propagation_controls import FILES as PROPAGATION_FILES

CONFIG = 'configs/hz_propagation_supervision_20261001.json'
PROTOCOL_SHA = '1b86f70f0fd80e741598e332f8187259320efc29577c12770626aae5ea39f452'
DONE = 'CHECKED_PROPAGATION_EXECUTION_CONDITIONAL_ON_TRUSTED_LOWERING'
PHASES = ('produce', 'check', 'receive')
LIMIT = 4*2**20
FILES = tuple(dict.fromkeys([*PROPAGATION_FILES, CONFIG,
    'docs/hz_checked_propagation_20261001_r5.json',
    'scoped_proof/io.py', 'scoped_proof/owned_bounded.py', 'source_enclosure/format.py',
    'scripts/hz_propagation_supervised.py', 'scripts/hz_propagation_worker.py']))


def required_hash(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('mandatory externally bound hash')
    return value


def protocol():
    if sha(ROOT/'docs/hz_checked_propagation_20261001_r5.json') != '981a506133f05a99349276ae99e2f53bca3f688caac135577b435fbd18912b69':
        raise ValueError('frozen propagation reference changed')
    return load(ROOT/CONFIG, PROTOCOL_SHA)


def sources():
    return {name: sha(ROOT/name) for name in FILES}


def finite(value, *, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError('finite nonnegative time/cost required')
    return value


def deadline_status(status, now, deadline):
    # Unresolved ownership is a mandatory stop, even if the budget also expired.
    return status if status == 'CLEANUP_INCOMPLETE' else 'TIMEOUT' if now >= deadline else status


def specification(case='retained_guard', control=''):
    spec = {'schema': 'HZ_PROPAGATION_SPEC_V1', 'case': case, 'control': control,
            'protocol_sha256': PROTOCOL_SHA, **protocol()['cases'][case]}
    validate_spec(spec)
    return spec


def validate_spec(spec):
    cfg = protocol()
    if (spec.get('case') not in cfg['cases'] or spec.get('control') not in ('', *cfg['faults'])
            or (spec['control'] and spec['case'] != 'retained_guard')
            or spec != {'schema': 'HZ_PROPAGATION_SPEC_V1', 'case': spec['case'],
                        'control': spec['control'], 'protocol_sha256': PROTOCOL_SHA,
                        **cfg['cases'][spec['case']]}):
        raise ValueError('only frozen synthetic requests admitted')


def phase_deadline(inv, phase):
    return (inv['work_deadline']-min(5., inv['budget']/4)
            if phase == 'produce' else inv['work_deadline'])


def run_deadline(inv, phase):
    return phase_deadline(inv, phase)-min(.25, inv['budget']/20)


def bind(root, spec, invocation_sha):
    inv = load(root/'invocation.json', required_hash(invocation_sha), LIMIT)
    validate_spec(spec)
    for name in ('start', 'deadline', 'work_deadline', 'budget'):
        finite(inv[name])
    if (inv['spec_sha256'] != identity(spec) or load(root/'spec.json') != spec
            or inv['sources'] != sources() or inv['deadline'] != inv['start']+inv['budget']
            or inv['work_deadline'] != inv['deadline']-min(1., inv['budget']/5)
            or not 0 < inv['budget'] <= 300 or type(inv['rss_limit']) is not int
            or not 0 < inv['rss_limit'] <= 2*2**30):
        raise ValueError('caller/source/deadline binding')
    return inv


def validate_payload(payload, spec, inv):
    p = payload['propagation']
    if (payload.get('schema') != 'HZ_PROPAGATION_PAYLOAD_V1'
            or payload.get('invocation') != inv['invocation']
            or payload.get('spec_sha256') != identity(spec)
            or p['scope_sha256'] != spec['scope_sha256'] or identity(p['scope']) != p['scope_sha256']
            or p.get('status') != 'PROPAGATED_TRUSTED_LEGACY_LOWERING'
            or p.get('network_or_complete_moe_proof') is not False
            or p.get('hard_budget_supervision') is not False):
        raise ValueError('propagation source or guarantee')
    events = p['events']
    if [e['layer_id'] for e in events] != spec['relu_layers']:
        raise ValueError('complete ReLU event roster')
    summaries = []
    for event, count in zip(events, spec['query_counts']):
        if event.get('completed') is not True or event['scope_sha256'] != p['scope_sha256']:
            raise ValueError('incomplete/foreign layer')
        package = event['package']
        if count == 0:
            if package is not None or event['status'] != 'NO_CONSTRAINTS':
                raise ValueError('unexpected checked obligation')
        else:
            from scripts.run_hz_checked_propagation_controls import validate_query_roster
            if package is None or event['status'] != 'CHECKED_SUPPORT_APPLIED':
                raise ValueError('missing used support evidence')
            validate_query_roster(package['batch'], event, p)
            checked = package['accepted']
            rows = checked['results']
            expected = [(q['id'], q['side']) for q in package['batch']['queries']]
            if (len(rows) != count or [(r['id'], r['side']) for r in rows] != expected
                    or checked['batch_sha256'] != identity(package['batch'])
                    or checked['candidate_sha256'] != identity(package['candidates'])
                    or checked.get('network_or_complete_moe_proof') is not False
                    or checked['status'] != 'CHECKED_GIVEN_HZ_CONTINUOUS_RELAXATION'):
                raise ValueError('checked roster/binding')
            for r in rows:
                Fraction(r['bound']); required_hash(r['lp_sha256'])
            summaries.append({'layer': event['layer_id'], 'result': checked})
    for value in (p['construction_seconds'], p['total_seconds'], *[e['total_seconds'] for e in events]):
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError('finite nested propagation costs')
    if p['total_seconds']+1e-8 < p['construction_seconds']+sum(e['total_seconds'] for e in events):
        raise ValueError('nested propagation accounting')
    return {'scope_sha256': p['scope_sha256'], 'output_sha256': identity(p['output']),
            'required_support_bounds': sum(spec['query_counts']), 'layers': summaries}


def exact_check(payload, spec, inv, deadline):
    from act.back_end.moe.check_batched_support import check_batch
    summary = validate_payload(payload, spec, inv)
    for event in payload['propagation']['events']:
        package = event['package']
        if package is None:
            continue
        tick(deadline)
        checked = check_batch(package['batch'], package['candidates'],
                              expected_batch_sha256=identity(package['batch']), deadline=deadline)
        if checked != package['accepted']:
            raise ValueError('used support bound differs from exact recheck')
    tick(deadline)
    return summary


def receive(root, spec, invocation_sha, payload_sha, checker_sha, deadline):
    tick(deadline)
    inv = bind(root, spec, invocation_sha)
    payload = load(root/'payload.json', required_hash(payload_sha), LIMIT)
    checked = load(root/'check.json', required_hash(checker_sha), LIMIT)
    summary = validate_payload(payload, spec, inv)
    expected = {'schema': 'HZ_PROPAGATION_CHECK_V1', 'invocation': inv['invocation'],
                'spec_sha256': identity(spec), 'payload_sha256': payload_sha, 'summary': summary,
                'complete_moe_proof': False}
    if checked != expected:
        raise ValueError('checker-output reception mismatch')
    tick(deadline)
    return {'status': DONE, 'invocation': inv['invocation'], 'spec_sha256': identity(spec),
            'payload_sha256': payload_sha, 'checker_sha256': checker_sha, **summary,
            'complete_moe_proof': False, 'gpu_execution': False,
            'trusted': ['network and guard lowering', 'shared factor semantics', 'legacy float propagation']}


def inventory(root):
    # Fault/event prefixes remain mandatory even when there is no payload.
    names = [p.name for p in root.iterdir() if p.is_file() and
             (p.suffix in ('.log', '.jsonl', '.partial') or p.name.startswith('prefix_'))]
    return {name: sha(root/name) for name in sorted(names)}


def supervise(root, spec, *, expected_request_sha256, budget=300., rss_limit=2*2**30):
    start = time.monotonic()
    validate_spec(spec)
    if (type(budget) not in (int, float) or not math.isfinite(budget) or not 0 < budget <= 300
            or type(rss_limit) is not int or not 0 < rss_limit <= 2*2**30
            or identity(spec) != required_hash(expected_request_sha256)):
        raise ValueError('bounded caller request')
    root = Path(root)
    if not root.is_absolute() or not root.resolve().is_relative_to(ROOT.parent/'baseline_runs'):
        raise ValueError('new project result directory required')
    root.mkdir(parents=True, exist_ok=False)
    inv = {'schema': 'HZ_PROPAGATION_INVOCATION_V1', 'invocation': uuid.uuid4().hex,
           'start': start, 'deadline': start+budget, 'work_deadline': start+budget-min(1., budget/5),
           'budget': budget, 'rss_limit': rss_limit, 'spec_sha256': identity(spec), 'sources': sources()}
    save(root/'spec.json', spec)
    inv_sha = save(root/'invocation.json', inv)['sha256']
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    stages, payload_sha, checker_sha, accepted = [], None, None, None
    status, error = 'ERROR', None
    try:
        for phase in PHASES:
            before = time.monotonic()
            bind(root, spec, inv_sha)
            command = [PYTHON, '-B']+(['-S'] if phase == 'receive' else [])+[
                '-m', 'scripts.hz_propagation_worker', phase, str(root),
                '--invocation-sha', inv_sha, '--deadline', str(run_deadline(inv, phase))]
            if phase in ('check', 'receive'):
                command += ['--payload-sha', required_hash(payload_sha)]
            if phase == 'receive':
                command += ['--checker-sha', required_hash(checker_sha)]
            if spec['control'] == 'launch_failure':
                command[0] = str(root/'missing-python')
            record = execute(command, root/(phase+'.log'), run_deadline=run_deadline(inv, phase),
                             cleanup_deadline=phase_deadline(inv, phase), env=env, rss_limit=rss_limit)
            record.update(phase=phase, executable=command[0], cpu_threads=1, cuda_visible_devices='',
                          start_seconds=before-start)
            stages.append(record)
            try:
                if record['status'] == 'COMPLETED' and phase in ('produce', 'check'):
                    path = root/('payload.json' if phase == 'produce' else 'check.json')
                    load(path, limit=LIMIT)
                    anchor = sha(path)  # parent hashes the actual bytes after cleanup
                    record['output_sha256'] = anchor
                    if phase == 'produce': payload_sha = anchor
                    else: checker_sha = anchor
            except Exception as exc:
                record['reception_error'] = repr(exc)
                record['status'] = 'ERROR'
            record['end_seconds'] = time.monotonic()-start
            if start+record['end_seconds'] >= phase_deadline(inv, phase) and record['status'] == 'COMPLETED':
                record['status'] = 'TIMEOUT'
            save(root/(phase+'_stage.json'), record)
            if record['status'] != 'COMPLETED':
                status = record['status']
                break
        else:
            accepted = receive(root, spec, inv_sha, payload_sha, checker_sha, inv['work_deadline'])
            if load(root/'accepted.json', limit=LIMIT) != accepted:
                raise ValueError('receiver mismatch')
            status = DONE
    except Exception as exc:
        status, error, accepted = 'ERROR', repr(exc), None
    status = deadline_status(status, time.monotonic(), inv['work_deadline'])
    if status != DONE:
        accepted = None
    terminal = {'schema': 'HZ_PROPAGATION_TERMINAL_V1', 'invocation_sha256': inv_sha,
                'status': status, 'error': error, 'stages': stages, 'accepted': accepted if status == DONE else None,
                'prefix_inventory': inventory(root), 'seconds': time.monotonic()-start}
    terminal_sha = save(root/'terminal.json', terminal)['sha256']
    if spec['control'] == 'late_publish' and status == DONE:
        save(root/'late_publish_reached.json', {'invocation': inv['invocation'], 'seconds': time.monotonic()-start})
        time.sleep(max(0., inv['deadline']-time.monotonic())+.02)
    end = time.monotonic()
    status = deadline_status(status, end, inv['deadline'])
    stage_cost = sum(s['seconds'] for s in stages)
    cost = {'schema': 'HZ_PROPAGATION_COST_V1', 'terminal_sha256': terminal_sha,
            'invocation_sha256': inv_sha, 'status': status, 'seconds': end-start,
            'stage_seconds': stage_cost, 'parent_seconds': end-start-stage_cost,
            'includes': 'creation/imports, propagation/support, serialization, check, receive, owned cleanup, publication',
            'late_publication_witness_sha256': sha(root/'late_publish_reached.json') if (root/'late_publish_reached.json').exists() else None,
            'nested_propagation_seconds_are_not_added': True, 'complete_moe_proof': False}
    cost_sha = save(root/'cost.json', cost)['sha256']
    end = time.monotonic()
    status = deadline_status(status, end, inv['deadline'])
    return {'status': status, 'invocation_sha256': inv_sha, 'terminal_sha256': terminal_sha,
            'cost_sha256': cost_sha, 'seconds': end-start, 'complete_moe_proof': False}


def audit(root, *, observation=None, recheck=False):
    root = Path(root)
    spec = load(root/'spec.json'); inv_sha = sha(root/'invocation.json')
    inv = bind(root, spec, inv_sha)
    term = load(root/'terminal.json'); cost = load(root/'cost.json')
    finite(term['seconds'])
    for name in ('seconds', 'stage_seconds', 'parent_seconds'):
        finite(cost[name])
    if (term['invocation_sha256'] != inv_sha or cost['invocation_sha256'] != inv_sha
            or cost['terminal_sha256'] != sha(root/'terminal.json') or term['prefix_inventory'] != inventory(root)):
        raise ValueError('terminal/cost/failed-prefix chain')
    stages = term['stages']
    if not stages or [s['phase'] for s in stages] != list(PHASES[:len(stages)]):
        raise ValueError('mandatory phase prefix')
    stage_paths = {p.name for p in root.glob('*_stage.json')}
    if stage_paths != {s['phase']+'_stage.json' for s in stages}:
        raise ValueError('deleted or invented phase prefix')
    previous = 0.
    anchors = {}
    cleanup_pending = False
    for i, s in enumerate(stages):
        phase = s['phase']
        if load(root/(phase+'_stage.json')) != s:
            raise ValueError('stage receipt differs')
        for field in ('seconds', 'execution_seconds', 'cleanup_seconds', 'start_seconds', 'end_seconds'):
            if type(s[field]) not in (int,float) or not math.isfinite(s[field]) or s[field] < 0:
                raise ValueError('invalid charged cost')
        if (s['start_seconds'] < previous or s['end_seconds'] < s['start_seconds']+s['seconds']-1e-8
                or abs(s['seconds']-s['execution_seconds']-s['cleanup_seconds']) > 1e-8
                or s['cuda_visible_devices'] != '' or s['cpu_threads'] != 1
                or s['run_deadline'] != run_deadline(inv, phase)
                or s['cleanup_deadline'] != phase_deadline(inv, phase)
                or s['escaped_descendants_or_driver_cleanup'] is not False):
            raise ValueError('execution/cost/environment binding')
        expected_python = str(root/'missing-python') if spec['control']=='launch_failure' else PYTHON
        if s['executable'] != expected_python:
            raise ValueError('execution interpreter')
        # A parent-anchored prefix stays mandatory if reception later expired.
        if 'output_sha256' in s:
            if phase not in ('produce','check'):
                raise ValueError('unexpected phase output anchor')
            name = 'payload.json' if phase == 'produce' else 'check.json'
            load(root/name, required_hash(s['output_sha256']), LIMIT)
            anchors[phase] = s['output_sha256']
        if s['status'] == 'COMPLETED':
            finite(s['exit_observed_at'])
            if (s['cleanup_status'] != 'LEADER_REAPED_NO_LIVE_GROUP' or s['remaining_group']['live']
                    or type(s['pid']) is not int or s['pid'] <= 0 or s['descendant_on_leader_exit'] is not False
                    or s['exit_observed_at'] >= run_deadline(inv, phase)
                    or s['exit_observed_at'] < inv['start']+s['start_seconds']
                    or s['returncode'] != 0 or inv['start']+s['end_seconds'] >= phase_deadline(inv, phase)):
                raise ValueError('successful phase with unresolved cleanup/deadline')
            if phase in ('produce','check'):
                required_hash(s.get('output_sha256'))
        elif i != len(stages)-1:
            raise ValueError('continued after failed phase')
        if s['cleanup_status'] == 'CLEANUP_INCOMPLETE':
            cleanup_pending = True
            if s['status'] != 'CLEANUP_INCOMPLETE' or i != len(stages)-1:
                raise ValueError('unresolved cleanup hidden')
        previous = s['end_seconds']
    if cleanup_pending and (term['status'] != 'CLEANUP_INCOMPLETE' or cost['status'] != 'CLEANUP_INCOMPLETE'):
        raise ValueError('unresolved cleanup terminal hidden')
    if (term['seconds'] < previous or cost['seconds'] < term['seconds']
            or abs(cost['stage_seconds']-sum(s['seconds'] for s in stages)) > 1e-8
            or cost['parent_seconds'] < 0 or abs(cost['seconds']-cost['stage_seconds']-cost['parent_seconds']) > 1e-8
            or cost['nested_propagation_seconds_are_not_added'] is not True or cost['complete_moe_proof'] is not False):
        raise ValueError('complete cost conservation')
    if cost['status'] == DONE and (cost['seconds'] >= inv['budget'] or term['status'] != DONE):
        raise ValueError('late/fabricated success')
    marker = root/'late_publish_reached.json'
    if spec['control'] == 'late_publish' and term['status'] == DONE:
        witness = load(marker, required_hash(cost.get('late_publication_witness_sha256')), LIMIT)
        if witness['invocation'] != inv['invocation']:
            raise ValueError('publication witness identity')
        finite(witness['seconds'])
        if not term['seconds'] <= witness['seconds'] <= cost['seconds']:
            raise ValueError('publication witness chronology')
    elif cost.get('late_publication_witness_sha256') is not None or marker.exists():
        raise ValueError('unexpected publication witness')
    if term['status'] == DONE:
        if len(stages) != 3 or any(s['status'] != 'COMPLETED' for s in stages):
            raise ValueError('success without complete stages')
        accepted = receive(root, spec, inv_sha, anchors['produce'], anchors['check'], time.monotonic()+30)
        if accepted != term['accepted'] or load(root/'accepted.json') != accepted:
            raise ValueError('accepted record differs')
        if recheck:
            exact_check(load(root/'payload.json', anchors['produce'], LIMIT), spec, inv, time.monotonic()+30)
    elif term['accepted'] is not None:
        raise ValueError('failed prefix accepted')
    if observation is not None:
        result = observation['result']
        finite(observation['begin']); finite(observation['end']); finite(result['seconds'])
        for key, expected in [('invocation_sha256', inv_sha), ('terminal_sha256', sha(root/'terminal.json')),
                              ('cost_sha256', sha(root/'cost.json'))]:
            if result[key] != expected:
                raise ValueError('external API observation binding')
        if (observation['begin'] > inv['start'] or observation['end']-inv['start'] < result['seconds']
                or result['seconds'] < cost['seconds']):
            raise ValueError('API return cost ordering')
        expected = deadline_status(cost['status'], observation['end'], inv['deadline'])
        if result['status'] != expected or result['complete_moe_proof'] is not False:
            raise ValueError('unobserved/late API success')
    return {'status': 'PASS', 'execution_status': None if observation is None else observation['result']['status'],
            'budget_acceptance_observed': observation is not None and observation['result']['status'] == DONE,
            'bounds_rechecked': sum(spec['query_counts']) if recheck and term['status'] == DONE else 0,
            'complete_moe_proof': False}
