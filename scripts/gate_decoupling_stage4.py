"""Fixed stage4 queue and descriptive attribution; no new numerical engine.

The complete catalog, including missing sources and unexecuted suffixes, is the
denominator. The diagnostic population is outcome-selected, not a confirmation
sample. Wilson intervals implement the user's descriptive decision rule.
"""
import time
ENTRY = time.monotonic()
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import gate_decoupling_worker as worker
from scoped_proof.io import load, save, sha
from scoped_source.development_work import POLICY


def calls(objects):
    if len({o['id'] for o in objects}) != len(objects):
        raise ValueError('duplicate catalog object')
    # Catalog is1a model/index order, then original B2 order, then1b model/index.
    # Alternate order within pairs; never reorder after a variant outcome.
    return [dict(object_id=o['id'], variant=v) for n, o in enumerate(objects)
        for v in (('V2', 'V3') if n % 2 == 0 else ('V3', 'V2'))]


def wilson(k, n):
    if not 0 <= k <= n:
        raise ValueError('event denominator')
    if not n:
        return None
    z = 1.959963984540054; p = k/n; d = 1+z*z/n
    center = (p+z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0., center-half), min(1., center+half)]


def group_summary(objects, rows, *, terminal):
    by = {(r['object_id'], r['variant']): r for r in rows}
    if len(by) != len(rows):
        raise ValueError('duplicate diagnostic call')
    details = []
    for obj in objects:
        pair = {v: by.get((obj['id'], v)) for v in ('V2', 'V3')}
        statuses = {v: 'UNEXECUTED' if r is None else r['status'] for v, r in pair.items()}
        values = {v: (r or {}).get('evidence') or {} for v, r in pair.items()}
        closed = {v: bool(r and r['accepted'] and values[v].get('closed')
            and statuses[v] == ('POSITIVE' if v == 'V2' else 'DIAGNOSTIC_CLOSED')) for v, r in pair.items()}
        timed_out = statuses['V3'] == 'TIMEOUT'
        v2_observed_unclosed = statuses['V2'] in ('UNKNOWN', 'TIMEOUT')
        phase = ((pair['V3'] or {}).get('last_phase') or {}).get('phase')
        details.append(dict(object_id=obj['id'], group=obj['group'], statuses=statuses,
            v2_closed=closed['V2'], v3_closed=closed['V3'], v3_timeout=timed_out,
            both_closed=closed['V2'] and closed['V3'],
            v3_closed_v2_observed_unclosed=closed['V3'] and v2_observed_unclosed,
            v3_timeout_phase=phase if timed_out else None,
            domains_equal=(values['V2']['domain_sha256'] == values['V3']['domain_sha256']
                if 'domain_sha256' in values['V2'] and 'domain_sha256' in values['V3'] else None),
            history_source_kind=obj.get('history_source_kind'),
            known_original_witness=bool(obj.get('known_saved_witness') or obj.get('known_replayed_sources'))))
    n = len(objects)
    counts = {k: sum(d[k] for d in details) for k in
        ('v3_timeout', 'both_closed', 'v3_closed_v2_observed_unclosed')}
    intervals = {k: wilson(v, n) for k, v in counts.items()}
    if not terminal:
        decision = 'PENDING_REGISTERED_QUEUE'
    elif n and intervals['v3_timeout'][0] > .5:
        decision = 'STOP_ALGORITHM_LINE_USER_RULE_A'
    elif n and intervals['both_closed'][0] > .3:
        decision = 'ENTER_STAGE5_USER_RULE_B'
    elif n and counts['v3_closed_v2_observed_unclosed'] > n/2:
        decision = 'MATH_CONTRACT_FOR_USER_REVIEW_RULE_C'
    else:
        decision = 'GRAY_REPORT'
    return dict(objects=n, counts=counts, wilson95=intervals, decision=decision,
        status_counts={v: dict(Counter(d['statuses'][v] for d in details)) for v in ('V2', 'V3')},
        v3_timeout_phases=dict(Counter(d['v3_timeout_phase'] or 'NO_PHASE_RECORD' for d in details if d['v3_timeout'])),
        domains_not_comparable=sum(d['domains_equal'] is None for d in details),
        domains_different=sum(d['domains_equal'] is False for d in details), details=details)


def summarize(objects, rows, *, terminal):
    expected = {(o['id'], v) for o in objects for v in ('V2', 'V3')}
    worker.require(all((r['object_id'], r['variant']) in expected for r in rows), 'catalog membership')
    result = {name: group_summary([o for o in objects if group is None or o['group'] == group],
        [r for r in rows if group is None or next(o for o in objects if o['id'] == r['object_id'])['group'] == group],
        terminal=terminal) for name, group in (('all', None), ('singlelayer', 'singlelayer'), ('multilayer', 'multilayer'))}
    return dict(groups=result, primary_decision=result['all']['decision'],
        actual_local_calls=len(rows), real_complete_calls=0,
        elapsed_call_seconds=sum(r['seconds'] for r in rows),
        full_catalog_denominator=True, missing_not_reclassified_as_failure=True,
        scope='FAILURE_SELECTED_LOCAL_HISTORY_DIAGNOSTIC_NOT_COMPLETE_REQUEST_OR_CONFIRMATION',
        inference='descriptive Wilson on model-input objects; repeated inputs/fixed models are not independent population draws')


def freeze(path, catalog, control):
    from gate_decoupling_control import CODE
    inventory = load(catalog); passed = load(control)
    worker.require(inventory['stage1b_complete_included'], 'completed audited1b in full catalog')
    worker.require(passed['status'] == 'PASS', 'semantic and worker integration controls')
    control_manifest = Path(control).parent/'manifest.json'
    # The run's copied manifest is authoritative, not a mutable convenience path.
    checked = load(control_manifest, passed['manifest_sha256'])
    worker.require(all(sha(ROOT/p) == h for p, h in checked['source_files'].items()), 'controlled sources')
    sources = tuple(dict.fromkeys((*CODE, 'scripts/gate_decoupling_catalog.py',
        'scripts/gate_decoupling_stage4.py', 'tests/test_gate_decoupling_stage4.py')))
    save(path, dict(schema='gate-decoupling-stage4-v1', policy=POLICY,
        source_files={p: sha(ROOT/p) for p in sources},
        backend_identity=worker.digest(worker.backend.source_identity()), genbab_config=worker.backend.CONFIG,
        catalog=dict(path=str(Path(catalog).resolve()), sha256=sha(catalog)),
        control=dict(path=str(Path(control).resolve()), sha256=sha(control)),
        calls=calls(inventory['objects']), budgets=dict(local_calls=250, request=300, stage=21*3600,
            artifacts=128*2**20, as_bytes=8*2**30),
        ordering='inventory order; adjacent V2/V3 with alternating within-object order; retained unexecuted suffix',
        endpoint=dict(primary='all fixed hard-core model-input objects',
            strata=['singlelayer', 'multilayer'], timeout_wilson_lower_above=.5,
            joint_closed_wilson_lower_above=.3, v3_only_observed_majority_above=.5,
            precedence=['a', 'b', 'c', 'gray'], missing='keep full denominator and separate status',
            timeout='all V3 TIMEOUT terminals; phase strata forbid unique-ReLU causal inference'),
        no_result_dependent_changes=True))


def freeze_supplement(path, catalog, control, prior_manifest, prior_root, review):
    from gate_decoupling_control import CODE
    from gate_decoupling_followup import supplement_objects, frozen_t1
    original = load(prior_manifest)
    inventory = load(catalog); old = load(review)['final_review']
    worker.require(old['primary_decision'] == 'GRAY_REPORT' and old['completed_calls'] == 250,
        'preserved original gray and completed250')
    worker.require(old['manifest_sha256'] == sha(prior_manifest)
        and original['catalog']['sha256'] == sha(catalog), 'original frozen population')
    previous = {str(p.resolve()): sha(p) for p in sorted(prior_root.glob('ledger_*.json'))}
    worker.require(len(previous) == 250, 'all250 original terminals')
    rows = [load(p, h) for p, h in previous.items()]
    chosen = supplement_objects(inventory['objects'], rows)
    worker.require(len(chosen) == 97 and sum(o['history'] is None for o in chosen) == 44,
        'original68 unexecuted plus29 source-missing,44 missing history IDs overall')
    ids = {o['id'] for o in chosen}
    table = [c for c in original['calls'] if c['object_id'] in ids]
    worker.require(len(table) == 194, 'two variants in retained original order')
    passed = load(control)
    worker.require(passed['status'] == 'PASS', 'reconstruction and reception controls')
    checked = load(control.parent/'manifest.json', passed['manifest_sha256'])
    worker.require(all(sha(ROOT/p) == h for p, h in checked['source_files'].items()), 'controlled sources')
    sources = tuple(dict.fromkeys((*CODE, 'scripts/gate_decoupling_catalog.py',
        'scripts/gate_decoupling_followup.py', 'tests/test_gate_decoupling_followup.py')))
    save(path, dict(schema='gate-decoupling-stage4-supplement-v1', policy=POLICY,
        source_files={p: sha(ROOT/p) for p in sources},
        backend_identity=worker.digest(worker.backend.source_identity()), genbab_config=worker.backend.CONFIG,
        catalog=dict(path=str(catalog.resolve()), sha256=sha(catalog)),
        control=dict(path=str(control.resolve()), sha256=sha(control)), calls=table,
        reconstruct_missing_history=True, prior_ledgers=previous,
        prior_manifest=dict(path=str(prior_manifest.resolve()), sha256=sha(prior_manifest)),
        prior_review=dict(path=str(review.resolve()), sha256=sha(review)),
        original_decision='GRAY_REPORT', entry_to_stage5='USER_APPROVED_NOT_GATE_PASSED',
        cascade_t1=frozen_t1(old['positives']),
        budgets=dict(local_calls=200, request=300, stage=17*3600, artifacts=128*2**20, as_bytes=8*2**30),
        ordering='filter original call table for68unexecuted+29missing; no variant or object reorder',
        reconstruction='canonical first CROWN-unexcluded history, not recovery of missing old unproved ID',
        endpoint='full193 supplementary descriptive table; old gray and user decision remain unchanged'))


def execute_queue(folder, spec, manifest, deadline, *, invoke=worker.invoke):
    """Straight-line queue; tool errors retain their prefix for authorized repair."""
    prior = spec.get('prior_accounting', {})
    if prior:
        worker.require(type(prior.get('attempts')) is int and prior['attempts'] >= 0
            and type(prior.get('bytes')) is int and prior['bytes'] >= 0
            and type(prior.get('seconds')) in (int,float) and math.isfinite(prior['seconds'])
            and prior['seconds'] >= 0, 'valid cumulative prior costs before any launch')
    rows = []; reason = 'ALL_REGISTERED_CALLS_EXECUTED'
    for index, call in enumerate(spec['calls']):
        if index+prior.get('attempts', 0) >= spec['budgets']['local_calls']:
            reason = 'LOCAL_CALL_CAP'; break
        if time.monotonic()+300 > deadline:
            reason = 'STAGE_TIME_CAP'; break
        if worker.size(folder)+prior.get('bytes', 0)+256*1024 >= spec['budgets']['artifacts']:
            reason = 'ARTIFACT_CAP'; break
        save(folder/f'started_{index:03d}.json', dict(index=index, call=call, clock=time.monotonic()))
        row = invoke(folder/f'call_{index:03d}', manifest, index, stage_deadline=deadline)
        worker.require((row['object_id'], row['variant']) == (call['object_id'], call['variant']), 'terminal matches call')
        save(folder/f'ledger_{index:03d}.json', row); rows.append(row)
        print(dict(index=index, object=call['object_id'], variant=call['variant'],
                   status=row['status'], seconds=row['seconds']), flush=True)
        if row['status'] == 'ERROR':
            reason = 'TOOL_OR_RECEPTION_ERROR'; break
    return rows, reason


def audit(folder, manifest, *, deadline=None, summarizer=None):
    from b1_observed_batch import session_ok
    deadline = time.monotonic()+300 if deadline is None else deadline
    spec = load(manifest); start = load(folder/'start.json')
    worker.require(start['manifest_sha256'] == sha(manifest), 'original frozen protocol')
    worker.require(all(sha(ROOT/p) == h for p, h in spec['source_files'].items()), 'same audited verifier source version')
    inventory = load(spec['catalog']['path'], spec['catalog']['sha256'])
    rows = []
    for index, path in enumerate(sorted(folder.glob('ledger_*.json'))):
        worker.tick(deadline)
        worker.require(path.name == f'ledger_{index:03d}.json', 'contiguous retained prefix')
        row = load(path); call = spec['calls'][index]; child = folder/f'call_{index:03d}'
        terminal = load(child/'terminal.json', row['terminal_sha256'])
        worker.require(all(row[k] == v for k, v in terminal.items()), 'unaltered terminal/ledger')
        worker.require((row['object_id'], row['variant']) == (call['object_id'], call['variant']), 'same call')
        worker.require(0 <= row['seconds'] <= 300, 'call cost')
        for name, h in row['files'].items():
            worker.require(sha(child/name) == h, 'raw receipt identity')
        if row['accepted']:
            worker.require(session_ok(load(child/'session.json')), 'owned on-time completed session')
            request = load(child/'request.json'); publication = load(child/'published.json')
            outcome = load(child/'outcome.json', publication['sha256'])
            worker.require(outcome['value'] == row['evidence']
                and outcome['request_sha256'] == sha(child/'request.json'), 'accepted raw outcome')
            worker.require(outcome['clock'] < request['worker_deadline']
                and publication['clock'] < request['worker_deadline'], 'original on-time publication')
            # This recheck does NOT extend the old execution deadline or grant success.
            worker.check_received(child, row['evidence'], spec, call, deadline)
        rows.append(row)
    end_path = folder/'complete.json'
    end = load(end_path) if end_path.exists() else None
    terminal = bool(end and end['stop_reason'] in ('ALL_REGISTERED_CALLS_EXECUTED', 'LOCAL_CALL_CAP', 'STAGE_TIME_CAP', 'ARTIFACT_CAP'))
    result = (summarizer or summarize)(inventory['objects'], rows, terminal=terminal)
    result.update(status='AUDITED_DIAGNOSTIC_TERMINALS' if end else 'AUDITED_PREFIX',
        stop_reason=None if end is None else end['stop_reason'],
        unexecuted=spec['calls'][len(rows):], bytes=worker.size(folder),
        mathematical_bound_reproof=False)
    starts = sorted(folder.glob('started_*.json'))
    for index, path in enumerate(starts):
        event = load(path)
        worker.require(path.name == f'started_{index:03d}.json' and event['index'] == index
            and event['call'] == spec['calls'][index], 'ordered attempts including failed starts')
    result.update(attempted_local_calls=len(starts), started_without_terminal=len(starts)-len(rows))
    result['unexecuted'] = spec['calls'][len(starts):]
    result['started_without_terminal_calls'] = spec['calls'][len(rows):len(starts)]
    for group in result['groups'].values():
        for detail in group['details']:
            for call in result['started_without_terminal_calls']:
                if detail['object_id'] == call['object_id']:
                    variant = call['variant']; detail['statuses'][variant] = 'STARTED_NO_TERMINAL'
                    group['status_counts'][variant]['UNEXECUTED'] -= 1
                    counts = group['status_counts'][variant]
                    counts['STARTED_NO_TERMINAL'] = counts.get('STARTED_NO_TERMINAL', 0)+1
    worker.require(len(rows) <= len(starts) <= min(spec['budgets'].get('local_calls', 250), len(rows)+1)
        and result['bytes'] <= spec['budgets']['artifacts'], 'stage quota')
    if end:
        worker.require(end['calls'] == len(rows) and end['seconds'] <= spec['budgets']['stage'], 'terminal accounting')
    full_path, stopped_path = folder/'final_accounting.json', folder/'stopped.json'
    full = load(full_path) if full_path.exists() else None
    stopped = load(stopped_path) if stopped_path.exists() else None
    if full:
        worker.require(full['audit_sha256'] == sha(folder/'audit.json')
            and full['seconds'] <= spec['budgets']['stage'], 'full stage accounting/audit binding')
    result['recorded_stage_seconds'] = (full or end or stopped or {}).get('seconds')
    result['recorded_cost_scope'] = ('THROUGH_FINAL_AUDIT_ACCOUNTING' if full else
        'THROUGH_BATCH_STOP_NOT_FINAL_AUDIT' if end or stopped else 'LIVE_PREFIX_ONLY')
    result['sum_terminal_cost_excludes_unfinished_call'] = len(starts) != len(rows)
    prior = spec.get('prior_accounting', {})
    if prior:
        worker.require(all(type(prior.get(k)) in (int, float) and prior[k] >= 0
            for k in ('attempts', 'seconds', 'bytes')), 'nonnegative retained prior cost')
        result['physical_attempts_including_prior'] = len(starts)+prior['attempts']
        result['cumulative_stage_seconds'] = (result['recorded_stage_seconds'] or 0)+prior['seconds']
        result['cumulative_artifact_bytes'] = result['bytes']+prior['bytes']
        worker.require(result['physical_attempts_including_prior'] <= spec['budgets']['local_calls']
            and result['cumulative_stage_seconds'] <= spec['budgets']['stage']
            and result['cumulative_artifact_bytes'] <= spec['budgets']['artifacts'],
            'continuation cannot reset any stage quota')
    if spec.get('schema') == 'gate-decoupling-stage4-supplement-v1':
        # Explicit new view; never alter or relabel a historical terminal.
        old = [load(p, h) for p, h in spec['prior_ledgers'].items()]
        merged = {(r['object_id'], r['variant']): r for r in old}
        for row in rows:
            merged[row['object_id'], row['variant']] = row
        result['supplementary_full193'] = summarize(inventory['objects'], list(merged.values()), terminal=terminal)
        result['original_decision'] = spec['original_decision']
        result['entry_to_stage5'] = spec['entry_to_stage5']
        result['new_history_provenance'] = {o['id']: 'RECONSTRUCTED_NOT_OLD_UNPROVED_ID'
            for o in inventory['objects'] if o['history'] is None and any(c['object_id'] == o['id'] for c in spec['calls'])}
    return result


def batch(folder, manifest, *, summarizer=None, entry=None):
    entry = ENTRY if entry is None else entry
    spec = load(manifest)
    worker.require(spec['schema'] in worker.SCHEMAS and spec['policy'] == POLICY, 'frozen stage4')
    worker.require(ROOT != Path('/data1/Kane/MOE/ACT') and (ROOT/'.frozen_commit').is_file(), 'immutable execution copy')
    worker.require(all(sha(ROOT/p) == h for p, h in spec['source_files'].items()), 'source identity')
    supplementary = spec['schema'] in worker.SCHEMAS[1:]
    worker.require(spec['budgets'] == dict(local_calls=200 if supplementary else 250, request=300, stage=(17 if supplementary else 21)*3600,
        artifacts=128*2**20, as_bytes=8*2**30), 'registered unchanged quotas')
    resource.setrlimit(resource.RLIMIT_AS, (8*2**30, 8*2**30)); os.nice(10)
    prior = spec.get('prior_accounting', {})
    folder.mkdir(); deadline = entry+spec['budgets']['stage']-prior.get('seconds', 0)
    save(folder/'start.json', dict(entry=entry, deadline=deadline, pid=os.getpid(),
        commit=(ROOT/'.frozen_commit').read_text().strip(), manifest_sha256=sha(manifest),
        execution_root=str(ROOT), started_at=time.time()))
    try:
        rows, reason = execute_queue(folder, spec, manifest, deadline)
        save(folder/'complete.json', dict(calls=len(rows), stop_reason=reason,
            seconds=time.monotonic()-entry, clock=time.monotonic()))
        result = audit(folder, manifest, deadline=deadline-2, summarizer=summarizer); save(folder/'audit.json', result)
        worker.require(time.monotonic() < deadline
            and worker.size(folder)+prior.get('bytes', 0)+65536 < worker.CAP, 'audit cost within stage')
        save(folder/'final_accounting.json', dict(seconds=time.monotonic()-entry,
            bytes_before_receipt=worker.size(folder), audit_sha256=sha(folder/'audit.json'), clock=time.monotonic()))
    except BaseException as error:
        save(folder/'stopped.json', dict(kind=type(error).__name__, message=str(error),
            started_calls=len(list(folder.glob('started_*.json'))),
            seconds=time.monotonic()-entry, clock=time.monotonic()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('freeze', 'freeze-supplement', 'batch', 'audit'))
    parser.add_argument('folder', type=Path); parser.add_argument('--manifest', type=Path)
    parser.add_argument('--catalog', type=Path); parser.add_argument('--control', type=Path)
    parser.add_argument('--prior-manifest', type=Path); parser.add_argument('--prior-root', type=Path)
    parser.add_argument('--review', type=Path); a = parser.parse_args()
    if a.mode == 'freeze': freeze(a.folder, a.catalog, a.control)
    elif a.mode == 'freeze-supplement': freeze_supplement(a.folder, a.catalog, a.control, a.prior_manifest, a.prior_root, a.review)
    elif a.mode == 'batch': batch(a.folder, a.manifest)
    else: print(audit(a.folder, a.manifest))
