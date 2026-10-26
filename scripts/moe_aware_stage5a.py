"""One stage5a freeze and analysis, reusing the existing queue and supervisor."""
import time
ENTRY = time.monotonic()
import argparse
from collections import Counter
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import gate_decoupling_worker as W
import gate_decoupling_stage4 as queue
import moe_aware_diagnostics as D
from scoped_proof.io import load, save, sha, tick
from scoped_source.development_work import POLICY


def call_table(records):
    rows = []
    for i, obj in enumerate(records['mb_eligible']):
        variants = ('MB_OFF', 'MB_ON') if i % 2 == 0 else ('MB_ON', 'MB_OFF')
        rows.extend(dict(object_id=obj['object_id'], variant=v) for v in variants)
    for i, oid in enumerate(records['mc_original_v3_only']):
        variants = ('MC_BASE', 'MC_2', 'MC_4')
        variants = variants[i % 3:]+variants[:i % 3]
        rows.extend(dict(object_id=oid, variant=v) for v in variants)
    return rows


def freeze(path, records_path, catalog, control):
    from gate_decoupling_control import CODE
    records, inventory, checked = load(records_path), load(catalog), load(control)
    W.require(checked['status'] == 'PASS' and checked['trained_calls'] == 0, 'ordinary integration tests passed')
    W.require(all(sha(ROOT/p) == h for p, h in checked['source_files'].items()), 'tested source identities')
    objects = {o['id']: o for o in inventory['objects']}
    W.require(len(records['mb_eligible']) == 11 and len(records['mc_original_v3_only']) == 27,
              'existing fixed mechanism populations')
    for obj in records['mb_eligible']:
        evidence = obj['census']
        census = load(evidence['path'], evidence['sha256'])
        W.require(census['request_sha256'] == evidence['request_sha256']
            and census['result']['observed_histories'] == obj['observed_histories'], 'bound original census')
        W.require(len(obj['observed_histories']) >= 2 and obj['object_id'] in objects, 'two sampled-native histories')
    for oid in records['mc_original_v3_only']:
        W.require(oid in objects and objects[oid]['source_kind'] == 'multilayer', 'registered multilayer identity')
    table = call_table(records)
    W.require(len(table) == 103 and len({(r['object_id'], r['variant']) for r in table}) == 103,
              '22 M-B plus81 M-C, no duplicate/reselected calls')
    sources = tuple(dict.fromkeys((*CODE, 'scripts/gate_decoupling_followup.py',
        'act/back_end/moe/moe_aware_bab.py', 'scripts/moe_aware_diagnostics.py',
        'scripts/moe_aware_stage5a.py', 'tests/test_moe_aware_bab.py', 'tests/test_moe_aware_diagnostics.py')))
    save(path, dict(schema=W.SCHEMAS[2], policy=POLICY,
        source_files={p: sha(ROOT/p) for p in sources},
        backend_identity=W.digest(W.backend.source_identity()), genbab_config=W.backend.CONFIG,
        catalog=dict(path=str(catalog.resolve()), sha256=sha(catalog)),
        population_record=dict(path=str(records_path.resolve()), sha256=sha(records_path)),
        control=dict(path=str(control.resolve()), sha256=sha(control)), calls=table,
        budgets=dict(local_calls=200, request=300, stage=17*3600, artifacts=128*2**20, as_bytes=8*2**30),
        mb_objects=[r['object_id'] for r in records['mb_eligible']], mc_objects=records['mc_original_v3_only'],
        source_policy='each call recomputes its own ranges; canonical new history where old ID missing',
        ordering='M-B catalog alternating arms, then M-C original27 rotating base/2/4; no outcome reorder',
        mb=dict(candidate_evaluations=20, alpha_lr=.5, beta_lr=.1, beta_start=.1,
                score='minimum of all guard-conditional property CROWN lower bounds'),
        mc=dict(initial_alpha_iterations=20, initial_all_cells_before_bab=True,
                score='minimum all-cell/all-endpoint guarded-spec INITIAL bound, not property lower bound',
                residual_order='cell then LO/HI', global_resolution_choice='2 if passes, else4 if passes, otherwise exclude'),
        decision=dict(additional_closed_min=5, paired_median_bound_improvement_strict_min=0,
            missing_scores='missing, never0 or infinity; report full population and complete-score pair count',
            gain='online positive vs contemporaneous executed nonpositive, same history/domain/properties',
            timeout_prefix='may contribute complete initial-bound diagnostic only; never online positive',
            stop='exclude a mechanism if neither fixed variant passes; no new samples or parameter tuning'),
        guarantee='NUMERICAL_POLICY_LOCAL_HISTORY_ONLY; no full-request advantage or G3 proof'))


def comparison(objects, indexed, off, on):
    details, deltas, gained, lost = [], [], [], []
    for oid in objects:
        a, b = indexed.get((oid, off)), indexed.get((oid, on))
        same = bool(a and b and a.get('identity') is not None and a['identity'] == b.get('identity'))
        positive_a = bool(a and a['status'] == 'POSITIVE' and a['accepted'])
        positive_b = bool(b and b['status'] == 'POSITIVE' and b['accepted'])
        measured = same and a.get('score') is not None and b.get('score') is not None
        delta = b['score']-a['score'] if measured else None
        if delta is not None: deltas.append(delta)
        if same and positive_b and not positive_a: gained.append(oid)
        if same and positive_a and not positive_b: lost.append(oid)
        details.append(dict(object_id=oid, same_identity=same,
            off='UNEXECUTED' if a is None else a['status'], on='UNEXECUTED' if b is None else b['status'],
            off_score=None if a is None else a.get('score'), on_score=None if b is None else b.get('score'),
            on_beta_nonzero=None if b is None else b.get('beta_nonzero'),
            on_beta_count=None if b is None else b.get('beta_count'),
            delta=delta))
    median = statistics.median(deltas) if deltas else None
    return dict(population=len(objects), gained=gained, lost=lost, complete_score_pairs=len(deltas),
        paired_median_bound_improvement=median,
        passes=len(gained) >= 5 and median is not None and median > 0, details=details)


def observed_row(folder, index, row, spec):
    """Time-bounded prefixes are descriptive only, not accepted successes."""
    child = folder/f'call_{index:03d}'
    call = spec['calls'][index]
    result = dict(status=row['status'], accepted=row['accepted'], score=None, identity=None)
    # The queue auditor first checks every original terminal-bound file hash.
    if 'domain.json' not in row['files']:
        return result
    saved = load(child/'domain.json', row['files']['domain.json'])
    obj = W.source_object(spec, call); G = D.gate_module()
    sid = W.digest(dict(object=obj, sources=spec['source_files'], backend=spec['backend_identity']))
    history = obj['history']
    if history is None:
        from gate_decoupling_followup import check_selection
        W.require('history_selection.json' in row['files'], 'bound reconstructed diagnostic prefix')
        selection = load(child/'history_selection.json', row['files']['history_selection.json'])
        history = check_selection(selection, obj['experts'], sid, time.monotonic()+10)
    domain = G.GateDomain(saved['source_id'], tuple(map(tuple, saved['history'])),
        tuple(map(tuple, saved['margins'])), tuple(map(tuple, saved['intervals'])))
    G.validate_domain(domain, history, expected_source_id=sid)
    W.require(G.domain_digest(domain) == saved['domain_sha256'], 'bound initial domain')
    result['identity'] = saved['domain_sha256']
    if row['accepted']:
        result['score'] = row['evidence'].get('score')
        if call['variant'].startswith('MB_'):
            result.update({k: row['evidence']['bounds'][k] for k in ('beta_nonzero', 'beta_count')})
    elif call['variant'].startswith('MC_') and 'initial_aggregate.json' in row['files']:
        initial = load(child/'initial_aggregate.json', row['files']['initial_aggregate.json'])
        pieces = []
        for i in range(len(D.cells_for(domain, call['variant']))):
            for v in ('V2_LO', 'V2_HI'):
                name = f'cell_{i}_{v}_initial.json'
                W.require(name in row['files'], 'all initial rows bound before timeout')
                pieces.append(load(child/name, row['files'][name]))
        recomputed = D.aggregate_mc(domain, call['variant'], pieces,
            obj['original_request']['model']['architecture']['classes']-1, time.monotonic()+10)
        W.require(initial == recomputed, 'complete prefix bound aggregate')
        result['score'] = initial['score']
        result['prefix_only_not_online_acceptance'] = True
    return result


def retained_calls(spec, deadline):
    """Recheck completed arm families across retained tool-error prefixes.

    A compatibility repair cannot mix engine identities within an object's
    comparison. Completed MB pairs / MC triples remain on their original
    engine; a failed new object and all its remaining arms use the repaired
    engine. Errors and probes remain in cumulative physical costs.
    """
    if 'continuation' not in spec:
        return {}
    from b1_observed_batch import session_ok
    parent = spec['continuation']; root = Path(parent['root'])
    original = load(parent['manifest'], parent['manifest_sha256'])
    start = load(root/'start.json', parent['start_sha256'])
    final = load(root/'final_accounting.json', parent['final_sha256'])
    audit = load(root/'audit.json', parent['audit_sha256'])
    indexed = retained_calls(original, deadline)
    kept = parent.get('completed_prefix', 22)
    W.require(isinstance(kept, int) and 0 <= kept < len(original['calls']),
        'completed parent prefix within frozen calls')
    W.require(final['audit_sha256'] == parent['audit_sha256']
        and start['manifest_sha256'] == parent['manifest_sha256']
        and audit['completed_calls'] == kept+1+len(indexed)
        and audit['stop_reason'] == 'TOOL_OR_RECEPTION_ERROR', 'bound stopped parent including error')
    W.require(all(sha(Path(start['execution_root'])/p) == h for p,h in original['source_files'].items()),
        'unchanged parent verifier sources')
    W.require(spec['calls'] == original['calls'][kept:]
        and all(spec[k] == original[k] for k in ('mb_objects','mc_objects','mb','mc','decision','budgets','catalog')),
        'same fixed remaining calls and endpoints; no scientific change')
    family = lambda c: (c['object_id'], 'MB' if c['variant'].startswith('MB_') else 'MC')
    W.require(not ({family(c) for c in original['calls'][:kept]}
                   & {family(c) for c in spec['calls']}),
        'retain whole same-engine arm families, never split a paired comparison')
    W.require(len(parent['ledger_sha256']) == kept+1, 'all old attempts, including error')
    for i in range(kept+1):
        tick(deadline);row = load(root/f'ledger_{i:03d}.json', parent['ledger_sha256'][f'ledger_{i:03d}.json'])
        child = root/f'call_{i:03d}';terminal = load(child/'terminal.json', row['terminal_sha256'])
        W.require(all(row[k] == v for k,v in terminal.items()), 'parent original terminal')
        W.require(all(sha(child/p) == h for p,h in row['files'].items()), 'parent original files')
        if i == kept:
            W.require(row['status'] == 'ERROR' and not row['accepted'], 'old error never upgraded')
            continue
        call = original['calls'][i]
        W.require((row['object_id'],row['variant']) == (call['object_id'],call['variant'])
            and row['status'] not in ('ERROR','UNSUPPORTED'), 'original completed call, no error upgrade')
        if row['accepted']:
            W.require(session_ok(load(child/'session.json')), 'accepted parent session')
            W.check_received(child, row['evidence'], original, call, deadline)
        else:
            W.require(row['status']=='TIMEOUT', 'unaccepted parent remains timeout')
        key=(row['object_id'],row['variant'])
        W.require(key not in indexed, 'no duplicate completed parent calls')
        indexed[key] = observed_row(root,i,row,original)
    prior=original.get('prior_accounting', {})
    W.require(spec['prior_accounting']['attempts'] >= kept+1+prior.get('attempts',0)
        and spec['prior_accounting']['seconds'] >= final['seconds']+prior.get('seconds',0)
        and spec['prior_accounting']['bytes'] >= final['bytes_before_receipt']+prior.get('bytes',0),
        'all prior costs retained, including previous continuation')
    return indexed


def summarizer(folder, spec):
    def summarize(_inventory, rows, *, terminal):
        prior = retained_calls(spec, time.monotonic()+60)
        indexed = dict(prior)
        for i,r in enumerate(rows):
            key = (r['object_id'], r['variant'])
            W.require(key not in indexed, 'completed parent calls are not rerun')
            indexed[key] = observed_row(folder,i,r,spec)
        mb = comparison(spec['mb_objects'], indexed, 'MB_OFF', 'MB_ON')
        mc2 = comparison(spec['mc_objects'], indexed, 'MC_BASE', 'MC_2')
        mc4 = comparison(spec['mc_objects'], indexed, 'MC_BASE', 'MC_4')
        complete = terminal and len(rows) == len(spec['calls'])
        return dict(groups={}, completed_calls=len(rows)+len(prior), planned_calls=len(spec['calls'])+len(prior),
            status_counts=dict(Counter(key[1]+':'+r['status'] for key,r in indexed.items())),
            seconds=sum(r['seconds'] for r in rows), mb=mb, mc2=mc2, mc4=mc4,
            result_view='unique fixed103; old errors and fault probes remain in physical-attempt/cost ledger',
            parent_M_B_count=sum(key[1].startswith('MB_') for key in prior),
            retained_completed_calls=len(prior),
            admitted_mechanisms=([*(['M-B'] if mb['passes'] else []),
                *(['M-C-2'] if mc2['passes'] else ['M-C-4'] if mc4['passes'] else [])] if complete else None),
            primary_decision='COMPLETE_FIXED_MECHANISM_GATE' if complete else 'PENDING',
            complete_MoE=False, formal_SAFE=False)
    return summarize


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('freeze', 'batch', 'audit'))
    parser.add_argument('folder', type=Path); parser.add_argument('--manifest', type=Path)
    parser.add_argument('--records', type=Path); parser.add_argument('--catalog', type=Path)
    parser.add_argument('--control', type=Path); a = parser.parse_args()
    if a.mode == 'freeze':
        freeze(a.folder, a.records, a.catalog, a.control)
    else:
        spec = load(a.manifest); summarize = summarizer(a.folder, spec)
        W.require(spec['schema'] == W.SCHEMAS[2], 'registered stage5a')
        if a.mode == 'batch': queue.batch(a.folder, a.manifest, summarizer=summarize, entry=ENTRY)
        else: print(queue.audit(a.folder, a.manifest, summarizer=summarize))
