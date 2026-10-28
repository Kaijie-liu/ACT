"""One revised M-B study, using the existing owned worker, queue and audit."""
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
import gate_decoupling_stage4 as Q
import moe_aware_diagnostics as D
from scoped_proof.io import load, save, sha, tick
from scoped_source.development_work import POLICY

VARIANTS = ('ALPHA_CONTINUE', 'GUARD_REFINE')


def aggregate(row, source_id, history, variant, properties, guards, deadline):
    use_guards = variant == 'GUARD_REFINE'
    W.require(variant in VARIANTS and row['engine_adapter'] ==
        'baseline_preserving_CROWN_guard_refinement_v1', 'registered refinement variant')
    result = D.aggregate_mb(row, source_id, history, 'MB_ON' if use_guards else 'MB_OFF',
                            properties, guards, deadline)
    W.require(row['baseline_limit'] == 20 and row['refinement_update_limit'] == 40
        and row['beta_start'] == 0. and row['beta_probe'] == (1e-4 if use_guards and guards else None),
        'frozen prefix, refinement and zero-start probe')
    base, extra = [], []
    counts = Counter()
    for entry in row['trace']:
        tick(deadline)
        W.require(entry['phase'] in ('baseline', 'refinement', 'probe'), 'known phase')
        values = D.vector(entry['lower'], properties)
        counts[entry['phase']] += 1
        (base if entry['phase'] == 'baseline' else extra).append(values)
    W.require(1 <= len(base) <= 20 and counts['refinement'] == 41
        and counts['probe'] == int(use_guards and guards > 0), 'all registered candidate phases')
    W.require(row['evaluations'] == {k: counts[k] for k in ('baseline', 'refinement', 'probe')}
        and row['candidate_evaluations'] == len(row['trace']), 'all candidate work retained')
    base_max = [max(r[j] for r in base) for j in range(properties)]
    extra_max = [max(r[j] for r in extra) for j in range(properties)]
    W.require(row['baseline_lower'] == row['zero_lower'] == base_max
        and row['refinement_lower'] == extra_max
        and row['lower'] == [max(a, b) for a, b in zip(base_max, extra_max)]
        and row['improvement'] == [a-b for a, b in zip(row['lower'], base_max)],
        'independently recomputed complete pointwise baseline retention')
    star = max(range(len(base)), key=lambda i: sum(base[i]))
    W.require(row['alpha_star_iteration'] == star
        and len(row['alpha_star_sha256']) == 64, 'evaluated alpha-star selection binding')
    return dict(**result, baseline_closed=all(v > 0 for v in base_max),
        baseline_score=min(base_max), refinement_score=min(extra_max),
        increment=min(row['lower'])-min(base_max),
        mechanism_attribution='extra-alpha continuation is a separate paid control')


def diagnose(obj, variant, spec, deadline, journal):
    model, tensors, domain, identity = D.prepare_source(obj, variant, spec, deadline, journal)
    W.require(domain is not None, 'same original eleven histories required')
    from act.back_end.moe.moe_aware_bab import history_objectives
    from act.back_end.moe.guard_dual_refinement import refined_conditional_bounds
    graph = journal.call('guard_graph', lambda: history_objectives(model, domain.history,
        tensors['rows'], tensors['thresholds'], source_id=domain.source_id))
    bounds = journal.call('baseline_and_refinement', lambda: refined_conditional_bounds(
        graph, tensors['center'], tensors['lower'], tensors['upper'], deadline=deadline,
        use_guards=variant == 'GUARD_REFINE',
        report=lambda name, data: journal.put(name+'_prefix.json', data)))
    journal.put('refined_bounds.json', bounds)
    accepted = aggregate(bounds, domain.source_id, identity['history'], variant,
                         len(tensors['rows']), graph.guard_count, deadline)
    return dict(**identity, **accepted, group=variant, bounds=bounds)


def check_received(folder, value, spec, call, deadline):
    tick(deadline)
    obj = W.source_object(spec, call)
    sid = W.digest(dict(object=obj, sources=spec['source_files'], backend=spec['backend_identity']))
    W.require(obj['history'] is not None and value['source_id'] == sid
        and value['group'] == call['variant'] and value['history'] == obj['history'],
        'same original model, materialized box, properties and history')
    saved = load(folder/'domain.json'); G = D.gate_module()
    domain = G.GateDomain(sid, tuple(map(tuple, saved['history'])),
        tuple(map(tuple, saved['margins'])), tuple(map(tuple, saved['intervals'])))
    G.validate_domain(domain, tuple(map(tuple, obj['history'])), expected_source_id=sid)
    n = obj['original_request']['model']['architecture']['classes']-1
    W.require(value['domain_sha256'] == saved['domain_sha256'] == G.domain_digest(domain)
        and value['property_count'] == n, 'complete bound domain and property set')
    row = load(folder/'refined_bounds.json')
    W.require(row == value['bounds'], 'saved complete refinement trace')
    result = aggregate(row, sid, obj['history'], call['variant'], n,
                       sum(2*(e-2) for e in obj['experts']), deadline)
    W.require(all(value[k] == v for k, v in result.items()), 'same received aggregation')


def freeze(path, control):
    original = ROOT/'docs/moe_aware_stage5a_freeze_20261027_r4.json'
    old, checked = load(original), load(control)
    W.require(checked['status'] == 'PASS', 'ordinary and author-engine controls')
    W.require(all(sha(ROOT/p) == h for p, h in checked['source_files'].items()), 'tested source')
    objects = {o['id']: o for o in load(old['catalog']['path'], old['catalog']['sha256'])['objects']}
    # Three original catalogue entries lacked IDs; the completed old M-B calls
    # recorded their reconstructed histories. Bind BOTH old arms, never select
    # a fresh/easier history or inherit either arm's numerical lower bound.
    old_root = Path('/data1/Kane/MOE/results/moe_aware_stage5a_20261026_r1')
    chosen = []
    for i, oid in enumerate(old['mb_objects']):
        obj = dict(objects[oid]); histories = []; bindings = []
        for j in (2*i, 2*i+1):
            child = old_root/f'call_{j:03d}'
            row = load(old_root/f'ledger_{j:03d}.json')
            terminal = load(child/'terminal.json', row['terminal_sha256'])
            W.require(row['object_id'] == oid and row['accepted']
                and all(row[k] == v for k, v in terminal.items()), 'completed original M-B object')
            saved = load(child/'source.json', row['files']['source.json'])
            W.require(saved['object'] == objects[oid], 'original model/input/property identity')
            domain = load(child/'domain.json', row['files']['domain.json'])
            W.require(domain['history'] == row['evidence']['history'], 'old executed history')
            histories.append(domain['history'])
            bindings.append(dict(path=str(child/'terminal.json'), sha256=row['terminal_sha256']))
        W.require(histories[0] == histories[1] and
            (obj['history'] is None or obj['history'] == histories[0]), 'same old paired history')
        obj['history'] = histories[0]
        if objects[oid]['history'] is None:
            obj['history_source_kind'] = 'BOUND_PREVIOUS_MB_RECONSTRUCTED_HISTORY'
        obj['original_sources'] = [*obj['original_sources'], *bindings]
        chosen.append(obj)
    W.require(len(chosen) == 11, 'unchanged eleven original objects')
    catalog_path = path.with_name(path.stem+'_sources.json')
    save(catalog_path, dict(objects=chosen, original_catalog=old['catalog'],
        policy='same executed old M-B histories, ranges independently recomputed'))
    table = [dict(object_id=oid, variant=v) for i, oid in enumerate(old['mb_objects'])
        for v in (VARIANTS if i % 2 == 0 else VARIANTS[::-1])]
    sources = dict.fromkeys((*old['source_files'], *checked['source_files']))
    save(path, dict(schema=W.SCHEMAS[3], policy=POLICY,
        source_files={p: sha(ROOT/p) for p in sources},
        backend_identity=W.digest(W.backend.source_identity()), genbab_config=W.backend.CONFIG,
        catalog=dict(path=str(catalog_path.resolve()), sha256=sha(catalog_path)),
        original_population=dict(path=str(original), sha256=sha(original)),
        control=dict(path=str(control.resolve()), sha256=sha(control)), calls=table,
        mb_objects=old['mb_objects'], budgets=dict(local_calls=30, request=300, stage=3*3600,
            artifacts=128*2**20, as_bytes=8*2**30),
        optimization=dict(baseline_evaluations=20, refinement_updates=40,
            beta_start=0., beta_probe=1e-4, alpha_lr=.5, beta_lr=.1,
            alpha_star='earliest prefix state maximizing sum of property bounds',
            retained='pointwise maximum over ALL baseline/refinement/probe candidates'),
        decision=dict(gained_histories_min=5, median_increment_strict_min=0,
            primary='GUARD_REFINE vs its same-call full20-step baseline',
            attribution='GUARD_REFINE vs paid equal-extra-alpha ALPHA_CONTINUE; no extra admission gate',
            missing='keep eleven-object denominator; no online upgrade from a saved prefix'),
        prior_accounting=dict(attempts=0, seconds=checked['seconds'], bytes=checked['bytes']),
        unchanged_b4='continues immutable R3; no changed T1, source, engine or old verdict'))


def summarize_for(spec):
    def summarize(objects, rows, *, terminal):
        indexed = {(r['object_id'], r['variant']): r for r in rows}
        details, gains, deltas = [], [], []
        for oid in spec['mb_objects']:
            pair = {v: indexed.get((oid, v)) for v in VARIANTS}
            values = {v: r['evidence'] if r and r['accepted'] else None for v, r in pair.items()}
            on, alpha = values['GUARD_REFINE'], values['ALPHA_CONTINUE']
            if on is not None:
                deltas.append(on['increment'])
                if on['closed'] and not on['baseline_closed']:
                    gains.append(oid)
            details.append(dict(object_id=oid,
                states={v: r['status'] if r else 'UNEXECUTED' for v, r in pair.items()},
                baseline=None if on is None else on['baseline_score'],
                refined=None if on is None else on['score'],
                increment=None if on is None else on['increment'],
                beta_nonzero=None if on is None else on['bounds']['beta_nonzero'],
                beta_count=None if on is None else on['bounds']['beta_count'],
                guard_minus_equal_alpha=None if on is None or alpha is None else on['score']-alpha['score'],
                baseline_max_drift=None if on is None or alpha is None else max(abs(a-b) for a,b in
                    zip(on['bounds']['baseline_lower'], alpha['bounds']['baseline_lower'])),
                costs={v: None if r is None else r['seconds'] for v,r in pair.items()}))
        median = statistics.median(deltas) if deltas else None
        complete = terminal and len(rows) == len(spec['calls'])
        passed = len(gains) >= 5 and median is not None and median > 0
        return dict(groups={}, population=11, details=details, gained=gains,
            complete_score_objects=len(deltas), median_increment=median,
            primary_decision=('PASS' if passed else 'STOP_MB_AND_MC') if complete else 'PENDING',
            completed_calls=len(rows), status_counts=dict(Counter(r['status'] for r in rows)),
            seconds=sum(r['seconds'] for r in rows), formal_SAFE=False, complete_MoE=False,
            guarantee='NUMERICAL_LOCAL_HISTORY_DIAGNOSTIC_NOT_COMPLETE_REQUEST_ADVANTAGE')
    return summarize


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('freeze', 'batch', 'audit'))
    parser.add_argument('folder', type=Path); parser.add_argument('--manifest', type=Path)
    parser.add_argument('--control', type=Path); args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.folder, args.control)
    else:
        spec = load(args.manifest)
        W.require(spec['schema'] == W.SCHEMAS[3], 'new refinement stage only')
        summarizer = summarize_for(spec)
        if args.mode == 'batch':
            Q.batch(args.folder, args.manifest, summarizer=summarizer, entry=ENTRY)
        else:
            print(Q.audit(args.folder, args.manifest, summarizer=summarizer))
