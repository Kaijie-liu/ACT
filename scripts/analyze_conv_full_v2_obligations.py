"""Stdlib-only, hash-bound saved-log analysis. No solver imports or queries."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/moe/results/conv_three_arm_full_20260915_v2'
PARENT = ROOT / 'act/pipeline/moe/results/conv_full_v2_review_20260915.json'
OUTPUT = ROOT / 'act/pipeline/moe/results/conv_full_v2_obligations_20260915.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def scope_keys(scope, pairs, label, classes=10):
    """A union query covers several obligations; it is not several solves."""
    row = scope['row']
    competitors = [i for i, v in enumerate(row) if v == -1]
    if len(competitors) != 1 or len(row) != classes:
        raise ValueError('malformed property row')
    competitor = competitors[0]
    if (competitor == label or scope['constant'] != 0
            or row != [1. if i == label else -1. if i == competitor else 0.
                       for i in range(classes)]):
        raise ValueError('wrong property identity')
    selected = [tuple(p) for p in scope['pairs']]
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(pairs):
        raise ValueError('wrong or repeated pair scope')
    return [(p, competitor) for p in selected]


def reusable_keys(snapshot):
    """Reconstruct eligible interval facts only, not execution or new proofs."""
    y = snapshot['identity']['property']['clean_prediction']
    policy = snapshot['scope']['numerical_policy']
    facts = set()
    for b in snapshot['branches']:
        bounds = b.get('proof_output_bounds')
        if bounds is None:
            continue
        lo, hi = bounds['lower'], bounds['upper']
        if len(lo) != 10 or len(hi) != 10 or any(not math.isfinite(v) for v in lo + hi) or any(a > b for a, b in zip(lo, hi)):
            raise ValueError('invalid interval')
        for j in range(10):
            if j == y:
                continue
            exact = Fraction(lo[y]) - Fraction(hi[j])
            slack = policy['outward_absolute'] + policy['outward_relative'] * max(abs(lo[y]), abs(hi[j]))
            lower = math.nextafter(float(exact) - slack, -math.inf)
            if lower > policy['safe_positive_margin']:
                facts.add((b['candidate'], j))
    if len(facts) != snapshot['available_fact_count']:
        raise ValueError('fact inventory does not reconstruct')
    return {(tuple(p), j) for p in snapshot['feasible_route_sets'] for j in range(10)
            if j != y and all((i, j) in facts for i in p)}


def parse(events, pairs, label):
    stack, properties, native = [], {}, {}
    assigned = set()
    for e in events:
        kind = e['kind']
        if kind in ('LOCAL_BEGIN', 'PROPERTY_BEGIN'):
            if kind == 'PROPERTY_BEGIN':
                keys = scope_keys(e['scope'], pairs, label)
                if assigned.intersection(keys):
                    raise ValueError('repeated property obligation in this no-retry protocol')
                assigned.update(keys)
                properties[e['seq']] = {'begin': e, 'end': None, 'replays': [], 'native': [], 'keys': keys}
            stack.append(e)
        elif kind in ('LOCAL_END', 'LOCAL_RAISE', 'PROPERTY_RESULT', 'PROPERTY_RAISE'):
            if not stack or stack[-1]['seq'] != e['token']:
                raise ValueError('unmatched/misnested scope terminal')
            b = stack.pop()
            if (b['kind'] == 'PROPERTY_BEGIN') != kind.startswith('PROPERTY_'):
                raise ValueError('wrong scope terminal type')
            if b['kind'] == 'PROPERTY_BEGIN':
                properties[e['token']]['end'] = e
        elif kind == 'PROPERTY_REPLAY':
            p = properties.get(e['token'])
            if p is None or p['end'] is None or p['end']['kind'] != 'PROPERTY_RESULT':
                raise ValueError('replay without property result')
            p['replays'].append(e)
        elif kind == 'NATIVE_READY':
            owner = next((b['seq'] for b in reversed(stack) if b['kind'] == 'PROPERTY_BEGIN'), None)
            category = properties[owner]['begin']['function'] if owner is not None else stack[-1]['function'] if stack else 'UNSCOPED_ROUTING_OR_OTHER'
            native[e['seq']] = {'ready': e, 'end': None, 'owner': owner, 'category': category}
            if owner is not None:
                properties[owner]['native'].append(e['seq'])
        elif kind in ('NATIVE_RETURN', 'NATIVE_RAISE', 'NATIVE_SKIPPED'):
            n = native.get(e['token'])
            if n is None or n['end'] is not None:
                raise ValueError('unmatched/duplicate native terminal')
            if kind != 'NATIVE_SKIPPED' and (e['effective'] <= 0 or e['entered'] + e['effective'] > n['ready']['deadline'] + 1e-9):
                raise ValueError('native allocation violation')
            n['end'] = e
    return properties, native, stack


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def analyze_row(row):
    directory = RAW / row['job_id']
    data = (directory / 'budget_journal.jsonl').read_bytes()
    if not data.endswith(b'\n'):
        raise ValueError('unexpected partial journal tail')
    events = [json.loads(l) for l in data.splitlines()]
    evidence = read(directory / 'package/evidence.json')
    snapshot = read(directory / 'common_facts.json')['payload']
    pairs = [tuple(p) for p in snapshot['feasible_route_sets']]
    label = snapshot['identity']['property']['clean_prediction']
    props, native, stack = parse(events, pairs, label)
    if stack:
        raise ValueError('complete V2 run retained an open property/local scope')
    eligible = reusable_keys(snapshot)
    required = {(p, j) for p in pairs for j in range(10) if j != label}
    obligation = {k: {'state': 'REUSABLE_FROM_COMMON_FACTS' if k in eligible else 'NO_PROPERTY_QUERY_OBSERVED',
                      'property_token': None} for k in required}
    queries = []
    for token, p in props.items():
        b, end = p['begin'], p['end']
        result = end.get('result', {}) if end else {}
        if set(p['keys']) & eligible:
            raise ValueError('solver scope includes already reusable obligation')
        state = 'QUERY_UNRETURNED' if end is None else 'QUERY_RAISED' if end['kind'] == 'PROPERTY_RAISE' else (
            'RETURNED_SOLVER_LIMIT' if result.get('solver_status') == 1 else
            'RETURNED_COMPLETE_RELAXATION' if result.get('reason', '').endswith('_RELAXATION') and result.get('solver_status') == 0 else
            'RETURNED_OTHER')
        for key in p['keys']:
            obligation[key] = {'state': state, 'property_token': token}
        calls = []
        for nt in p['native']:
            n = native[nt]; ne = n['end']
            calls.append({'token': nt, 'terminal': ne['kind'] if ne else None,
                          'result': ne.get('result') if ne else None,
                          'observed_seconds': ne['clock_elapsed'] - ne['entered'] if ne and 'entered' in ne else None,
                          'passed_seconds': ne.get('effective') if ne else None})
        # Weighted raw native objective omits the HZ center. Use its recorded
        # solver_dual_objective; never interpret a positive raw coefficient-only objective.
        dual = result.get('solver_dual_objective')
        origin = 'recorded_weighted_objective_including_center'
        if b['function'] == 'solve_monolithic_weighted_top2_f0':
            dual = calls[0]['result'].get('mip_dual_bound') if len(calls) == 1 and calls[0]['result'] else None
            origin = 'recorded_monolithic_native_objective_including_selector_centers'
        queries.append({'token': token, 'function': b['function'], 'scope': b['scope'],
                        'state': state, 'result': result, 'exception': end.get('exception') if end else None,
                        'started_seconds': b['clock_elapsed'], 'deadline_seconds': b['deadline'],
                        'observed_seconds': end['clock_elapsed'] - b['clock_elapsed'] if end else None,
                        'native_calls': calls,
                        'diagnostic_dual': dual if finite(dual) else None, 'dual_source': origin,
                        'positive_unaccepted_dual': finite(dual) and dual > 1e-7 and result.get('status') != 'SAFE',
                        'replay_observations': [e['result'] for e in p['replays']]})
    categories = defaultdict(lambda: {'calls': 0, 'observed_seconds': 0., 'status_counts': Counter(), 'missing_dual': 0})
    over = []
    for token, n in native.items():
        entry = categories[n['category']]; entry['calls'] += 1
        end = n['end']; r = end.get('result', {}) if end else {}
        entry['status_counts'][str(r.get('status', 'UNAVAILABLE'))] += 1
        entry['missing_dual'] += not finite(r.get('mip_dual_bound'))
        if end and 'entered' in end:
            entry['observed_seconds'] += end['clock_elapsed'] - end['entered']
            over.append({'token': token, 'category': n['category'],
                         'seconds': max(0., end['clock_elapsed'] - n['ready']['deadline'])})
    t2 = evidence['tier2']
    states = [e for e in events if e['kind'] == 'STATE']
    f0entry = next((e['clock_elapsed'] for e in states if e['record'].get('active_stage') in ('TIER2_F0_RUNNING', 'MONOLITHIC_F0_RUNNING')), None)
    return {
        **{k: row[k] for k in ('job_id', 'method', 'rank', 'dataset_index', 'status', 'wall_seconds')},
        'reason': evidence['verdict']['reason'], 'pairs': pairs,
        'selected_path': evidence['route_complexity_schedule']['selected_path'],
        'common_fact_count': snapshot['available_fact_count'],
        'common_fact_completion_seconds': snapshot['completion_elapsed_seconds'],
        'f0_entry_seconds': f0entry, 'first_property_seconds': queries[0]['started_seconds'] if queries else None,
        'package_partial_rows_censored': t2.get('partial_rows_censored', False),
        'package_stop': t2.get('stopped_at'), 'required_pair_property_count': len(required),
        'obligation_counts': dict(Counter(v['state'] for v in obligation.values())),
        'obligations': [{'pair': list(p), 'competitor': j, **obligation[p, j]} for p, j in sorted(required)],
        'property_queries': queries,
        'native_categories': dict(categories), 'native_calls': len(native),
        'native_unreturned_tokens': [t for t, n in native.items() if n['end'] is None],
        'native_overrun_count_over_1ms': sum(o['seconds'] > .001 for o in over),
        'largest_observed_native_overrun': max(over, key=lambda o: o['seconds']) if over else None,
        'source_hashes': {n: sha(directory / n) for n in ('request.json', 'terminal.json', 'package/evidence.json',
                                                       'common_facts.json', 'budget_journal.jsonl')}}


def aggregate(rows):
    props = [q for r in rows for q in r['property_queries']]
    counts = Counter()
    for r in rows:
        counts.update(r['obligation_counts'])
    cat = defaultdict(lambda: {'calls': 0, 'observed_seconds': 0.})
    for r in rows:
        for name, v in r['native_categories'].items():
            cat[name]['calls'] += v['calls']; cat[name]['observed_seconds'] += v['observed_seconds']
    return {'requests': len(rows), 'pair_strata': dict(Counter('single' if len(r['pairs']) == 1 else 'multiple' for r in rows)),
            'property_begins': len(props), 'property_states': dict(Counter(q['state'] for q in props)),
            'property_reasons': dict(Counter(q['result'].get('reason', q['exception'] or 'UNAVAILABLE') for q in props)),
            'obligation_counts': dict(counts), 'required_pair_properties': sum(r['required_pair_property_count'] for r in rows),
            'native_categories': dict(cat),
            'property_native_observed_seconds': sum(c['observed_seconds'] or 0. for q in props for c in q['native_calls']),
            'request_observed_seconds': sum(r['wall_seconds'] for r in rows),
            'queries_with_diagnostic_dual': sum(q['diagnostic_dual'] is not None for q in props),
            'positive_unaccepted_dual_queries': [{'job_id': r['job_id'], 'token': q['token'], 'scope': q['scope'],
                                                  'value': q['diagnostic_dual']} for r in rows for q in r['property_queries'] if q['positive_unaccepted_dual']],
            'native_unreturned_count': sum(len(r['native_unreturned_tokens']) for r in rows),
            'native_overrun_count_over_1ms': sum(r['native_overrun_count_over_1ms'] for r in rows),
            'largest_overrun': max(({'job_id': r['job_id'], **r['largest_observed_native_overrun']} for r in rows if r['largest_observed_native_overrun']), key=lambda v: v['seconds'], default=None),
            'first_property_mean_median_seconds': [mean(v), median(v)] if (v := [r['first_property_seconds'] for r in rows if r['first_property_seconds'] is not None]) else None}


def analyze():
    parent = read(PARENT)
    if parent['status'] != 'PASS':
        raise ValueError('unaccepted parent archive')
    actual = {str(p.relative_to(RAW)) for p in RAW.rglob('*') if p.is_file()}
    if actual != {v['path'] for v in parent['artifact_inventory']}:
        raise ValueError('raw inventory changed')
    for v in parent['artifact_inventory']:
        p = RAW / v['path']
        if sha(p) != v['sha256'] or p.stat().st_size != v['bytes']:
            raise ValueError(f'raw artifact changed: {v["path"]}')
    rows = [analyze_row(r) for r in parent['terminals'] if r['method'] != 'crown']
    return {'schema': 'CONV_FULL_V2_SAVED_OBLIGATION_ANALYSIS_V1', 'status': 'PASS', 'issues': [],
            'parent_archive_sha256': sha(PARENT), 'raw_inventory_unchanged': True,
            'all_act': aggregate(rows),
            'timeouts': {arm: aggregate([r for r in rows if r['method'] == arm and r['status'] == 'TIMEOUT']) for arm in ('adaptive', 'monolithic')},
            'rows': rows, 'cross_arm_witness_context': parent['witness_context'],
            'scope': 'Saved logs only; no bound/replay/optimization queries. Query records are not request verdicts. '
                     'Scopes expanded for coverage accounting, not independent solve counts. Common facts denote '
                     'eligibility, not proof that execution consumed them. No observed property query can mean '
                     'early valid UNSAFE or exhaustion before its query, not a failed solve. Diagnostic duals '
                     'lack independent certification and do not pass the frozen optimal-status gate. '
                     'Native times are disjoint observed entry-return durations; inclusive local spans are not summed. '
                     'Positive/negative native objectives with missing affine centers are not margin bounds.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); result = analyze()
    if args.check:
        if read(OUTPUT) != json.loads(json.dumps(result)):
            raise ValueError('saved analysis differs')
    else:
        # Exclusive write, no raw-directory mutation; completed file is deterministic.
        with OUTPUT.open('x') as f:
            json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    print(json.dumps({'status': result['status'], 'requests': result['all_act']['requests'],
                      'property_states': result['all_act']['property_states'],
                      'timeout_requests': {k: v['requests'] for k, v in result['timeouts'].items()},
                      'positive_unaccepted_dual_queries': len(result['all_act']['positive_unaccepted_dual_queries'])}, indent=2))
