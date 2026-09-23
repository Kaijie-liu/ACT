"""Saved-only descriptive accounting; no model, data, solver or new queries.

Missing executions remain missing (not zero-time, UNKNOWN, or negatives).
Positive/decided set comparisons apply ONLY to pairs with both arms returned
normally; numerical evidence grades are deliberately not equated.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


ARMS = ('act', 'author')
NORMAL = {'POSITIVE', 'BACKEND_POSITIVE', 'UNKNOWN', 'UNSAFE_REPLAYED'}
POSITIVE = {'POSITIVE', 'BACKEND_POSITIVE'}
DECIDED = POSITIVE | {'UNSAFE_REPLAYED'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timing(values):
    return {'n': len(values), 'sum': sum(values) if values else None,
            'mean': mean(values) if values else None,
            'median': median(values) if values else None}


def summarize(archive):
    if archive['audit'] != 'PASS' or archive['issues'] != 0:
        raise ValueError('requires accepted terminal audit')
    rows = archive['rows']
    lookup = {(r['id'], r['arm']): r for r in rows}
    ids = list(dict.fromkeys(r['id'] for r in rows))
    if len(lookup) != len(rows) or set(lookup) != {(i, a) for i in ids for a in ARMS}:
        raise ValueError('missing or duplicate roster row')
    for r in rows:
        s = r.get('seconds')
        if r['status'] == 'NOT_STARTED_AFTER_ERROR':
            if s is not None:
                raise ValueError('unexecuted request has execution time')
        elif s is None or not math.isfinite(s) or s < 0:
            raise ValueError('missing or invalid attempted-request cost')

    groups = {}
    for dataset in ('all', *sorted({i.rsplit('_', 1)[0] for i in ids})):
        group_ids = [i for i in ids if dataset == 'all' or i.rsplit('_', 1)[0] == dataset]
        entry = {}
        for arm in ARMS:
            rr = [lookup[i, arm] for i in group_ids]
            attempted = [r for r in rr if r['status'] != 'NOT_STARTED_AFTER_ERROR']
            entry[arm] = {
                'registered_denominator': len(rr), 'attempted': len(attempted),
                'terminal_counts': dict(Counter(r['status'] for r in rr)),
                'charged_seconds_all_attempted': timing([r['seconds'] for r in attempted]),
                'charged_seconds_by_status': {s: timing([r['seconds'] for r in attempted if r['status'] == s])
                    for s in sorted({r['status'] for r in attempted})},
                'grades': dict(Counter(r.get('grade', 'NONE') for r in rr)),
                'raw_backend_counts': dict(Counter(r.get('result', {}).get('backend_status', 'not_recorded')
                    for r in attempted)),
                'peak_sampled_group_rss_bytes': max((r['receipt']['peak_sampled_group_rss_bytes']
                    for r in attempted), default=None),
            }
        paired = [i for i in group_ids if all(lookup[i, a]['status'] in NORMAL for a in ARMS)]
        entry['both_returned_normally_ids'] = paired
        entry['not_comparable_ids'] = [i for i in group_ids if i not in paired]
        for name, statuses in (('positive', POSITIVE), ('decided', DECIDED)):
            a, b = ({i for i in paired if lookup[i, arm]['status'] in statuses} for arm in ARMS)
            entry[name+'_sets_on_normal_pairs'] = {
                'intersection': sorted(a & b), 'act_only': sorted(a - b), 'author_only': sorted(b - a)}
        entry['paired_act_minus_author_seconds_normal_pairs'] = timing([
            lookup[i, 'act']['seconds'] - lookup[i, 'author']['seconds'] for i in paired])
        groups[dataset] = entry

    act_details = []
    for r in rows:
        if r['arm'] != 'act' or r['status'] not in NORMAL:
            continue
        d = r['details']
        ex = d['experts']
        props = [p for e in ex for p in e['properties']]
        queries = [q for e in ex for q in e['queries'] if q['scope']['phase'] == 'expanded']
        act_details.append({
            'id': r['id'], 'status': r['status'], 'candidates': r['result']['candidates'],
            'unresolved_routes': r['result']['unresolved'],
            'all_recorded_nonzero_accepted': bool(d['nonzero']) and all(q['accepted'] for q in d['nonzero']),
            'properties_recorded': len(props),
            'property_status_counts': dict(Counter(p['status'] for p in props)),
            'blocked_properties': [p for p in props if p['status'] != 'infeasible'],
            'expanded_query_terminal_counts': dict(Counter(q['terminal'] for q in queries)),
            'expanded_query_seconds': timing([q['return_elapsed_seconds'] for q in queries]),
            'expert_seconds_before_publication': sum(e['expert_elapsed_before_publication'] for e in ex),
            'base_seconds': sum(e['base_seconds'] for e in ex),
            'native_property_calls': sum(e['native_property_calls'] for e in ex),
            'nonzero_native_calls': sum(q['native_invoked'] for q in d['nonzero']),
        })
    return {
        'scope': 'Fixed stopped cohort; normal-pair comparisons descriptive, not complete-cohort superiority.',
        'numerical_guarantees_equated': False, 'new_solves': 0,
        'groups': groups, 'act_obligation_details': act_details, 'batch_cost': archive['cost'],
        'interpretation': 'Local query TIMEOUT is an execution limit, not proof of relaxation insufficiency or model unsafety.',
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    v = summarize(json.loads(a.archive.read_text()))
    v.update(archive_sha256=digest(a.archive), analysis_source_sha256=digest(__file__))
    with a.output.open('x') as f:
        json.dump(v, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps(v['groups'], indent=2))
