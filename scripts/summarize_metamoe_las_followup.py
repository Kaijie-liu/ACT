"""Saved-only repaired-cohort accounting and bounded query-cost diagnosis.

No checkpoint, dataset, solver or new query is loaded. The frozen execution
and auditor are unchanged. Timeouts remain in full-cohort outcome/cost tables;
normal-pair timing is a separate, explicitly conditional statistic.
"""
import argparse
import copy
import json
import math
from pathlib import Path

from summarize_metamoe_checked_small import (
    ARMS, DECIDED, POSITIVE, digest, summarize, timing,
)


def set_comparison(rows, ids, statuses):
    lookup = {(r['id'], r['arm']): r for r in rows}
    a, b = ({i for i in ids if lookup[i, arm]['status'] in statuses} for arm in ARMS)
    return {'intersection': sorted(a & b), 'act_only': sorted(a - b),
            'author_only': sorted(b - a)}


def base_cost(expert):
    """Fallback audit schema stores time on the query, not base_seconds."""
    if 'base_seconds' in expert:
        return expert['base_seconds']
    base = [q for q in expert['queries'] if q['scope']['phase'] == 'base']
    if len(base) != 1 or not math.isfinite(base[0]['return_elapsed_seconds']):
        raise ValueError('missing base-query cost; cannot substitute zero')
    return base[0]['return_elapsed_seconds']


def query_observation(q, native, raw_result=None):
    """Native-start is entry into MILP, not evidence of search progress."""
    start, end = q['started_monotonic'], q['deadline_monotonic']
    if not all(math.isfinite(v) for v in (start, end)) or end < start:
        raise ValueError('invalid query clocks')
    if bool(native) != q['native_started']:
        raise ValueError('native-start evidence mismatch')
    if native and (native['token'] != q['token'] or
                   not start <= native['started_monotonic'] < end):
        raise ValueError('native identity or clock')
    result = raw_result if raw_result is not None else q['native_result']
    if q['native_result'] is not None and raw_result is not None and q['native_result'] != raw_result:
        raise ValueError('native result differs from query ledger')
    if result and (result['token'] != q['token'] or native is None or
                   result['finished_monotonic'] < native['started_monotonic']):
        raise ValueError('native-result identity or clock')
    return {
        'scope': q['scope'], 'terminal': q['terminal'],
        'returned_status': q['returned_status'],
        'local_allocation_seconds': end - start,
        'observed_return_seconds': q['return_elapsed_seconds'],
        'native_started': bool(native),
        'pre_native_seconds': native['started_monotonic'] - start if native else None,
        'remaining_at_native_entry_seconds': end - native['started_monotonic'] if native else None,
        'native_options': native['options'] if native else None,
        'native_return_recorded': result is not None,
        'native_result_in_query_record': q['native_result'] is not None,
        'native_status': result['status'] if result else None,
        'native_seconds_on_return': result['native_seconds'] if result else None,
        'native_effective_options_on_return': result.get('effective_options') if result else None,
        # Worker timestamps precede JSON encoding/fsync/atomic publication.
        # They do not establish when the parent could receive the result.
        'native_finish_timestamp_before_deadline': result['finished_monotonic'] < end if result else None,
        'native_dual_bound': result.get('mip_dual_bound') if result else None,
        'native_candidate_recorded': bool(result and result.get('candidate_sha256')),
        'accepted_after_receipt': q['accepted_after_receipt'],
        'interpretation': ('No saved native return; native time is censored, not zero. '
                           'No relaxation-impossibility or model-unsafety inference.'
                           if result is None else
                           'Raw native evidence retained; only accepted_after_receipt controls acceptance. '
                           'A pre-publication finish timestamp is not a receipt timestamp.'),
    }


def analyze(archive, root):
    # The historical summarizer expects an object for absent killed results.
    # Normalize a private copy, never the frozen archive or terminal record.
    normalized = copy.deepcopy(archive)
    for r in normalized['rows']:
        if r.get('result') is None:
            r['result'] = {}
        for e in (r.get('details') or {}).get('experts', []):
            e['base_seconds'] = base_cost(e)
    value = summarize(normalized)
    value['scope'] = 'Observed-cohort repaired-version followup; no new holdout or historical-result splice.'
    rows = archive['rows']
    ids = list(dict.fromkeys(r['id'] for r in rows))
    lookup = {(r['id'], r['arm']): r for r in rows}
    for name, group in value['groups'].items():
        members = [i for i in ids if name == 'all' or i.rsplit('_', 1)[0] == name]
        attempted = [i for i in members if all(
            lookup[i, a]['status'] != 'NOT_STARTED_AFTER_ERROR' for a in ARMS)]
        group['both_attempted_ids'] = attempted
        group['missing_execution_ids'] = [i for i in members if i not in attempted]
        group['positive_sets_on_registered_inputs'] = set_comparison(rows, members, POSITIVE)
        group['decided_sets_on_registered_inputs'] = set_comparison(rows, members, DECIDED)
        group['paired_act_minus_author_seconds_all_attempted_pairs'] = timing([
            lookup[i, 'act']['seconds'] - lookup[i, 'author']['seconds'] for i in attempted])
        group['comparison_note'] = (
            'Registered-input sets report observed positives, not false/unsafe negatives; '
            'timeouts and errors stay in denominators. Missing execution is separately marked. '
            'All-attempted timing compares incurred costs, not equal successful work.')

    def read_bound(relative):
        path = root / relative
        item = archive['files'].get(str(relative))
        if item is None or digest(path) != item['sha256']:
            raise ValueError('unbound saved diagnostic artifact: ' + str(relative))
        return json.loads(path.read_text())

    details = []
    for row in rows:
        if row['arm'] != 'act' or row['status'] not in ('POSITIVE', 'UNKNOWN', 'UNSAFE_REPLAYED'):
            continue
        for ei, expert in enumerate(row['details'].get('experts', [])):
            folder = Path(row['id'] + '_act') / 'protected' / f'evaluation_{ei:03d}'
            queries = []
            for qi, summary in enumerate(expert['queries']):
                if summary['scope']['phase'] == 'base':
                    continue
                qdir = folder / f'query_{qi:03d}'
                q = read_bound(qdir / 'return.json')
                # Effective late-publication status is recorded by the auditor.
                late = qdir / 'late_return_rejected.json'
                if str(late) in archive['files']:
                    read_bound(late)
                    q.update(returned_status='unknown', accepted_after_receipt=False,
                             terminal='RETURN_PUBLICATION_DEADLINE')
                if any(q[k] != v for k, v in summary.items()):
                    raise ValueError('audited query mismatch')
                native = read_bound(qdir / 'native_started.json') if q['native_started'] else None
                raw_path = qdir / 'native_result.json'
                raw = read_bound(raw_path) if str(raw_path) in archive['files'] else None
                if q['native_result'] is not None and raw is None:
                    raise ValueError('missing native result artifact')
                queries.append(query_observation(q, native, raw))
            details.append({
                'id': row['id'], 'evaluation': ei, 'request_status': row['status'],
                'request_seconds': row['seconds'], 'expert_allocation_seconds': expert['allocation_seconds'],
                'expert_seconds': expert['expert_elapsed_before_publication'],
                'base_query_seconds': base_cost(expert), 'base_status': expert['base_status'],
                'base_kind': expert['queries'][0]['terminal'],
                'property_status_counts': expert['property_status_counts'],
                'blocked_properties': [p for p in expert['properties'] if p['status'] != 'infeasible'],
                'queries': queries,
            })
    value['saved_query_diagnosis'] = details
    value['diagnostic_scope'] = ('No optimality reconstruction or new solving. Lack of native return '
        'cannot separate presolve/search/precision causes. Query and expert limits differ from outer limit.')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    v = analyze(json.loads(args.archive.read_text()), args.root)
    v.update(archive_sha256=digest(args.archive), analysis_source_sha256=digest(__file__),
             historical_summary_source_sha256=digest(Path(__file__).with_name('summarize_metamoe_checked_small.py')))
    with args.output.open('x') as f:
        json.dump(v, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps(v['groups']['all'], indent=2))
