"""Terminal accounting helpers, no real selection or automatic experiment launch."""
from collections import Counter
from pathlib import Path
from modular_supervised.flow import audit, costs


def loop(jobs, run, publish):
    ids = [j['job_id'] for j in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate job identity')
    rows, stopped = [], False
    for job in jobs:
        if stopped:
            row = {'status': 'NOT_RUN_AFTER_ERROR', 'complete_independent_check': False}
        else:
            try:
                row = run(job)
            except Exception as exc:
                row = {'status': 'ERROR', 'complete_independent_check': False, 'error': repr(exc)}
        stopped |= row['status'] == 'ERROR'
        row = {**row, 'job_id': job['job_id']}
        publish(row)
        rows.append(row)
    return rows


def summarize(rows, roots):
    """roots is an explicit job→directory map; unknown/missing costs are never zero."""
    if len({r['job_id'] for r in rows}) != len(rows) or set(roots) - {r['job_id'] for r in rows}:
        raise ValueError('terminal roster')
    records, times, stopped = [], [], False
    for row in rows:
        if stopped != (row['status'] == 'NOT_RUN_AFTER_ERROR'):
            raise ValueError('stop-after-error roster')
        job = row['job_id']
        if job in roots:
            v = audit(Path(roots[job]))
            if v['status'] != row['status'] or v['complete_independent_check'] != row['complete_independent_check']:
                raise ValueError('terminal drift')
            cost = costs(Path(roots[job]))
            times.append(cost['whole_supplied_LP_seconds'])
        else:
            if row['status'] not in ('ERROR', 'NOT_RUN_AFTER_ERROR'):
                raise ValueError('missing request directory')
            cost = None
        records.append({'job_id': job, 'status': row['status'], 'costs': cost})
        stopped |= row['status'] == 'ERROR'
    return {'denominator': len(rows), 'status_counts': dict(Counter(r['status'] for r in rows)),
            'costed_requests': len(times), 'uncosted_requests': len(rows) - len(times),
            'sum_observed_request_seconds': sum(times), 'records': records,
            'scope': 'request publication clocks only; resource waits and fresh audits reported separately'}
