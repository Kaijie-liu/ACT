"""Archive all frozen denominators, including stop-after-error and missing costs."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import math
import time
from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest
from sparse_supervised.study import verify, select, audit_saved, OUTPUT, FREEZE, REVIEW
from sparse_supervised.flow import STAGES

DEST = ROOT / 'docs/sparse_supervised_real_v1_execution_results.json'


def aggregate(rows):
    if len(rows) != 4 or len({r['job_id'] for r in rows}) != 4:
        raise ValueError('four-job roster')
    stopped = False
    for row in rows:
        status = row['status']
        if stopped != (status == 'NOT_RUN_AFTER_ERROR'):
            raise ValueError('error-stop denominator')
        if row['complete_independent_check'] != (status == 'CHECKED_LP_DIAGNOSTIC'):
            raise ValueError('terminal/check consistency')
        d, c = row['diagnostic'], row['costs']
        if status == 'NOT_RUN_AFTER_ERROR' and (d is not None or c is not None):
            raise ValueError('unexecuted request assigned evidence/cost')
        if row['complete_independent_check'] != (d is not None):
            raise ValueError('missing or spurious check')
        if d:
            if d['network_SAFE'] or d['network_UNSAFE']:
                raise ValueError('network claim')
            if d['primal_status'] != 'EXACT_FEASIBLE' and d['upper_bound'] is not None:
                raise ValueError('inexact point promoted')
            if d['primal_status'] == 'EXACT_FEASIBLE' and (d['upper_bound'] is None or any(d['violation_counts'].values())):
                raise ValueError('feasibility evidence')
            if d['lower_bound'] is not None or d['exact_gap'] is not None or d['exact_optimality']:
                raise ValueError('this frozen path supplies no dual')
        if c:
            values = [c[k] for k in ('whole_supplied_LP_seconds','observed_phase_sum_seconds','residual_seconds')]
            if any(not math.isfinite(x) or x < 0 for x in values) or abs(values[0]-values[1]-values[2])>1e-8:
                raise ValueError('whole cost identity')
            if c['native_calls'] not in (None,1) or (c['native_calls'] is None) != (c['native_seconds'] is None):
                raise ValueError('recorded native call cost')
            phases = c['phases']
            if set(phases) != set(STAGES):
                raise ValueError('phase coverage')
            observed = [p['seconds'] for p in phases.values() if p['seconds'] is not None]
            if any(not math.isfinite(x) or x < 0 for x in observed) or abs(sum(observed)-values[1])>1e-8:
                raise ValueError('phase cost identity')
        stopped |= status == 'ERROR'
    known = [r['costs'] for r in rows if r['costs'] is not None]
    return {'denominator':4, 'status_counts':dict(Counter(r['status'] for r in rows)),
            'complete_checks':sum(r['complete_independent_check'] for r in rows),
            'checked_upper_bounds':sum(r['diagnostic'] is not None and r['diagnostic']['upper_bound'] is not None for r in rows),
            'checked_nonpositive_upper_bounds':sum(r['diagnostic'] is not None and r['diagnostic']['upper_bound'] is not None
                                                   and Fraction(r['diagnostic']['upper_bound'])<=0 for r in rows),
            'recorded_native_calls':sum(c['native_calls'] for c in known if c['native_calls'] is not None),
            'native_count_missing':sum(r['costs'] is None or r['costs']['native_calls'] is None for r in rows),
            'cost_records':len(known), 'uncosted_requests':4-len(known),
            'total_supplied_LP_seconds':sum(c['whole_supplied_LP_seconds'] for c in known),
            'phase_costs':{p:{'observed_seconds':sum(c['phases'][p]['seconds'] for c in known if c['phases'][p]['seconds'] is not None),
                               'with_record':sum(c['phases'][p]['seconds'] is not None for c in known),
                               'missing_of_four':4-sum(c['phases'][p]['seconds'] is not None for c in known)} for p in STAGES},
            'network_SAFE':False,'network_UNSAFE':False}


def review():
    begin = time.monotonic()
    v, summary = verify(), audit_saved()
    if v['jobs'] != select():
        raise ValueError('frozen selection drift')
    rows = []
    for job, original in zip(v['jobs'], summary['rows']):
        if job['job_id'] != original['job_id']:
            raise ValueError('ordered roster')
        root = OUTPUT / job['job_id']
        logs = {p: (root/(p+'.log')).read_text() for p in STAGES if (root/(p+'.log')).exists() and
                p in ('capture','map','construct')}
        # No new native calls, bound computation or basis reconstruction here.
        construction = read(root/'construction.json') if (root/'construction.json').exists() else None
        row = {k:original[k] for k in ('job_id','status','complete_independent_check')}
        row.update(dataset_index=job['dataset_index'],pair=job['statement']['pair'],
                   property_index=job['statement']['property_index'],statement_sha256=job['statement_sha256'],
                   prior_checked_lower_context_only=job['prior_checked_lower'],
                   diagnostic=original.get('diagnostic'),costs=original.get('costs'),logs=logs,
                   construction=None if construction is None else {k:construction[k] for k in
                       ('status','error','seconds','operations','stats','attempts','solver_calls')},
                   saved_evidence={name:(root/name).exists() for name in ('prepared.json','native/input.json',
                       'native/submission.json','native/raw_native.json','native/capture.json',
                       'mapping.json','construction.json','bundle.json','check.log')})
        rows.append(row)
    stats = aggregate(rows)
    if stats['status_counts'] != summary['status_counts'] or abs(stats['total_supplied_LP_seconds']-
            summary['cost_totals']['diagnostic_publication_seconds'])>1e-8:
        raise ValueError('saved summary disagreement')
    files = list(OUTPUT.rglob('*')) + [FREEZE, REVIEW]
    return {'schema':'SPARSE_BASIS_DIAGNOSTIC_ARCHIVE_V1','status':'PASS','issues':[],
            'execution_status':summary['status'],'execution_head':read(OUTPUT/'launch.json')['head'],
            'rows':rows,'aggregates':stats,'cost_totals':summary['cost_totals'],
            'automatic_final_audit_seconds':read(OUTPUT/'summary.json')['final_audit_seconds'],
            'artifact_sha256':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in sorted(files) if p.is_file()},
            'archival_sources':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')},
            'independent_archival_seconds':time.monotonic()-begin,
            'new_solver_or_basis_construction_or_bound_check_calls':0,'original_records_unchanged':True,
            'scope':'saved terminal/cost evidence, not LP feasibility or network proof; missing values remain null'}


if __name__ == '__main__':
    import json
    if DEST.exists():
        raise FileExistsError('no archive overwrite')
    result=review();save_new(DEST,result)
    print(json.dumps({k:result[k] for k in ('status','execution_status','aggregates')},indent=2))
