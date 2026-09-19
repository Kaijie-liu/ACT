"""Read-only saved-record review; no new optimization or bound checking."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import time
from single_check_portable.execution import ROOT,read,save_new
from portable_proof.runtime import digest
from lp_diagnostic.study import verify,audit_saved,OUTPUT,FREEZE,REVIEW

DEST=ROOT/'docs/lp_diagnostic_v1_execution_results.json'

def aggregate(rows):
    if len(rows)!=4 or len({r['job_id'] for r in rows})!=4:raise ValueError('four-job roster')
    for r in rows:
        d=r['diagnostic'];c=r['costs']
        if r['complete_independent_check']!=(r['status']=='CHECKED_LP_DIAGNOSTIC'):
            raise ValueError('terminal acceptance')
        if d:
            if not r['complete_independent_check'] or d['network_SAFE'] or d['network_UNSAFE']:
                raise ValueError('proof scope')
            if d['primal_status']!='EXACT_FEASIBLE' and d['upper_bound'] is not None:
                raise ValueError('nonexact point promoted')
            if d['primal_status']=='EXACT_FEASIBLE' and any(d['violation_counts'].values()):
                raise ValueError('infeasible witness')
            if d['exact_optimality'] and (d['upper_bound'] is None or d['lower_bound'] is None or
                    Fraction(d['lower_bound'])!=Fraction(d['upper_bound'])):raise ValueError('optimality claim')
        if c and abs(c['whole_diagnostic_seconds']-c['observed_phase_sum_seconds']-c['residual_clock_seconds'])>1e-8:
            raise ValueError('cost identity')
    known=[r['costs'] for r in rows if r['costs']]
    return {'denominator':4,'status_counts':dict(Counter(r['status'] for r in rows)),
        'classification_counts':dict(Counter(r['diagnostic']['classification'] for r in rows if r['diagnostic'])),
        'complete_checks':sum(r['complete_independent_check'] for r in rows),
        'checked_upper_bounds':sum(r['diagnostic'] is not None and r['diagnostic']['upper_bound'] is not None for r in rows),
        'exact_optimality_count':sum(bool(r['diagnostic'] and r['diagnostic']['exact_optimality']) for r in rows),
        'recorded_native_calls':sum(c['native_calls'] for c in known if c['native_calls'] is not None),
        'native_count_missing':sum(r['costs'] is None or r['costs']['native_calls'] is None for r in rows),
        'total_diagnostic_seconds':sum(c['whole_diagnostic_seconds'] for c in known),
        'cost_records':len(known),
        'phase_totals_seconds':{p:sum(c['phases'][p]['seconds'] for c in known)
            for p in ('load','propose','package','check') if all(c['phases'][p]['seconds'] is not None for c in known)},
        'residual_seconds':sum(c['residual_clock_seconds'] for c in known),
        'nested_native_seconds':sum(c['native_seconds'] for c in known if c['native_seconds'] is not None)}

def review():
    begin=time.monotonic();v=verify();s=audit_saved();saved=read(OUTPUT/'summary.json')
    if v['jobs']!=__import__('lp_diagnostic.study',fromlist=['select']).select():raise ValueError('selection drift')
    rows=[]
    for job,original in zip(v['jobs'],s['rows']):
        if job['job_id']!=original['job_id']:raise ValueError('job order')
        root=OUTPUT/job['job_id'];native_path=root/'proposal/native.json'
        native=read(native_path) if native_path.exists() else None
        d=original.get('diagnostic');lp=sizes=None
        if (root/'prepared.json').exists():
            lp=read(root/'prepared.json')['lp']
            sizes={'variables':len(lp['c']),'A_rows':len(lp['b']),'E_rows':len(lp['h'])}
        row={k:original[k] for k in ('job_id','status','complete_independent_check')}
        row.update(dataset_index=job['dataset_index'],pair=job['statement']['pair'],property_index=job['statement']['property_index'],
            statement_sha256=job['statement_sha256'],prior_checked_lower=job['prior_checked_lower'],
            diagnostic=d,costs=original.get('costs'),dimensions=sizes,
            native=None if native is None else {k:native[k] for k in ('success','status','message','iterations',
                'objective_without_offset','granted_seconds','native_seconds')})
        # Decimal values are display only; exact checker strings above are authoritative.
        row['display']=None if d is None else {
            'prior_lower':float(Fraction(job['prior_checked_lower'])),
            'checked_lower':None if d['lower_bound'] is None else float(Fraction(d['lower_bound'])),
            'candidate_objective_NOT_UPPER_UNLESS_FEASIBLE':None if d['candidate_objective_NOT_UPPER_UNLESS_FEASIBLE'] is None else
                float(Fraction(d['candidate_objective_NOT_UPPER_UNLESS_FEASIBLE'])),
            'maximum_violations':{k:float(Fraction(x)) for k,x in d['maximum_exact_violations'].items()}}
        rows.append(row)
    stats=aggregate(rows)
    if stats['status_counts']!=s['status_counts'] or stats['classification_counts']!=s['classification_counts']:
        raise ValueError('aggregate reconstruction')
    if abs(stats['total_diagnostic_seconds']-s['cost_totals']['diagnostic_publication_seconds'])>1e-8:
        raise ValueError('total cost drift')
    inventory={str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in OUTPUT.rglob('*') if p.is_file()}
    inventory.update({str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in (FREEZE,REVIEW)})
    return {'schema':'LP_DIAGNOSTIC_ARCHIVE_V1','status':'PASS','issues':[],
        'execution_status':s['status'],'execution_head':read(OUTPUT/'launch.json')['head'],
        'rows':rows,'aggregates':stats,'cost_totals':s['cost_totals'],
        'automatic_final_audit_seconds':saved['final_audit_seconds'],
        'artifact_sha256':inventory,'archival_sources':{str(p.relative_to(ROOT)):digest(p.read_bytes())
            for p in Path(__file__).parent.glob('*.py')},
        'independent_archival_seconds':time.monotonic()-begin,'new_solver_or_model_or_bound_check_calls':0,
        'scope':'saved-record reconstruction only; exact LP checks were run during frozen execution; upstream lowering remains trusted',
        'original_records_unchanged':True}

if __name__=='__main__':
    import json
    if DEST.exists():raise FileExistsError('archive already exists')
    result=review();save_new(DEST,result)
    print(json.dumps({k:result[k] for k in ('status','execution_status','aggregates','independent_archival_seconds')},indent=2))
