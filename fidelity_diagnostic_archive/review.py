"""Reconstruct identities/costs; never solve or reconstruct a new basis."""
import json
from pathlib import Path
import time
from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest
from lp_sandwich.check import identity
from fidelity_supervised.study import verify, select, audit_saved, OUTPUT, FREEZE, REVIEW
from fidelity_supervised.flow import STAGES
from fidelity_supervised.native import OPTIONS
from sparse_diagnostic_archive.review import aggregate
from fidelity_diagnostic_archive.structure import input_bit_inventory, basis_inventory

DEST=ROOT/'docs/fidelity_supervised_real_v2_execution_results.json'


def imported_evidence(root):
    """A partial import is retained, not silently promoted to complete fidelity."""
    names=('import_status.json','import.json','preflight.json','capture.json','raw_native.json')
    saved={n:read(root/'native'/n) if (root/'native'/n).exists() else None for n in names}
    status, imp, pre, cap, raw=(saved[n] for n in names)
    result={'import_status':None if status is None else status['status'],
            'full_readback_checked':False,'before_after_match':None,
            'import_seconds':None if imp is None else imp['seconds'],
            'preflight':pre,'native_model_status':None if raw is None else raw['model_status'],
            'basis_valid':None if raw is None else raw['basis_valid'],
            'value_valid':None if raw is None else raw['value_valid'],
            'native_objective_untrusted':None if raw is None else raw['native_objective']}
    if imp is not None:
        expected=read(root/'native/input.json')['submitted']
        if (status is None or imp['status']!=status['status'] or
                imp['options']!=OPTIONS or status['options']!=OPTIONS or
                imp['submitted_sha256']!=identity(expected) or
                status['submitted_sha256']!=identity(expected)):
            raise ValueError('import identity')
        result['full_readback_checked']=(imp['status']=='HighsStatus.kOk' and imp['readback']==expected)
        result['matrix_entries']=sum(len(r['entries']) for r in expected['rows'])
        if cap is not None:
            if not result['full_readback_checked'] or any(cap[k]!=expected for k in
                ('submitted','readback_before','readback_after')):
                raise ValueError('completed capture without full fidelity')
            result['before_after_match']=True
    if cap is not None and imp is None:
        raise ValueError('capture without import evidence')
    return result


def review():
    begin=time.monotonic();v=verify();summary=audit_saved()
    if v['jobs']!=select():raise ValueError('ordered frozen selection drift')
    rows=[]
    for job,original in zip(v['jobs'],summary['rows']):
        if job['job_id']!=original['job_id']:raise ValueError('roster order')
        root=OUTPUT/job['job_id']
        construction=read(root/'construction.json') if (root/'construction.json').exists() else None
        mapping=read(root/'mapping.json') if (root/'mapping.json').exists() else None
        prepared=read(root/'prepared.json') if (root/'prepared.json').exists() else None
        row={k:original[k] for k in ('job_id','status','complete_independent_check')}
        row.update(dataset_index=job['dataset_index'],pair=job['statement']['pair'],
            property_index=job['statement']['property_index'],statement_sha256=job['statement_sha256'],
            prior_checked_lower_context_only=job['prior_checked_lower'],
            diagnostic=original.get('diagnostic'),costs=original.get('costs'),
            import_evidence=imported_evidence(root),
            mapping=None if mapping is None else {k:mapping.get(k) for k in ('status','reason')},
            input_bit_inventory=None if prepared is None else input_bit_inventory(prepared['lp']),
            basis_inventory=basis_inventory(mapping),
            construction=None if construction is None else {k:construction[k] for k in
                ('status','error','seconds','operations','stats','attempts','solver_calls')},
            logs={name:(root/(name+'.log')).read_text() for name in ('capture','map','construct')
                   if (root/(name+'.log')).exists()},
            saved_evidence={name:(root/name).exists() for name in ('prepared.json','native/input.json',
                'native/preflight.json','native/import_status.json','native/import.json',
                'native/submission.json','native/raw_native.json','native/capture.json','mapping.json',
                'construction.json','bundle.json','check.log')})
        rows.append(row)
    stats=aggregate(rows)
    if stats['status_counts']!=summary['status_counts'] or abs(stats['total_supplied_LP_seconds']-
        summary['cost_totals']['diagnostic_publication_seconds'])>1e-8:
        raise ValueError('summary/cost disagreement')
    stats.update(imports_with_complete_fidelity=sum(r['import_evidence']['full_readback_checked'] for r in rows),
                 complete_before_after_readbacks=sum(r['import_evidence']['before_after_match'] is True for r in rows))
    files=list(OUTPUT.rglob('*'))+[FREEZE,REVIEW]
    return {'schema':'FIDELITY_BASIS_DIAGNOSTIC_ARCHIVE_V2','status':'PASS','issues':[],
        'execution_status':summary['status'],'execution_head':read(OUTPUT/'launch.json')['head'],
        'rows':rows,'aggregates':stats,'cost_totals':summary['cost_totals'],
        'automatic_final_audit_seconds':read(OUTPUT/'summary.json')['final_audit_seconds'],
        'artifact_sha256':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in sorted(files) if p.is_file()},
        'archival_sources':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')},
        'independent_archival_seconds':time.monotonic()-begin,
        'new_solver_or_basis_construction_or_bound_check_calls':0,'original_records_unchanged':True,
        'scope':'saved import/terminal/cost reconstruction; not an independent new feasibility or network proof'}


if __name__=='__main__':
    if DEST.exists():raise FileExistsError('no overwrite')
    out=review();save_new(DEST,out)
    print(json.dumps({k:out[k] for k in ('status','execution_status','aggregates')},indent=2))
