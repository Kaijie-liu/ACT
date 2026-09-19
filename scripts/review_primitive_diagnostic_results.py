"""Fresh read-only review of the four frozen primitive diagnostic outcomes.

Replays identity/cost auditing and derives bit-growth context from saved logs.
No native solve, basis reconstruction, new bound or change to frozen sources.
"""
import json
from pathlib import Path
import time
from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest
from primitive_diagnostic import contract as C
from primitive_diagnostic.archive import collect

ARCHIVE=ROOT/'docs/primitive_diagnostic_v1_execution_results.json'
DEST=ROOT/'docs/primitive_diagnostic_v1_execution_review.json'


def review():
    begin=time.monotonic();saved=read(ARCHIVE);fresh=collect()
    if {k:v for k,v in saved.items() if k!='seconds'}!={k:v for k,v in fresh.items() if k!='seconds'}:
        raise ValueError('fresh archive reconstruction mismatch')
    old=read(C.PRIOR)
    old_rows={row['job_id']:row for row in old['rows']}
    rows=[]
    for row in saved['rows']:
        name=row['job_id'];root=C.OUTPUT/name
        c=read(root/'construction.json');t=row['costs'];ar=c['arithmetic'];stop=ar['first_limit']
        # Result-specific checks: not generic scientific acceptance gates.
        if (row['status']!='LIMIT' or c['status']!='LIMIT' or c['bundle'] is not None or
                c['row_residuals'] is not None or row['complete_independent_check'] or row['diagnostic'] is not None):
            raise ValueError('four unresolved outcomes/evidence mismatch')
        if (stop['phase']!='elimination' or stop['operation']!='row_product' or
                stop['kind']!='integer bits' or stop['cap']!=C.POLICY['arithmetic']['max_bits'] or
                stop['observed']<=stop['cap'] or ar['max_integer_bits']!=stop['observed']):
            raise ValueError('recorded bit-limit context mismatch')
        if any((root/p).exists() for p in ('bundle.json','portable','check.log')):
            raise ValueError('unresolved run unexpectedly supplied a proof')
        if t['native_calls']!=1 or not t['arithmetic_progress']['complete'] or t['arithmetic_progress']['censored']:
            raise ValueError('construction/native accounting incomplete')
        if any(t['phases'][p]['seconds'] is not None for p in ('package','check')):
            raise ValueError('unreached phase imputed cost')
        events=[read(p) for p in sorted((root/'arithmetic_events').glob('*.json'))]
        entry=next(e for e in events if e['phase']=='elimination')
        if entry['arithmetic']['first_limit'] is not None or entry['arithmetic']['max_integer_bits']>stop['cap']:
            raise ValueError('already failed before elimination')
        if any(e['phase'] in ('back_substitution','candidate') for e in events):
            raise ValueError('unexpected progress beyond recorded stop')
        comp=row['arithmetic_detail']['comparison']
        prior=old_rows[name]['construction']
        rows.append({'job_id':name,'status':row['status'],'old_recorded_pivots':prior['stats']['pivots'],
            'new_recorded_pivots':c['stats']['pivots'],'bit_cap':stop['cap'],
            'recorded_integer_bits_at_elimination_entry':entry['arithmetic']['max_integer_bits'],
            'first_limit':stop,'operations':c['operations'],'fill_insertions':c['stats']['fill_insertions'],
            'peak_live_nnz':c['stats']['peak_live_nnz'],'comparison':comp,
            'constructor_seconds':c['seconds'],'native_seconds':t['native_seconds'],
            'supplied_LP_seconds':t['whole_supplied_LP_seconds'],
            'arithmetic_phase_seconds':{s['phase']:s['seconds'] for s in c['journal']['segments']},
            'serialization_seconds':t['construction_serialization']['seconds'],
            'journal_events':len(events),'network_SAFE':False,'network_UNSAFE':False})
    if len(rows)!=4:raise ValueError('all four denominator rows required')
    return {'status':'PASS','issues':[],'archive':C.ref(ARCHIVE),'sources':C.sources(),
        'reviewer_sha256':digest(Path(__file__).read_bytes()),'artifact_files':len(saved['artifact_sha256']),
        'rows':rows,'same_basis_count':sum(r['comparison']['basis_structure_equal'] is True for r in rows),
        'same_system_count':sum(r['comparison']['assembled_system_equal'] is True for r in rows),
        'checked_feasible_upper_bounds':saved['aggregates']['checked_upper_bounds'],
        'observed_stop':'integer cross-product before row subtraction/content reduction',
        'not_established':['exact solution needs more than4096bits','LP infeasible','network unsafe',
                           'raising time or cap solves the proof','all exact algorithms fail'],
        'new_native_or_basis_or_bound_calls':0,'seconds':time.monotonic()-begin,
        'scope':'independent process replay of records, identities and costs; not an independent execution proof of elimination'}


if __name__=='__main__':
    if DEST.exists():raise FileExistsError('retain previous review')
    result=review();save_new(DEST,result)
    print(json.dumps({k:result[k] for k in ('status','issues','artifact_files','same_basis_count',
          'same_system_count','checked_feasible_upper_bounds')},indent=2))
