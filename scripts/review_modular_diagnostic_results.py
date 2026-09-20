"""Read-only review of the four completed modular diagnostics; no new solving.

Result-specific assertions describe this frozen run, never new acceptance gates.
"""
from collections import Counter, defaultdict
import json
from math import prod
from pathlib import Path
import time
from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest
from modular_diagnostic import contract as C
from modular_diagnostic.archive import collect

ARCHIVE=ROOT/'docs/modular_diagnostic_v1_execution_results.json'
DEST=ROOT/'docs/modular_diagnostic_v1_execution_review.json'


def derive(row, construction, events, old_construction):
    """Saved records only. This is not independent proof of modular execution."""
    c=construction;t=row['costs'];s=c['stats'];a=c['arithmetic'];stop=a['first_limit']
    if (row['status']!='LIMIT' or c['status']!='LIMIT' or c['bundle'] is not None or
            c['row_residuals'] is not None or row['complete_independent_check'] or
            row['diagnostic'] is not None or c['network_SAFE'] or c['network_UNSAFE']):
        raise ValueError('unresolved outcome or evidence mismatch')
    if (stop['kind']!='operations' or stop['cap']!=C.POLICY['arithmetic']['operations'] or
            stop['observed']!=stop['cap']+1 or c['operations']!=stop['observed'] or
            (stop['phase'],stop['operation']) not in
            (('finite_field','map'),('finite_field','elimination'),('prime_schedule','trial_division'))):
        raise ValueError('recorded operation-limit context')
    if (s['max_field_product_bits']>60 or
            max(a[k] for k in ('max_integer_bits','max_fraction_numerator_bits',
                               'max_fraction_denominator_bits'))>C.POLICY['arithmetic']['max_bits']):
        raise ValueError('unexpected bit-limit evidence')
    counts=Counter(r.get('status','INCOMPLETE_FIELD_ROUND') for r in s['rounds'])
    merged=[r for r in s['rounds'] if 'modulus_bits' in r]
    if (not 0<s['primes_used']<=s['primes_tried']<C.POLICY['arithmetic']['primes'] or
            len(s['rounds'])!=s['primes_tried'] or len(merged)!=s['primes_used'] or
            counts['RECONSTRUCTION_INCOMPLETE']!=s['primes_used'] or
            counts['INCOMPLETE_FIELD_ROUND']!=s['primes_tried']-s['primes_used'] or
            set(counts)-{'RECONSTRUCTION_INCOMPLETE','INCOMPLETE_FIELD_ROUND'} or
            s['bad_denominator_primes'] or s['singular_primes'] or s['residual_rejections'] or
            s['reconstruction_attempts']!=s['primes_used']):
        raise ValueError('round/reconstruction accounting')
    modulus=prod(r['prime'] for r in merged)
    if modulus.bit_length()!=s['max_modulus_bits']:
        raise ValueError('recorded CRT modulus bits')
    if (t['native_calls']!=1 or not t['arithmetic_progress']['complete'] or
            t['arithmetic_progress']['censored'] or
            any(t['phases'][p]['seconds'] is not None for p in ('package','check'))):
        raise ValueError('complete unresolved cost accounting')
    if (len(events)!=c['journal']['event_count'] or not events[-1]['closed'] or
            events[-1]['stats']!=s or events[-1]['arithmetic']!=a or
            events[-1]['operations']!=c['operations'] or
            any(e['phase'] in ('exact_residual','candidate') for e in events)):
        raise ValueError('unexpected candidate/residual progress')
    seconds=defaultdict(float);windows=Counter()
    for segment in c['journal']['segments']:
        seconds[segment['phase']]+=segment['seconds']
        windows[segment['phase']]+=1
    if abs(sum(seconds.values())-c['journal']['observed_phase_seconds'])>1e-7:
        raise ValueError('cyclic phase sum (do not overwrite repeated phases)')
    comp=row['arithmetic_detail']['comparison']
    return {'job_id':row['job_id'],'status':row['status'],'first_limit':stop,
        'primes_tried':s['primes_tried'],'primes_merged':s['primes_used'],
        'round_status_counts':dict(counts),'max_modulus_bits':s['max_modulus_bits'],
        'max_field_product_bits':s['max_field_product_bits'],'arithmetic':a,
        'recorded_field_pivots_total':s['pivots'],
        'pivot_interpretation':'sum over all prime factorizations, not a single basis dimension',
        'fill_insertions_total':s['fill_insertions'],'peak_live_nnz':s['peak_live_nnz'],
        'comparison':comp,'old_primitive_first_limit':old_construction['arithmetic']['first_limit'],
        'constructor_seconds':c['seconds'],'native_seconds':t['native_seconds'],
        'supplied_LP_seconds':t['whole_supplied_LP_seconds'],
        'arithmetic_phase_seconds':dict(seconds),'arithmetic_phase_windows':dict(windows),
        'serialization_seconds':t['construction_serialization']['seconds'],
        'journal_events':len(events),'complete_original_equation_checks':0,
        'checked_feasible_upper_bounds':0,'network_SAFE':False,'network_UNSAFE':False}


def review():
    begin=time.monotonic();saved=read(ARCHIVE);fresh=collect()
    if {k:v for k,v in saved.items() if k!='seconds'}!={k:v for k,v in fresh.items() if k!='seconds'}:
        raise ValueError('fresh archive reconstruction mismatch')
    previous={r['job_id']:r for r in read(C.PRIOR)['rows']}
    rows=[]
    for row in saved['rows']:
        root=C.OUTPUT/row['job_id']
        if any((root/p).exists() for p in ('bundle.json','portable','check.log')):
            raise ValueError('unexpected proof for unresolved request')
        c=read(root/'construction.json')
        events=[read(p) for p in sorted((root/'arithmetic_events').glob('*.json'))]
        old=previous[row['job_id']]['arithmetic_detail']['construction']
        rows.append(derive(row,c,events,old))
    if [r['job_id'] for r in rows]!=C.POLICY['job_ids']:raise ValueError('ordered four-result roster')
    return {'status':'PASS','issues':[],'archive':C.ref(ARCHIVE),'sources':C.sources(),
        'reviewer_sha256':digest(Path(__file__).read_bytes()),'artifact_files':len(saved['artifact_sha256']),
        'rows':rows,'same_basis_count':sum(r['comparison']['basis_structure_equal'] is True for r in rows),
        'same_system_count':sum(r['comparison']['assembled_system_equal'] is True for r in rows),
        'checked_feasible_upper_bounds':saved['aggregates']['checked_upper_bounds'],
        'observed_stop':'shared operation cap before a complete reconstructed vector',
        'not_established':['exact solution requires more than4096bits','all128 primes would suffice',
            'LP infeasible','network unsafe','increased time or cap would close the proof'],
        'new_native_or_basis_or_bound_calls':0,'seconds':time.monotonic()-begin,
        'scope':'record/identity/cost reconstruction and modular-round accounting, not an independent proof of elimination'}


if __name__=='__main__':
    if DEST.exists():raise FileExistsError('retain prior review')
    result=review();save_new(DEST,result)
    print(json.dumps({k:result[k] for k in ('status','issues','artifact_files','same_basis_count',
          'same_system_count','checked_feasible_upper_bounds')},indent=2))
