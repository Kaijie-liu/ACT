"""Read-only finite-prefix costs; not another solve or incomplete-proof upgrade."""
import argparse
import json
from fractions import Fraction
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scoped_proof.io import load,sha,save
from scripts.h2_capacity_full import derive as execution_archive

ARCHIVE=ROOT/'docs/h2_capacity_full_execution_20261001_r1.json'


def derive():
    archive=load(ARCHIVE)
    if execution_archive()!=archive: raise ValueError('full execution archive changed')
    rows=[]
    for call in archive['calls']:
        root=Path(call['call']['root']); path=root/'produce_events.jsonl'
        if sha(path)!=call['events']['produce']['sha256']: raise ValueError('trace binding')
        lines=path.read_bytes().splitlines(); events=[]
        for i,line in enumerate(lines):
            try: events.append(json.loads(line))
            except (ValueError,UnicodeDecodeError):
                if i!=len(lines)-1 or not call['events']['produce']['truncated_final_line']: raise
        terminal=load(root/'terminal.json',call['terminal_sha256'])
        stage=terminal['stages'][0]; attempts=[]; current=None; duty=None
        operations={e['operation']:e for e in events if e['event']=='EXIT' and not e['operation'].startswith('native_candidate_')}
        bases=[e for e in events if e['operation']=='pair_assembled']
        for e in events:
            op=e['operation']
            if op in ('endpoint_begin','mccormick_begin'):
                duty={k:e[k] for k in ('pair','competitor','origin')}
                duty['weight']=e.get('weight')
            if e['event']=='ENTER' and op.startswith('native_candidate_'):
                if current is not None: raise ValueError('overlapping proposal attempts')
                current={'index':int(op.removeprefix('native_candidate_')),'duty':duty,'start':e['elapsed'],'marks':{}}
            elif current is not None and e['event']=='TRACE':
                current['marks'][op]=e
            elif current is not None and e['event'] in ('EXIT','EXIT_ERROR') and op.startswith('native_candidate_'):
                current.update(completed=e['event']=='EXIT',seconds=e['seconds'],end=e['elapsed'])
                attempts.append(current); current=None
        if current is not None:
            current.update(completed=False,seconds=None,
                attempt_to_owned_cutoff_bracket_seconds=stage['end_seconds']-current['start'])
            attempts.append(current)
        if [a['index'] for a in attempts]!=list(range(len(attempts))): raise ValueError('proposal sequence')
        candidates=[]
        for a in attempts:
            marks=a.pop('marks'); spans={}
            for name,left,right in (
                ('exact_matrix_validation','adapter_validation_begin','adapter_validation_complete'),
                ('conversion_and_identity','adapter_validation_complete','native_call_boundary'),
                ('native_boundary_to_return','native_call_boundary','native_solver_return'),
                ('candidate_exact_evaluation','candidate_exact_evaluation_begin','candidate_independent_bound_begin'),
                ('independent_bound_check','candidate_independent_bound_begin','candidate_independent_bound_complete')):
                spans[name]=marks[right]['elapsed']-marks[left]['elapsed'] if left in marks and right in marks else None
            a['measured_spans_seconds']=spans
            a['lp_sha256']=marks.get('adapter_validation_complete',{}).get('lp_sha256')
            a['solver_return_status']=marks.get('native_solver_return',{}).get('status')
            a['native_side_check_completed']='candidate_independent_bound_complete' in marks
            f=root/'candidates'/f"{a['index']:05d}.json"
            if f.exists():
                value=load(f); cert=value['certificate']
                if (value['invocation']!=call['call']['invocation'] or value['lp_sha256']!=a['lp_sha256'] or
                    cert['lp_sha256']!=a['lp_sha256'] or not a['completed'] or not a['native_side_check_completed']):
                    raise ValueError('published prefix candidate binding')
                candidates.append({'index':a['index'],'duty':a['duty'],'sha256':sha(f),'lp_sha256':a['lp_sha256'],
                    'claimed_lower_bound':cert['claimed_lower_bound'],
                    'lower_bound_approx':float(Fraction(cert['claimed_lower_bound'])),
                    'interpretation':'producer-side exact check recorded, not independently replayed as a complete portable request'})
        files=sorted(p.name for p in (root/'candidates').glob('*.json'))
        if files!=[f"{c['index']:05d}.json" for c in candidates]: raise ValueError('extra candidate file')
        rows.append({'mode':call['mode'],'api_seconds':call['call']['seconds'],
            'stage_seconds':call['stage_seconds'],'parent_publication_seconds':call['parent_publication_seconds'],
            'sampled_peak_rss':call['sampled_peak_rss'],'counts':call['events']['produce']['counts'],
            'completed_upstream_events':{k:v['seconds'] for k,v in operations.items() if not k.startswith('candidate_publication_')},
            'pair_bases':bases,'attempts':attempts,'published_candidates':candidates,
            'independently_checked_feasible_point_or_upper_bound':False,
            'unique_failure_cause_established':False,'LP_uncertifiability_established':False,'model_unsafety_established':False})
    if [v['base_sha256'] for v in rows[0]['pair_bases']]!=[v['base_sha256'] for v in rows[1]['pair_bases']]:
        raise ValueError('observed pair base mismatch')
    return {'schema':'H2_CAPACITY_PREFIX_ANALYSIS_V1','execution_archive_sha256':sha(ARCHIVE),
        'scope':'existing two synthetic timeout prefixes only; no new solves or numerical reclassification',
        'rows':rows,'new_solves':0,'complete_output_proofs':0,'native_float_SAFE':False,
        'interpretation':'capacity not completed within frozen budget; negative dual lower bounds alone do not establish LP impossibility or model unsafety',
        'timing':'nested spans are components, not additive to API/stage totals; unfinished attempt bracket is censored and not exact solver CPU time'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--output',type=Path); g.add_argument('--check',type=Path); a=p.parse_args(); result=derive()
    if a.output: save(a.output,result)
    if a.check and load(a.check)!=result: raise ValueError('prefix analysis drift')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},sort_keys=True))
