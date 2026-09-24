"""Saved-only archive replay plus exact real router-bound differential.

No solver/query/model load. Tap completed checker outputs without changing or
skipping their arithmetic; this is not an independent human review.
"""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def comparable(result):
    keys=('higher','lower','checked_lower_bound','residual_box_term','nonzero_residual_coordinates','status')
    return {'bounds':[{k:b[k] for k in keys} for b in result['bounds']],
        'pairs':[{k:v for k,v in p.items() if k not in ('lp_sha256','evidence_sha256')} for p in result['pairs']],
        'needed_experts':result['needed_experts']}


def replay():
    if not sys.flags.no_site:raise ValueError('python -S required')
    import scripts.archive_residual_proof_comparison as archive
    import residual_proof.check as checker
    from scoped_proof.io import load, sha
    from source_enclosure.format import identity
    start=time.monotonic();observed={};original=checker.route_check
    def capture(*args,**kwargs):
        value=original(*args,**kwargs)
        key=kwargs['mode'];normalized=comparable(value)
        if key in observed and observed[key]!=normalized:raise ValueError('same mode route check changed')
        observed[key]=normalized
        return value
    checker.route_check=capture
    try:result,fresh=archive.collect()
    finally:checker.route_check=original
    if archive.stable(result)!=archive.stable(load(archive.ARCHIVE)) or archive.stable(fresh)!=archive.stable(load(archive.AUDIT)):
        raise ValueError('saved archive replay mismatch')
    if set(observed)!={'pairwise','shared'} or observed['pairwise']!=observed['shared']:
        raise ValueError('real router exact differential mismatch')
    if any(n.split('.')[0] in ('torch','numpy','scipy','highspy','act') for n in sys.modules):
        raise ValueError('model or native solver imported')
    value=observed['shared']
    return {'status':'PASS','issues':0,'config_sha256':result['config_sha256'],
        'raw_files_rehashed':len(result['files']),'frozen_sources_checked':result['frozen_source_files_checked'],
        'source_signatures':fresh['source_signatures'],'same_source_verified':fresh['same_source_verified'],
        'ordered_bounds_exactly_equal':len(value['bounds']),'pair_decisions_exactly_equal':len(value['pairs']),
        'normalized_router_evidence_sha256':identity(value),
        'retained_pairs':[p['pair'] for p in value['pairs'] if p['status']=='RETAINED'],
        'both_original_terminals_preserved':True,'new_complete_output_certificates':0,
        'new_model_queries':0,'new_native_solver_calls':0,'replay_script_sha256':sha(__file__),
        'separate_replay_seconds':time.monotonic()-start,
        'scope':'offline exact recheck/differential only; no retroactive online acceptance'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write-report',action='store_true');a=p.parse_args()
    result=replay()
    if a.write_report:
        from scoped_proof.io import save
        save(ROOT/'docs/residual_proof_execution_replay_20260925_r1.json',result)
    print(result)
