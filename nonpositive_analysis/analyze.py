"""Exact decomposition of saved candidates. No model loading or optimization."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import time

from act.back_end.solver.lp_certificate import identity, rational
from act.back_end.solver.sparse_lp_certificate import evaluate
from portable_proof.runtime import compact, digest
from single_check_portable.execution import ROOT, read, save_new

ARCHIVE = ROOT/'docs/reuse_supervised_v1_execution_results.json'
RAW = ROOT/'data/moe/results/reuse_supervised_comparison_20260916_v1'
UNRESOLVED = 'UNRESOLVED_CANDIDATE_VS_LP_RELAXATION'


def number(v):
    return {'exact': str(v), 'approx': float(v)}


def decompose(lp, cert, nc, nb):
    """Reference checker first; then independently account for its exact bound."""
    lower, residual = evaluate(lp, cert)
    if rational(cert['claimed_lower_bound']) > lower:
        raise ValueError('invalid claimed bound')
    if len(residual) != nc+nb+2:
        raise ValueError('weighted variable layout')
    terms = {'offset': rational(lp.get('offset', 0)),
             'base_inequalities': F(0), 'mccormick_inequalities': F(0), 'equalities': F(0)}
    if lp['A']['shape'][0] < 4:
        raise ValueError('missing McCormick rows')
    for i, (b, y) in enumerate(zip(lp['b'], cert['inequality_dual'])):
        key = 'mccormick_inequalities' if i >= len(lp['b'])-4 else 'base_inequalities'
        terms[key] += rational(b)*rational(y)
    terms['equalities'] = sum((rational(h)*rational(z) for h,z in zip(lp['h'],cert['equality_dual'])), F(0))
    box = [r*min(rational(a),rational(b)) if r >= 0 else r*max(rational(a),rational(b))
           for r,a,b in zip(residual,lp['lower'],lp['upper'])]
    groups = {}
    for name, first, last in [('continuous',0,nc),('relaxed_binary',nc,nc+nb),
                               ('lambda',nc+nb,nc+nb+1),('product',nc+nb+1,nc+nb+2)]:
        rr, bb = residual[first:last], box[first:last]
        groups[name] = {'nonzero':sum(r != 0 for r in rr),
                        'residual_l1':number(sum(map(abs,rr),F(0))),
                        'box_contribution':number(sum(bb,F(0)))}
    base = sum(terms.values(), F(0)); correction = sum(box,F(0))
    if lower != base+correction:
        raise ValueError('exact accounting mismatch')
    return {'lower':number(lower), 'base_without_box_term_NOT_A_BOUND':number(base),
            'dual_terms':{k:number(v) for k,v in terms.items()},
            'residual_box_term':number(correction), 'residual_groups':groups,
            'largest_box_terms':[{'variable':i,'residual':str(residual[i]),'contribution':number(box[i])}
                                 for i in sorted(range(len(box)),key=lambda j:abs(box[j]),reverse=True)[:5]]}


def bound_file(path, inventory, refs):
    path=Path(path); name=str(path.relative_to(ROOT)); raw=path.read_bytes()
    if inventory.get(name)!=digest(raw):
        raise ValueError('sealed artifact mismatch: '+name)
    refs[name]=digest(raw)
    return json.loads(raw)


def reference(source, ref, inventory, refs):
    p=(source/ref['file']).resolve()
    if not p.is_relative_to(source.resolve()) or digest(p.read_bytes())!=ref['sha256']:
        raise ValueError('source reference identity')
    return bound_file(p,inventory,refs)


def run():
    started=time.monotonic(); archive=read(ARCHIVE); inventory=archive['artifact_sha256']; refs={}
    if archive['status']!='PASS':raise ValueError('archive not accepted')
    results=[]; requests=[]
    for ar in [r for r in archive['rows'] if r['arm']=='reuse_on']:
        root=RAW/ar['job_id']; source=root/'source'
        manifest=bound_file(source/'manifest.json',inventory,refs)
        checked=bound_file(root/'tail/check.log',inventory,refs)['result']
        queries=bound_file(source/'query_log.json',inventory,refs)
        if digest(compact(checked))!=ar['full_result_sha256']:
            raise ValueError('archived checker result mismatch')
        threshold=rational(manifest['positive_threshold'])
        seen=set(); local=[]
        for ob in manifest['obligations']:
            key=(tuple(ob['pair']),ob['property_index'])
            if key in seen or ob['kind']!='residual' or ob['weighted_status']!='PROPOSED':
                raise ValueError('duplicate/nonresidual/unproposed output')
            seen.add(key)
            ex=reference(source,ob['weighted'],inventory,refs)
            cert=reference(source,ob['certificate'],inventory,refs)
            saved=[r for r in checked['obligations'] if (tuple(r['pair']),r['property_index'])==key]
            if len(saved)!=1:raise ValueError('checker coverage mismatch')
            if (list(map(rational,ex['q']))!=list(map(rational,ob['property']['q'])) or rational(ex['offset'])!=rational(ob['property']['constant']) or
                ex['gate']!=ob['gate_bounds'] or ex['difference']!=ob['difference_bounds'] or
                ex['source_sha256']!=identity(ex['source'])):raise ValueError('property/range/source mismatch')
            nc,nb=ex['source']['Gc']['shape'][1],ex['source']['Gb']['shape'][1]
            parts=decompose(ex['lp'],cert,nc,nb); lower=rational(parts['lower']['exact'])
            if lower!=rational(saved[0]['lower_bound']):raise ValueError('independent exact bound differs')
            positive=lower>threshold
            if positive != (saved[0]['state']=='CHECKED_RATIONAL_POSITIVE'):raise ValueError('classification mismatch')
            # Inventory the certificates actually saved, not the solver's in-memory result.
            if set(cert)!={'lp_sha256','inequality_dual','equality_dual','claimed_lower_bound'}:
                raise ValueError('unexpected evidence fields: review attribution before proceeding')
            gate=list(map(rational,ex['gate'])); diff=list(map(rational,ex['difference']))
            row={'dataset_index':ar['dataset_index'],'request_id':manifest['request_id'],
                 'pair':ob['pair'],'property_index':ob['property_index'],
                 'clean_label':manifest['request']['clean_prediction'],
                 'competitor':ob['property']['q'].index(-1),
                 'gate':list(map(number,gate)),'difference':list(map(number,diff)),
                 'difference_crosses_zero':diff[0]<=0<=diff[1],
                 'gate_width':number(gate[1]-gate[0]),'difference_width':number(diff[1]-diff[0]),
                 'positive':positive,'certificate_fields':sorted(cert),
                 'saved_primal_point':False,'independently_checked_lp_upper_bound':None,
                 'independently_checked_optimality':False,
                 'attribution':'CHECKED_POSITIVE' if positive else UNRESOLVED,
                 'weighted_export_sha256':ob['weighted']['sha256'],
                 'weighted_certificate_sha256':ob['certificate']['sha256'], **parts}
            local.append(row);results.append(row)
            print(f"input={ar['dataset_index']} property={ob['property_index']} lower={float(lower):.8g}",flush=True)
        if len(seen)!=checked['required_obligations'] or len(seen)!=len(checked['obligations']):
            raise ValueError('missing necessary obligations')
        blocking=sorted([r for r in local if not r['positive']],key=lambda r:rational(r['lower']['exact']))
        # Gate supports are hash-bound provenance. Their checks were made by the sealed
        # full checker; this analysis does not infer a tighter sigmoid range.
        gate_records={}
        for name,s in manifest['supports'].items():
            if s['kind']=='router_order':
                cert=reference(source,s['certificate'],inventory,refs)
                reference(source,s['export'],inventory,refs)
                gate_records[name]={'checked_claim':cert['claimed_lower_bound'],
                                   'role':'lower(r_a-r_b)' if name.endswith('_lo') else 'lower(r_b-r_a)',
                                   'source_sha256':s['source_sha256']}
        requests.append({'dataset_index':ar['dataset_index'],'pairs':manifest['routes']['feasible'],
                         'required':len(local),'positive':len(local)-len(blocking),'nonpositive':len(blocking),
                         'blockers_most_negative_first':[r['property_index'] for r in blocking],
                         'gate_order_evidence':gate_records,'saved_query_records':len(queries)})
    if len(results)!=36 or sum(not r['positive'] for r in results)!=30:
        raise ValueError('sealed denominator changed')
    return {'schema':'SAVED_NONPOSITIVE_ANALYSIS_V1','status':'PASS',
            'archive_sha256':digest(ARCHIVE.read_bytes()),'artifact_sha256':refs,
            'sources':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')},
            'requests':requests,'obligations':results,'total':36,'positive':6,'nonpositive':30,
            'unseparated_nonpositive':30,'solver_calls':0,'model_calls':0,'new_candidates':0,
            'exact_saved_bound_rechecks':36,'analysis_seconds':time.monotonic()-started,
            'interpretation':'All 30 remain unresolved: no saved primal point or independently checked LP upper bound. PROPOSED is not an exact optimality certificate. A nonpositive lower bound does not prove LP impossibility or model unsafety.',
            'residual_caveat':'Box residual term is mandatory dual accounting, not solely roundoff; bound-variable multipliers are not stored. Removing it is NOT a valid lower bound.',
            'decision':'No identified causal range/envelope obstruction; no representation change or new solving justified by this analysis alone.',
            'trusted_base':'Original full checker provenance; network-to-HZ, guard lowering and route exclusions remain trusted.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args()
    if a.destination.exists():raise FileExistsError('preserve previous analysis')
    save_new(a.destination,run())
