"""Post-result, no-solve diagnostics; never a feasible-point or safety checker."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

from act.back_end.solver.sparse_lp_certificate import rows


def point_terms(obligation, point, n):
    """Evaluate u,d,lambda,w at an UNVERIFIED stored LP vector, exactly."""
    if obligation['gate'] != ['0', '1'] or len(point) != n:
        raise ValueError('registered universal gate and vector size required')
    x = list(map(F, point))
    lo, hi = map(F, obligation['difference']); d0 = (lo + hi) / 2
    if lo > hi:
        raise ValueError('reversed difference range')
    planes = [dict(r) for r in rows(obligation['A_extra'], n)]
    # The second registered plane has s=1, t=hi: d_coeff*x + hi*lambda - w.
    if (len(planes) != 4 or planes[1].get(n-2) != hi or
            planes[1].get(n-1) != -1 or F(obligation['b_extra'][1]) != hi-d0):
        raise ValueError('not the registered second McCormick plane')
    objective = list(rows(obligation['objective'], n))
    if len(objective) != 1:
        raise ValueError('one property objective required')
    coeff = dict(objective[0])
    if coeff.get(n-1) != 1 or coeff.get(n-2, 0) != 0:
        raise ValueError('not u+w objective')
    u = F(obligation['offset']) + sum((v*x[j] for j,v in coeff.items() if j < n-2), F(0))
    d = d0 + sum((v*x[j] for j,v in planes[1].items() if j < n-2), F(0))
    lam, w = x[-2:]; product = lam*d
    return {k:str(v) for k,v in dict(u=u, d=d, gate_value=lam, relaxed_product=w,
        relaxed_objective=u+w, product_replaced_objective=u+product,
        product_minus_relaxed_product=product-w).items()}


def analyze(root):
    start = __import__('time').monotonic()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    review = json.loads((root/'review.json').read_bytes())
    if review['status'] != 'PASS' or review['issues'] or not review['complete']:
        raise ValueError('completed independent review required')
    src = root/'relocated'; manifest = json.loads((src/'manifest.json').read_bytes())
    if sha(src/'manifest.json') != review['manifest_sha256']:
        raise ValueError('reviewed package changed')
    for name,h in manifest['files'].items():
        p = (src/name).resolve()
        if not p.is_relative_to(src.resolve()) or sha(p) != h:
            raise ValueError('reviewed file changed')
    base = json.loads((src/'source/lp_base.json').read_bytes())
    obs = json.loads((src/'source/obligations.json').read_bytes())
    checked = {r['competitor']:r for r in review['result']['rows']}; result = []
    for obligation,entry in zip(obs['rows'],manifest['outcomes'],strict=True):
        k = obligation['competitor']
        if entry['competitor'] != k:
            raise ValueError('outcome order changed')
        item = {'competitor':k, 'gate':obligation['gate'], 'difference':obligation['difference'],
                'bound_status':checked[k]['status'], 'point_terms':None}
        if entry['file'] is not None:
            record = json.loads((src/entry['file']).read_bytes())
            if record['lp_sha256'] != checked[k]['lp_sha256']:
                raise ValueError('reviewed LP changed')
            item['candidate_sha256'] = sha(src/entry['file'])
            item['solver_seconds'] = record['solver']['native_seconds']
            if 'checked_lower_bound' in checked[k]:
                item.update({key:checked[k][key] for key in
                    ('checked_lower_bound','dual_constant','residual_box_correction')})
                item['reported_objective_minus_checked_bound'] = str(
                    F(record['solver']['reported_objective_including_offset']) - F(item['checked_lower_bound']))
            if record['approximate_primal_not_checked'] is not None:
                item['point_terms'] = point_terms(obligation,record['approximate_primal_not_checked'],base['variables'])
        result.append(item)
    return {'schema':'NEW_LP_SAVED_POINT_DIAGNOSTIC_V1','review_sha256':sha(root/'review.json'),
        'manifest_sha256':review['manifest_sha256'],'analysis_script_sha256':sha(Path(__file__)),
        'required_obligations':len(result),'rows':result,'solver_calls':0,
        'exact_primal_feasibility_checked':False,'network_forward_calls':0,'production_verdict_changed':False,
        'scope':'Post-result arithmetic on unverified LP vectors. Product substitution is not softmax or ReLU validation; no unsafe or LP-gap proof.',
        'seconds':__import__('time').monotonic()-start}


if __name__ == '__main__':
    print(json.dumps(analyze(Path(sys.argv[1]).resolve()),indent=2,sort_keys=True))
