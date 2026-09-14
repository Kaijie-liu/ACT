"""Independent supplied-LP checks and complete-request obligation aggregation.

No torch, SciPy, exporter or model execution. Trust boundaries in TRUSTED
remain assumptions, NOT independently checked facts.
"""
import hashlib
import itertools
import json
from pathlib import Path
from fractions import Fraction
from act.back_end.solver.lp_certificate import identity, rational
from act.back_end.solver.check_hz_lp_export import check_export

TRUSTED = ['network_and_input_to_HZ', 'membership_and_pair_guard_lowering',
           'router_infeasibility_exclusions', 'F0_outer_HZ_construction_and_floating_coefficients']
RATIONAL_TRUSTED = TRUSTED[:-1]


def property_row(classes, clean, index):
    other = [i for i in range(classes) if i != clean][index]
    q = [0]*classes; q[clean] = 1; q[other] = -1
    return q


def order_envelope(lower, negative_upper):
    """Exact dyadic sigmoid enclosure from checked score ORDER only."""
    if lower is not None and negative_upper is not None and lower > -negative_upper:
        raise ValueError('inconsistent checked router range')
    return [0.5 if lower is not None and lower >= 0 else 0,
            0.5 if negative_upper is not None and negative_upper >= 0 else 1]


def aggregate(manifest, read):
    direct = manifest['schema'] == 'request_lp_rational_v3'
    trusted = RATIONAL_TRUSTED if direct else TRUSTED
    if manifest['schema'] not in ('request_lp_v1','request_lp_order_v2','request_lp_rational_v3') or manifest['trusted_base'] != trusted:
        raise ValueError('unknown proof contract')
    request = manifest['request']; rid = identity(request)
    if manifest['request_id'] != rid or request['tie_policy'] != 'ANY_LEGAL_TOPK' or request['top_k'] != 2:
        raise ValueError('request/tie identity mismatch')
    routes = manifest['routes']; groups = [routes[k] for k in ('feasible','infeasible','unresolved')]
    listed = [tuple(p) for group in groups for p in group]
    all_pairs = list(itertools.combinations(range(request['experts']),2))
    if sorted(listed) != all_pairs: raise ValueError('route partition missing/duplicate/noncanonical')
    if not routes['exact'] or routes['unresolved']:
        return {'status':'UNKNOWN', 'reason':'INCOMPLETE_ROUTE_COVERAGE', 'trusted_base':trusted}
    required = {(tuple(p),i) for p in routes['feasible'] for i in range(request['classes']-1)}
    records = manifest['obligations']
    if len(records) != len(required) or {(tuple(r['pair']),r['property_index']) for r in records} != required:
        raise ValueError('output obligation inventory missing/duplicate')
    cache = {}
    def bound(key, kind, scope, prop, q):
        r = manifest['proofs'][key]
        if (r['request_id'] != rid or r['kind'] != kind or r['scope'] != scope or r['property_index'] != prop):
            raise ValueError('proof reused outside request/domain/property scope')
        export = read(r['export'])
        if export['q'] != q or rational(export['offset']) != 0: raise ValueError('wrong output property')
        if key not in cache:
            if r['status'] != 'CHECKED':
                cache[key] = None
            else:
                cert = read(r['certificate'])
                result = check_export(export,cert,expected_source_sha256=r['hz_sha256'])
                cache[key] = Fraction(result['bound']['checked_lower_bound'])
        return cache[key]
    accepted = []; counts = {'reused':0,'residual':0,'unknown':0}
    threshold = rational(manifest['positive_threshold'])
    if threshold != rational(1e-7): raise ValueError('acceptance threshold changed')
    for row in records:
        pair, prop = row['pair'], row['property_index']; q = property_row(request['classes'],request['clean_prediction'],prop)
        value = None
        if row['kind'] == 'reused':
            if len(row['sources']) != 2: raise ValueError('reuse requires two experts')
            values = [bound(key,'expert',{'membership':i},prop,q) for key,i in zip(row['sources'],pair)]
            if all(v is not None and v > threshold for v in values): value = min(values)
            else: raise ValueError('reuse lacks positive checked facts')
        elif row['kind'] == 'residual':
            qdiff = q + [-v for v in q]
            lo = bound(row['difference_lower'],'difference',{'pair':pair},prop,qdiff)
            neg_hi = bound(row['difference_upper'],'difference',{'pair':pair},prop,[-v for v in qdiff])
            if lo is None or neg_hi is None: raise ValueError('residual lacks checked disagreement range')
            if rational(row['difference_bounds'][0]) > lo or rational(row['difference_bounds'][1]) < -neg_hi:
                raise ValueError('disagreement range rounded inward')
            if rational(row['difference_bounds'][0]) > rational(row['difference_bounds'][1]):
                raise ValueError('reversed disagreement range')
            gate=[0,1]
            if manifest['schema'] in ('request_lp_order_v2','request_lp_rational_v3'):
                qm=[0]*request['experts'];qm[pair[0]]=1;qm[pair[1]]=-1
                low=bound(row['gate_lower'],'router_order',{'pair':pair},None,qm)
                neg=bound(row['gate_upper'],'router_order',{'pair':pair},None,[-v for v in qm])
                gate=order_envelope(low,neg)
            if row['lambda_bounds'] != gate: raise ValueError('unproved nonlinear gate range')
            if direct:
                from act.back_end.solver.check_rational_mccormick import check_construction
                item = manifest['proofs'][row['source']]
                if (item['request_id']!=rid or item['kind']!='rational_weighted' or
                    item['scope']!={'pair':pair} or item['property_index']!=prop):
                    raise ValueError('rational proof outside scope')
                lower_item=manifest['proofs'][row['difference_lower']]
                upper_item=manifest['proofs'][row['difference_upper']]
                # Scope alone is insufficient: both support proofs must refer
                # to the very same pre-F0 joint HZ and factor frame.
                if lower_item['hz_sha256']!=upper_item['hz_sha256'] or item['hz_sha256']!=lower_item['hz_sha256']:
                    raise ValueError('different shared factor sources')
                record=read(item['export'])
                certificate=read(item['certificate']) if item['status']=='CHECKED' else None
                checked=check_construction(record,certificate,source_hash=item['hz_sha256'],q=q,offset=0,
                                           gate=gate,difference=row['difference_bounds'])
                value=Fraction(checked['bound']['checked_lower_bound']) if certificate is not None else None
            else:
                value = bound(row['source'],'weighted',{'pair':pair},prop,[1])
        elif row['kind'] != 'unknown': raise ValueError('unknown obligation kind')
        if value is not None and value > threshold:
            accepted.append(value); counts[row['kind']] += 1
        else: counts['unknown'] += 1
    return {'status':('CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_LOWERING'
                      if required and len(accepted)==len(required) else 'UNKNOWN'),
            'required':len(required), 'counts':counts,
            'minimum_checked_bound':str(min(accepted)) if len(accepted)==len(required) and accepted else None,
            'trusted_base':trusted, 'scope':'All required supplied LP obligations checked; NOT full-network independent reproof or deployed floating-point proof.'}


def check_directory(root, *, expected_request_id=None):
    root = Path(root).resolve(); manifest = json.loads((root/'manifest.json').read_text())
    if expected_request_id is not None and manifest['request_id'] != expected_request_id:
        raise ValueError('frozen request identity mismatch')
    def read(ref):
        path = (root/ref['file']).resolve()
        if not path.is_relative_to(root): raise ValueError('artifact escapes request directory')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref['sha256']: raise ValueError('artifact hash mismatch')
        return json.loads(raw)
    return aggregate(manifest,read)
