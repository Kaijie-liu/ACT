"""Independent saved-control recheck; no producer, optimizer, or model import."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]


def read(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('hash mismatch: '+str(path))
    return json.loads(raw)


def review(directory, result_hash):
    started = time.monotonic()
    if not sys.flags.no_site:
        raise ValueError('use python -S')
    def guard(event, args):
        if event == 'import' and args[0].split('.')[0] in {
            'torch','numpy','scipy','highspy','gurobipy','decimal'}:
            raise ImportError('forbidden checker dependency')
        if event.startswith(('subprocess.', 'socket.')) or event in {'os.system','os.fork','os.exec'}:
            raise PermissionError('external execution forbidden')
    sys.addaudithook(guard)
    for name in ('act','act.back_end','act.back_end.solver','act.pipeline','act.pipeline.moe'):
        if name in sys.modules:
            raise ValueError('fresh process required')
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT.joinpath(*name.split('.')))]
        sys.modules[name] = module
    from checked_gate.checker import check
    from act.pipeline.moe.check_request_lp import check_directory
    from act.back_end.solver.check_hz_lp_export import check_export
    from act.back_end.solver.check_rational_mccormick import check_construction

    result = read(directory/'result.json', result_hash)
    parent = read(ROOT/'act/pipeline/moe/results/request_lp_act_only_review_20260915_r1.json',
                  result['parent_review_sha256'])
    if parent['status'] != 'PASS' or parent['issues']:
        raise ValueError('parent status')
    raw = ROOT/'data/moe/results/request_lp_act_only_20260915_r1'
    def old(name):
        return read(raw/name, parent['raw_hashes'][name])
    expected = {k:v for k,v in parent['raw_hashes'].items()
                if k.startswith('seed1_4018/') and k.endswith('.json')}
    if result['parent_dependencies'] != expected:
        raise ValueError('dependency inventory mismatch')
    for name, digest in expected.items():
        read(raw/name, digest)
    m = old('seed1_4018/manifest.json')
    if result['request_id'] != m['request_id'] or result['pair'] != [3,5] or result['property_index'] != 1:
        raise ValueError('request binding')
    original = check_directory(raw/'seed1_4018', expected_request_id=m['request_id'])
    if original != result['baseline']:
        raise ValueError('baseline mismatch')
    if original['counts'] != {'reused':25, 'residual':1, 'unknown':1} or original['required'] != 27:
        raise ValueError('frozen obligation inventory')
    row = next(r for r in m['obligations'] if r['pair']==[3,5] and r['property_index']==1)
    def source(key, kind):
        ref = m['proofs'][key][kind]
        return old('seed1_4018/'+ref['file'])
    endpoints = []
    for key in (row['gate_lower'], row['gate_upper']):
        r = m['proofs'][key]
        checked = check_export(source(key,'export'), source(key,'certificate'),
                               expected_source_sha256=r['hz_sha256'])
        endpoints.append(F(checked['bound']['checked_lower_bound']))
    margin = [str(endpoints[0]), str(-endpoints[1])]
    if result['margin'] != margin:
        raise ValueError('fresh router bound differs from supplied margin')
    context = {'request_id':m['request_id'], 'ordered_pair':[3,5],
        'margin_lower_proof':m['proofs'][row['gate_lower']]['certificate']['sha256'],
        'margin_negative_upper_proof':m['proofs'][row['gate_upper']]['certificate']['sha256']}
    if set(result['artifacts']) != {'gate_proof.json','weighted_export.json','weighted_certificate.json'}:
        raise ValueError('new dependency inventory')
    def new(name):
        return read(directory/name, result['artifacts'][name]['sha256'])
    gate = check(new('gate_proof.json'), expected_context=context, expected_margin=margin)['gate']
    old_weighted = source(row['source'], 'export')
    old_dual = source(row['source'], 'certificate')
    dual = new('weighted_certificate.json')
    if any(old_dual[k] != dual[k] for k in ('inequality_dual','equality_dual')):
        raise ValueError('not fixed multipliers')
    checked = check_construction(new('weighted_export.json'), dual,
        source_hash=old_weighted['source_sha256'], q=old_weighted['q'],
        offset=old_weighted['offset'], gate=gate, difference=old_weighted['difference'])
    bound = F(checked['bound']['checked_lower_bound'])
    if (gate != result['new_gate'] or str(bound) != result['new_lower_bound'] or
        old_dual['claimed_lower_bound'] != result['old_lower_bound'] or
        (bound > F.from_float(1e-7)) != result['new_obligation_positive'] or
        result['complete_strict_network_certificate'] is not False or
        result['production_verdict_changed'] is not False):
        raise ValueError('result overclaim or mismatch')
    return {'status':'PASS', 'issues':[], 'result_sha256':result_hash,
        'fresh_router_certificates_checked':2, 'original_required_obligations':27,
        'original_positive_obligations':26, 'new_obligation_positive':bound > F.from_float(1e-7),
        'new_bound':str(bound), 'new_gate':gate,
        'fixed_duals_verified':True, 'production_verdict_changed':False,
        'scope':'Saved-source local diagnostic; not end-to-end strict/high-accuracy/network SAFE.',
        'review_seconds':time.monotonic()-started}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path); p.add_argument('--result-hash', required=True)
    args=p.parse_args()
    print(json.dumps(review(args.directory.resolve(), args.result_hash), indent=2))
