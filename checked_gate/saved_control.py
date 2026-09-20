"""Fixed saved-request diagnostic: no optimizer, no new model/LP search.

Rechecks the old complete obligation inventory, tightens one gate using exact
range evidence, and re-evaluates the SAME dual multipliers on the new LP.
This is not a fresh-generation or high-accuracy experiment.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT/'act/pipeline/moe/results/request_lp_act_only_review_20260915_r1.json'
RAW = ROOT/'data/moe/results/request_lp_act_only_20260915_r1'
CASE = 'seed1_4018'
KEY = 's3_5_p1_rational'


def checked_read(relative, archived):
    path = RAW / relative
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != archived[relative]:
        raise ValueError('archived identity mismatch: ' + relative)
    return json.loads(raw)


def run(output):
    start = time.monotonic()
    if not sys.flags.no_site:
        raise ValueError('use python -S')
    output.mkdir(exist_ok=False)
    # Only pure checker/builder modules; ACT's eager Torch initializers are not
    # used. No model, numerical package, solver, or external process allowed.
    def guard(event, args):
        if event == 'import' and args[0].split('.')[0] in {'torch','numpy','scipy','highspy','gurobipy'}:
            raise ImportError('numerical/model import forbidden')
        if event.startswith(('subprocess.', 'socket.')) or event in {'os.system','os.fork','os.exec'}:
            raise PermissionError('external execution forbidden')
    sys.addaudithook(guard)
    for name in ('act','act.back_end','act.back_end.solver','act.pipeline','act.pipeline.moe'):
        if name in sys.modules:
            raise ValueError('fresh process required')
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT.joinpath(*name.split('.')))]
        sys.modules[name] = module
    from checked_gate.propose import propose
    from checked_gate.checker import check
    from act.pipeline.moe.check_request_lp import check_directory
    from act.back_end.solver.rational_mccormick import build
    from act.back_end.solver.check_rational_mccormick import check_construction
    from act.back_end.solver.sparse_lp_certificate import evaluate
    from act.back_end.solver.lp_certificate import identity

    review = json.loads(REVIEW.read_text())
    if review['status'] != 'PASS' or review['issues']:
        raise ValueError('parent audit failure')
    hashes = review['raw_hashes']
    # Freeze every consumed old dependency against the archived review, not
    # merely the potentially mutable manifest's self-declared hashes.
    consumed = {}
    for name, digest in hashes.items():
        if name.startswith(CASE+'/') and name.endswith('.json'):
            raw = (RAW/name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError('parent dependency drift: '+name)
            consumed[name] = digest
    manifest = checked_read(CASE+'/manifest.json', hashes)
    rid = manifest['request_id']
    baseline = check_directory(RAW/CASE, expected_request_id=rid)
    if baseline['required'] != 27 or baseline['counts'] != {'reused':25,'residual':1,'unknown':1}:
        raise ValueError('not the frozen 26/27 control')
    old_case = next(c for c in review['cases'] if c['model']=='seed1' and c['dataset_index']==4018)
    blocking = [o for o in old_case['obligations'] if not o['positive']]
    if len(blocking)!=1 or blocking[0]['pair']!=[3,5] or blocking[0]['property_index']!=1:
        raise ValueError('blocking obligation changed')
    row = next(r for r in manifest['obligations'] if r['pair']==[3,5] and r['property_index']==1)
    if row['source'] != KEY:
        raise ValueError('weighted source mismatch')
    proofs = manifest['proofs']
    lo, neg_hi = (F(proofs[row[k]]['checked_lower_bound']) for k in ('gate_lower','gate_upper'))
    context = {'request_id': rid, 'ordered_pair':[3,5],
               'margin_lower_proof': proofs[row['gate_lower']]['certificate']['sha256'],
               'margin_negative_upper_proof': proofs[row['gate_upper']]['certificate']['sha256']}
    margin = [str(lo), str(-neg_hi)]
    gate_proof = propose(context, margin)
    gate_check = check(gate_proof, expected_context=context, expected_margin=margin)
    old = checked_read(CASE+'/'+proofs[KEY]['export']['file'], hashes)
    dual = checked_read(CASE+'/'+proofs[KEY]['certificate']['file'], hashes)
    record = build(old['source'], old['q'], old['offset'], gate_check['gate'], old['difference'])
    new_dual = {**dual, 'lp_sha256': identity(record['lp'])}
    bound, _ = evaluate(record['lp'], new_dual)
    new_dual['claimed_lower_bound'] = str(bound)
    verified = check_construction(record, new_dual, source_hash=old['source_sha256'],
        q=old['q'], offset=old['offset'], gate=gate_check['gate'], difference=old['difference'])
    if F(verified['bound']['checked_lower_bound']) != bound:
        raise ValueError('bound check mismatch')
    files = {'gate_proof.json':gate_proof, 'weighted_export.json':record,
             'weighted_certificate.json':new_dual}
    written = {}
    for name, value in files.items():
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        (output/name).write_bytes(raw)
        written[name] = {'sha256':hashlib.sha256(raw).hexdigest(), 'bytes':len(raw)}
    result = {'schema':'CHECKED_GATE_SAVED_CONTROL_V1',
        'scope':'Postselected supplied-HZ fixed-dual diagnostic; not production timing, high-accuracy or end-to-end strict proof.',
        'parent_review_sha256':hashlib.sha256(REVIEW.read_bytes()).hexdigest(),
        'request_id':rid, 'pair':[3,5], 'property_index':1,
        'baseline':baseline, 'margin':margin, 'old_gate':old['gate'], 'new_gate':gate_check['gate'],
        'old_lower_bound':dual['claimed_lower_bound'], 'new_lower_bound':str(bound),
        'same_dual_multipliers':True, 'candidate_search_calls':0, 'model_calls':0,
        'new_obligation_positive':bound > F.from_float(1e-7),
        'complete_strict_network_certificate':False, 'production_verdict_changed':False,
        'artifacts':written, 'parent_dependencies':consumed,
        'elapsed_seconds':time.monotonic()-start,
        'remaining_trusted_base':baseline['trusted_base']}
    (output/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'parent_dependencies','artifacts'}},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args=parser.parse_args()
    path=args.output.resolve()
    if not path.is_relative_to(Path('/data1/Kane/MOE')):
        raise ValueError('output outside project write scope')
    run(path)
