"""Immutable identities for two sign-certificate feasibility controls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/moe/results/conv_three_arm_full_20260915_v2'
DEFAULT = ROOT / 'data/moe/results/conv_sign_lp_20260915_r1'
PROTOCOL = ROOT / 'scripts/conv_sign_lp_protocol.json'
FREEZE = ROOT / 'act/pipeline/moe/results/conv_sign_lp_freeze_20260915_r1.json'
PARENT = ROOT / 'act/pipeline/moe/results/conv_full_v2_review_20260915.json'
ANALYSIS = ROOT / 'act/pipeline/moe/results/conv_full_v2_obligations_20260915.json'
FILES = ('scripts/conv_sign_lp_protocol.json', 'scripts/conv_sign_lp_contract.py',
         'scripts/run_conv_sign_lp.py', 'scripts/check_conv_sign_lp.py',
         'scripts/test_conv_sign_lp.py', 'docs/conv_sign_lp_r1.md',
         'scripts/budget_contract_v2.py', 'scripts/f0_timing_trace.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def source_identity():
    paths = sorted(set(ROOT / p for p in FILES) | set((ROOT / 'act').rglob('*.py')))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def parents():
    p = read(PROTOCOL)
    if sha(PARENT) != p['parent_archive_sha256'] or sha(ANALYSIS) != p['parent_analysis_sha256']:
        raise ValueError('parent archive changed')
    inventory = read(PARENT)['artifact_inventory']
    if {str(f.relative_to(RAW)) for f in RAW.rglob('*') if f.is_file()} != {v['path'] for v in inventory}:
        raise ValueError('raw inventory changed')
    for v in inventory:
        if sha(RAW / v['path']) != v['sha256']: raise ValueError('raw artifact changed')
    return p


def jobs():
    p = parents(); rows = {r['job_id']: r for r in read(ANALYSIS)['rows']}
    out = []
    for case in p['cases']:
        reqpath = RAW / case['job_id'] / 'request.json'; req = read(reqpath)
        query = rows[case['job_id']]['property_queries'][0]
        q = [1 if i == req['sample']['label'] else -1 if i == case['competitor'] else 0 for i in range(10)]
        scope = {'pairs': [case['pair']], 'row': q, 'constant': 0}
        if (req['sample']['dataset_index'] != case['dataset_index'] or query['scope'] != scope
                or not query['positive_unaccepted_dual']):
            raise ValueError('frozen first-property control mismatch')
        out.append({'case': case, 'parent_request': req, 'parent_request_sha256': sha(reqpath),
                    'expected_scope': scope, 'parent_query_token': query['token'],
                    'historical_diagnostic_dual': query['diagnostic_dual']})
    return out


def verify_freeze():
    f = read(FREEZE)
    if f['protocol'] != parents() or f['sources'] != source_identity() or f['jobs'] != jobs():
        raise ValueError('sign-control freeze drift')
    return f


def validate_job(directory):
    directory = Path(directory).resolve(); freeze = verify_freeze()
    if directory.parent != DEFAULT:
        raise ValueError('worker outside frozen evidence root')
    expected = next((j for j in freeze['jobs'] if j['case']['job_id'] == directory.name), None)
    if expected is None: raise ValueError('unregistered control')
    job = read(directory / 'job.json')
    if job != {**expected, 'protocol':freeze['protocol'], 'freeze_sha256':sha(FREEZE)}:
        raise ValueError('worker job/selection identity drift')
    req = job['parent_request']
    if sha(req['config']['path']) != req['config']['sha256']:
        raise ValueError('original method config changed')
    return job
