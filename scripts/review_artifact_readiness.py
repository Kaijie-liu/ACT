"""Inventory/copy/check an EXISTING conditional bundle; no proof generation.

Writes only a new retained directory under /data1/Kane/MOE and a new receipt.
This is a local relocation check, not an empirical clean-install/release test.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '282e0507b94ac2f6105096e43ea7c0f7a0d38bc5'
REVIEW = 'docs/portable_conv_proof_v1_review.json'
OUTPUT = ROOT/'docs/reviewer_artifact_readiness_20260921.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(value, why):
    if not value:
        raise ValueError(why)


def inventory(directory, old):
    raw = (directory/'bundle.json').read_bytes()
    require(sha(raw) == old['bundle_sha256'], 'old bundle identity mismatch')
    meta = json.loads(raw)
    records = []
    for name, digest in dict(meta['files'], **{'bundle.json': sha(raw)}).items():
        p = (directory/name).resolve()
        require(p.is_relative_to(directory.resolve()) and p.is_file(), 'outside/missing bundle file')
        data = p.read_bytes()
        require(sha(data) == digest, 'bundle file identity mismatch')
        records.append({'path': name, 'sha256': digest, 'bytes': len(data)})
    require(sum(r['bytes'] for r in records) == old['bundle_bytes'], 'bundle byte count mismatch')
    return meta, records


def run():
    require(not OUTPUT.exists(), 'receipt exists; never overwrite an earlier check')
    old_bytes = (ROOT/REVIEW).read_bytes()
    require(old_bytes == subprocess.check_output(['git', 'show', BASELINE+':'+REVIEW], cwd=ROOT),
            'old review changed')
    old = json.loads(old_bytes)
    source = Path(old['raw_root'])/'relocated'
    meta, files = inventory(source, old)
    scratch = Path(tempfile.mkdtemp(prefix='review-artifact-20260921-', dir='/data1/Kane/MOE'))
    start = time.monotonic()
    target = scratch/'bundle'
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    copy_seconds = time.monotonic()-start
    copied_meta, copied_files = inventory(target, old)
    require(copied_meta == meta and copied_files == files, 'copied bundle drift')
    command = [sys.executable, '-I', '-S', str(target/'verify.py'),
               '--bundle-hash', old['bundle_sha256'], '--statement-hash', old['statement_sha256']]
    started = time.monotonic()
    try:
        p = subprocess.run(command, cwd=scratch, env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'},
                           text=True, capture_output=True, timeout=180)
        terminal = {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr,
                    'timeout': False}
    except subprocess.TimeoutExpired as exc:
        terminal = {'returncode': None, 'stdout': str(exc.stdout or ''), 'stderr': str(exc.stderr or ''),
                    'timeout': True}
    wall = time.monotonic()-started
    terminal['wall_seconds'] = wall
    terminal['command'] = command
    terminal_path = scratch/'terminal.json'
    terminal_path.write_text(json.dumps(terminal, indent=2)+'\n')
    require(not terminal['timeout'] and terminal['returncode'] == 0,
            'relocation check failed; terminal and copied bundle retained')
    result = json.loads(terminal['stdout'])
    require(result['result'] == meta['expected_result'], 'complete conditional result changed')
    require(result['isolated'] and result['site_disabled'] and not result['solver_imported'],
            'checker isolation flags')
    receipt = {'schema': 'REVIEWER_ARTIFACT_READINESS_V1', 'status': 'LOCAL_RELOCATION_CHECK_PASSED',
               'starting_head': BASELINE, 'source_review': {'path': REVIEW, 'sha256': sha(old_bytes)},
               'bundle_sha256': old['bundle_sha256'], 'statement_sha256': old['statement_sha256'],
               'files': files, 'bundle_bytes': sum(r['bytes'] for r in files),
               'scratch_directory': str(scratch), 'terminal_sha256': sha(terminal_path.read_bytes()),
               'copy_seconds': copy_seconds, 'check_process_wall_seconds': wall,
               'complete_result_matches_old': True, 'positive_obligations': old['positive_obligations'],
               'required_obligations': old['required_obligations'],
               'minimum_lower_bound': result['result']['minimum_lower_bound'],
               'trusted_upstream_lowering': True, 'deployed_float_SAFE': False,
               'source_complete_positive_proof': False, 'new_solver_calls': 0,
               'new_model_forwards': 0, 'new_source_propagations': 0, 'proof_regenerated': False,
               'input98_followup': 'STOP_INPUT98_FOLLOWUP', 'external_release_performed': False,
               'clean_environment_install_tested': False, 'weights_or_dataset_distributed': False,
               'negative_controls': 'Not rerun here; four previous rehashed semantic/content rejection controls are bound by source_review.',
               'remaining': ['PI-approved reviewer access/distribution',
                             'Clean environment empirical installation and real verification rerun',
                             'Checkpoint/data availability and redistribution permissions']}
    with OUTPUT.open('x') as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write('\n')
    return receipt


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, sort_keys=True))
