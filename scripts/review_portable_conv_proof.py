"""Build/copy/check input98 once, then reject four mutations; no solves."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from portable_proof.runtime import Store, compact, digest, original_bytes
from scripts.build_portable_conv_proof import ROOT, SOURCE, build


def save(path, value):
    path.write_bytes(original_bytes(value))


def check(directory, meta, statement_sha, target):
    start = time.monotonic()
    proc = subprocess.run([sys.executable, '-I', '-S', str(directory / 'verify.py'),
                           '--bundle-hash', digest(original_bytes(meta)),
                           '--statement-hash', statement_sha],
                          cwd=directory.parent, env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'},
                          text=True, capture_output=True, timeout=180)
    result = {'returncode': proc.returncode, 'wall_seconds': time.monotonic() - start,
              'stdout': proc.stdout, 'stderr': proc.stderr}
    save(target, result)
    return result


def run(root):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = build(root / 'packed')
    # New location is outside ACT; no dependency on a sibling checkout.
    relocated = root / 'relocated'
    start = time.monotonic()
    shutil.copytree(root / 'packed', relocated)
    report['copy_seconds'] = time.monotonic() - start
    meta = json.loads((relocated / 'bundle.json').read_bytes())
    good = check(relocated, meta, report['statement_sha256'], root / 'positive.json')
    if good['returncode']:
        raise RuntimeError('relocated check failed; preserved positive.json')
    report['positive'] = json.loads(good['stdout'])
    report['check_process_wall_seconds'] = good['wall_seconds']
    report['matches_original_independent_result'] = report['positive']['result'] == json.loads((SOURCE / 'independent.json').read_bytes())
    negatives = []
    for kind in ('missing_obligation', 'changed_source', 'changed_property', 'damaged_archive'):
        dest = root / kind
        shutil.copytree(relocated, dest)
        m = copy.deepcopy(meta)
        if kind == 'damaged_archive':
            with (dest / 'evidence.zip').open('ab') as f:
                f.write(b'DAMAGED')
        else:
            store = Store(dest / 'evidence.zip')
            logical = m['logical_files']['manifest.json']
            manifest = copy.deepcopy(store.decode(store.get(logical['root'])))
            store.zip.close()
            if kind == 'missing_obligation':
                manifest['obligations'].pop()
            elif kind == 'changed_source':
                manifest['joint_source']['sha256'] = '0' * 64
            else:
                manifest['obligations'][0]['q'][0] = 0
            raw = compact(manifest)
            sha = digest(raw)
            with zipfile.ZipFile(dest / 'evidence.zip', 'a', compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr(sha, raw)
            original_sha = digest(original_bytes(manifest))
            logical.update(root=sha, original_sha256=original_sha)
            m['manifest']['sha256'] = original_sha
            m['files']['evidence.zip'] = digest((dest / 'evidence.zip').read_bytes())
        save(dest / 'bundle.json', m)
        result = check(dest, m, report['statement_sha256'], root / (kind + '.json'))
        if result['returncode'] == 0:
            raise AssertionError('accepted mutation: ' + kind)
        negatives.append({'kind': kind, 'rejected': True, 'wall_seconds': result['wall_seconds'],
                          'reason': result['stderr'].strip().splitlines()[-1],
                          'transport_rehashed': kind != 'damaged_archive'})
    report['negative_controls'] = negatives
    report['execution_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    report['raw_root'] = str(root)
    report['status'] = 'PASS'
    save(root / 'REVIEW.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    if not args.root.resolve().is_relative_to(Path('/data1/Kane/MOE')) or args.root.resolve().is_relative_to(ROOT):
        raise ValueError('new authorized directory OUTSIDE the checkout required')
    print(json.dumps(run(args.root), indent=2, sort_keys=True))
