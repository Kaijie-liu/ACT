"""Archive synthetic source-construction controls; never load a real request."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'docs/scoped_source_controls_20260924_r1.json'
SUITES = ['scoped_source.tests', 'source_enclosure.tests', 'source_enclosure.portable_tests', 'full_source.tests']


def identities():
    files = [Path(__file__).resolve()]
    for folder in ('scoped_source', 'source_enclosure', 'full_source', 'router_source', 'upstream_source'):
        files.extend(sorted((ROOT/folder).glob('*.py')))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true'); args = parser.parse_args()
    if args.check:
        doc = json.loads(OUTPUT.read_text())
        if doc['source_hashes'] != identities() or doc['returncode'] != 0 or doc['tests_passed'] != 35:
            raise ValueError('control artifact identity/status drift')
        print('PASS: source construction control archive identity (not a real proof)'); return
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    started = time.monotonic(); command = [sys.executable, '-m', 'unittest', *SUITES, '-v']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
    output = result.stdout + result.stderr
    counts = re.search(r'Ran (\d+) tests', output)
    doc = {'status': 'PASS' if result.returncode == 0 else 'FAIL', 'command': command,
        'returncode': result.returncode, 'tests_passed': int(counts.group(1)) if counts and result.returncode == 0 else 0,
        'seconds': time.monotonic()-started, 'output': output, 'source_hashes': identities(),
        'new_generic_controls': 17, 'reused_rule_regressions': 18,
        'scope': 'synthetic declared Linear/ReLU/Flatten top2 source and all output LP constructions',
        'real_model_requests': 0, 'real_bound_queries': 0, 'positive_certificates': 0,
        'real_execution_gate': 'NOT_FROZEN: outer resource supervision and new-bound acceptance/aggregation still required'}
    with OUTPUT.open('x') as handle: json.dump(doc, handle, indent=2, sort_keys=True); handle.write('\n')
    print(json.dumps({k: doc[k] for k in ('status', 'tests_passed', 'seconds', 'real_model_requests')}))
    if result.returncode: raise SystemExit(result.returncode)


if __name__ == '__main__': main()
