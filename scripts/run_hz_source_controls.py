"""Finite source-to-HZ controls; fresh proposals, solver-free saved recheck."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'configs/hz_source_connection_20261001.json'
PROTOCOL_SHA = 'd3a0b4b48d535bec3f7f9018797612c88523d52edd1dff0db105db86c2b17046'
FILES = [PROTOCOL, 'configs/h2_source_controls_20260930.json', 'docs/hz_source_connection_design_20261001.md',
         'scoped_source/hz_source_build.py', 'scoped_source/hz_source_check.py', 'scoped_source/test_hz_source.py',
         'scripts/run_hz_source_controls.py', 'scoped_source/graph.py', 'scoped_source/endpoint_source_controls.py',
         'scoped_source/sparse_controls.py', 'source_enclosure/format.py', 'source_enclosure/produce.py',
         'source_enclosure/check.py', 'upstream_source/checker.py', 'router_source/checker.py',
         'act/back_end/solver/solver_hz.py', 'act/back_end/hybridz_tf/tf_mlp.py',
         'act/back_end/solver/hz_lp_export.py', 'act/back_end/solver/check_hz_lp_export.py',
         'act/back_end/moe/weighted_top2.py', 'act/back_end/moe/hz_endpoints.py',
         'act/back_end/moe/check_hz_endpoints.py', 'act/back_end/moe/batched_support.py',
         'act/back_end/moe/check_batched_support.py', 'scoped_source/rowwise_bound.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, separators=(',', ':'), allow_nan=False)
        f.write('\n')


def protocol():
    if sha(ROOT/PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('frozen protocol identity changed')
    return load(ROOT/PROTOCOL)


def run(root):
    config = protocol()
    root.mkdir(parents=True, exist_ok=False)
    bindings = {}
    for name in FILES:
        dest = root/'implementation'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
        bindings[name] = sha(dest)
    save(root/'implementation.json', bindings)
    from scoped_source.test_hz_source import SourceConnectionTests
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SourceConnectionTests)
    expected = sorted('scoped_source.test_hz_source.SourceConnectionTests.test_'+n for n in config['controls'])
    if [t.id() for t in suite] != expected:
        raise ValueError('control inventory changed')
    start = time.monotonic()
    with (root/'tests.log').open('x') as log:
        class Result(unittest.TextTestResult):
            outcomes = []
            def startTest(self, t):
                self.started = time.monotonic(); self.state = 'PASS'; super().startTest(t)
            def addError(self, t, e):
                self.state = 'ERROR'; super().addError(t, e)
            def addFailure(self, t, e):
                self.state = 'FAIL'; super().addFailure(t, e)
            def addSkip(self, t, e):
                self.state = 'SKIP'; super().addSkip(t, e)
            def addExpectedFailure(self, t, e):
                self.state = 'EXPECTED_FAILURE'; super().addExpectedFailure(t, e)
            def addUnexpectedSuccess(self, t):
                self.state = 'UNEXPECTED_SUCCESS'; super().addUnexpectedSuccess(t)
            def stopTest(self, t):
                self.outcomes.append({'test': t.id(), 'status': self.state, 'seconds': time.monotonic()-self.started})
                super().stopTest(t)
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=Result).run(suite)
    save(root/'observations.json', SourceConnectionTests.observations)
    summary = {'status': 'PASS' if result.wasSuccessful() and len(result.outcomes) == len(expected)
               and all(row['status'] == 'PASS' for row in result.outcomes) else 'FAIL',
               'tests': result.testsRun, 'outcomes': result.outcomes, 'protocol_sha256': PROTOCOL_SHA,
               'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'implementation_sha256': sha(root/'implementation.json'),
               'observations_sha256': sha(root/'observations.json'), 'tests_log_sha256': sha(root/'tests.log'),
               'seconds': time.monotonic()-start, 'native_solves': 0, 'cuda_calls': 0, 'real_requests': 0,
               'hard_budget_supervision': False, 'performance_claim': False,
               'exceptional_tests': {k: len(getattr(result, k)) for k in
                                     ('errors', 'failures', 'skipped', 'expectedFailures', 'unexpectedSuccesses')}}
    save(root/'summary.json', summary)
    return summary


def check_summary(summary, config):
    expected = sorted('scoped_source.test_hz_source.SourceConnectionTests.test_'+n for n in config['controls'])
    if (summary['status'] != 'PASS' or summary['protocol_sha256'] != PROTOCOL_SHA
            or summary['tests'] != len(expected) or [r['test'] for r in summary['outcomes']] != expected
            or any(r['status'] != 'PASS' for r in summary['outcomes'])
            or summary['exceptional_tests'] != {k: 0 for k in
                                               ('errors', 'failures', 'skipped', 'expectedFailures', 'unexpectedSuccesses')}):
        raise ValueError('complete non-exceptional control outcomes required')
    for key in ('native_solves', 'cuda_calls', 'real_requests'):
        if type(summary[key]) is not int or summary[key] != 0:
            raise ValueError('execution scope counter')
    for key in ('hard_budget_supervision', 'performance_claim'):
        if summary[key] is not False:
            raise ValueError('execution scope flag')


def check_coverage(name, item, checked, observations):
    if name == 'partial_proof':
        expected = deepcopy(observations['tied_partial_reuse']['package'])
        expected['proof']['pairs'][-1]['candidates'] = None
        if (item['package'] != expected or checked['status'] != 'UNKNOWN_MISSING_EVIDENCE'
                or (checked['required'], checked['checked_endpoints'], checked['missing_endpoints']) != (18, 15, 3)):
            raise ValueError('partial control must omit exactly the final pair')
    else:
        expected = sum(len(p['batch']['queries']) for p in item['package']['endpoint_request']['pairs'])
        if checked['missing_endpoints'] != 0 or checked['checked_endpoints'] != expected:
            raise ValueError('normal source requires complete fresh endpoint evidence')


def audit(root):
    from source_enclosure.format import identity
    from scoped_source.endpoint_source_controls import cases
    from scoped_source.sparse_controls import exact_routes
    from scoped_source.hz_source_check import check
    from fractions import Fraction as F
    config = protocol()
    summary = load(root/'summary.json')
    bindings = load(root/'implementation.json')
    if set(bindings) != set(FILES):
        raise ValueError('implementation inventory')
    for path, digest in bindings.items():
        if sha(root/'implementation'/path) != digest or sha(ROOT/path) != digest:
            raise ValueError('saved/checker execution identity differs')
    for field, file in [('implementation_sha256', 'implementation.json'), ('observations_sha256', 'observations.json'),
                        ('tests_log_sha256', 'tests.log')]:
        if summary[field] != sha(root/file):
            raise ValueError('archive anchor')
    check_summary(summary, config)
    obs = load(root/'observations.json')
    if set(obs) != set(config['cases']) | {'partial_proof'}:
        raise ValueError('all fixed source cases and partial evidence required')
    sources = {n: d for n, d, _ in cases()}
    out = []
    for name in config['cases']+['partial_proof']:
        item = obs[name]
        source = sources['tied_partial_reuse' if name == 'partial_proof' else name]
        if item['source'] != source:
            raise ValueError('fixed source modified')
        checked = check(source, item['package'], expected_source_sha256=identity(source), deadline=time.monotonic()+300)
        if checked != item['checked']:
            raise ValueError('saved source/endpoint result differs')
        check_coverage(name, item, checked, obs)
        if name != 'partial_proof':
            cost = item['cost_seconds']
            if (item['source_sha256'] != identity(source)
                    or set(cost) != {'construction', 'proposals', 'serialization', 'check', 'total'}
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in cost.values())
                    or cost['total'] > 300 or sum(v for k, v in cost.items() if k != 'total') > cost['total']+1e-6):
                raise ValueError('source/cooperative cost inventory')
            encoded = json.dumps(item['package'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            if len(encoded) != item['package_bytes']:
                raise ValueError('package byte accounting')
        elif checked['status'] != 'UNKNOWN_MISSING_EVIDENCE':
            raise ValueError('partial cannot establish complete positivity')
        out.append({'case': name, 'source_sha256': identity(source), 'package_sha256': identity(item['package']),
                    'status': checked['status'], 'positive': checked['positive'], 'required': checked['required'],
                    'checked_endpoints': checked['checked_endpoints'], 'source_steps': checked['checked_source_steps'],
                    'affine_error_factors': checked['affine_error_factors'],
                    'minimum': min((r['lower_bound'] for r in checked['results'] if r['lower_bound'] is not None),
                                   key=F, default=None), 'cost_seconds': item.get('cost_seconds')})
    witnesses = [exact_routes(sources['weighted_sign'], [x]) for x in (-1, F(-1, 2), 1)]
    if obs['weighted_sign']['route_witnesses'] != witnesses:
        raise ValueError('route witness declaration binding')
    return {'status': 'PASS', 'issues': 0, 'protocol_sha256': PROTOCOL_SHA,
            'summary_sha256': sha(root/'summary.json'), 'tests': len(config['controls']), 'cases': out,
            'checked_packages': len(out), 'route_witnesses': witnesses, 'native_solves': 0,
            'cuda_calls': 0, 'real_requests': 0, 'deployed_float_SAFE': False,
            'hard_budget_supervision': False, 'performance_claim': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'audit'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.directory) if args.mode == 'run' else audit(args.directory)
    if args.output:
        save(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
