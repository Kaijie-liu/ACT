"""Archive finite HybridZ integration controls and recheck stored support only."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'configs/hz_checked_propagation_20261001.json'
PROTOCOL_SHA = 'd61ffac532ca2480e74b4845daf2e6a70f26ec482a880466c1a5313e5b86a36a'
FILES = [PROTOCOL, 'docs/hz_checked_propagation_design_20261001.md',
         'act/back_end/moe/checked_propagation.py', 'act/back_end/moe/test_checked_propagation.py',
         'scripts/run_hz_checked_propagation_controls.py',
         'act/back_end/moe/batched_support.py', 'act/back_end/moe/check_batched_support.py',
         'act/back_end/hybridz_tf/hybridz_tf.py', 'act/back_end/hybridz_tf/tf_mlp.py',
         'act/back_end/moe/hz_routing.py', 'act/back_end/solver/hz_lp_export.py',
         'act/back_end/solver/check_hz_lp_export.py', 'scoped_source/rowwise_bound.py',
         'scoped_source/rowwise_native.py', 'act/back_end/analyze.py', 'act/config/config.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, sort_keys=True, separators=(',', ':'), allow_nan=False)
        f.write('\n')


def read(path):
    return json.loads(path.read_text())


def names():
    if sha(ROOT/PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('frozen protocol changed')
    return sorted('act.back_end.moe.test_checked_propagation.CheckedPropagationTests.test_'+s
                  for s in read(ROOT/PROTOCOL)['controls'])


def validate_query_roster(batch, event, propagation):
    from scoped_source.rowwise_bound import identity
    expected = [(f"{event['layer_id']}:{row}:{side}", side)
                for row in event['selected'] for side in ('min', 'max')]
    if ([(q['id'], q['side']) for q in batch['queries']] != expected
            or batch['context']['layer'] != event['layer_id']
            or batch['context']['caller_scope'] != propagation['scope_sha256']
            or batch['context']['guard'] != identity(propagation['scope']['entry'])
            or batch['source_sha256'] != event['preactivation_sha256']
            or batch['context']['request'] != propagation['scope']['request']
            or batch['context']['domain'] != identity(propagation['scope']['source'])):
        raise ValueError('layer/guard query binding')
    for q, row in zip(batch['queries'], [row for row in event['selected'] for _ in (0, 1)]):
        if ([Fraction(v) for v in q['q']] != [int(j == row) for j in range(len(batch['source']['c']))]
                or Fraction(q['offset']) != 0):
            raise ValueError('wrong preactivation property')


def run(dest):
    required = names()
    if not dest.is_relative_to(ROOT.parent/'baseline_runs'):
        raise ValueError('new baseline_runs directory required')
    dest.mkdir(parents=True, exist_ok=False)
    bindings = {}
    for relative in FILES:
        target = dest/'implementation'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
        bindings[relative] = sha(target)
    save(dest/'implementation.json', bindings)
    from act.back_end.moe.test_checked_propagation import CheckedPropagationTests
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CheckedPropagationTests)
    if [t.id() for t in suite] != required:
        raise ValueError('frozen test inventory')
    start = time.monotonic()
    with (dest/'tests.log').open('x') as log:
        class Result(unittest.TextTestResult):
            outcomes = []
            def startTest(self, t):
                self.begin = time.monotonic(); self.status = 'PASS'
                super().startTest(t)
            def addError(self, t, err):
                self.status = 'ERROR'; super().addError(t, err)
            def addFailure(self, t, err):
                self.status = 'FAIL'; super().addFailure(t, err)
            def addSkip(self, t, reason):
                self.status = 'SKIP'; super().addSkip(t, reason)
            def addExpectedFailure(self, t, err):
                self.status = 'EXPECTED_FAILURE'; super().addExpectedFailure(t, err)
            def addUnexpectedSuccess(self, t):
                self.status = 'UNEXPECTED_SUCCESS'; super().addUnexpectedSuccess(t)
            def stopTest(self, t):
                self.outcomes.append({'test': t.id(), 'status': self.status,
                                      'seconds': time.monotonic()-self.begin})
                super().stopTest(t)
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=Result).run(suite)
    save(dest/'observations.json', CheckedPropagationTests.observations)
    summary = {'status': 'PASS' if result.wasSuccessful() and all(r['status']=='PASS' for r in result.outcomes) else 'FAIL', 'tests': result.testsRun,
               'outcomes': result.outcomes, 'head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
               'protocol_sha256': PROTOCOL_SHA, 'implementation_sha256': sha(dest/'implementation.json'),
               'observations_sha256': sha(dest/'observations.json'), 'log_sha256': sha(dest/'tests.log'),
               'seconds': time.monotonic()-start, 'native_solves': 0, 'gpu_executions': 0,
               'real_requests': 0, 'hard_budget_supervision': False, 'complete_moe_proofs': 0}
    save(dest/'summary.json', summary)
    return summary


def audit(dest):
    summary, bindings = read(dest/'summary.json'), read(dest/'implementation.json')
    if (set(bindings) != set(FILES) or summary['protocol_sha256'] != PROTOCOL_SHA
            or sha(dest/'implementation.json') != summary['implementation_sha256']
            or [r['test'] for r in summary['outcomes']] != names() or summary['tests'] != 14):
        raise ValueError('control/source inventory')
    if (summary['status'] != 'PASS' or any(r['status'] != 'PASS' for r in summary['outcomes'])):
        raise ValueError('all frozen controls must pass without skips')
    for relative, expected in bindings.items():
        if sha(dest/'implementation'/relative) != expected:
            raise ValueError('saved implementation changed')
    for relative in ('act/back_end/moe/check_batched_support.py', 'scoped_source/rowwise_bound.py',
                     'act/back_end/solver/check_hz_lp_export.py'):
        if sha(ROOT/relative) != bindings[relative]:
            raise ValueError('checker version differs from execution')
    for name in ('observations',):
        if sha(dest/(name+'.json')) != summary[name+'_sha256']:
            raise ValueError('saved evidence changed')
    if sha(dest/'tests.log') != summary['log_sha256']:
        raise ValueError('saved control log changed')
    if any(summary[k] != 0 for k in ('native_solves', 'gpu_executions', 'real_requests', 'complete_moe_proofs')):
        raise ValueError('control scope changed')
    from act.back_end.moe.check_batched_support import check_batch
    from scoped_source.rowwise_bound import identity
    checked, queries = 0, 0
    obs = read(dest/'observations.json')
    required = {'retained_guard', 'negative_relu', 'disabled', 'guard_discarded',
                'two_layers', 'five_rows', 'capacity_fallback', 'candidate_failure'}
    if set(obs) != required:
        raise ValueError('mandatory observation inventory')
    signatures = {
        'retained_guard': (['CHECKED_SUPPORT_APPLIED'], [2], 0),
        'negative_relu': (['CHECKED_SUPPORT_APPLIED'], [2], 0),
        'disabled': ([], [], 1),
        'guard_discarded': (['NO_CONSTRAINTS'], [0], 1),
        'two_layers': (['CHECKED_SUPPORT_APPLIED']*2, [4, 4], None),
        'five_rows': (['CHECKED_SUPPORT_APPLIED'], [8], 4),
        'capacity_fallback': (['CAPACITY_NATIVE_FALLBACK'], [0], 129),
        'candidate_failure': (['CANDIDATE_REJECTED_NATIVE_FALLBACK'], [0], 1)}
    for name, p in obs.items():
        statuses, counts, n_bin = signatures[name]
        if ([e['status'] for e in p['events']] != statuses
                or [0 if e['package'] is None else len(e['package']['accepted']['results']) for e in p['events']] != counts
                or (n_bin is not None and p['output']['Gb']['shape'][1] != n_bin)):
            raise ValueError('mandatory propagation/proof signature changed')
        if (identity(p['scope']) != p['scope_sha256'] or p['network_or_complete_moe_proof']
                or p['hard_budget_supervision']):
            raise ValueError('propagation scope changed')
        if p['total_seconds'] < p['construction_seconds']+sum(e['total_seconds'] for e in p['events']):
            raise ValueError('nested costs inconsistent')
        for e in p['events']:
            if not e['completed'] or e['scope_sha256'] != p['scope_sha256']:
                raise ValueError('incomplete or foreign propagation event')
            package = e['package']
            if package is None:
                if e['status'] == 'CHECKED_SUPPORT_APPLIED':
                    raise ValueError('missing applied support proof')
                continue
            if e['status'] != 'CHECKED_SUPPORT_APPLIED':
                raise ValueError('proof contradicts execution status')
            batch = package['batch']
            validate_query_roster(batch, e, p)
            accepted = check_batch(batch, package['candidates'], expected_batch_sha256=identity(batch), deadline=time.monotonic()+300)
            if accepted != package['accepted']:
                raise ValueError('saved exact support changed')
            checked += 1; queries += len(accepted['results'])
    outcome = {'status': 'PASS', 'execution_status': summary['status'], 'batches_rechecked': checked,
               'bounds_rechecked': queries, 'summary_sha256': sha(dest/'summary.json'),
               'scope': 'structural archive plus given-HZ bounds; not independent network lowering',
               'complete_moe_proofs': 0}
    return outcome


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'audit']); parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = run(args.directory.resolve()) if args.action == 'run' else audit(args.directory.resolve())
    print(json.dumps(result, sort_keys=True))
    sys.exit(0 if result['status'] == 'PASS' else 1)
