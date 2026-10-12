"""One frozen CPU row-control suite and producer-free saved-evidence audit.

Maximum three repair attempts, new directories r1/r2/r3; no automatic retries.
Each normal control includes creation/differential/live/check costs under 300 s.
Auxiliary controls have 30 s. Suite envelope 1800 s (cooperative, not a watchdog).
This is not a performance experiment, source integration or real-model admission.
"""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from scoped_proof.io import ROOT, load, save, sha
from scoped_source import hz_sparse_row_fixtures as fixtures
from scoped_source import check_hz_sparse_rows as checker

DESIGN = 'docs/hz_sparse_rows_design_20261002.md'
DESIGN_SHA = '53c18bf92537df3557aa387105074e1188c8a2957714e899b9c98de6a30a9cae'
FREEZE_HEAD = '910c0ab12a295bc8e61f2f622da117248cc95cfc'
FILES = (DESIGN, 'scoped_source/check_hz_sparse_rows.py', 'scoped_source/hz_sparse_rows.py',
         'scoped_source/hz_sparse_row_fixtures.py', 'scoped_source/test_hz_sparse_rows.py',
         'scripts/run_hz_sparse_rows.py', 'scoped_source/check_hz_binary64.py',
         'scoped_source/hz_binary64.py', 'scoped_source/test_hz_binary64.py',
         'source_enclosure/format.py', 'upstream_source/checker.py', 'router_source/checker.py',
         'scoped_source/graph.py', 'scoped_proof/io.py', 'act/back_end/solver/solver_hz.py',
         'act/back_end/solver/hz_lp_export.py')
LEGACY_SHA = {'scoped_source/test_hz_binary64.py': '457211351b142c74cdffbdeb1490ea22152e76b6b6a5e3b2875fcafbe0df835a',
              'scoped_source/hz_binary64.py': 'ac753309027be11964c449f4a4bd5421c74ece1f9d87184abc06195b8eb7c6d0',
              'scoped_source/check_hz_binary64.py': 'c52da0f3a31a675d2e5c020028bd14fc9ecb9e8f3bceb2a3f454cddd241551c9'}
TESTS = tuple(f'test_{i:02d}_{name}' for i, name in enumerate(
    ('legacy', 'wide', 'chunks', 'identity', 'binding_csr_layout', 'inward',
     'arithmetic_resources', 'deadline_mutation', 'independence_inventory'), 1))
FAULTS = ('producer_chunk', 'checker_chunk', 'producer_final_identity', 'checker_final_identity',
          'producer_source_mutation', 'checker_source_mutation', 'checker_proof_mutation', 'live_expired')
ARITHMETIC = tuple('overflow_' + str(i) for i in range(2)) + fixtures.RESOURCE_CASES + ('checker_intermediate',)


def audit_inventory(obs, rejects, aux):
    if set(obs) != set(fixtures.NORMAL) or set(rejects) != set(fixtures.MUTATIONS) or set(aux) != set(FAULTS + ARITHMETIC):
        raise ValueError('complete frozen inventory required')
    for name, saved in rejects.items():
        if checker.compact(saved['input']) != checker.compact(fixtures.mutation(name, obs)):
            raise ValueError('concrete mutation binding: ' + name)
    for name in FAULTS:
        if aux[name]['error'] != fixtures.FAULT_MESSAGES[name]:
            raise ValueError('specific fault message: ' + name)


def run(root):
    if root.name not in tuple(f'hz_sparse_rows_20261012_r{i}' for i in (1, 2, 3)):
        raise ValueError('one of three finite repair-attempt directories required')
    if sha(ROOT / DESIGN) != DESIGN_SHA:
        raise ValueError('precommitted row design changed')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != FREEZE_HEAD:
        raise ValueError('separate frozen execution identity required')
    for name, expected in LEGACY_SHA.items():
        if sha(ROOT/name) != expected:
            raise ValueError('legacy finite implementation changed')
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    bindings = {n: sha(ROOT/n) for n in FILES}
    for name in FILES:
        target = root/'implementation'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    save(root/'implementation.json', bindings)
    importing = time.monotonic()
    from scoped_source import test_hz_sparse_rows as tests
    imported = time.monotonic()
    names = ['scoped_source.test_hz_sparse_rows.SparseRowTests.' + n for n in TESTS]
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(tests.SparseRowTests)
    if [t.id() for t in suite] != names:
        raise ValueError('frozen test inventory')
    with (root/'tests.log').open('x') as out:
        class Result(unittest.TextTestResult):
            def __init__(self, *a, **k): super().__init__(*a, **k); self.outcomes = []
            def startTest(self, t): self.state = 'PASS'; super().startTest(t)
            def addError(self, t, e): self.state = 'ERROR'; super().addError(t, e)
            def addFailure(self, t, e): self.state = 'FAIL'; super().addFailure(t, e)
            def addSkip(self, t, e): self.state = 'SKIP'; super().addSkip(t, e)
            def addExpectedFailure(self, t, e): self.state = 'EXPECTED_FAILURE'; super().addExpectedFailure(t, e)
            def addUnexpectedSuccess(self, t): self.state = 'UNEXPECTED_SUCCESS'; super().addUnexpectedSuccess(t)
            def stopTest(self, t): self.outcomes.append({'test': t.id(), 'status': self.state}); super().stopTest(t)
        result = unittest.TextTestRunner(stream=out, verbosity=2, resultclass=Result, failfast=True).run(suite)
    ended = time.monotonic()
    blobs = {'observations': tests.OBS, 'rejections': tests.REJECTS, 'auxiliary': tests.AUX}
    digests = {n + '_sha256': save(root/(n + '.json'), v)['sha256'] for n, v in blobs.items()}
    published = time.monotonic()
    exceptions = {k: len(getattr(result, k)) for k in ('errors', 'failures', 'skipped', 'expectedFailures', 'unexpectedSuccesses')}
    ok = result.wasSuccessful() and result.outcomes == [{'test': n, 'status': 'PASS'} for n in names]
    ok = ok and published - start < 1800
    summary = {'status': 'PASS' if ok else 'FAIL', 'tests': result.testsRun, 'outcomes': result.outcomes,
               'exceptions': exceptions, 'head': head, 'design_sha256': DESIGN_SHA,
               'implementation_sha256': sha(root/'implementation.json'), 'tests_log_sha256': sha(root/'tests.log'),
               **digests, 'times': [start, importing, imported, ended, published],
               'import_seconds': imported-importing, 'suite_seconds': ended-imported,
               'archive_seconds': published-ended, 'total_seconds': published-start,
               'real_requests': 0, 'native_solves': 0, 'cuda_calls': 0,
               'performance_claim': False, 'hard_supervision': False}
    save(root/'summary.json', summary)
    return summary


def clocks(times, deadline, budget):
    if (type(times) is not list or len(times) < 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in times)
            or any(a > b for a, b in zip(times, times[1:]))
            or deadline != times[0]+budget or times[-1] >= deadline):
        raise ValueError('finite ordered cost/deadline inventory')


def finite_costs(cost):
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in cost.values()):
        raise ValueError('finite nonnegative numeric cost required')


def expected_counts(ref, proof, chunk):
    source, target = ref['state']['hz'], proof['target']['state']['hz']
    out = {'source_parses': 1, 'target_parses': 1, 'residual_union_entries': 0, 'residual_chunks': 0,
           'layout_entries': 0, 'layout_chunks': 0, 'sparse_sort_calls': 0, 'sparse_sort_keys': 0}
    for name, raw in (('source', source), ('target', target)):
        out[name+'_matrix_coefficients'] = sum(len(raw[k]['data']) for k in checker.MATRICES)
        out[name+'_vector_coefficients'] = sum(len(raw[k]) for k in checker.VECTORS)
    nc = len(ref['state']['continuous_ids'])
    for key in checker.MATRICES:
        a, b = source[key], target[key]
        for i in range(a['shape'][0]):
            left = set(a['indices'][a['indptr'][i]:a['indptr'][i+1]])
            right = b['indices'][b['indptr'][i]:b['indptr'][i+1]]
            n = len(left | {j for j in right if key.endswith('b') or j < nc})
            out['residual_union_entries'] += n; out['residual_chunks'] += (n+chunk-1)//chunk
            out['sparse_sort_calls'] += 1; out['sparse_sort_keys'] += n
            out['layout_entries'] += len(right); out['layout_chunks'] += (len(right)+chunk-1)//chunk
    return out


def _aux_math(name, item):
    """Check saved arithmetic refusal reasons without importing the producer."""
    from fractions import Fraction as F
    ref = (fixtures.overflow_patterns()[int(name[-1])] if name.startswith('overflow_')
           else fixtures.resource_case('intermediate_bits' if name == 'checker_intermediate' else name))
    expected_input = {'reference': ref, 'expected': checker.identity(ref), 'owner': 'resource'}
    if name == 'checker_intermediate': expected_input['proof'] = fixtures.zero_resource_proof(ref)
    if checker.compact(item['input']) != checker.compact(expected_input):
        raise ValueError('arithmetic reference binding')
    clocks(item['times'], item['deadline'], 30)
    if item['reached'] is not True:
        raise ValueError('arithmetic refusal not reached')
    if name.startswith('overflow_'):
        key = 'c' if name == 'overflow_0' else 'ub'
        if F(ref['state']['hz'][key][0]) <= F(sys.float_info.max):
            raise ValueError('overflow source does not exceed finite binary64')
        reason = 'no finite binary64 coefficient' if key == 'c' else 'no finite outward binary64 endpoint'
        if item['error_type'] != 'ValueError' or item['error'] != reason:
            raise ValueError('overflow refusal reason')
        return
    work = checker.Work(time.monotonic()+30)
    try:
        if name == 'checker_intermediate':
            checker.check(ref, item['input']['proof'], expected_reference_sha256=checker.identity(ref),
                          expected_owner='resource', deadline=work.deadline)
        else:
            parsed = checker.capture(ref, work, 'audit_owned')[0]
            h, _, _ = checker.parse_owned(parsed, work)
            if name != 'intermediate_bits':
                raise AssertionError('expected admission refusal absent')
            # All three positive operands round to zero. Independently add the
            # exact residual; each stored operand passes the 4096-bit admission.
            total = F(0)
            for value in h['Gc'][0].values():
                if value >= F(1, 2**1075): raise ValueError('wrong underflow resource witness')
                total = checker.bounded(total + value)
    except checker.ResourceRefusal as exc:
        if (item['error_type'], item['error']) != (type(exc).__name__, str(exc)):
            raise ValueError('resource reason changed')
    else:
        raise ValueError('resource refusal not reproduced')


def audit(root):
    from upstream_source.checker import hz
    from scoped_source import check_hz_binary64 as old
    start = time.monotonic()
    s = load(root/'summary.json'); binding = load(root/'implementation.json', s['implementation_sha256'])
    obs = load(root/'observations.json', s['observations_sha256'])
    rejects = load(root/'rejections.json', s['rejections_sha256'])
    aux = load(root/'auxiliary.json', s['auxiliary_sha256'])
    names = ['scoped_source.test_hz_sparse_rows.SparseRowTests.'+n for n in TESTS]
    if (s['status'] != 'PASS' or s['tests'] != 9 or s['outcomes'] != [{'test': n, 'status': 'PASS'} for n in names]
            or s['exceptions'] != {k: 0 for k in ('errors', 'failures', 'skipped', 'expectedFailures', 'unexpectedSuccesses')}
            or s['head'] != FREEZE_HEAD or s['design_sha256'] != DESIGN_SHA or set(binding) != set(FILES)
            or binding[DESIGN] != DESIGN_SHA or sha(root/'tests.log') != s['tests_log_sha256']
            or any(type(s[k]) is not int or s[k] != 0 for k in ('real_requests', 'native_solves', 'cuda_calls'))
            or s['performance_claim'] is not False or s['hard_supervision'] is not False):
        raise ValueError('frozen outcomes/identity/scope')
    for name, expected in binding.items():
        if sha(ROOT/name) != expected or sha(root/'implementation'/name) != expected:
            raise ValueError('implementation binding: ' + name)
    for name, expected in LEGACY_SHA.items():
        if binding[name] != expected: raise ValueError('legacy drift')
    clocks(s['times'], s['times'][0]+1800, 1800)
    finite_costs({k: s[k] for k in ('import_seconds', 'suite_seconds', 'archive_seconds', 'total_seconds')})
    if (s['import_seconds'] != s['times'][2]-s['times'][1] or s['suite_seconds'] != s['times'][3]-s['times'][2]
            or s['archive_seconds'] != s['times'][4]-s['times'][3] or s['total_seconds'] != s['times'][4]-s['times'][0]):
        raise ValueError('suite component cost')
    audit_inventory(obs, rejects, aux)
    points, rows, intervals = 0, {}, []
    for name in fixtures.NORMAL:
        ref, assignments, owner, chunk = fixtures.fixture(name); r = obs[name]; p = r['proof']
        if r['reference'] != ref or r['owner'] != owner or r['chunk'] != chunk:
            raise ValueError('fixed normal source/owner/chunk')
        deadline = time.monotonic()+300
        with patch.object(checker, 'parse_owned', wraps=checker.parse_owned) as parses, \
                patch.object(checker, 'capture', wraps=checker.capture) as captures:
            result = checker.check(ref, p, expected_reference_sha256=checker.identity(ref),
                                   expected_owner=owner, deadline=deadline, chunk_size=chunk)
        if checker.compact(result) != checker.compact(r['checked']) or result['work'] != expected_counts(ref, p, chunk):
            raise ValueError('saved exact result/work count')
        if parses.call_count != 2 or captures.call_count != 2 or r['measured_parse_calls'] != 2 or r['measured_capture_calls'] != 2:
            raise ValueError('one source/target parse required')
        h, _, _ = checker.parse_owned(checker.capture(p['target'], checker.Work(deadline), 'snapshot')[0], checker.Work(deadline), target=True)
        if hz(r['snapshot']) != h or r['snapshot']['exact'] is not False:
            raise ValueError('actual HZ snapshot mismatch')
        if name in fixtures.NORMAL[:8]:
            previous = r['legacy']
            got = old.check(ref, previous['proof'], expected_reference_sha256=checker.identity(ref), expected_owner=owner, deadline=deadline)
            if got != previous['checked'] or fixtures.math_view(p) != fixtures.math_view(previous['proof']) or r['snapshot'] != previous['snapshot']:
                raise ValueError('old/new differential')
            for k in ('compensation', 'inequalities', 'added_continuous', 'binary_preserved'):
                if got[k] != result[k]: raise ValueError('old/new exact residual difference')
        elif r['legacy'] is not None:
            raise ValueError('wide row has no admitted old producer')
        if name.startswith('wide_'):
            from fractions import Fraction as F
            if fixtures.math_view(p) != fixtures.expected_wide(ref):
                raise ValueError('frozen analytic wide target')
            if (result['work']['source_matrix_coefficients'] != 12288
                    or result['work']['target_matrix_coefficients'] != 12291
                    or result['work']['residual_union_entries'] != 12288
                    or r['embeddings'][0]['target']['continuous'][-3:] != [str(F(2**42, 2**42+7)), '0', '0']):
                raise ValueError('frozen wide counts/extension')
        if len(r['embeddings']) != len(assignments): raise ValueError('point inventory')
        for item, (a, b) in zip(r['embeddings'], assignments):
            if item['source'] != {'continuous': a, 'binary': b}: raise ValueError('fixed assignment')
            point = checker.check_embedding(ref, p, item['source'], item['target'], expected_reference_sha256=checker.identity(ref),
                                            expected_owner=owner, deadline=deadline, chunk_size=chunk)
            if item['checked'] != point: raise ValueError('saved point result')
            points += 1
        clocks(r['times'], r['deadline'], 300)
        components = ('create', 'legacy_differential', 'produce', 'instantiate', 'serialize', 'check', 'points')
        finite_costs(r['cost'])
        if len(r['times']) != 8 or set(r['cost']) != set(components + ('total',)):
            raise ValueError('complete normal cost inventory')
        for key, a, b in zip(components, r['times'], r['times'][1:]):
            if r['cost'][key] != b-a: raise ValueError('normal component cost')
        if r['cost']['total'] != r['times'][-1]-r['times'][0]: raise ValueError('normal total cost')
        intervals.append((r['times'][0], r['times'][-1]))
        rows[name] = {'reference_sha256': checker.identity(ref), 'target_sha256': p['target_sha256'],
                      'proof_sha256': checker.identity(p), 'work': result['work'], 'compensation': result['compensation'],
                      'inequalities': result['inequalities'], 'cost_seconds': r['cost']}
    if obs['wide_1']['proof'] != obs['wide_256']['proof']:
        raise ValueError('chunk-dependent mathematical object')
    for name, r in rejects.items():
        clocks(r['times'], r['deadline'], 30); item = r['input']
        intervals.append((r['times'][0], r['times'][-1]))
        try:
            checker.check(item['reference'], item['proof'], expected_reference_sha256=item['expected'],
                          expected_owner=item['owner'], deadline=time.monotonic()+30)
        except (ValueError, TypeError, KeyError) as exc:
            if (type(exc).__name__, str(exc)) != (r['error_type'], r['error']) or item['reason'] not in str(exc):
                raise ValueError('mutation reason changed')
        else: raise ValueError('saved mutation accepted')
    for name in ARITHMETIC:
        _aux_math(name, aux[name])
    for name in FAULTS:
        r = aux[name]; clocks(r['times'], r['deadline'], 30)
        mutation = name.endswith('_mutation')
        expected_reached = ['before_live'] if name == 'live_expired' else ['final_identity'] if name.endswith('_final_identity') else [
            'producer_residual_chunk' if name.startswith('producer') else 'residual_chunk']
        if (r['error_type'] != ('ValueError' if mutation else 'TimeoutError') or r['reached'] != expected_reached
                or r['error'] != fixtures.FAULT_MESSAGES[name]
                or (not mutation and r['simulated_clock_after'] != r['deadline']+1)
                or (not name.endswith('_final_identity') and name != 'live_expired' and r['chunk_tick_calls'] != 2)):
            raise ValueError('specific fault injection/terminal')
        if r['source_before'] != obs['affine_sum']['checked']['reference_sha256'] or r['proof_before'] != obs['affine_sum']['checked']['proof_sha256']:
            raise ValueError('fault source anchors')
        if name.endswith('_source_mutation') != (r['source_before'] != r['source_after']) or name.endswith('_proof_mutation') != (r['proof_before'] != r['proof_after']):
            raise ValueError('fault mutation not observed')
        mutated_source, mutated_proof = deepcopy(obs['affine_sum']['reference']), deepcopy(obs['affine_sum']['proof'])
        if name.endswith('_source_mutation'): mutated_source['state']['hz']['c'][0] = '2'
        if name.endswith('_proof_mutation'): mutated_proof['target']['state']['hz']['c'][0] = '2'
        if r['source_after'] != checker.identity(mutated_source) or r['proof_after'] != checker.identity(mutated_proof):
            raise ValueError('specific late mutation binding')
    intervals += [(v['times'][0], v['times'][-1]) for v in aux.values()]
    intervals.sort()
    if (any(a[1] > b[0] for a, b in zip(intervals, intervals[1:]))
            or any(not s['times'][2] <= a <= b <= s['times'][3] for a, b in intervals)):
        raise ValueError('overlapping or uncharged control calls')
    if any(n in sys.modules for n in ('scoped_source.hz_sparse_rows', 'scoped_source.hz_binary64',
                                     'scoped_source.test_hz_sparse_rows', 'numpy', 'scipy', 'torch')):
        raise ValueError('audit imported producer or numeric libraries')
    return {'status': 'PASS', 'root': str(root), 'summary_sha256': sha(root/'summary.json'),
            'freeze_head': FREEZE_HEAD, 'design_sha256': DESIGN_SHA, 'tests': 9,
            'normal_executions': len(rows), 'embeddings_checked': points, 'negative_inputs_rechecked': len(rejects),
            'arithmetic_refusals_checked': len(ARITHMETIC), 'fault_receipts_checked': len(FAULTS), 'cases': rows,
            'audit_seconds': time.monotonic()-start, 'run_total_seconds': s['total_seconds'],
            'real_requests': 0, 'native_solves': 0, 'cuda_calls': 0, 'performance_claim': False,
            'hard_supervision': False, 'all_six_goal_gates_remain_open': True}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['run', 'audit']); p.add_argument('root', type=Path)
    p.add_argument('--report', type=Path); a = p.parse_args()
    if not a.root.is_absolute() or not a.root.resolve().is_relative_to(ROOT.parent/'baseline_runs'):
        raise ValueError('new project control directory required')
    result = run(a.root) if a.action == 'run' else audit(a.root)
    if a.report: save(a.report, result)
    print({'status': result['status'], 'tests': result['tests'], 'root': str(a.root)})
