"""Nine frozen groups for sparse-row controls; no model, optimizer or CUDA."""
import ast
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from source_enclosure.format import empty, pack
from scoped_source import check_hz_sparse_rows as checker
from scoped_source import hz_sparse_rows as producer
from scoped_source import hz_sparse_row_fixtures as fixtures
from scoped_source import check_hz_binary64 as legacy_check
from scoped_source import hz_binary64 as legacy
from scoped_source import test_hz_binary64 as legacy_fixtures

OBS, REJECTS, AUX = {}, {}, {}
TESTS = tuple(f'test_{i:02d}_{name}' for i, name in enumerate(
    ('legacy', 'wide', 'chunks', 'identity', 'binding_csr_layout', 'inward',
     'arithmetic_resources', 'deadline_mutation', 'independence_inventory'), 1))
FAULTS = ('producer_chunk', 'checker_chunk', 'producer_final_identity', 'checker_final_identity',
          'producer_source_mutation', 'checker_source_mutation', 'checker_proof_mutation', 'live_expired')
ARITHMETIC = tuple('overflow_' + str(i) for i in range(2)) + fixtures.RESOURCE_CASES + ('checker_intermediate',)


def checked(ref, proof, owner, deadline, chunk=256):
    return checker.check(ref, proof, expected_reference_sha256=checker.identity(ref),
                         expected_owner=owner, deadline=deadline, chunk_size=chunk)


class SparseRowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OBS.clear(); REJECTS.clear(); AUX.clear()
        for name in fixtures.NORMAL:
            times = [time.monotonic()]; deadline = times[0] + 300
            ref, points, owner, chunk = fixtures.fixture(name)
            times.append(time.monotonic())
            r = {'reference': ref, 'owner': owner, 'chunk': chunk, 'deadline': deadline, 'times': times}
            OBS[name] = r  # Preserve the actual prefix if construction fails.
            old = None
            if not name.startswith('wide_'):
                if legacy_fixtures.patterns()[name] != (ref, points):
                    raise AssertionError('old frozen reference/points changed')
                p = legacy.produce(ref, expected_reference_sha256=checker.identity(ref), owner=owner, deadline=deadline)
                result = legacy_check.check(ref, p, expected_reference_sha256=checker.identity(ref),
                                             expected_owner=owner, deadline=deadline)
                live, snapshot = legacy.instantiate(p['target'], expected_target_sha256=p['target_sha256'], deadline=deadline)
                old = {'proof': p, 'checked': result, 'snapshot': snapshot}
                del live
            r['legacy'] = old
            times.append(time.monotonic())
            proof = producer.produce(ref, expected_reference_sha256=checker.identity(ref), owner=owner,
                                     deadline=deadline, chunk_size=chunk)
            r['proof'] = proof; times.append(time.monotonic())
            live, snapshot = producer.instantiate(proof['target'], expected_target_sha256=proof['target_sha256'], deadline=deadline)
            r['snapshot'] = snapshot; del live; times.append(time.monotonic())
            proof = json.loads(checker.compact(proof)); r['proof'] = proof
            times.append(time.monotonic())
            with patch.object(checker, 'parse_owned', wraps=checker.parse_owned) as parses, \
                    patch.object(checker, 'capture', wraps=checker.capture) as captures:
                r['checked'] = checked(ref, proof, owner, deadline, chunk)
                r['measured_parse_calls'], r['measured_capture_calls'] = parses.call_count, captures.call_count
            times.append(time.monotonic()); r['embeddings'] = []
            for a, b in points:
                assignment = {'continuous': a, 'binary': b}
                extension = fixtures.extend(ref, proof, assignment)
                receipt = checker.check_embedding(ref, proof, assignment, extension,
                    expected_reference_sha256=checker.identity(ref), expected_owner=owner, deadline=deadline, chunk_size=chunk)
                r['embeddings'].append({'source': assignment, 'target': extension, 'checked': receipt})
            times.append(time.monotonic())
            if times[-1] >= deadline:
                raise TimeoutError('complete normal control cutoff')
            r['cost'] = {k: b-a for k, a, b in zip(
                ('create', 'legacy_differential', 'produce', 'instantiate', 'serialize', 'check', 'points'), times, times[1:])}
            r['cost']['total'] = times[-1] - times[0]

    def test_01_legacy(self):
        self.assertEqual(set(OBS), set(fixtures.NORMAL))
        for name in fixtures.NORMAL[:8]:
            r = OBS[name]; old = r['legacy']
            self.assertEqual(fixtures.math_view(r['proof']), fixtures.math_view(old['proof']))
            self.assertEqual(r['snapshot'], old['snapshot'])
            for field in ('compensation', 'inequalities', 'added_continuous', 'binary_preserved'):
                self.assertEqual(r['checked'][field], old['checked'][field])
        ref = fixtures.owned(empty(1), [f'x/{i}' for i in range(9)], [])
        with self.assertRaisesRegex(ValueError, 'capacity'):
            legacy_check.parse(ref)

    def test_02_wide(self):
        d = F(7, 2**96)
        for name in ('wide_256', 'wide_1'):
            r = OBS[name]; p = r['proof']; got = r['checked']
            self.assertEqual(fixtures.math_view(p), fixtures.expected_wide(r['reference']))
            self.assertEqual([F(v['radius']) for v in p['output_errors']], [F(1, 2**54)+d, d])
            self.assertEqual(F(p['equality_errors'][0]['radius']), d)
            self.assertEqual(p['target']['state']['hz']['ub'], ['1'])
            self.assertEqual(r['embeddings'][0]['target']['continuous'][-3:], [str(F(2**42, 2**42+7)), '0', '0'])
            stats = got['work']
            self.assertEqual(stats['source_parses'], 1); self.assertEqual(stats['target_parses'], 1)
            self.assertEqual(stats['source_matrix_coefficients'], 12288)
            self.assertEqual(stats['target_matrix_coefficients'], 12291)
            self.assertEqual(stats['residual_union_entries'], 12288)
            self.assertEqual(stats['layout_entries'], 12291)
            self.assertEqual(stats['sparse_sort_keys'], 12288)
            self.assertEqual((r['measured_parse_calls'], r['measured_capture_calls']), (2, 2))

    def test_03_chunks(self):
        a, b = (OBS[n] for n in ('wide_1', 'wide_256'))
        self.assertEqual(a['reference'], b['reference'])
        self.assertEqual(a['proof'], b['proof'])
        self.assertEqual(a['snapshot'], b['snapshot'])
        self.assertEqual(a['embeddings'], b['embeddings'])
        for k in ('compensation', 'inequalities'):
            self.assertEqual(a['checked'][k], b['checked'][k])
        self.assertEqual(a['checked']['work']['residual_chunks'], 12288)
        self.assertEqual(b['checked']['work']['residual_chunks'], 48)

    def test_04_identity(self):
        r = OBS['wide_256']; p = r['proof']
        self.assertEqual(p['maps']['flat'], list(range(1536)) + list(range(1539, 3075)))
        self.assertEqual(p['target']['ownership']['binary'], ['expert0'] * 1536)
        self.assertEqual(p['target']['ownership']['continuous'], ['shared'] * 1536 + ['wide/row'] * 3)
        self.assertEqual(OBS['identity']['reference'], OBS['identity']['proof']['target'])
        self.assertEqual(OBS['zero_empty']['checked']['added_continuous'], 0)
        self.assertEqual(OBS['shared_private']['proof']['maps']['flat'], [0, 1, 2, 5, 6])

    def reject_mutation(self, name):
        start = time.monotonic(); deadline = start + 30
        item = fixtures.mutation(name, OBS); before = time.monotonic()
        try:
            checker.check(item['reference'], item['proof'], expected_reference_sha256=item['expected'],
                          expected_owner=item['owner'], deadline=deadline)
        except (ValueError, TypeError, KeyError) as exc:
            self.assertIn(item['reason'], str(exc))
            end = time.monotonic()
            REJECTS[name] = {'input': item, 'error_type': type(exc).__name__, 'error': str(exc),
                             'times': [start, before, end], 'deadline': deadline}
            self.assertLess(end, deadline)
        else:
            self.fail('concrete mutation accepted: ' + name)

    def test_05_binding_csr_layout(self):
        for name in fixtures.MUTATIONS:
            if 'inward' not in name and name not in ('output_sign', 'equality_sign') and not name.startswith('target_only_'):
                self.reject_mutation(name)

    def test_06_inward(self):
        for name in fixtures.MUTATIONS:
            if 'inward' in name or name in ('output_sign', 'equality_sign') or name.startswith('target_only_'):
                self.reject_mutation(name)

    def test_07_arithmetic_resources(self):
        self.assertEqual(fixtures.overflow_patterns(), legacy_fixtures.overflow_patterns())
        self.assertEqual(OBS['subnormal']['checked']['added_continuous'], 3)
        for name in ARITHMETIC:
            start = time.monotonic(); deadline = start + 30
            ref = (fixtures.overflow_patterns()[int(name[-1])] if name.startswith('overflow_')
                   else fixtures.resource_case('intermediate_bits' if name == 'checker_intermediate' else name))
            item = {'reference': ref, 'owner': 'resource', 'expected': checker.identity(ref)}
            before = time.monotonic()
            try:
                if name == 'checker_intermediate':
                    # Zero target: no large literals, but the complete exact residual sum grows.
                    p = fixtures.zero_resource_proof(ref)
                    item['proof'] = p
                    checked(ref, p, 'resource', deadline)
                else:
                    producer.produce(ref, expected_reference_sha256=item['expected'], owner='resource', deadline=deadline)
            except ValueError as exc:
                self.assertIn('finite' if name.startswith('overflow_') else 'capacity', str(exc))
                if name in ('intermediate_bits', 'checker_intermediate'):
                    self.assertEqual(str(exc), 'exact intermediate capacity')
                end = time.monotonic()
                AUX[name] = {'input': item, 'error_type': type(exc).__name__, 'error': str(exc),
                             'times': [start, before, end], 'deadline': deadline, 'reached': True}
                self.assertLess(end, deadline)
            else: self.fail('arithmetic/resource refusal missing: ' + name)

    def test_08_deadline_mutation(self):
        for name in FAULTS:
            real_time = time.monotonic; start = real_time(); deadline = start + 30
            r = deepcopy(OBS['affine_sum']); ref, p, owner = r['reference'], r['proof'], r['owner']
            expected = checker.identity(ref); now = [start]; reached = []; calls = [0]
            original_tick, original_identity = checker.Work.tick, checker.identity
            module = producer if name.startswith('producer') or name == 'live_expired' else checker
            phase = 'producer_residual_chunk' if name.startswith('producer') else 'residual_chunk'
            def tick(work, label):
                if label == phase:
                    calls[0] += 1
                    if calls[0] == 2:
                        if name.endswith('_chunk'):
                            reached.append(label); now[0] = deadline + 1
                        elif name.endswith('_source_mutation'):
                            reached.append(label); ref['state']['hz']['c'][0] = '2'
                        elif name.endswith('_proof_mutation'):
                            reached.append(label); p['target']['state']['hz']['c'][0] = '2'
                return original_tick(work, label)
            def hashed(value):
                result = original_identity(value)
                if name.endswith('_final_identity') and value is ref:
                    reached.append('final_identity'); now[0] = deadline + 1
                return result
            before = real_time()
            try:
                with patch.object(checker.time, 'monotonic', side_effect=lambda: now[0]), \
                        patch.object(checker.Work, 'tick', tick), patch.object(module, 'identity', hashed):
                    if name == 'live_expired':
                        reached.append('before_live'); now[0] = deadline + 1
                        producer.instantiate(p['target'], expected_target_sha256=p['target_sha256'], deadline=deadline)
                    elif name.startswith('producer'):
                        producer.produce(ref, expected_reference_sha256=expected, owner=owner, deadline=deadline, chunk_size=1)
                    else:
                        checker.check(ref, p, expected_reference_sha256=expected, expected_owner=owner, deadline=deadline, chunk_size=1)
            except (TimeoutError, ValueError) as exc:
                mutation = name.endswith('_mutation')
                self.assertEqual(type(exc), ValueError if mutation else TimeoutError)
                self.assertEqual(str(exc), fixtures.FAULT_MESSAGES[name])
                self.assertEqual(len(reached), 1)
                end = real_time()
                AUX[name] = {'error_type': type(exc).__name__, 'error': str(exc), 'reached': reached,
                             'times': [start, before, end], 'deadline': deadline,
                             'simulated_clock_after': now[0], 'chunk_tick_calls': calls[0],
                             'source_before': expected, 'source_after': checker.identity(ref),
                             'proof_before': OBS['affine_sum']['checked']['proof_sha256'],
                             'proof_after': checker.identity(p)}
                self.assertLess(end, deadline)
            else: self.fail('fault was not exercised: ' + name)

    def test_09_independence_inventory(self):
        r = OBS['identity']
        with patch.object(producer, 'produce', side_effect=AssertionError), \
                patch.object(producer, 'nearest', side_effect=AssertionError):
            checked(r['reference'], r['proof'], r['owner'], time.monotonic()+30)
        tree = ast.parse(Path(checker.__file__).read_text())
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): modules.update(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom): modules.add(node.module)
        self.assertEqual(modules, {'fractions', 'hashlib', 'json', 'math', 're', 'time'})
        self.assertEqual(set(REJECTS), set(fixtures.MUTATIONS))
        self.assertEqual(set(AUX), set(FAULTS + ARITHMETIC))
        from scripts.run_hz_sparse_rows import audit_inventory
        audit_inventory(OBS, REJECTS, AUX)
        for which, original in enumerate((OBS, REJECTS, AUX)):
            broken = dict(original); broken.pop(next(iter(broken)))
            args = [OBS, REJECTS, AUX]; args[which] = broken
            with self.assertRaisesRegex(ValueError, 'inventory'): audit_inventory(*args)
        changed = deepcopy(REJECTS)
        changed['owner']['input'] = deepcopy(changed['proof_binding']['input'])
        with self.assertRaisesRegex(ValueError, 'mutation binding'):
            audit_inventory(OBS, changed, AUX)
        for name in FAULTS:
            changed = deepcopy(AUX)
            changed[name]['error'] = 'unrelated changed deadline'
            with self.assertRaisesRegex(ValueError, 'specific fault message'):
                audit_inventory(OBS, REJECTS, changed)


if __name__ == '__main__':
    unittest.main()
