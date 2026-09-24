"""Exact no-cache differential, isolation, semantic mutations and cutoff controls."""
import copy
from fractions import Fraction as F
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_source.build import build as original_build
from scoped_source.check import check
from source_enclosure.format import identity, sparse
from upstream_source.checker import csr
from source_construction_lab.cache import MatrixParser
from source_construction_lab.build import construct, bind
from source_construction_lab.fixtures import document

ROOT = Path(__file__).resolve().parents[1]


def parser(**kwargs):
    return MatrixParser('request-A', enabled=True, tick=lambda: None, **kwargs)


class ParserControls(unittest.TestCase):
    def setUp(self):
        self.matrix = sparse([{0: F(1, 3), 2: F(-2)}, {}], 3)

    def test_reference_and_mutable_input_output_isolation(self):
        cache = parser()
        result = cache.csr(self.matrix, [2, 3])
        result[0][0] = F(77)
        self.assertEqual(cache.csr(self.matrix), csr(self.matrix))
        self.matrix['data'][0] = '2/3'
        self.assertEqual(cache.csr(self.matrix)[0][0], F(2, 3))
        self.assertEqual(cache.stats()['hits'], 1)

    def test_hash_collision_and_scope_binding(self):
        with patch('source_construction_lab.cache.digest', return_value='collision'):
            cache = parser(); cache.csr(self.matrix)
            self.matrix['data'][0] = '11'
            with self.assertRaises(ValueError): cache.csr(self.matrix)
        cache = parser(); cache.csr(self.matrix)
        cache._scope = 'wrong-request'
        # Even a maliciously transplanted entry cannot authorize a hit.
        old_key, value = next(iter(cache._items.items()))
        cache._items[('wrong-request', old_key[1], old_key[2])] = value
        with self.assertRaises(ValueError): cache.csr(self.matrix)

    def test_immutable_retained_rows(self):
        cache = parser(); cache.csr(self.matrix)
        value = next(iter(cache._items.values()))
        with self.assertRaises((TypeError, AttributeError)): value.rows[0][0] = (0, F(9))

    def test_shape_obligation_rechecked_on_hit(self):
        cache = parser(); cache.csr(self.matrix)
        with self.assertRaises(ValueError): cache.csr(self.matrix, [2, 4])

    def test_malformed_after_warming(self):
        for key, replacement in [('indices', [0, 0]), ('indptr', [0, 2]),
                                  ('data', ['NaN', '1']), ('shape', [2, -1])]:
            with self.subTest(key=key):
                cache = parser(); cache.csr(self.matrix)
                bad = copy.deepcopy(self.matrix); bad[key] = replacement
                with self.assertRaises((ValueError, OverflowError)): cache.csr(bad)

    def test_capacity_eviction_and_oversized(self):
        cache = parser(limits={'entries': 1, 'bytes': 10000, 'cells': 100})
        cache.csr(self.matrix); cache.csr(sparse([{1: F(4)}], 3))
        self.assertEqual(cache.stats()['evictions'], 1)
        cache.clear(); self.assertEqual(cache.stats()['live_cells'], 0)
        tiny = parser(limits={'entries': 1, 'bytes': 1, 'cells': 1})
        self.assertEqual(tiny.csr(self.matrix), csr(self.matrix))
        self.assertEqual(tiny.stats()['live_entries'], 0)

    def test_deadline_on_hit_and_disabled_mode(self):
        cache = parser(); cache.csr(self.matrix)
        cache._tick = lambda: (_ for _ in ()).throw(TimeoutError('expired'))
        with self.assertRaises(TimeoutError): cache.csr(self.matrix)
        off = MatrixParser('request-A', enabled=False, tick=lambda: None)
        for _ in range(2): self.assertEqual(off.csr(self.matrix), csr(self.matrix))
        self.assertEqual((off.stats()['hits'], off.stats()['live_entries']), (0, 0))

    def test_random_exact_csr_differential(self):
        rng = random.Random(944)
        cache = parser()
        for _ in range(20):
            rows = [{j: F(rng.randrange(-4, 5), rng.randrange(1, 9))
                     for j in range(7) if rng.random() < .4} for _ in range(4)]
            matrix = sparse(rows, 7)
            for _ in range(2): self.assertEqual(cache.csr(matrix, [4, 7]), csr(matrix, [4, 7]))


class ConstructionControls(unittest.TestCase):
    def test_complete_byte_differential_and_independent_checks(self):
        cases = [dict(experts=2, classes=2, width=1, depth=0),
            dict(experts=3, classes=4, width=3, depth=2),
            dict(experts=4, classes=3, width=2, depth=1, tied=True),
            dict(experts=3, classes=3, width=2, depth=2, constant=True),
            dict(experts=3, classes=2, width=2, depth=1, radius='0'),
            dict(experts=8, classes=10, width=1, depth=0, tied=True)]
        for case in cases:
            with self.subTest(case=case):
                doc = document(**case); before = identity(doc)
                expected = original_build(doc, expected_source_sha256=before, deadline=time.monotonic()+30)
                for mode in ('reference', 'uncached', 'cached'):
                    actual, report = construct(doc, expected_source_sha256=before,
                        deadline=time.monotonic()+30, mode=mode)
                    self.assertEqual(actual, expected)
                    self.assertEqual(identity(actual), identity(expected))
                    self.assertEqual(identity(doc), before)
                    checked = check(doc, actual, expected_source_sha256=before, deadline=time.monotonic()+30)
                    self.assertEqual(checked['output_obligations'], case['experts']*(case['experts']-1)//2*(case['classes']-1))
                    self.assertFalse(checked['complete_output_positive_proof'])
                    if report['parser'] is not None:
                        self.assertEqual(report['parser']['live_entries'], 0)

    def test_source_and_output_mutations_still_rejected(self):
        doc = document(experts=3, classes=3, width=2, depth=1, tied=True)
        bundle, _ = construct(doc, expected_source_sha256=identity(doc), deadline=time.monotonic()+30, mode='cached')
        for mode in ('pair', 'property', 'guard', 'source'):
            bad = copy.deepcopy(bundle); changed = copy.deepcopy(doc)
            if mode == 'pair': bad['pairs'].pop()
            if mode == 'property': bad['pairs'][0]['obligations']['rows'].pop()
            if mode == 'guard': bad['pairs'][0]['guarded']['hz']['ub'][-1] = '-1'
            if mode == 'source': changed['request']['radius'] = '1/9'
            with self.assertRaises(ValueError):
                check(changed, bad, expected_source_sha256=identity(doc), deadline=time.monotonic()+30)

    def test_no_global_substitution(self):
        from upstream_source import checker
        from full_source import lift
        before = (checker.csr, lift.affine.__globals__['unpack'], original_build.__globals__['affine'])
        doc = document(experts=2, classes=2, width=2, depth=1)
        construct(doc, expected_source_sha256=identity(doc), deadline=time.monotonic()+30, mode='cached')
        self.assertEqual(before, (checker.csr, lift.affine.__globals__['unpack'], original_build.__globals__['affine']))
        with self.assertRaises(ValueError): bind(original_build, absent=1)

    def test_error_deadline_and_cleanup(self):
        doc = document(experts=2, classes=2, width=2, depth=1)
        with self.assertRaises(ValueError):
            construct(doc, expected_source_sha256='0'*64, deadline=time.monotonic()+30, mode='cached')
        with self.assertRaises(TimeoutError):
            construct(doc, expected_source_sha256=identity(doc), deadline=time.monotonic()-1, mode='cached')
        cleared = []
        original = MatrixParser.clear
        def record(cache):
            original(cache); cleared.append(cache.stats()['live_entries'])
        with patch.object(MatrixParser, 'clear', record):
            def emit(event):
                if event['event'] == 'ENTER' and event['operation'] == 'pair_guard':
                    raise RuntimeError('controlled after-prefix exception')
            with self.assertRaises(RuntimeError):
                construct(doc, expected_source_sha256=identity(doc), deadline=time.monotonic()+30, mode='cached', emit=emit)
        self.assertEqual(cleared, [0])

    def test_fresh_stdlib_checker_and_nested_costs(self):
        doc = document(experts=3, classes=3, width=2, depth=1)
        bundle, report = construct(doc, expected_source_sha256=identity(doc), deadline=time.monotonic()+30, mode='cached')
        times = report['timings']
        self.assertLessEqual(sum(r['exclusive_seconds'] for r in times.values()), times['construction']['inclusive_seconds'] + .001)
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as folder:
            from scoped_proof.io import save
            p = Path(folder)/'synthetic.json'; save(p, [doc, bundle])
            code = ('import json,sys,time; from scoped_source.check import check; '
                'd,b=json.load(open(sys.argv[1])); '
                'r=check(d,b,expected_source_sha256=sys.argv[2],deadline=time.monotonic()+20); '
                'assert not any(n.split(".")[0] in ("torch","numpy","scipy","act","source_construction_lab") for n in sys.modules); '
                'print(r["output_obligations"])')
            result = subprocess.run([sys.executable, '-S', '-c', code, str(p), identity(doc)],
                cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=25)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), '6')


class SupervisionControls(unittest.TestCase):
    def run_case(self, command=None, budget=10.):
        from source_construction_lab.study import run_one
        from scoped_proof.io import load
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as folder:
            root = Path(folder)/'run'
            result = run_one(root, dict(experts=2, classes=2, width=1, depth=0),
                'cached', budget=budget, commands=command)
            terminal, cost = load(root/'terminal.json'), load(root/'cost.json')
            self.assertFalse(terminal['positive_certificate'])
            self.assertAlmostEqual(cost['stage_seconds']+cost['overhead_seconds'], cost['end_to_end_seconds'])
            self.assertGreaterEqual(result['seconds'], cost['end_to_end_seconds'])
            return result, terminal, cost

    def test_full_pipeline_accounting(self):
        result, terminal, _ = self.run_case()
        self.assertEqual(result['status'], 'COMPLETED_CONSTRUCTION_CHECK_ONLY')
        self.assertEqual([p['phase'] for p in terminal['phases']], ['build', 'check'])

    def test_deadline_preserves_partial_and_no_proof(self):
        command = lambda *args: [sys.executable, '-S', '-c', 'import time; time.sleep(3)']
        result, terminal, _ = self.run_case(command, .2)
        self.assertEqual(result['status'], 'TIMEOUT')
        self.assertEqual(len(terminal['phases']), 1)
        self.assertEqual(terminal['phases'][0]['returncode'], -9)

    def test_exception_records_terminal(self):
        command = lambda *args: [sys.executable, '-S', '-c', 'raise RuntimeError("controlled failure")']
        result, terminal, _ = self.run_case(command)
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(terminal['phases'][0]['status'], 'ERROR')

    def test_successful_exit_without_evidence_rejected(self):
        command = lambda *args: [sys.executable, '-S', '-c', 'pass']
        result, terminal, _ = self.run_case(command)
        self.assertEqual(result['status'], 'ERROR')
        self.assertIsNotNone(terminal['error'])


if __name__ == '__main__': unittest.main()
