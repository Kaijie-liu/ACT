"""Serialization, identity, cutoff, partial evidence and unchanged proof controls."""
import copy
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from bounded_evidence import stream
from bounded_evidence.worker import POLICY
from scoped_proof.io import ROOT, PYTHON, load, save, sha


def deadline():
    return time.monotonic() + 60


def command(phase, root, end):
    from residual_proof.supervisor import command as old
    if phase == 'construct':
        return [PYTHON, '-S', '-m', 'bounded_evidence.worker', str(root), '--deadline', str(end)]
    return old(phase, root, end, load(root / 'spec.json'))


class SerializationControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/data1/Kane/MOE')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def check(self, obj, block=65536):
        folder = self.root / str(len(list(self.root.iterdir())))
        folder.mkdir()
        original = save(folder / 'old.json', obj)
        stats = {}
        new = stream.save(folder / 'new.json', obj, deadline=deadline(), metrics=stats, block_bytes=block)
        self.assertEqual(original, new)
        self.assertEqual((folder / 'old.json').read_bytes(), (folder / 'new.json').read_bytes())
        self.assertEqual(new['sha256'], sha(folder / 'new.json'))
        self.assertLessEqual(stats['max_block_bytes'], block)
        self.assertLessEqual(stats['max_fragment_bytes'], 12 * stream.STRING_CHARS)
        self.assertEqual(stats['bytes_written'], new['bytes'])
        self.assertTrue(stats['published'] and stats['returned'])
        self.assertFalse((folder / 'new.json.partial').exists())

    def test_byte_identity_scalars_unicode_rationals_and_signed_zero(self):
        values = [None, False, True, 0, -1, 10**1000, 0., -0., 1e-300,
                  1.7976931348623157e308, '', '\x00\x1f\t\n\r"\\/é中😀\ud800\udfff',
                  '1234567890123456789/987654321', [], {}, (), [1, {'z': (2, 3), 'a': None}]]
        for v in values:
            self.check(v, block=128)

    def test_long_strings_and_many_csr_entries_are_fragmented(self):
        self.check({'matrix': {'shape': [5000, 5000], 'data': ['12/7'] * 5000,
                              'indices': list(range(5000)), 'indptr': list(range(5001))},
                    'text': '中😀\udfff\\' * 20000}, block=4096)

    def test_random_nested_and_finite_float_byte_differential(self):
        rng = random.Random(20260925)
        values = [rng.uniform(-1e200, 1e200) for _ in range(300)]
        values += [math.ldexp(rng.uniform(-1, 1), rng.randrange(-1022, 1023)) for _ in range(300)]
        self.check({'empty': {}, 'nested': [[{'x': x, 'a': [False, None, 'ø']} for x in values[:50]]],
                    'floats': values, 'keys': {'中': 1, '\ud800': 2, '😀': 3}})

    def test_aliases_allowed_cycles_rejected(self):
        shared = {'matrix': [1, 2, 3]}
        self.check([shared, shared])
        cycle = []; cycle.append(cycle)
        with self.assertRaises(ValueError):
            stream.save(self.root / 'cycle', cycle, deadline=deadline())
        self.assertFalse((self.root / 'cycle').exists())

    def test_nonfinite_unsupported_and_resource_shape_fail_closed(self):
        deep = []
        for _ in range(70): deep = [deep]
        values = [float('nan'), float('inf'), -float('inf'), {1: 'coerced-key'},
                  b'bytes', set(), {str(i): i for i in range(4097)}, {'x' * 4097: 1},
                  1 << 12001, deep]
        for i, obj in enumerate(values):
            path = self.root / str(i)
            with self.assertRaises((ValueError, TypeError)):
                stream.save(path, obj, deadline=deadline())
            self.assertFalse(path.exists())

    def test_deadline_before_and_during_write_retains_only_partial(self):
        path = self.root / 'a'
        with self.assertRaises(TimeoutError): stream.save(path, 1, deadline=time.monotonic() - 1)
        self.assertFalse(path.with_name('a.partial').exists())
        flag = [False]
        original = os.write
        def tick():
            if flag[0]: raise TimeoutError('injected elapsed deadline')
        def write(fd, data):
            n = original(fd, data); flag[0] = True; return n
        with patch.object(stream, 'clock', return_value=tick), patch.object(stream.os, 'write', write):
            with self.assertRaises(TimeoutError): stream.save(path, 'x' * 100000, deadline=deadline())
        self.assertFalse(path.exists()); self.assertGreater((self.root / 'a.partial').stat().st_size, 0)

    def test_short_write_hash_and_zero_write_rejection(self):
        original = os.write
        with patch.object(stream.os, 'write', lambda fd, b: original(fd, b[:7])):
            self.check({'x': '中😀' * 500}, block=128)
        with patch.object(stream.os, 'write', return_value=0), self.assertRaises(OSError):
            stream.save(self.root / 'zero', [1, 2], deadline=deadline())
        self.assertFalse((self.root / 'zero').exists())

    def test_fsync_link_failures_no_success_receipt(self):
        for event in ('fsync', 'link'):
            path = self.root / event
            with patch.object(stream.os, event, side_effect=OSError('injected')), self.assertRaises(OSError):
                stream.save(path, {'ok': True}, deadline=deadline())
            self.assertFalse(path.exists()); self.assertTrue(path.with_name(event + '.partial').exists())

    def test_expiration_after_link_is_unreceipted_not_accepted(self):
        flag = [False]; original = os.link; metrics = {}
        def tick():
            if flag[0]: raise TimeoutError('expired after atomic link')
        def link(a, b):
            original(a, b); flag[0] = True
        with patch.object(stream, 'clock', return_value=tick), patch.object(stream.os, 'link', link):
            with self.assertRaises(TimeoutError):
                stream.save(self.root / 'late', 1, deadline=deadline(), metrics=metrics)
        self.assertTrue((self.root / 'late').exists())
        self.assertTrue(metrics['published']); self.assertFalse(metrics['returned'])

    def test_existing_final_partial_symlink_and_publish_race_never_overwrite(self):
        for mode in ('final', 'partial', 'symlink', 'race'):
            path = self.root / mode
            prior = self.root / (mode + '.prior'); prior.write_bytes(b'original')
            original = os.link
            def raced(a, b):
                original(prior, b); original(a, b)
            if mode == 'final': os.link(prior, path)
            if mode == 'partial': os.link(prior, path.with_name(path.name + '.partial'))
            if mode == 'symlink': path.with_name(path.name + '.partial').symlink_to(prior)
            with patch.object(stream.os, 'link', raced if mode == 'race' else original):
                with self.assertRaises(FileExistsError): stream.save(path, [1], deadline=deadline())
            self.assertEqual(prior.read_bytes(), b'original')
            if mode in ('final', 'race'): self.assertEqual(path.read_bytes(), b'original')

    def test_invalid_deadlines_and_block_sizes(self):
        for end in (True, float('nan'), float('inf'), None, time.monotonic() + 301):
            with self.assertRaises(ValueError): stream.save(self.root / 'bad', {}, deadline=end)
        for i, size in enumerate((True, 0, 63, 65537)):
            with self.assertRaises(ValueError):
                stream.save(self.root / str(i), {}, deadline=deadline(), block_bytes=size)


class PipelineControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from residual_proof.tests import fixture
        from residual_proof.supervisor import supervise
        dest = os.environ.get('BOUNDED_CONTROL_EVIDENCE_ROOT')
        cls.temp = None if dest else tempfile.TemporaryDirectory(dir='/data1/Kane/MOE')
        cls.root = Path(dest if dest else cls.temp.name)
        if dest: cls.root.mkdir(parents=True, exist_ok=False)
        assets = cls.root / 'assets'; assets.mkdir()
        cls.spec = fixture(assets)
        cls.spec['evidence_publication'] = POLICY
        cls.good = cls.root / 'good'
        cls.result = supervise(cls.good, cls.spec, budget=300, command_factory=command)
        from scoped_proof.evidence import POSITIVE
        if cls.result['status'] != POSITIVE:
            raise AssertionError(str(cls.result) + '\n' + '\n'.join(p.read_text() for p in cls.good.glob('*.log')))

    @classmethod
    def tearDownClass(cls):
        if cls.temp is not None: cls.temp.cleanup()

    def setUp(self):
        self.work = self.root / self._testMethodName; self.work.mkdir()

    def test_full_pipeline_unchanged_checks_bytes_and_cost(self):
        from residual_proof.audit import audit
        report = audit(self.good)
        self.assertEqual((report['required'], report['checked_bounds'], report['discharged_by_exclusion']), (6, 2, 4))
        bundle = load(self.good / 'construction.json')
        record = save(self.work / 'legacy.json', bundle)
        self.assertEqual(record, load(self.good / 'construction_receipt.json')['construction'])
        self.assertEqual((self.work / 'legacy.json').read_bytes(), (self.good / 'construction.json').read_bytes())
        metrics = load(self.good / 'construction_serialization.json')
        self.assertEqual(metrics['record'], record)
        cost = load(self.good / 'cost.json')
        self.assertEqual(cost['budget_seconds'], 300)
        self.assertLessEqual(metrics['seconds_including_fsync_and_publication'], report['stage_seconds']['construct'])
        self.assertAlmostEqual(cost['stage_seconds'] + cost['overhead_seconds'], cost['end_to_end_seconds'])
        self.assertGreaterEqual(self.result['seconds'], cost['end_to_end_seconds'])

    def test_relocation_fresh_solver_free_full_check(self):
        target = self.work / 'relocated'; shutil.copytree(self.good, target)
        # No original checkpoint/data paths are used by this saved-only audit.
        code = ('import sys; from residual_proof.audit import audit; r=audit(sys.argv[1]); '
                'assert r["complete_output_positive_proof"]; '
                'assert not any(n.split(".")[0] in ("torch","numpy","scipy","highspy","act") for n in sys.modules)')
        run = subprocess.run([PYTHON, '-S', '-c', code, str(target)], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_removed_obligation_changed_factor_guard_or_property_rejected(self):
        from residual_proof.check import check
        from source_enclosure.format import identity
        doc = load(self.good / 'source.json')
        original = load(self.good / 'construction.json')
        for mode in ('missing', 'coefficient', 'factor', 'pair'):
            bundle = copy.deepcopy(original)
            p = bundle['pairs'][0]
            if mode == 'missing': p['obligations']['rows'].pop()
            if mode == 'coefficient': p['projected']['hz']['c'][0] = '1000'
            if mode == 'factor': p['guarded']['continuous_ids'][0] = 'foreign-factor'
            if mode == 'pair': p['pair'] = [0, 1]
            file = self.work / mode
            stream.save(file, bundle, deadline=deadline())
            with self.assertRaises((ValueError, KeyError)):
                check(doc, load(file), invocation=original['invocation'],
                      expected_source_sha256=identity(doc), deadline=deadline())

    def test_no_acceptance_from_partial_final_or_wrong_receipt(self):
        from residual_proof.audit import audit
        for mode in ('partial', 'receipt_missing', 'wrong_run', 'wrong_digest'):
            root = self.work / mode; shutil.copytree(self.good, root)
            if mode == 'partial': (root / 'construction.json').rename(root / 'construction.json.partial')
            elif mode == 'receipt_missing': (root / 'construction_receipt.json').unlink()
            else:
                obj = load(root / 'construction_receipt.json')
                if mode == 'wrong_run': obj['invocation'] = 'other'
                else: obj['construction']['sha256'] = '0' * 64
                (root / 'construction_receipt.json').unlink()
                save(root / 'construction_receipt.json', obj)
            with self.assertRaises((ValueError, FileNotFoundError)): audit(root)

    def test_watchdog_partial_and_exception_charged_without_acceptance(self):
        from residual_proof.supervisor import supervise
        from residual_proof.audit import audit
        for fault in ('hang', 'write_error'):
            def factory(phase, root, end):
                if phase == 'construct':
                    return [PYTHON, '-S', '-m', 'bounded_evidence.probe', fault, str(root), '--deadline', str(end)]
                return command(phase, root, end)
            root = self.work / fault
            result = supervise(root, self.spec, budget=8, command_factory=factory)
            self.assertEqual(result['status'], 'TIMEOUT' if fault == 'hang' else 'ERROR')
            self.assertFalse(result['complete_output_positive_proof'])
            self.assertGreater((root / 'construction.json.partial').stat().st_size, 0)
            self.assertFalse((root / 'construction.json').exists())
            report = audit(root)
            self.assertFalse(report['complete_output_positive_proof'])
            cost = load(root / 'cost.json')
            self.assertGreater(cost['stage_seconds'], 0)
            self.assertAlmostEqual(cost['stage_seconds'] + cost['overhead_seconds'], cost['end_to_end_seconds'])
            stages = load(root / 'terminal.json')['stages']
            self.assertEqual(stages[-1]['phase'], 'construct')
            self.assertTrue(stages[-1]['cleanup_included'])
            from scoped_proof.supervisor import group_rss
            self.assertFalse(group_rss(stages[-1]['pid'])[1])


if __name__ == '__main__': unittest.main()
