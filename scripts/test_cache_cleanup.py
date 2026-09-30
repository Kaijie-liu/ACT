"""Destructive operations are confined to disposable fixtures, never live caches."""
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cleanup', HERE / 'clean_regenerable_caches.py')
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cache-control-', dir=HERE.parent / 'data/moe/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.target = 'cache/pip/http-v2'
        self.relative = 'a/b/c/d/e/' + 'a' * 56 + '.body'
        self.file = self.base / self.target / self.relative
        self.file.parent.mkdir(parents=True)
        self.file.write_bytes(b'regenerable wheel bytes')
        old = time.time() - 2 * cleanup.AGE
        os.utime(self.file, (old, old))

    def plan(self):
        return cleanup.plan(self.base, time.time(), [])

    def execute(self, plan, busy=()):
        with (self.base / 'receipt.jsonl').open('x') as receipt:
            cleanup.apply(self.base, plan, receipt, busy)

    def test_plan_readonly_apply_and_keep_evidence(self):
        evidence = self.base / 'failed_result.json'
        evidence.write_text('KEEP')
        document = self.plan()
        self.assertTrue(self.file.exists())
        self.assertEqual(len(document['entries']), 1)
        self.execute(document)
        self.assertFalse(self.file.exists())
        self.assertEqual(evidence.read_text(), 'KEEP')

    def test_recent_retained(self):
        os.utime(self.file, None)
        self.assertEqual(self.plan()['entries'], [])

    def test_mutated_file_refused(self):
        document = self.plan()
        self.file.write_bytes(b'new cache download')
        with self.assertRaises(ValueError):
            self.execute(document)
        self.assertTrue(self.file.exists())

    def test_protected_path_refused_before_unlink(self):
        document = self.plan()
        entry = dict(document['entries'][0], target='baseline_weights')
        document['entries'].append(entry)
        with self.assertRaises(ValueError):
            self.execute(document)
        self.assertTrue(self.file.exists())

    def test_traversal_refused(self):
        document = self.plan()
        document['entries'][0]['relative'] = '../../model.pt'
        with self.assertRaises(ValueError):
            self.execute(document)

    def test_symlink_leaf_and_parent_retained(self):
        self.file.unlink()
        self.file.symlink_to('/etc/hosts')
        self.assertEqual(self.plan()['entries'], [])
        self.file.unlink()
        directory = self.file.parent
        directory.rmdir()
        directory.symlink_to('/tmp')
        self.assertEqual(self.plan()['entries'], [])

    def test_hardlink_retained(self):
        os.link(self.file, self.base / 'other')
        self.assertEqual(self.plan()['entries'], [])

    def test_active_cache_refused(self):
        document = self.plan()
        with self.assertRaises(ValueError):
            self.execute(document, [self.target])
        self.assertTrue(self.file.exists())

    def test_duplicate_refused(self):
        document = self.plan()
        document['entries'] *= 2
        with self.assertRaises(ValueError):
            self.execute(document)
        self.assertTrue(self.file.exists())

    def test_unknown_format_and_orphan_bytecode_retained(self):
        unknown = self.base / self.target / 'checkpoint.pt'
        unknown.write_bytes(b'model')
        os.utime(unknown, (0, 0))
        orphan = self.base / '.pycache' / 'not-present-anywhere.cpython-312.pyc'
        orphan.parent.mkdir()
        orphan.write_bytes(b'orphan')
        os.utime(orphan, (0, 0))
        self.assertEqual(len(self.plan()['entries']), 1)


if __name__ == '__main__':
    unittest.main()
