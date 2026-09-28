"""Navigation regression tests only: no model/data/solver/environment imports."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
IMPORT = importlib.util.spec_from_file_location('layout', ROOT / 'scripts/check_project_layout.py')
m = importlib.util.module_from_spec(IMPORT)
IMPORT.loader.exec_module(m)


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.spec = m.read_json(ROOT / m.SPEC)

    def test_catalogue_matches(self):
        self.assertEqual((ROOT / m.CATALOG).read_text(), m.render_catalog(self.spec))

    def test_duplicate_and_escaping_roots_rejected(self):
        for name in ('act', '../outside', '/root', '.', ''):
            changed = copy.deepcopy(self.spec)
            changed['groups'][1]['roots'].append(name)
            with self.assertRaises(ValueError):
                m.directory_owners(changed)

    def test_all_repository_checks(self):
        result = m.check()
        self.assertEqual(result['frozen_sources_unchanged'], 619)
        self.assertEqual(result['experiments_started'], 0)

    def test_workspace_unknown_and_ambiguous_rejected(self):
        with self.assertRaises(ValueError):
            m.workspace_category('unregistered-run', self.spec)
        changed = copy.deepcopy(self.spec)
        changed['workspace_rules'].append(changed['workspace_rules'][0])
        with self.assertRaises(ValueError):
            m.workspace_category('ACT', changed)

    def test_temp_name_is_not_deletion_permission(self):
        category = m.workspace_category('test_missing_obligation_xyz', self.spec)
        rule = next(r for r in self.spec['workspace_rules'] if r['category'] == category)
        self.assertIn('失败证据', rule['policy'])
        self.assertEqual(m.workspace_category('portable_conv98_20260915_v1', self.spec),
                         '审阅与可搬迁制品')

    def test_hash_and_line_changes_rejected(self):
        m.check_history(ROOT, self.spec['history'])
        for key, value in [('sha256', '0' * 64), ('lines', 1)]:
            changed = dict(self.spec['history'], **{key: value})
            with self.assertRaises(ValueError):
                m.check_history(ROOT, changed)

    def test_paths_cannot_escape(self):
        for relative in ('../../outside', '/etc/passwd'):
            with self.assertRaises(ValueError):
                m.local_path(ROOT, relative)

    def test_relative_links_and_missing_files(self):
        # Test fixtures live only in a disposable task directory under MOE.
        with tempfile.TemporaryDirectory(prefix='project-layout-', dir=ROOT) as raw:
            root = Path(raw)
            (root / 'target.md').write_text('target\n')
            doc = root / 'index.md'
            doc.write_text('[ok](target.md) [web](https://example.org) [anchor](#local)\n')
            self.assertEqual(m.check_links(root, ['index.md']), (1, 0))
            doc.write_text('[missing](missing.md)\n')
            with self.assertRaises(ValueError):
                m.check_links(root, ['index.md'])

    def test_json_duplicate_keys_rejected(self):
        with tempfile.TemporaryDirectory(prefix='project-layout-', dir=ROOT) as raw:
            path = Path(raw) / 'bad.json'
            path.write_text('{"schema":1,"schema":2}')
            with self.assertRaises(ValueError):
                m.read_json(path)


if __name__ == '__main__':
    unittest.main()
