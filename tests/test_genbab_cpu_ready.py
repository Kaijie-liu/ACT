"""B2 inventory/branch reception controls; no external numerical imports."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('genbab_cpu_ready',
    Path(__file__).resolve().parents[1]/'scripts/genbab_cpu_ready.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class ReadyControls(unittest.TestCase):
    def test_vendored_metadata_is_not_last_name_wins(self):
        rows = [dict(name='packaging', version='26.3', location='/env/site-packages'),
                dict(name='packaging', version='26.0', location='/env/site-packages/setuptools/_vendor')]
        for ordered in (rows, rows[::-1]):
            mod.check_inventory(ordered, Path('/env/site-packages'), {'packaging': '26.3'})

    def test_wrong_top_level_rejected(self):
        with self.assertRaises(ValueError):
            mod.check_inventory([dict(name='packaging', version='26.0', location='/env/site-packages')],
                                Path('/env/site-packages'), {'packaging': '26.3'})

    def test_no_plain_crown_acceptance(self):
        record = dict(status='verified', stats={'bab': []},
                      graph=[dict(kind='BoundSigmoid')], split_activations={})
        with self.assertRaises(ValueError):
            mod.accepted_smoke(record, '')

    def test_nonlinear_branch_evidence(self):
        record = dict(status='verified', stats={'bab': [[0, .1, 2, .2]]},
                      graph=[dict(kind='BoundSigmoid')],
                      split_activations={'margin': [dict(kind='BoundSigmoid')]})
        self.assertEqual(mod.accepted_smoke(record, 'splitting decisions\nBranching points'), 2)
        record['status'] = 'unknown'
        with self.assertRaises(ValueError):
            mod.accepted_smoke(record, 'splitting decisions\nBranching points')

    def test_guard_rows_preserve_ties(self):
        for x in (-.5, .5):
            y = [0, x, -x, -.5]
            self.assertTrue(all(sum(a*b for a, b in zip(row, y)) <= rhs
                                for row, rhs in zip(mod.C, mod.RHS)))
        self.assertEqual(mod.C[0], [1, 0, 0, 0])
        self.assertFalse(mod.CONFIG['bab.branching.input_split.enable'])
        self.assertEqual(mod.CONFIG['general.complete_verifier'], 'bab')


if __name__ == '__main__':
    unittest.main()
