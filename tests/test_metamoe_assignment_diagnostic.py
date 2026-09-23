import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_metamoe_current_assignment import query_groups
from act.back_end.solver.isolated_feasibility import csr_arrays


class ExactQueryIdentityControls(unittest.TestCase):
    def row(self, index, value=1., threshold=0.):
        return {'row': index, 'arrays': {**csr_arrays(sp.csr_matrix([[value, 0.]])),
            'lb': np.array([threshold]), 'ub': np.array([np.inf])}}

    def test_identical_constraints_group_without_dropping_obligations(self):
        result = query_groups([self.row(0), self.row(1), self.row(2, 2.)], 'model-a')
        self.assertEqual([r['rows'] for r in result], [[0, 1], [2]])

    def test_one_ulp_difference_not_merged(self):
        out = query_groups([self.row(0), self.row(1, np.nextafter(1., 2.))], 'model-a')
        self.assertEqual(len(out), 2)

    def test_different_threshold_not_merged(self):
        out = query_groups([self.row(0), self.row(1, threshold=1e-8)], 'model-a')
        self.assertEqual(len(out), 2)

    def test_model_identity_changes_key(self):
        a = query_groups([self.row(0)], 'model-a')
        b = query_groups([self.row(0)], 'model-b')
        self.assertNotEqual(a[0]['identity'], b[0]['identity'])

    def test_missing_or_reordered_property_refused(self):
        for rows in ([self.row(1)], [self.row(1), self.row(0)]):
            with self.assertRaises(ValueError):
                query_groups(rows, 'model-a')

    def test_missing_fields_and_nan_refused(self):
        rows = [self.row(0)]
        del rows[0]['arrays']['ub']
        with self.assertRaises(ValueError):
            query_groups(rows, 'model-a')
        with self.assertRaises(ValueError):
            query_groups([self.row(0, np.nan)], 'model-a')


if __name__ == '__main__':
    unittest.main()
