import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_metamoe_current_assignment import check_point, recover_and_compare, scalar_rows, csr


def matrix(a, prefix=''):
    from scipy.sparse import csr_matrix
    v = csr_matrix(a, dtype=float)
    return {prefix+'data': v.data, prefix+'indices': v.indices,
            prefix+'indptr': v.indptr, prefix+'shape': np.array(v.shape)}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.z = {**matrix([[1., -1.]]), **matrix([[1., 1.]], 'value_'),
                  'integrality': np.array([0, 1]), 'var_lb': np.array([-1., 0.]),
                  'var_ub': np.ones(2), 'row_lb': np.zeros(1), 'row_ub': np.zeros(1),
                  'value_center': np.zeros(1)}
        self.point = np.zeros(2)
        self.mapping = {**matrix([[1.]], 'Gc_'), **matrix([[0.]], 'Gb_'), 'center':np.zeros(1)}
        self.replay = {'point':np.zeros(1), 'center':np.zeros(1), 'lower':-np.ones(1),
                       'upper':np.ones(1), 'source':np.ones(1), 'padded':np.ones(1),
                       'represented':np.zeros(1), 'rows':np.ones((1,1)), 'thresholds':np.array([.1])}

    def test_valid_numeric_point_not_source_witness(self):
        self.assertEqual(check_point(self.z,self.point)['rows'],1)
        result=recover_and_compare(self.z,self.point,self.mapping,self.replay)
        self.assertEqual(result['status'],'HZ_SOURCE_POINT_MISMATCH')
        self.assertFalse(result['full_model_violation'])

    def test_partial_or_nan_point(self):
        for x in (np.zeros(1),np.array([np.nan,0.])):
            with self.assertRaises(ValueError): check_point(self.z,x)

    def test_bounds_or_integrality(self):
        for x in (np.array([2.,2.]),np.array([.5,.5])):
            with self.assertRaises(ValueError): check_point(self.z,x)

    def test_all_base_rows_checked(self):
        with self.assertRaisesRegex(ValueError,'row residual'):
            check_point(self.z,np.array([1.,0.]))

    def test_input_map_tamper(self):
        changed=copy.deepcopy(self.mapping); changed['center'] += .1
        with self.assertRaisesRegex(ValueError,'recovery'):
            recover_and_compare(self.z,self.point,changed,self.replay)

    def test_output_map_tamper(self):
        changed=copy.deepcopy(self.z); changed['value_center'] += .1
        with self.assertRaisesRegex(ValueError,'output map'):
            recover_and_compare(changed,self.point,self.mapping,self.replay)

    def test_outside_box(self):
        self.replay['lower']=np.ones(1)
        with self.assertRaisesRegex(ValueError,'outside'):
            recover_and_compare(self.z,self.point,self.mapping,self.replay)

    def test_changed_padded_expert(self):
        self.replay['padded'] += .1
        with self.assertRaisesRegex(ValueError,'source/padded'):
            recover_and_compare(self.z,self.point,self.mapping,self.replay)

    def test_nonfinite_forward(self):
        self.replay['source'][:]=np.inf; self.replay['padded'][:]=np.inf
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            recover_and_compare(self.z,self.point,self.mapping,self.replay)

    def test_unsorted_row_evaluated_as_stored(self):
        q=csr({'data':np.array([2.,1.]),'indices':np.array([1,0]),
               'indptr':np.array([0,2]),'shape':np.array([1,2])})
        self.assertEqual(scalar_rows(q,np.array([3.,4.]))[0],11.)


if __name__ == '__main__': unittest.main()
