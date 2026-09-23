from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from metamoe_assignment_replay import assess_point


class ReplayControls(unittest.TestCase):
    def args(self):
        return dict(point=np.array([0.]),lower=np.array([-1.]),upper=np.array([1.]),
            original=np.array([2.,1.]),represented=np.array([2.,1.]),
            rows=np.array([[1.,-1.]]),thresholds=np.array([1e-7]))

    def test_safe_point_not_safe_domain(self):
        r=assess_point(**self.args())
        self.assertEqual(r['status'],'NO_VIOLATION_AT_POINT')
        self.assertTrue(r['not_a_robustness_certificate'])

    def test_abstract_violation_not_original_unsafe(self):
        a=self.args(); a['represented']=np.array([0.,1.])
        r=assess_point(**a)
        self.assertEqual(r['status'],'HZ_SOURCE_POINT_MISMATCH')
        self.assertFalse(r['full_model_violation'])

    def test_original_violation_can_be_replayed(self):
        a=self.args(); a['original']=np.array([0.,1.])
        self.assertTrue(assess_point(**a)['full_model_violation'])

    def test_outside_box_never_accepted(self):
        a=self.args(); a['point']=np.array([1.01]); a['original']=np.array([0.,1.])
        self.assertFalse(assess_point(**a)['full_model_violation'])

    def test_nonfinite_not_accepted(self):
        a=self.args(); a['original']=np.array([np.nan,1.])
        self.assertEqual(assess_point(**a)['status'],'INVALID_REPLAY_ARRAYS')

    def test_bad_shape_not_accepted(self):
        a=self.args(); a['represented']=np.zeros(3)
        self.assertEqual(assess_point(**a)['status'],'INVALID_REPLAY_ARRAYS')


if __name__=='__main__': unittest.main()
