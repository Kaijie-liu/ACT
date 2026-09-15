"""Supplemental regression for the actual NumPy property boundary (not Torch)."""
import unittest
from unittest.mock import patch
import numpy as np
from act.back_end.moe.weighted_top2 import linear_safety_rows
from act.front_end.specs import OutputSpec,OutKind
from act.back_end.solver.lp_certificate import rational
from scripts.test_conv_pre_f0_r2 import PreF0Tests


class ScalarBoundaryTests(unittest.TestCase):
    def test_actual_property_scalars(self):
        rows=linear_safety_rows(OutputSpec(kind=OutKind.TOP1_ROBUST,y_true=0),10)
        self.assertEqual(len(rows),9)
        for q,c in rows:
            self.assertIs(type(q[0]),np.float64)
            with self.assertRaises(ValueError):rational(q[0])
            canonical=[float(v) for v in q]
            self.assertEqual(canonical,list(q));self.assertEqual(sum(canonical),0)
            for v in canonical:self.assertIn(rational(v),(-1,0,1))
            self.assertEqual(float(c),0)

    def test_capture_with_numpy_instead_of_mock_torch_scalars(self):
        # Reuse the full construction-only test with the real scalar kind.
        # All model/front-end operations stay mocked; this is not a new query.
        with patch('torch.tensor',side_effect=lambda value,**kw:np.asarray(value,dtype=np.float64)):
            PreF0Tests('test_capture_stops_before_float_f0').test_capture_stops_before_float_f0()


if __name__=='__main__':unittest.main()
