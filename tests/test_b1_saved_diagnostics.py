"""Captured-matrix decoding and attribution controls, without solver calls."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('saved_diagnostics',
    Path(__file__).resolve().parents[1]/'scripts/b1_saved_diagnostics.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fixture():
    arrays = {}
    def arr(name,values,kind='<f8'):
        a = np.array(values,dtype=kind)
        arrays[name] = dict(dtype=kind,count=a.size,base64=base64.b64encode(a.tobytes()).decode())
        return name
    for name,values in [('objective',[2.,-1.]),('row_lb',[-np.inf]),('row_ub',[1.]),
                        ('var_lb',[0.,0.]),('var_ub',[1.,1.])]:
        arr('native.'+name,values)
    arr('native.integrality',[0,1],'<i8')
    mat = dict(shape=[1,2],data=arr('A.data',[1.,1.]),
        indices=arr('A.indices',[0,1],'<i8'),indptr=arr('A.indptr',[0,2],'<i8'))
    return dict(schema='NATIVE_F0_DIAGNOSTIC_CAPTURE_V1',
        native=dict(variables=2,objective_offset=(.25).hex()),arrays=arrays,
        matrices={'native.A':mat},scope={'request':'fixed'},sources={})


class SavedControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/data1/Kane/MOE')
        self.path=Path(self.tmp.name)/'source.json'
        raw=json.dumps(fixture()).encode();self.path.write_bytes(raw)
        self.digest=hashlib.sha256(raw).hexdigest()
        self.source=m.StoredProblem(self.path,self.digest,time.monotonic()+60)

    def tearDown(self): self.tmp.cleanup()

    def test_same_matrix_constant_and_readonly(self):
        s=self.source
        self.assertEqual(s.A.toarray().tolist(),[[1.,1.]])
        self.assertEqual(s.residuals([0.,1.],relaxed=False)['objective'],-.75)
        self.assertFalse(s.objective.flags.writeable)
        self.assertEqual(s.integrality.tolist(),[0,1])

    def test_fractional_lp_point_not_integer_witness(self):
        self.assertTrue(self.source.residuals([0.,.5],relaxed=True)['numerical_feasible_1e7'])
        r=self.source.residuals([0.,.5],relaxed=False)
        self.assertFalse(r['numerical_feasible_1e7'])
        self.assertFalse(r['exact_feasibility_checked'])
        self.assertEqual(r['integrality_violation'],.5)

    def test_row_and_box_violations(self):
        r=self.source.residuals([1.,1.],relaxed=False)
        self.assertEqual(r['row_violation'],1.)
        self.assertFalse(r['numerical_feasible_1e7'])
        self.assertEqual(self.source.residuals([-1.,0.],relaxed=False)['box_violation'],1.)

    def test_identity_and_deadline(self):
        with self.assertRaises(ValueError):
            m.StoredProblem(self.path,'0'*64,time.monotonic()+60)
        with self.assertRaises(TimeoutError):
            m.StoredProblem(self.path,self.digest,time.monotonic()-1)
        self.source.deadline=time.monotonic()+.5
        with self.assertRaises(TimeoutError):
            self.source.remaining(120)

    @unittest.skipUnless(os.environ.get('B1_NATIVE_CONTROL')=='1','explicit tiny native control')
    def test_native_lp_and_milp_same_matrix_with_offset(self):
        # Exactly two fixed synthetic calls; no checkpoint/dataset/propagation.
        for relaxed in (True,False):
            result=self.source.solve_same_matrix(relaxed=relaxed,time_limit=2.)
            self.assertEqual(result['status'],0)
            self.assertAlmostEqual(result['primal'],-.75)
            self.assertLess(result['numerical_lower'],-.75)
            self.assertGreater(result['numerical_lower'],-.751)
            self.assertTrue(result['candidate']['numerical_feasible_1e7'])
            self.assertEqual(result['native_calls'],1)
            self.assertFalse(result['repairs_online'])


if __name__ == '__main__': unittest.main()
