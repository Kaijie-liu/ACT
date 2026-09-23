import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_metamoe_assignment_layers import compare


class LayerAuditTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'layer':i,'kind':'test','n_out':1,'max_abs_difference':0.,
                    'point_disagrees_at_1e_9':False} for i in range(26)]
        self.arr={f'{i}_{s}':np.ones(1) for i in range(26) for s in ('ir','hz')}
        self.arr.update(source=np.array([2.]),final_represented=np.ones(1))
    def test_source_mismatch_not_hz_mismatch(self):
        result=compare(self.rows,self.arr)
        self.assertIsNone(result['first_disagreeing_layer']);self.assertEqual(result['final_ir_vs_source'],1.)
    def test_missing_or_reordered(self):
        with self.assertRaises(ValueError):compare(self.rows[:-1],self.arr)
        with self.assertRaises(ValueError):compare(self.rows[::-1],self.arr)
        del self.arr['0_hz']
        with self.assertRaises(ValueError):compare(self.rows,self.arr)
    def test_modified_array(self):
        self.arr['3_hz'] += 1.
        with self.assertRaises(ValueError):compare(self.rows,self.arr)
    def test_nan(self):
        self.arr['3_hz'][:]=np.nan
        with self.assertRaises(ValueError):compare(self.rows,self.arr)


if __name__=='__main__':unittest.main()
