from copy import deepcopy
from pathlib import Path
import unittest
from native_import_audit.inventory import inspect
from native_import_audit.probe import model, compare, submit
from lp_sandwich.tests import fixture, csr
from lp_sandwich.check import identity


class Controls(unittest.TestCase):
    def data(self):
        lp={'matrix_format':'csr_v1','c':[-1.,0.],'offset':0.,'lower':[0.,0.],'upper':[1.,1.],
            'E':csr([],2),'h':[],'A':csr([[1.,1e-10]],2),'b':[0.]}
        s=deepcopy(fixture()['statement']);s['lp_sha256']=identity(lp)
        return {'lp':lp,'statement':s,'submitted':model(1e-10)}

    def test_small_inventory(self):
        v=inspect(self.data(),{'small_matrix_value':1e-9,'large_matrix_value':1e15,
                              'infinite_cost':1e20,'infinite_bound':1e20})
        self.assertEqual(v['small_matrix_count'],1)
        self.assertEqual(v['matrix']['count'],2)
        self.assertTrue(v['saved_conversion_reproduced'])

    def test_original_identity_and_saved_submission(self):
        opts={'small_matrix_value':1e-9,'large_matrix_value':1e15,'infinite_cost':1e20,'infinite_bound':1e20}
        for change in ('lp','submitted'):
            v=self.data()
            if change=='lp':v['lp']['b'][0]=1
            else:v['submitted']['rows'][0]['entries'][1][1]=0
            with self.assertRaises(ValueError):inspect(v,opts)

    def test_duplicate_index_rejected(self):
        v=self.data();v['lp']['A']['indices']=[0,0];v['statement']['lp_sha256']=identity(v['lp'])
        with self.assertRaises(ValueError):inspect(v,{'small_matrix_value':1e-9})

    def test_matrix_and_nonmatrix_mutations_distinguished(self):
        a=model(1e-10);b=deepcopy(a);b['rows'][0]['entries'].pop()
        d=compare(a,b);self.assertEqual(len(d['matrix_changes']),1)
        self.assertTrue(d['other_fields_unchanged'] and d['row_bounds_unchanged'])
        b['upper'][0]=2;self.assertFalse(compare(a,b)['other_fields_unchanged'])
        b['rows'][0]['upper']=1;self.assertFalse(compare(a,b)['row_bounds_unchanged'])

    def test_real_sized_input_rejected_before_import(self):
        a=model(1e-10);a['cost']=[0.]*7397
        with self.assertRaises(ValueError):submit('not_analytic',a,Path('/data1/Kane/MOE/ACT/data/moe/results/SHOULD_NOT_EXIST_import_control'))


if __name__=='__main__':unittest.main()
