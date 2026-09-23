import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from metamoe_assignment_layers import concrete_layer, evaluate_factors


class LayerControls(unittest.TestCase):
    def layer(self,kind,**params):return SimpleNamespace(kind=kind,params=params)
    def test_input(self):
        x=torch.arange(4,dtype=torch.float64)
        self.assertTrue(torch.equal(concrete_layer(self.layer('INPUT'),None,x),x))
    def test_affine_relu(self):
        x=torch.tensor([-1.,2.],dtype=torch.float64)
        self.assertTrue(torch.equal(concrete_layer(self.layer('SCALE',a=torch.tensor([2.,-1.])),x,x),torch.tensor([-2.,-2.])))
        self.assertTrue(torch.equal(concrete_layer(self.layer('BIAS',c=torch.tensor([1.,2.])),x,x),torch.tensor([0.,4.])))
        self.assertTrue(torch.equal(concrete_layer(self.layer('RELU'),x,x),torch.tensor([0.,2.])))
    def test_conv(self):
        x=torch.arange(4,dtype=torch.float64)
        y=concrete_layer(self.layer('CONV2D',input_shape=(1,1,2,2),weight=torch.ones((1,1,2,2),dtype=torch.float64)),x,x)
        self.assertEqual(y.item(),6.)
    def test_pool_and_dense(self):
        x=torch.arange(4,dtype=torch.float64)
        self.assertEqual(concrete_layer(self.layer('AVGPOOL2D',input_shape=(1,1,2,2),kernel_size=2),x,x).item(),1.5)
        self.assertEqual(concrete_layer(self.layer('DENSE',weight=torch.ones((1,4),dtype=torch.float64)),x,x).item(),6.)
    def test_ambiguous_or_unknown_refused(self):
        x=torch.ones(4)
        for l in (self.layer('SCALE',a=torch.ones(2)),self.layer('UNKNOWN')):
            with self.assertRaises(ValueError):concrete_layer(l,x,x)
    def test_prefix_factor_layout(self):
        h=SimpleNamespace(n_cont=1,n_bin=1,c=np.array([1.]),Gc=sp.csr_matrix([[2.]]),Gb=sp.csr_matrix([[3.]]))
        self.assertEqual(evaluate_factors(h,np.array([.5,99.,1.]),2)[0],5.)
        with self.assertRaises(ValueError):evaluate_factors(h,np.array([.5]),1)


if __name__=='__main__':unittest.main()
