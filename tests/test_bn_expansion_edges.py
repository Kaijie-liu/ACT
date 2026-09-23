"""Source/IR graph regressions; no native queries or trained checkpoint."""
import sys
import unittest
from pathlib import Path
import torch
from torch import nn
from act.pipeline.verification.torch2act import _LayerGraphBuilder
from act.util.device_manager import initialize_device
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from metamoe_assignment_layers import concrete_layer


class BranchedBN(nn.Module):
    def __init__(self, root=False, shared=False):
        super().__init__();self.root=root;self.shared=shared
        self.fc=nn.Linear(2,2,dtype=torch.float64)
        self.bn1=nn.BatchNorm1d(2,eps=0.,dtype=torch.float64)
        self.bn2=nn.BatchNorm1d(2,eps=0.,dtype=torch.float64)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[2.,0.],[0.,3.]]));self.fc.bias.fill_(.5)
            self.bn1.weight.copy_(torch.tensor([2.,-3.]));self.bn1.bias.copy_(torch.tensor([1.,2.]))
            self.bn2.weight.copy_(torch.tensor([-2.,4.]));self.bn2.bias.copy_(torch.tensor([3.,-1.]))
    def forward(self,x):
        a=self.fc(x)
        left=self.bn1(a)
        right=(self.bn1 if self.shared else self.bn2)(x if self.root else a)
        return left+right


def ir_forward(model, point):
    layers,preds,succs=_LayerGraphBuilder(model,tuple(point.shape),dtype=torch.float64,sample_input=point).build_layer_graph()
    values={}
    for layer in layers:
        ps=preds[layer.id]
        if layer.kind in ('ADD','SGM_ADD'):
            if len(ps)!=2:raise ValueError('binary add predecessors')
            out=values[ps[0]]+values[ps[1]]
        else:
            if len(ps)>1:raise ValueError('unexpected multi predecessor')
            x=values[ps[0]] if ps else point.reshape(-1)
            out=concrete_layer(layer,x,point)
        values[layer.id]=out
        if layer.params.get('is_batchnorm_decomposition'):
            if ps:
                assert list(layers[ps[0]].out_vars)==list(layer.in_vars)
            else:
                assert list(layer.in_vars)==list(range(point.numel()))
        for pred in ps:assert layer.id in succs[pred]
    return values[layers[-1].id]


class BNGraphControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):initialize_device('cpu','float64')
    def check(self,m,shape):
        m=m.double().eval()
        for seed in range(3):
            torch.manual_seed(seed);x=torch.randn(shape,dtype=torch.float64)
            with torch.no_grad():expected=m(x).reshape(-1)
            torch.testing.assert_close(ir_forward(m,x),expected,atol=1e-12,rtol=1e-12)
    def test_shared_predecessor_not_previous_branch(self):self.check(BranchedBN(),(1,2))
    def test_late_root_branch_not_previous_branch(self):self.check(BranchedBN(root=True),(1,2))
    def test_reused_module_has_distinct_expansion(self):self.check(BranchedBN(shared=True),(1,2))
    def test_two_bn_in_sequence(self):
        self.check(nn.Sequential(BranchedBN().bn1,BranchedBN().bn2,nn.ReLU()),(1,2))
    def test_nonaffine_bn(self):
        bn=nn.BatchNorm2d(2,affine=False,dtype=torch.float64)
        bn.running_mean.copy_(torch.tensor([1.,-2.]));bn.running_var.copy_(torch.tensor([.4,2.]))
        self.check(nn.Sequential(bn,nn.Flatten()),(1,2,2,3))
    def test_bn1d_spatial(self):self.check(nn.Sequential(BranchedBN().bn1,nn.Flatten()),(1,2,3))
    def test_bn3d(self):self.check(nn.Sequential(nn.BatchNorm3d(2),nn.Flatten()),(1,2,2,2,2))
    def test_bn_free_unchanged(self):self.check(nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Linear(3,2)),(1,2))


if __name__=='__main__':unittest.main()
