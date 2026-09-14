from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
import torch

from act.back_end.moe.conv_factory import ConvOutputMoEConfig, build_conv_output_moe
from act.back_end.moe import load_output_moe_checkpoint, build_act_moe_program
from act.pipeline.moe.experiment1 import _propagate_component
from act.config.config import HybridZConfig
from act.front_end.specs import OutputSpec,OutKind
from act.util.device_manager import initialize_device


class ConvFactoryTests(unittest.TestCase):
    def setUp(self):
        initialize_device('cpu','float64');torch.set_num_threads(1)

    def test_rng_and_versioned_roundtrip(self):
        cfg=ConvOutputMoEConfig(input_shape=(1,8,8),num_classes=2,num_experts=3,channels=(2,4),hidden=4,router_pool=2)
        before=torch.get_rng_state().clone();net=build_conv_output_moe(cfg).double().eval()
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        x=torch.linspace(.1,.9,64).reshape(1,1,8,8)
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as root:
            path=Path(root)/'model.pt'
            torch.save({'format':'act-output-conv-moe-v1','factory_config':asdict(cfg),'state_dict':net.state_dict()},path)
            loaded,_=load_output_moe_checkpoint(path)
            self.assertTrue(torch.equal(net(x),loaded.double().eval()(x)))
        output,decision=net.forward_with_routing(x)
        values=torch.stack([e(x) for e in net.experts],1)
        selected=torch.gather(values,1,decision.indices.unsqueeze(-1).expand(-1,-1,2))
        self.assertTrue(torch.equal(output,(decision.weights.unsqueeze(-1)*selected).sum(1)))

    def test_actual_conv_pool_point_lowering(self):
        cfg=ConvOutputMoEConfig(input_shape=(1,8,8),num_classes=2,num_experts=2,channels=(2,4),hidden=4,router_pool=2)
        net=build_conv_output_moe(cfg).double().eval()
        # Dyadic control avoids inconsistent singleton bounds from independent
        # BLAS/conv reductions. It does not weaken the production acceptance gate.
        with torch.no_grad():
            for parameter in net.parameters():parameter.copy_(parameter.mul(16).round().div(16))
        x=(torch.arange(64,dtype=torch.float64)%16/16).reshape(1,1,8,8)
        program=build_act_moe_program(net,center=x,lower=x,upper=x,
            output_spec=OutputSpec(kind=OutKind.TOP1_ROBUST,y_true=[0]))
        hzcfg=HybridZConfig(max_input_dim=1,guarded_support_enabled=False,expert_property_solver_backend='scipy')
        for concrete,component in [(net.router,program.router),*zip(net.experts,program.experts)]:
            hz=_propagate_component(component,hybridz_config=hzcfg).output_hz
            expected=concrete(x).detach().numpy().reshape(-1)
            import numpy as np
            np.testing.assert_allclose(np.asarray(hz.c).reshape(-1),expected,rtol=0,atol=1e-12)
            self.assertTrue(hz.exact)

    def test_nondegenerate_box_keeps_sparse_conv_relations(self):
        from act.back_end.solver.solver_hz import SparseHZono,sparse_hz_fast_bounds
        cfg=ConvOutputMoEConfig(input_shape=(1,8,8),num_classes=2,num_experts=2,channels=(2,4),hidden=4,router_pool=2)
        net=build_conv_output_moe(cfg).double().eval(); x=torch.full((1,1,8,8),.5)
        program=build_act_moe_program(net,center=x,lower=x-.01,upper=x+.01,
            output_spec=OutputSpec(kind=OutKind.TOP1_ROBUST,y_true=[0]))
        hzcfg=HybridZConfig(max_input_dim=1,guarded_support_enabled=False,expert_property_solver_backend='scipy')
        for concrete,component in [(net.router,program.router),*zip(net.experts,program.experts)]:
            hz=_propagate_component(component,hybridz_config=hzcfg).output_hz
            self.assertIsInstance(hz,SparseHZono); self.assertTrue(hz.exact)
            bound=sparse_hz_fast_bounds(hz)
            for delta in (-.01,0,.01):
                y=concrete(x+delta).detach().reshape(-1)
                self.assertTrue(bool((y>=bound.lb.reshape(-1)-1e-12).all()))
                self.assertTrue(bool((y<=bound.ub.reshape(-1)+1e-12).all()))

    def test_bad_topology_rejected(self):
        for args in ({'input_shape':(3,31,32)},{'channels':(16,)},{'num_experts':1},{'router_pool':3}):
            with self.assertRaises(ValueError):ConvOutputMoEConfig(**args)


if __name__=='__main__':unittest.main()
