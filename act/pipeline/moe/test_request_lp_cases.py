import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from act.back_end.moe import GateKind, OutputMoEFactoryConfig, build_output_moe
from act.pipeline.moe.request_lp_cases import generate
from act.pipeline.moe.check_request_lp import check_directory, RATIONAL_TRUSTED
from act.util.device_manager import initialize_device


class GenericRequestTests(unittest.TestCase):
    def fixture(self):
        initialize_device('cpu','float64'); torch.set_num_threads(1)
        net=build_output_moe(OutputMoEFactoryConfig(input_shape=(2,),num_classes=2,
            num_experts=3,top_k=2,gate=GateKind.SELECTED_SOFTMAX,
            router_hidden=(),expert_hidden=(),seed=7)).cpu().double().eval()
        with torch.no_grad():
            net.router[1].weight.zero_(); net.router[1].bias.zero_()
            for expert,margin in zip(net.experts,[-.2,1.,2.]):
                expert[1].weight.zero_(); expert[1].bias.copy_(torch.tensor([margin,0.]))
        tensors={'center':torch.full((1,2),.5),'lower':torch.full((1,2),.4),'upper':torch.full((1,2),.6)}
        request={'classes':2,'experts':3,'top_k':2,'tie_policy':'ANY_LEGAL_TOPK','clean_prediction':0}
        return net,tensors,request

    def test_generic_all_ties_shared_rational_and_portable_check(self):
        net,tensors,request=self.fixture()
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as root:
            with patch('act.back_end.moe.weighted_top2.build_weighted_top2_f0',side_effect=AssertionError('float F0 called')):
                generate(net,tensors,request,Path(root))
            result=check_directory(root)
            self.assertEqual(result['required'],3)
            self.assertEqual(result['counts'],{'reused':1,'residual':2,'unknown':0})
            self.assertEqual(result['trusted_base'],RATIONAL_TRUSTED)
            # Standard-library-only checker from a fresh process (-S removes site packages).
            repo=Path(__file__).resolve().parents[3]
            done=subprocess.run([sys.executable,'-S',str(repo/'scripts/check_moe_request_lp.py'),root],
                                cwd=root,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            self.assertEqual(json.loads(done.stdout),result)
            m=json.loads((Path(root)/'manifest.json').read_text())
            changed=copy.deepcopy(m); changed['obligations'].pop()
            (Path(root)/'manifest.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError):check_directory(root)

    def test_failed_proposals_remain_unknown_without_floating_fallback(self):
        net,tensors,request=self.fixture()
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as root:
            with patch('act.back_end.solver.lp_certificate.propose',side_effect=ValueError('registered limit')):
                generate(net,tensors,request,Path(root))
            checked=check_directory(root)
            self.assertEqual(checked['status'],'UNKNOWN')
            self.assertEqual(checked['counts']['unknown'],3)


if __name__=='__main__': unittest.main()
