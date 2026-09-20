"""Actual module/checkpoint interface control; no forward or solver execution."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class Capture(unittest.TestCase):
    def test_real_checkpoint_types_and_pinned_state(self):
        import torch
        from act.back_end.moe.conv_factory import ConvOutputMoEConfig,build_conv_output_moe
        from act.pipeline.moe.staged_verifier import _model_state_identity,_tensor_identity
        from router_source.capture import capture,sha
        from router_source.checker import check,compact,digest
        from router_source.propose import propose
        config=ConvOutputMoEConfig(input_shape=(1,8,8),num_classes=2,num_experts=3,
                                  channels=(1,1),hidden=2,router_pool=2)
        model=build_conv_output_moe(config)
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp);ck=root/'model.pt';inp=root/'input.pt'
            torch.save({'format':'act-output-conv-moe-v1','factory_config':asdict(config),
                        'state_dict':model.state_dict()},ck)
            model=model.double().eval()
            center=torch.full((1,1,8,8),.5,dtype=torch.float64)
            tensors={'center':center,'lower':center-.01,'upper':center+.01};torch.save(tensors,inp)
            request={'experts':3,'classes':2,'top_k':2,'tie_policy':'ANY_LEGAL_TOPK',
                     'model_state':_model_state_identity(model),
                     **{k:_tensor_identity(v) for k,v in tensors.items()}}
            job={'parent_request':{'subject':{'checkpoint':str(ck),'checkpoint_sha256':sha(ck)},
                                   'tensors':{'path':str(inp),'sha256':sha(inp)}}}
            with patch.object(torch.nn.Module,'_call_impl',side_effect=AssertionError('forward called')):
                doc=capture(job,{'request':request})
            result=check(doc,propose(doc),expected_request=request,expected_source_sha256=digest(compact(doc)))
            self.assertEqual(result['all_pairs'],3)


if __name__=='__main__':unittest.main()
