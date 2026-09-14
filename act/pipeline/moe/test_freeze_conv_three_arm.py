import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock

from act.pipeline.moe import freeze_conv_three_arm as freeze
from act.pipeline.moe import conv_three_arm_worker as worker
from act.pipeline.moe.conv_training import atomic_json,sha


class ConvThreeArmFreezeTests(unittest.TestCase):
    def test_selection_ignores_route_and_bound_metadata(self):
        rows=[dict(dataset_index=i,label=0,prediction=(i==2),route_count=100-i,bound=i) for i in range(7)]
        self.assertEqual(freeze.choose_indices(rows,{0,1},3,2),([0,1],[3,4,5]))
        for r in rows:r.update(route_count=-1,bound=float('nan'))
        self.assertEqual(freeze.choose_indices(rows,{0,1},3,2),([0,1],[3,4,5]))

    def test_insufficient_clean_inputs_is_not_replacement(self):
        with self.assertRaises(ValueError):freeze.choose_indices([],set(),30,2)

    def test_job_schedule_balances_each_position(self):
        jobs=freeze.jobs([dict(dataset_index=i) for i in range(30)])
        self.assertEqual(len(jobs),90);self.assertEqual(len({j['job_id'] for j in jobs}),90)
        for arm in freeze.ARMS:
            for pos in range(3):self.assertEqual(sum(j['method']==arm and j['position']==pos for j in jobs),10)
        for i in range(30):self.assertEqual({j['method'] for j in jobs if j['rank']==i},set(freeze.ARMS))

    def test_eight_expert_or_unknown_method_rejected(self):
        request={'protocol':'CONV_FAMILY_THREE_ARM_R1','method':'adaptive','epsilon':2/255,
                 'topology':{'num_experts':4,'top_k':2,'classes':10}}
        worker.validate_request(request)
        with self.assertRaises(ValueError):worker.validate_request({**request,'method':'legacy'})
        with self.assertRaises(ValueError):worker.validate_request({**request,'topology':{'num_experts':8,'top_k':2,'classes':10}})
        with self.assertRaises(ValueError):worker.validate_request({**request,'epsilon':1/255})

    def test_monolithic_dispatch_does_not_fall_through_to_external(self):
        import torch
        from act.pipeline.moe import staged_verifier as staged
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            root=Path(d);cfg=root/'config.json';atomic_json(cfg,{'comparison_method':'monolithic_f0'})
            request={'protocol':'CONV_FAMILY_THREE_ARM_R1','method':'monolithic','epsilon':2/255,
                'topology':{'num_experts':4,'top_k':2,'classes':10},'config':{'path':str(cfg),'sha256':sha(cfg)},
                'subject':{'checkpoint':'unchanged.pt','checkpoint_sha256':'abc'},
                'sample':{'label':0,'dataset_index':12},'head':'frozen'}
            atomic_json(root/'request.json',request)
            net=Mock(experts=[1,2,3,4]);net.spec.top_k=2
            report=Mock(evidence={})
            with patch.object(worker,'load',return_value=(net,{'center':torch.zeros(1,3,32,32)})),\
                 patch.object(worker,'frontend') as external,\
                 patch.object(staged,'verify_staged_linf',return_value=report) as verify,\
                 patch.object(staged,'write_evidence_package') as write:
                worker.worker(root,123.)
                external.assert_not_called();verify.assert_called_once();write.assert_called_once()
                self.assertEqual(verify.call_args.args[3]['comparison_method'],'monolithic_f0')
                self.assertEqual(verify.call_args.kwargs['budget_started_at'],123.)

    def test_freeze_has_no_overwrite_path(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d,patch.object(freeze,'RAW',Path(d)):
            with self.assertRaises(FileExistsError):freeze.freeze()

    def test_protocol_keeps_evidence_levels_and_fair_facts(self):
        cfg=json.loads(freeze.PROTOCOL.read_text())
        self.assertEqual(cfg['execution']['outer_seconds_per_request'],300)
        self.assertEqual(cfg['execution']['full_requests'],30*3)
        self.assertNotEqual(cfg['arms']['adaptive']['evidence_level'],cfg['arms']['crown']['evidence_level'])
        a=json.loads((freeze.ROOT/cfg['arms']['adaptive']['config']).read_text())
        b=json.loads((freeze.ROOT/cfg['arms']['monolithic']['config']).read_text())
        a.pop('comparison_method');b.pop('comparison_method');self.assertEqual(a,b)
        self.assertEqual(a['route_complexity_schedule']['multi_pair_tier1_fraction'],.25)


if __name__=='__main__':unittest.main()
