import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import torch

from act.pipeline.moe import conv_training as training
from act.pipeline.moe import conv_training_supervisor as supervisor
from act.back_end.moe.conv_factory import ConvOutputMoEConfig, build_conv_output_moe
from act.back_end.moe.factory import load_output_moe_checkpoint


class ConvTrainingTests(unittest.TestCase):
    def setUp(self):training.setup(17,1,'cpu')

    def test_split_is_complete_disjoint_and_repeatable(self):
        a,b=training.split_indices(50000,.1,17)
        self.assertEqual((len(a),len(b)),(45000,5000));self.assertFalse(set(a)&set(b))
        self.assertEqual(len(set(a+b)),50000)
        self.assertEqual((a,b),training.split_indices(50000,.1,17))
        self.assertNotEqual(a,training.split_indices(50000,.1,18)[0])

    def test_earliest_tie_and_no_test_based_selection(self):
        rows=[dict(epoch=i,validation_correct=v,test_accuracy=t) for i,v,t in [(1,3,.9),(2,5,.1),(3,5,1.)]]
        self.assertEqual(training.selected_epoch(rows),2)
        with self.assertRaises(ValueError):training.selected_epoch([])

    def test_atomic_json_rejects_nonfinite_and_preserves_previous(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            path=Path(d)/'x.json';training.atomic_json(path,dict(value=1))
            with self.assertRaises(ValueError):training.atomic_json(path,dict(value=float('nan')))
            self.assertEqual(json.loads(path.read_text()),dict(value=1))

    def test_checkpoint_immutable_and_optimizer_continuation(self):
        cfg=ConvOutputMoEConfig(input_shape=(1,8,8),num_classes=2,num_experts=2,channels=(2,4),hidden=4,router_pool=2)
        model=build_conv_output_moe(cfg)
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        data=[(torch.rand(3,1,8,8),torch.tensor([0,1,0]))]
        metrics=training.train_epoch(model,data,opt,'cpu',.01)
        self.assertGreater(metrics['router_max_update'],0)
        self.assertEqual(sum(metrics['route_counts']),6)
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            path=Path(d)/'epoch.pt'
            from dataclasses import asdict
            training.save_checkpoint(path,dict(format='act-output-conv-moe-v1',factory_config=asdict(cfg),
                state_dict=model.state_dict(),optimizer=opt.state_dict()))
            with self.assertRaises(FileExistsError):training.save_checkpoint(path,{})
            other,payload=load_output_moe_checkpoint(path)
            opt2=torch.optim.AdamW(other.parameters());opt2.load_state_dict(payload['optimizer'])
            training.train_epoch(model,data,opt,'cpu',.01)
            training.train_epoch(other,data,opt2,'cpu',.01)
            for a,b in zip(model.parameters(),other.parameters()):self.assertTrue(torch.equal(a,b))

    def test_json_transient_retry_and_persistent_failure(self):
        with patch.object(Path,'read_text',side_effect=['{','{','{"ok":1}']):
            self.assertEqual(supervisor.read_json_retry('x',delay=0),{'ok':1})
        with patch.object(Path,'read_text',return_value='{') as read:
            with self.assertRaises(ValueError):supervisor.read_json_retry('x',delay=0)
            self.assertEqual(read.call_count,3)

    def test_dead_worker_is_not_infinite_wait(self):
        process=Mock(returncode=-9);process.poll.return_value=-9
        with self.assertRaisesRegex(RuntimeError,'exited -9'):
            supervisor.watch(process,Path('/unused'),'training',100)

    def test_resource_gate_waits_then_proceeds_or_exhausts(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            root=Path(d)
            with patch.object(supervisor.subprocess,'check_output',side_effect=['100','9000']),patch.object(supervisor.time,'sleep') as sleep:
                self.assertEqual(supervisor.resource_wait(root,'training')['free_gpu_mib'],9000)
                sleep.assert_called_once_with(30)
            with patch.object(supervisor.subprocess,'check_output',return_value='100'):
                with self.assertRaises(TimeoutError):supervisor.resource_wait(root,'training',limit=0)

    def test_live_stale_worker_recorded_not_killed(self):
        process=Mock(returncode=0,pid=123);process.poll.side_effect=[None,0]
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            root=Path(d);training.atomic_json(root/'heartbeat.json',{'phase':'train'})
            import os,time
            os.utime(root/'heartbeat.json',(time.time()-2000,time.time()-2000))
            with patch.object(supervisor.time,'sleep'):supervisor.watch(process,root,'training',3000)
            record=json.loads((root/'supervisor.json').read_text())
            self.assertEqual(record['status'],'STALLED_SUSPECTED')
            self.assertGreater(record['heartbeat_age_seconds'],1800)
            process.terminate.assert_not_called()

    def test_cosine_lr_ends_zero_without_shortening_recipe(self):
        p=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([p],lr=.001)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,100)
        for _ in range(100):opt.step();sch.step()
        self.assertEqual(opt.param_groups[0]['lr'],0.)


if __name__=='__main__':unittest.main()
