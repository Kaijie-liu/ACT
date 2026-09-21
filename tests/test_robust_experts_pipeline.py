import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robust_experts_supervised_pipeline import STAGES, terminal
from recent_moe_deployment import sha256, supervise
from robust_experts_workflow_control import write


class PipelineControls(unittest.TestCase):
    def folder(self):
        return tempfile.TemporaryDirectory(prefix='robust-pipeline-',dir='/data1/Kane/MOE/tmp')

    def complete(self,root):
        for s in STAGES:
            write(root/(s+'.json'),{'status':'COMPLETED','arm':'dense','config_sha256':'frozen'})
            write(root/(s+'_finished.json'),{'returncode':0})
        write(root/'inner_terminal.json',{'status':'PIPELINE_COMPLETED','arm':'dense',
            'config_sha256':'frozen','stage_hashes':{s:sha256(root/(s+'.json')) for s in STAGES}})

    def result(self,root,status='COMPLETED'):
        return terminal(root,{'status':status,'execution_including_preflight_seconds':1,
            'total_with_postflight_seconds':1.2},'frozen','dense','control')

    def test_complete(self):
        with self.folder() as p:
            root=Path(p);self.complete(root)
            self.assertTrue(self.result(root)['accepted'])

    def test_partial_never_accepted(self):
        with self.folder() as p:
            root=Path(p)
            write(root/'train.json',{'status':'COMPLETED'})
            self.assertFalse(self.result(root)['accepted'])

    def test_timeout_late_complete_never_accepted(self):
        with self.folder() as p:
            root=Path(p);self.complete(root)
            self.assertEqual(self.result(root,'TIMEOUT')['status'],'TIMEOUT')
            self.assertFalse(self.result(root,'ERROR')['accepted'])

    def test_bad_identity_and_mutation(self):
        with self.folder() as p:
            root=Path(p);self.complete(root)
            self.assertFalse(terminal(root,{'status':'COMPLETED',
                'execution_including_preflight_seconds':1,'total_with_postflight_seconds':1},
                'other','dense','control')['accepted'])
            # Intentional mutation confined to this generated test fixture.
            (root/'train.json').write_text('{}')
            self.assertFalse(self.result(root)['accepted'])

    def test_outer_deadline_retains_partial(self):
        with self.folder() as p:
            root=Path(p)/'attempt'
            code="import time;print('partial train',flush=True);time.sleep(10)"
            r=supervise([sys.executable,'-c',code],p,root,.3,'CONTROL')
            self.assertEqual(r['status'],'TIMEOUT')
            self.assertIn('partial train',(root/'stdout.txt').read_text())
            self.assertGreaterEqual(r['total_with_postflight_seconds'],r['execution_including_preflight_seconds'])

    def test_exception_and_nonoverwrite(self):
        with self.folder() as p:
            root=Path(p)/'attempt'
            r=supervise([sys.executable,'-c',"raise ValueError('fixture')"],p,root,10,'CONTROL')
            self.assertEqual(r['status'],'ERROR')
            with self.assertRaises(FileExistsError):supervise([sys.executable,'-c','pass'],p,root,10,'CONTROL')

    def test_recipe_native_limits(self):
        from robust_experts_supervised_pipeline import configuration
        raw=json.loads(Path('configs/recent_moe/robust_experts_paper_training_recipe_r1.json').read_text())
        for recipe in raw['recipes'].values():
            cfg=configuration(recipe,Path('/data1/Kane/MOE/not-run'),'/data1/Kane/MOE/data','control')
            self.assertEqual(cfg.datamodule.batch_size,640)
            self.assertEqual(cfg.datamodule.num_workers,2)
            self.assertEqual(cfg.attack_batch_size,256)
            self.assertEqual(cfg.model.scheduler.T_max,200)
            self.assertEqual(cfg.model.attack.steps,7)
            self.assertEqual(cfg.trainer.max_epochs,2)
            self.assertEqual(cfg.callbacks.checkpoint.save_top_k,-1)

    def test_state_digest_serializable_identity(self):
        import torch
        from robust_experts_supervised_pipeline import state_digest
        a={'weight':torch.tensor([1.,2.]),'counter':torch.tensor(1)}
        self.assertEqual(len(json.loads(json.dumps(state_digest(a)))),64)
        self.assertEqual(state_digest(a),state_digest(dict(reversed(list(a.items())))))
        self.assertNotEqual(state_digest(a),state_digest({**a,'counter':torch.tensor(2)}))


if __name__=='__main__':unittest.main()
