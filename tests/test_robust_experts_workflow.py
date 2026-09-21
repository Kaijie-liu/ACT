import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from robust_experts_workflow_control import compose, terminal


class WorkflowControls(unittest.TestCase):
    def test_late_partial_missing_not_pass(self):
        receipt = {'status': 'COMPLETED', 'execution_including_preflight_seconds': 1.,
                   'total_with_postflight_seconds': 1.1}
        result = {'status': 'NATIVE_WORKFLOW_CONTROL_PASS'}
        self.assertTrue(terminal(receipt, result)['accepted'])
        for bad in [None, {}, {'status': 'PREPARED'}]:
            self.assertEqual(terminal(receipt, bad)['status'], 'ERROR')
        for status in ['TIMEOUT', 'ERROR', 'SOURCE_CHANGED']:
            value = terminal({**receipt, 'status': status}, result)
            self.assertFalse(value['accepted'])
            self.assertFalse(value['formal_SAFE'])

    def test_native_config_math_and_local_locations(self):
        try:
            import pytorch_lightning
        except ImportError:
            self.skipTest('native composition tested in isolated Robust Experts environment')
        repo = Path('/data1/Kane/MOE/baselines/robust_experts_compat_20260921_r1')
        root = Path('/data1/Kane/MOE/baseline_runs/robust_experts_config_control_only')
        data = Path('/data1/Kane/MOE/baseline_data/robust_experts_20260921_r2')
        _, cfg, saved = compose(repo, root, data)
        self.assertEqual(cfg.model.model.k, 1)
        self.assertEqual(cfg.model.model.num_experts, 4)
        self.assertEqual(cfg.model.attack.steps, 7)
        self.assertEqual(cfg.model.attack.eps, .03137)
        self.assertEqual(cfg.model.optimizer.lr, .1)
        self.assertEqual(cfg.model.optimizer.momentum, .9)
        self.assertEqual(cfg.model.optimizer.weight_decay, .0005)
        self.assertEqual(cfg.model.model.balancing_loss, .5)
        self.assertEqual(cfg.trainer.limit_train_batches, 1)
        self.assertEqual(cfg.trainer.limit_test_batches, 1)
        self.assertEqual(cfg.datamodule.num_workers, 0)
        self.assertFalse(cfg.use_clearml)
        self.assertFalse(cfg.wandb)
        self.assertEqual(set(cfg.logger), {'csv'})
        self.assertEqual(cfg.logger.csv.save_dir, str(root))
        self.assertEqual(cfg.datamodule.data_dir, str(data))
        self.assertEqual(len(saved['datamodule']['train_transforms']['transforms']), 5)


if __name__ == '__main__':
    unittest.main()
