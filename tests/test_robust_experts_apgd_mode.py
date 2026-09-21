"""Native torchattacks control, with no real-data training or model selection."""
import copy
import importlib.util
from pathlib import Path
import unittest
import torch


class APGDModeControls(unittest.TestCase):
    def setUp(self):
        try:
            import torchattacks
        except ImportError:
            self.skipTest('run in frozen Robust Experts environment')
        torch.set_num_threads(2)
        torch.manual_seed(812)
        self.model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1),
            torch.nn.BatchNorm2d(4), torch.nn.Flatten(), torch.nn.Linear(4*4*4, 3)).eval()
        self.x = torch.rand(2, 3, 4, 4)
        self.y = self.model(self.x).argmax(1)

    def run_attack(self, model, suffix):
        path = Path('/data1/Kane/MOE/baselines')/('robust_experts_compat_20260921_'+suffix)/'src/utils/attack.py'
        spec = importlib.util.spec_from_file_location('native_attack_'+suffix, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.auto_pgd(model, torch.nn.Identity(), self.x, self.y, steps=4)[0]

    def test_native_restore_drift_and_unchanged_attack_tensor(self):
        original, fixed = copy.deepcopy(self.model), copy.deepcopy(self.model)
        before = copy.deepcopy(self.model.state_dict())
        old_adv = self.run_attack(original, 'r1')
        new_adv = self.run_attack(fixed, 'r3')
        self.assertTrue(original.training)
        self.assertFalse(fixed.training)
        self.assertTrue(torch.equal(old_adv, new_adv))
        self.assertTrue(all(torch.equal(v, fixed.state_dict()[k]) for k, v in before.items()))
        original(old_adv)
        fixed(new_adv)
        self.assertEqual(int(original[1].num_batches_tracked), 1)
        self.assertEqual(int(fixed[1].num_batches_tracked), 0)
        self.assertTrue(all(torch.equal(v, fixed.state_dict()[k]) for k, v in before.items()))

    def test_training_caller_still_restored_as_training(self):
        self.model.train()
        self.run_attack(self.model, 'r3')
        self.assertTrue(self.model.training)
        self.assertTrue(self.model[1].training)


if __name__ == '__main__':
    unittest.main()
