import copy
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_robust_experts_saved_state_v2 import check_chained_state


class ChainedStateControls(unittest.TestCase):
    def fixture(self):
        state = {'weight': torch.tensor([1.])}
        chain = {'T_max': 1, 'exponent': .9, '_last_lr': [0.], '_schedulers': [
            {'last_epoch': 1, 'base_lrs': [.1], 'start_factor': .01, 'total_iters': 0},
            {'last_epoch': 1, 'base_lrs': [.1], '_last_lr': [0.]}]}
        checkpoint = {'epoch': 0, 'global_step': 1, 'state_dict': copy.deepcopy(state),
            'lr_schedulers': [chain], 'optimizer_states': [{'param_groups': [{'params': [0], 'lr': 0.}],
                'state': {0: {'momentum_buffer': torch.tensor([.1])}}}]}
        return checkpoint, state, {'status': 'NATIVE_WORKFLOW_CONTROL_PASS'}

    def test_both_child_positions_without_fabricating_top_level(self):
        checkpoint, state, result = self.fixture()
        before = copy.deepcopy(checkpoint['lr_schedulers'])
        value = check_chained_state(checkpoint, state, result)
        self.assertTrue(value['accepted'])
        self.assertEqual(checkpoint['lr_schedulers'], before)
        self.assertNotIn('last_epoch', checkpoint['lr_schedulers'][0])

    def test_bad_child_or_lr_rejected(self):
        for field in ['first', 'second', 'last_lr', 'missing', 'horizon']:
            checkpoint, state, result = self.fixture()
            chain = checkpoint['lr_schedulers'][0]
            if field in ['first', 'second']:
                chain['_schedulers'][int(field == 'second')]['last_epoch'] = 0
            elif field == 'last_lr':
                chain['_last_lr'] = [.1]
            elif field == 'missing':
                chain['_schedulers'].pop()
            else:
                chain['T_max'] = 100
            with self.assertRaises(ValueError):
                check_chained_state(checkpoint, state, result)

    def test_tensor_check_not_bypassed(self):
        checkpoint, state, result = self.fixture()
        state['weight'][0] = 2.
        with self.assertRaises(ValueError):
            check_chained_state(checkpoint, state, result)


if __name__ == '__main__':
    unittest.main()
