import copy
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_robust_experts_saved_state import check_state


class SavedStateControls(unittest.TestCase):
    def fixture(self):
        state = {'weight': torch.tensor([1., 2.])}
        return ({'epoch': 0, 'global_step': 1, 'state_dict': copy.deepcopy(state),
                 'optimizer_states': [{'param_groups': [{'params': [0]}],
                     'state': {0: {'momentum_buffer': torch.tensor([.1, .2])}}}],
                 'lr_schedulers': [{'last_epoch': 1}]}, state,
                {'status': 'NATIVE_WORKFLOW_CONTROL_PASS'})

    def test_complete(self):
        value = check_state(*self.fixture())
        self.assertEqual(value['prediction_tensors'], 1)
        self.assertFalse(value['full_training_resume_proven'])

    def test_mutated_or_missing_prediction(self):
        for missing in [False, True]:
            checkpoint, prediction, result = self.fixture()
            if missing:
                prediction.clear()
            else:
                prediction['weight'][0] = 3.
            with self.assertRaises(ValueError):
                check_state(checkpoint, prediction, result)

    def test_nonfinite_or_wrong_momentum_binding(self):
        for bad in ['nonfinite', 'binding', 'missing']:
            checkpoint, prediction, result = self.fixture()
            state = checkpoint['optimizer_states'][0]['state']
            if bad == 'nonfinite':
                state[0]['momentum_buffer'][0] = float('nan')
            elif bad == 'binding':
                state[9] = state.pop(0)
            else:
                state.clear()
            with self.assertRaises(ValueError):
                check_state(checkpoint, prediction, result)

    def test_position_and_failed_control(self):
        for bad in ['step', 'scheduler', 'failed']:
            checkpoint, prediction, result = self.fixture()
            if bad == 'step':
                checkpoint['global_step'] = 2
            elif bad == 'scheduler':
                checkpoint['lr_schedulers'][0]['last_epoch'] = 0
            else:
                result['status'] = 'ERROR'
            with self.assertRaises(ValueError):
                check_state(checkpoint, prediction, result)


if __name__ == '__main__':
    unittest.main()
