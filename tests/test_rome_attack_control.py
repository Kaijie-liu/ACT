import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rome_autoattack_control import terminal, validate


class RomeControls(unittest.TestCase):
    def test_outer_precedence_and_missing_output(self):
        receipt = {'status': 'COMPLETED', 'execution_including_preflight_seconds': 2.,
                   'total_with_postflight_seconds': 2.1}
        good = {'status': 'ATTACK_EVALUATION_COMPLETED'}
        self.assertTrue(terminal(receipt, good)['result_accepted'])
        for missing in [None, {}, {'status': 'PREPARED'}]:
            self.assertEqual(terminal(receipt, missing)['status'], 'ERROR')
        for state in ['TIMEOUT', 'ERROR', 'SOURCE_CHANGED']:
            actual = terminal({**receipt, 'status': state}, good)
            self.assertFalse(actual['result_accepted'])
            self.assertEqual(actual['status'], state)
            self.assertEqual(actual['total_with_postflight_seconds'], 2.1)

    def test_unimplemented_config_and_nonfinite_budget_reject(self):
        base = {'norm': 'Linf', 'version': 'standard', 'batch_size': 1,
                'source_s_b': [4, 6], 'epsilon': 8/255}
        for key, value in [('norm', 'L2'), ('batch_size', 2), ('source_s_b', [4, 2]),
                           ('epsilon', float('nan')), ('epsilon', -1)]:
            with self.assertRaises(ValueError):
                validate({**base, key: value})


if __name__ == '__main__':
    unittest.main()
