import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dual_rs_epoch_pipeline import check_epoch_state, summarize_terminal
from dual_rs_training_state import cpu_tree, rng_state


class EpochControls(unittest.TestCase):
    def test_completed_epoch_and_mutations(self):
        m = torch.nn.Linear(2, 1)
        o = torch.optim.AdamW(m.parameters())
        s = torch.optim.lr_scheduler.MultiStepLR(o, [30, 60, 1000], gamma=.5)
        for _ in range(2):
            o.zero_grad()
            m(torch.ones(1, 2)).sum().backward()
            o.step()
        s.step()
        value = cpu_tree({'schema': 'dual_rs_completed_epoch_v1', 'binding': {'test': 'a'},
            'model': m.state_dict(), 'optimizer': o.state_dict(), 'scheduler': s.state_dict(),
            'rng': rng_state(), 'completed_epoch': 1, 'global_step': 2})
        check_epoch_state(value, {'test': 'a'}, 1, 2)
        bads = []
        for key in ['rng', 'scheduler', 'optimizer']:
            bad = copy.deepcopy(value)
            del bad[key]
            bads.append(bad)
        bad = copy.deepcopy(value)
        bad['scheduler']['last_epoch'] = 0
        bads.append(bad)
        bad = copy.deepcopy(value)
        bad['global_step'] = 1
        bads.append(bad)
        bad = copy.deepcopy(value)
        bad['model']['weight'].fill_(float('nan'))
        bads.append(bad)
        bad = copy.deepcopy(value)
        del bad['rng']['numpy']
        bads.append(bad)
        for bad in bads:
            with self.assertRaises(ValueError):
                check_epoch_state(bad, {'test': 'a'}, 1, 2)
        with self.assertRaises(ValueError):
            check_epoch_state(value, {'test': 'b'}, 1, 2)

    def test_outer_timeout_error_and_missing_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            receipt = {'status': 'COMPLETED', 'execution_including_preflight_seconds': .5,
                       'total_with_postflight_seconds': .6}
            self.assertEqual(summarize_terminal(root, receipt)['status'], 'ERROR')
            (root / 'terminal.json').write_text(json.dumps({'status': 'CONTROL_PASS'}))
            self.assertEqual(summarize_terminal(root, receipt)['status'], 'CONTROL_PASS')
            for status in ['TIMEOUT', 'ERROR', 'SOURCE_CHANGED']:
                result = summarize_terminal(root, {**receipt, 'status': status})
                self.assertEqual(result['status'], status)
                self.assertTrue(result['partial_evidence_preserved'])
                self.assertEqual(result['with_postflight_seconds'], .6)


if __name__ == '__main__':
    unittest.main()
