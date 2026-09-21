import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from metamoe_unfolded_export import export_unfolded, session


class UnfoldedControls(unittest.TestCase):
    def test_eval_bn_preserved(self):
        import torch
        import numpy as np
        torch.set_num_threads(2)
        torch.manual_seed(100)
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1),
                                    torch.nn.BatchNorm2d(4), torch.nn.ReLU()).eval()
        model[1].running_mean.fill_(.3)
        model[1].running_var.fill_(.7)
        x = torch.randn(1, 3, 8, 8)
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            p = Path(tmp) / 'test.onnx'
            info = export_unfolded(model, x, p)
            self.assertEqual(info['onnx_bn_count'], 1)
            runtime = session(p)
            for probe in [x, torch.zeros_like(x)]:
                with torch.no_grad():
                    ref = model(probe).numpy()
                out = runtime.run(None, {'input': probe.numpy()})[0]
                self.assertLess(float(np.max(np.abs(out-ref))), 1e-4)

    def test_training_rejected(self):
        import torch
        with self.assertRaisesRegex(ValueError, 'eval-only'):
            export_unfolded(torch.nn.BatchNorm2d(3), torch.zeros(1, 3, 8, 8), 'unused')

    def test_batch_dependent_rejected(self):
        import torch
        with self.assertRaisesRegex(ValueError, 'batch-dependent'):
            export_unfolded(torch.nn.BatchNorm2d(3, track_running_stats=False).eval(),
                            torch.zeros(1, 3, 8, 8), 'unused')


if __name__ == '__main__':
    unittest.main()
