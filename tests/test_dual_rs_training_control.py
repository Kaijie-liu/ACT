import copy
import hashlib
import json
import random
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dual_rs_training_state as state
from dual_rs_training_control import summarize_terminal
from recent_moe_deployment import supervise


class TrainingStateControls(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        self.model, self.opt, self.scheduler = self.build()
        self.batch = [torch.randn(4, 3), torch.zeros(4), torch.ones(4), torch.ones(4, 3)]
        self.binding = {"loader_batch": 4, "materialized_batch_sha256": state.digest(self.batch), "scope": "toy"}
        self.step()
        self.saved = state.snapshot(self.model, self.opt, self.scheduler, self.binding,
                                    {"epoch": 0, "batch_index": 0, "next_chunk": 1, "global_step": 1}, self.batch)

    @staticmethod
    def build():
        model = torch.nn.Linear(3, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [30, 60, 1000], gamma=.5)
        return model, opt, scheduler

    def step(self):
        self.opt.zero_grad()
        # Exercise all three CPU RNG families, not only model weights.
        loss = self.model(torch.randn(4, 3) + random.random() + np.random.rand()).square().mean()
        loss.backward()
        self.opt.step()

    def test_restore_exact_next_update(self):
        self.step()
        expected = state.snapshot(self.model, self.opt, self.scheduler, self.binding,
                                  {"epoch": 0, "batch_index": 0, "next_chunk": 2, "global_step": 2}, self.batch)
        self.model, self.opt, self.scheduler = self.build()
        state.restore(self.saved, self.model, self.opt, self.scheduler, self.binding)
        self.step()
        actual = state.snapshot(self.model, self.opt, self.scheduler, self.binding, expected["cursor"], self.batch)
        self.assertEqual(state.digest(expected), state.digest(actual))

    def test_missing_state_binding_cursor_and_batch_rejected(self):
        corruptions = []
        for key in ["optimizer", "scheduler", "rng", "batch"]:
            bad = copy.deepcopy(self.saved)
            del bad[key]
            corruptions.append(bad)
        bad = copy.deepcopy(self.saved)
        bad["binding"]["scope"] = "other-request"
        corruptions.append(bad)
        bad = copy.deepcopy(self.saved)
        bad["cursor"]["next_chunk"] = 0
        corruptions.append(bad)
        bad = copy.deepcopy(self.saved)
        bad["batch"][0][0, 0] += 1
        corruptions.append(bad)
        bad = copy.deepcopy(self.saved)
        del bad["rng"]["python"]
        corruptions.append(bad)
        for bad in corruptions:
            with self.subTest(keys=list(bad)):
                with self.assertRaises(ValueError):
                    state.validate(bad, self.binding)

    def test_atomic_checkpoint_hash_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "checkpoint.pt"
            record = state.save_snapshot(target, self.saved)
            loaded = state.load_snapshot(target, record["file_sha256"], self.binding)
            self.assertEqual(state.digest(loaded), state.digest(self.saved))
            with self.assertRaises(ValueError):
                state.save_snapshot(target, self.saved)
            with self.assertRaises(ValueError):
                state.load_snapshot(target, "0" * 64, self.binding)

    def test_partial_save_not_published(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "checkpoint.pt"
            with patch.object(torch, "save", side_effect=RuntimeError("disk failure")):
                with self.assertRaises(RuntimeError):
                    state.save_snapshot(target, self.saved)
            self.assertFalse(target.exists())
            self.assertTrue(target.with_suffix(".pt.partial").exists())
            with self.assertRaises(ValueError):
                state.save_snapshot(target, self.saved)

    def test_epoch_scheduler_milestone_replay(self):
        for _ in range(29):
            self.step()
            self.scheduler.step()
        saved = state.snapshot(self.model, self.opt, self.scheduler, self.binding,
                               self.saved["cursor"], self.batch)
        self.assertEqual(self.opt.param_groups[0]["lr"], .01)
        self.step()
        self.scheduler.step()
        expected = (state.digest(self.model.state_dict()), state.digest(self.opt.state_dict()),
                    state.digest(self.scheduler.state_dict()))
        self.assertEqual(self.opt.param_groups[0]["lr"], .005)
        self.model, self.opt, self.scheduler = self.build()
        state.restore(saved, self.model, self.opt, self.scheduler, self.binding)
        self.step()
        self.scheduler.step()
        self.assertEqual(expected, (state.digest(self.model.state_dict()), state.digest(self.opt.state_dict()),
                                    state.digest(self.scheduler.state_dict())))


class OuterControlTests(unittest.TestCase):
    def run_child(self, code, seconds):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name) / "run"
        r = supervise([sys.executable, "-c", code], temp.name, root, seconds, "TEST")
        return root, r

    def test_exception_stops_and_accounts(self):
        root, r = self.run_child("raise RuntimeError('intentional')", 5)
        terminal = summarize_terminal(root, r)
        self.assertEqual(terminal["status"], "ERROR")
        self.assertTrue(all(s["status"] == "NOT_STARTED" for s in terminal["stages"]))
        self.assertGreaterEqual(terminal["total_with_postflight_seconds"], terminal["execution_including_preflight_seconds"])

    def test_deadline_retains_partial_no_late_success(self):
        # The nested worker inherits the parent's process group, just like phases.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            code = ("import subprocess,sys;from pathlib import Path;"
                    f"Path({str(root / 'reference_started.json')!r}).write_text('{{}}');"
                    f"Path({str(root / 'after_step1.pt.partial')!r}).write_bytes(b'partial');"
                    "subprocess.run([sys.executable,'-c','import time;time.sleep(10)'])")
            r = supervise([sys.executable, "-c", code], d, root, .4, "TEST")
            terminal = summarize_terminal(root, r)
            self.assertEqual(terminal["status"], "TIMEOUT")
            self.assertEqual([s["status"] for s in terminal["stages"]], ["INTERRUPTED", "NOT_STARTED", "NOT_STARTED"])
            self.assertIn("after_step1.pt.partial", terminal["partial_files_retained"])
            self.assertFalse((root / "audit.json").exists())

    def test_outer_timeout_cannot_be_promoted_by_success_files(self):
        root, receipt = self.run_child("import time;time.sleep(10)", .2)
        for phase in ["reference", "resume", "audit"]:
            (root / f"{phase}_finished.json").write_text(json.dumps({"phase": phase, "returncode": 0}))
        (root / "audit.json").write_text(json.dumps({"audit": "PASS"}))
        self.assertEqual(summarize_terminal(root, receipt)["status"], "TIMEOUT")


class NativeLossFailureControls(unittest.TestCase):
    def test_kl_target_underflow_is_detectable_without_training(self):
        import torch.nn.functional as F
        counts = {}
        for dtype in [torch.float32, torch.float64]:
            logits = torch.tensor([[-150., 150., 0.], [-140., 140., 0.]], dtype=dtype, requires_grad=True)
            parts = torch.chunk(logits, 2, 0)
            target = sum(F.softmax(x, 1) for x in parts) / 2
            loss = sum(F.kl_div(F.log_softmax(x, 1), target, reduction="sum") for x in parts) / 2
            grad, = torch.autograd.grad(loss, logits)
            self.assertTrue(torch.isfinite(loss))
            counts[dtype] = int((~torch.isfinite(grad)).sum())
        self.assertGreater(counts[torch.float32], 0)
        self.assertEqual(counts[torch.float64], 0)


if __name__ == "__main__":
    unittest.main()
