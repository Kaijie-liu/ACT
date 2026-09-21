import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/recent_moe_deployment.py"
spec = importlib.util.spec_from_file_location("deployment", SCRIPT)
deployment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deployment)


class DeploymentControls(unittest.TestCase):
    def run_control(self, code, seconds=10):
        with tempfile.TemporaryDirectory(prefix="moe-deploy-control-") as temp:
            path = Path(temp) / "attempt"
            result = deployment.supervise([sys.executable, "-c", code], temp, path,
                                          seconds, "CONTROL")
            self.assertEqual(result, json.loads((path / "receipt.json").read_text()))
            self.assertGreaterEqual(result["total_with_postflight_seconds"],
                                    result["execution_including_preflight_seconds"])
            with self.assertRaises(FileExistsError):
                deployment.supervise([sys.executable, "-c", "pass"], temp, path, 10, "CONTROL")
            return result, (path / "stdout.txt").read_text()

    def test_success(self):
        result, text = self.run_control("print('only deployment, not certified accuracy')")
        self.assertEqual(result["status"], "COMPLETED")
        self.assertIn("not certified", text)

    def test_error(self):
        result, _ = self.run_control("raise RuntimeError('intentional')")
        self.assertEqual(result["status"], "ERROR")

    def test_timeout_retains_partial_output(self):
        result, text = self.run_control("import time;print('partial',flush=True);time.sleep(5)", .3)
        self.assertEqual(result["status"], "TIMEOUT")
        self.assertEqual(text.strip(), "partial")

    def test_invalid_budget(self):
        for value in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                deployment.supervise([], ".", "/unused", value, "CONTROL")


if __name__ == "__main__":
    unittest.main()
