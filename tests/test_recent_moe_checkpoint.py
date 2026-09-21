"""Identity rejection must work before importing Torch or unpickling anything."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/recent_moe_metamoe_checkpoint_smoke.py"


class CheckpointIdentityControls(unittest.TestCase):
    def invoke(self, repo, path):
        return subprocess.run([sys.executable, "-S", str(SCRIPT), "--repo", str(repo),
                               "--checkpoint", str(path), "--sha256", "0" * 64],
                              capture_output=True, text=True, timeout=5)

    def test_wrong_path_rejected_before_loading(self):
        with tempfile.TemporaryDirectory(prefix="moe-checkpoint-control-") as temp:
            result = self.invoke(temp, Path(temp) / "outside.pth")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("only pinned author artifacts accepted", result.stderr)
        self.assertNotIn("import torch", result.stderr)

    def test_wrong_hash_rejected_before_loading(self):
        with tempfile.TemporaryDirectory(prefix="moe-checkpoint-control-") as temp:
            path = Path(temp) / "paper/artifacts/not-a-pickle.pth"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"not a model")
            result = self.invoke(temp, path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checkpoint identity mismatch", result.stderr)
        self.assertNotIn("import torch", result.stderr)


if __name__ == "__main__":
    unittest.main()
