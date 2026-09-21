import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dual_rs_public_weights import fetch


class DownloadControls(unittest.TestCase):
    def test_matching_hash_and_atomic_final(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, target = root / "source", root / "target"
            source.write_bytes(b"public-control")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            row = fetch(source.as_uri(), target, digest)
            self.assertEqual(row["sha256"], digest)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertFalse((root / "target.part").exists())

    def test_wrong_hash_preserves_partial_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, target = root / "source", root / "target"
            source.write_bytes(b"control")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                fetch(source.as_uri(), target, "0" * 64)
            self.assertFalse(target.exists())
            self.assertTrue((root / "target.part").exists())

    def test_no_replacement_or_silent_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ["target", "partial.part"]:
                (root / name).write_bytes(b"retained")
            for target in [root / "target", root / "partial"]:
                with self.assertRaisesRegex(ValueError, "target exists"):
                    fetch("unused://not-contacted", target)
            self.assertEqual((root / "target").read_bytes(), b"retained")


if __name__ == "__main__":
    unittest.main()
