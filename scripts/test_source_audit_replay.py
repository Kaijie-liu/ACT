"""Controls for separating informational live hashes, never scientific facts."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('replay', ROOT / 'scripts/check_source_audit_replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class Replay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT.parent, prefix='source-replay-control-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'source.py').write_bytes(b'new implementation')
        self.digest = hashlib.sha256(b'new implementation').hexdigest()
        self.saved = dict(frozen_sources=dict(reviewed_call_path={
            'source.py': dict(frozen_sha256='0' * 64, current_sha256='0' * 64,
                              current_equals_frozen=True)}), inward_inputs=100, gain_count=23,
                          boxes=[dict(lower='-1/2', upper='1/2')])
        self.fresh = copy.deepcopy(self.saved)
        self.fresh['frozen_sources']['reviewed_call_path']['source.py'].update(
            current_sha256=self.digest, current_equals_frozen=False)

    def test_only_verified_live_identity_may_differ(self):
        saved = copy.deepcopy(self.saved)
        fresh = copy.deepcopy(self.fresh)
        result = replay.compare(self.saved, self.fresh, self.root)
        self.assertEqual(len(result['separately_validated_live_identity_changes']), 2)
        self.assertEqual(result['ignored_scientific_fields'], [])
        self.assertEqual(self.saved, saved)
        self.assertEqual(self.fresh, fresh)

    def test_scientific_field_or_frozen_identity_changes_rejected(self):
        for mutate in (lambda d: d.update(inward_inputs=99),
                       lambda d: d.update(gain_count=22),
                       lambda d: d['boxes'][0].update(lower='-1'),
                       lambda d: d['frozen_sources']['reviewed_call_path']['source.py'].update(frozen_sha256='1' * 64)):
            changed = copy.deepcopy(self.fresh)
            mutate(changed)
            with self.assertRaises(ValueError):
                replay.compare(self.saved, changed, self.root)

    def test_unverified_live_identity_or_missing_path_rejected(self):
        for mutate in (lambda r: r.update(current_sha256='1' * 64),
                       lambda r: r.update(current_equals_frozen=True)):
            changed = copy.deepcopy(self.fresh)
            mutate(changed['frozen_sources']['reviewed_call_path']['source.py'])
            with self.assertRaises(ValueError):
                replay.compare(self.saved, changed, self.root)
        changed = copy.deepcopy(self.fresh)
        changed['frozen_sources']['reviewed_call_path'].clear()
        with self.assertRaises(ValueError):
            replay.compare(self.saved, changed, self.root)


if __name__ == '__main__':
    unittest.main()
