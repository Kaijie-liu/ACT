"""Saved-record archiving must compare content, not audit wall-clock values."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from archive_metamoe_protected_smoke_r1 import archive


class ArchiveControls(unittest.TestCase):
    def exercise(self, mutate):
        with tempfile.TemporaryDirectory(prefix='protected-archive-', dir='/data1/Kane/MOE') as tmp:
            root = Path(tmp)
            cfg, review = root/'config.json', root/'review.json'
            cfg.write_text(json.dumps({'output_root': str(root/'raw'), 'files': {}}))
            saved = {'audit': 'PASS', 'row': {'status': 'UNKNOWN'}, 'files': {},
                     'separate_audit_seconds': .1}
            review.write_text(json.dumps(saved))
            fresh = copy.deepcopy(saved)
            fresh['separate_audit_seconds'] = .2
            mutate(fresh)
            with patch('archive_metamoe_protected_smoke_r1.audit', return_value=fresh):
                return archive(cfg, review)

    def test_wall_clock_difference_is_not_result_drift(self):
        out = self.exercise(lambda d: None)
        self.assertEqual(out['row']['status'], 'UNKNOWN')
        self.assertEqual(out['original_separate_audit_seconds'], .1)
        self.assertEqual(out['reconstruction_audit_seconds'], .2)

    def test_changed_outcome_cannot_be_archived(self):
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.exercise(lambda d: d['row'].update(status='SAFE'))


if __name__ == '__main__':
    unittest.main()
