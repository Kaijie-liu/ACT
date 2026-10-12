"""Audit/terminal mutations over the unchanged R1 archive; no numerical rerun."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.audit_affine_reference_controls import audit
from scripts.run_affine_reference_controls import execute

ARCHIVE=Path('/data1/Kane/MOE/runs/hz_affine_references_20261013_r1')


class ArchiveControls(unittest.TestCase):
    def test_identity_numeric_and_cost_mutations_rejected(self):
        changes=[('mixed.receipt.json',lambda d:d.__setitem__('source_path','/tmp/unrelated')),
                 ('deep.receipt.json',lambda d:d['work'].__setitem__('arithmetic',-1)),
                 ('deep.operator.json',lambda d:d['forward'][0].__setitem__(0,999.)),
                 ('relocated-deep.stdout',lambda d:d.__setitem__('source','0'*64)),
                 ('terminal.json',lambda d:d['calls'][1].__setitem__('parent_observed_seconds',.000001))]
        for name,mutate in changes:
            with self.subTest(name=name),tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
                root=Path(tmp)/'copy'; shutil.copytree(ARCHIVE,root)
                p=root/name; d=json.loads(p.read_bytes()); mutate(d); p.write_text(json.dumps(d))
                with self.assertRaises(ValueError): audit(root)

    def test_missing_constraint_cannot_be_hidden_by_new_forest_hash(self):
        from scoped_source.hz_request_store import compact,digest
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp)/'copy'; shutil.copytree(ARCHIVE,root)
            p=root/'deep.forest.json'; d=json.loads(p.read_bytes()); d['roles']['pairs'].pop(); raw=compact(d); p.write_bytes(raw)
            p=root/'deep.receipt.json'; d=json.loads(p.read_bytes()); d['forest_sha256']=digest(raw); p.write_text(json.dumps(d))
            with self.assertRaises(ValueError): audit(root)

    def test_launch_exception_retains_terminal_and_cost(self):
        for exc,status in [(OSError('simulated launch failure'),'ERROR'),
                           (subprocess.TimeoutExpired('simulated',300),'TIMEOUT')]:
            with self.subTest(status=status),tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
                root=Path(tmp)/'attempt'
                with patch('scripts.run_affine_reference_controls.subprocess.check_output',side_effect=['f'*40,'']),\
                     patch('scripts.run_affine_reference_controls.subprocess.run',side_effect=exc):
                    with self.assertRaises(type(exc)): execute(root)
                t=json.loads((root/'terminal.json').read_bytes())
                self.assertEqual(t['status'],'FAILED_OR_REFUSED'); self.assertEqual(len(t['calls']),1)
                c=t['calls'][0]; self.assertEqual(c['status'],status); self.assertIsNone(c['returncode'])
                self.assertGreater(c['parent_observed_seconds'],0)
                self.assertGreaterEqual(t['outer_before_terminal_seconds'],c['parent_observed_seconds'])


if __name__=='__main__': unittest.main()
