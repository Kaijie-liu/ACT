"""Only tiny interface controls, not the two frozen R1 profiling objects."""
import copy
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_proof.io import ROOT, PYTHON
from source_construction_lab.fixtures import document
from source_cost_controls.profile import run


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/data1/Kane/MOE')
        self.root = Path(self.temp.name)
        # Explicitly NOT small(E4/w4/d1) or medium(E8/w8/d2) from frozen R1.
        self.doc = document(experts=2, classes=2, width=1, depth=0, seed=91)
    def tearDown(self): self.temp.cleanup()

    def test_whole_diagnostic_and_fresh_unchanged_checker(self):
        r = run(self.doc, self.root / 'good', deadline=time.monotonic() + 30)
        self.assertEqual(r['source_check']['original_output_obligations'], 1)
        self.assertEqual(r['source_check']['output_obligations'], 1)
        self.assertFalse(r['complete_output_positive_proof'])
        for phase in r['phases']:
            self.assertAlmostEqual(phase['seconds'], phase['component_seconds'] + phase['other_seconds'])
        code = ('import sys,time; from pathlib import Path; from scoped_proof.io import load; '
                'from residual_proof.check import check; p=Path(sys.argv[1]); r=load(p/"report.json"); '
                'v=check(load(p/"source.json"),load(p/"construction.json"),invocation="profile_v2", '
                'expected_source_sha256=r["source_sha256"],deadline=time.monotonic()+30); '
                'assert v==r["source_check"]; '
                'assert not any(n.split(".")[0] in ("torch","numpy","scipy","act","highspy") for n in sys.modules)')
        out = subprocess.run([PYTHON, '-S', '-c', code, str(self.root / 'good')], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)

    def test_wrapped_missing_and_wrong_bindings_still_rejected(self):
        from shared_route_residual import propose
        original = propose.propose
        for mode in ('list', 'empty', 'wrong_run'):
            def corrupt(*args, **kwargs):
                c = original(*args, **kwargs)
                if mode == 'list': return [c]
                if mode == 'empty': return {}
                c = copy.deepcopy(c); c['binding']['invocation'] = 'different'; return c
            with patch.object(propose, 'propose', corrupt), self.assertRaises(ValueError):
                run(self.doc, self.root / mode, deadline=time.monotonic() + 30)
            self.assertFalse((self.root / mode / 'report.json').exists())

    def test_expired_inherited_deadline_no_result(self):
        with self.assertRaises(TimeoutError):
            run(self.doc, self.root / 'late', deadline=time.monotonic() - 1)
        self.assertFalse((self.root / 'late').exists())

    def test_wrong_source_binding_remains_error(self):
        doc = copy.deepcopy(self.doc); doc['request']['model_state']['sha256'] = '0' * 64
        with self.assertRaises(ValueError): run(doc, self.root / 'bad', deadline=time.monotonic() + 30)
        self.assertFalse((self.root / 'bad/report.json').exists())


if __name__ == '__main__': unittest.main()
