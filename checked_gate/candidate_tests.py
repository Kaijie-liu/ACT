import os
from pathlib import Path
import sys
import tempfile
import subprocess
import time
import unittest

from checked_gate.candidate_run import execute


class Lifecycle(unittest.TestCase):
    def test_fresh_checker_import_fraction_dependency(self):
        code=('from checked_gate.bootstrap import setup; setup(checker=True); '
              'from fractions import Fraction; '
              'from checked_gate.replacement import check; print(Fraction(1,3))')
        result=subprocess.run([sys.executable,'-S','-c',code],capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),'1/3')

    def test_complete_exception_deadline_and_partial(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as directory:
            root=Path(directory)
            for name,code,cap,want in [
                ('good','print("done")',2,'COMPLETED'),
                ('error','raise RuntimeError("controlled")',2,'ERROR'),
                ('deadline','import time; print("partial",flush=True); time.sleep(10)',.15,'TIMEOUT')]:
                with self.subTest(name=name):
                    row=execute([sys.executable,'-S','-c',code],root/name,time.monotonic()+cap,os.environ.copy())
                    self.assertEqual(row['state'],want)
                    self.assertGreater(row['seconds'],0)
            self.assertIn('partial',(root/'deadline').read_text())
            row=execute([sys.executable,'-c','raise AssertionError'],root/'expired',time.monotonic()-1,os.environ.copy())
            self.assertFalse(row['started']); self.assertFalse((root/'expired').exists())


if __name__=='__main__': unittest.main()
