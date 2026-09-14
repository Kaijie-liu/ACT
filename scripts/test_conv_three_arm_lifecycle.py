"""Post-smoke test-harness repair; preserves the frozen execution/test hashes.

The historical test does exists() then read() on /proc/PID/stat. Kernel reaping
can remove that file between the calls. This replacement accepts ESRCH/ENOENT
and briefly allows an already SIGKILLed child to reach its terminal state.
No production code, request, budget or archived smoke outcome changes.
"""
import errno
import os
from pathlib import Path
import sys
import tempfile
import time

from scripts import test_conv_three_arm as historical
from scripts.conv_three_arm_contract import ROOT
from scripts.run_conv_three_arm import execute


def gone_or_zombie(path):
    try:
        return Path(path).read_text().split()[2] == 'Z'
    except OSError as exc:
        if exc.errno in (errno.ESRCH, errno.ENOENT):
            return True
        raise


class ContractTests(historical.ContractTests):
    def test_process_group_deadline(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as d:
            path=Path(d);pidfile=path/'child.pid'
            command=[sys.executable,'-c',
                'import subprocess,time,pathlib,sys; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                f'pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)']
            code,expired=execute(command,path/'log',time.monotonic(),os.environ,budget=1)
            self.assertTrue(expired);self.assertNotEqual(code,0)
            pid=int(pidfile.read_text());stat=Path(f'/proc/{pid}/stat')
            until=time.monotonic()+1
            while not gone_or_zombie(stat) and time.monotonic()<until:
                time.sleep(.01)
            self.assertTrue(gone_or_zombie(stat))

    def test_reaping_race_and_unrelated_io_errors(self):
        from unittest.mock import patch
        for number in (errno.ESRCH, errno.ENOENT):
            with patch.object(Path,'read_text',side_effect=OSError(number,'gone')):
                self.assertTrue(gone_or_zombie('/proc/test/stat'))
        with patch.object(Path,'read_text',side_effect=PermissionError(errno.EACCES,'denied')):
            with self.assertRaises(PermissionError):gone_or_zombie('/proc/test/stat')
