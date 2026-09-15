import math
from pathlib import Path
import tempfile
import time
import os
import sys
import unittest
from scripts.optional_evidence_budget import EvidenceBudget, EvidenceBudgetExpired, terminal_status


class BudgetTests(unittest.TestCase):
    def test_one_clock_across_stages(self):
        now=[10.]
        b=EvidenceBudget(10, clock=lambda:now[0])
        now[0]=42
        self.assertEqual(b.remaining(),268)
        now[0]=250
        with self.assertRaises(EvidenceBudgetExpired):b.grant(60,80)
        self.assertEqual(b.remaining(2),58)

    def test_no_new_budget_at_each_query(self):
        b=EvidenceBudget(0,clock=lambda:215)
        self.assertEqual(b.grant(60,80),5)

    def test_deadline_and_missing_check_fail_closed(self):
        self.assertEqual(terminal_status('CHECKED_CONDITIONAL',300.001,300,True),'TIMEOUT')
        self.assertEqual(terminal_status('CHECKED_CONDITIONAL',199,300,False),'UNKNOWN_INCOMPLETE_EVIDENCE')
        self.assertEqual(terminal_status('CHECKED_CONDITIONAL',199,300,True),'CHECKED_CONDITIONAL')

    def test_invalid_budget(self):
        for started,total in [(math.inf,300),(0,math.nan),(0,-1),(100,300)]:
            with self.assertRaises(ValueError):EvidenceBudget(started,total,clock=lambda:0)

    def test_outer_watchdog_kills_overrunning_child(self):
        from scripts.run_optional_evidence_dev import stage
        root=Path(__file__).resolve().parents[1]/'data/moe/results'
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            budget=EvidenceBudget(time.monotonic(),total=2.15)
            result=stage([sys.executable,'-S','-c','import time; time.sleep(5)'],Path(tmp),'overrun',budget,dict(os.environ))
            self.assertEqual(result['state'],'OUTER_TIMEOUT')
            self.assertLess(result['elapsed_seconds'],2)


if __name__=='__main__':unittest.main()
