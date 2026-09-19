from copy import deepcopy
import unittest
from sparse_supervised.flow import STAGES
from sparse_diagnostic_archive.review import aggregate


def failed():
    c={'whole_supplied_LP_seconds':2.5,'observed_phase_sum_seconds':2.,'residual_seconds':.5,
       'native_calls':None,'native_seconds':None,
       'phases':{p:{'seconds':1. if p in ('load','capture') else None} for p in STAGES}}
    return [{'job_id':str(i),'status':'ERROR' if i==0 else 'NOT_RUN_AFTER_ERROR',
             'complete_independent_check':False,'diagnostic':None,'costs':c if i==0 else None} for i in range(4)]


class Controls(unittest.TestCase):
    def test_stopped_roster_and_missing_cost(self):
        a=aggregate(failed())
        self.assertEqual(a['denominator'],4)
        self.assertEqual(a['cost_records'],1)
        self.assertEqual(a['native_count_missing'],4)
        self.assertEqual(a['checked_upper_bounds'],0)
        self.assertEqual(a['phase_costs']['check']['missing_of_four'],4)

    def test_missing_denominator_and_continued_run_rejected(self):
        with self.assertRaises(ValueError):aggregate(failed()[:1])
        rows=failed();rows[1]['status']='TIMEOUT'
        with self.assertRaises(ValueError):aggregate(rows)

    def test_missing_cost_not_zero_and_bad_sum(self):
        rows=failed();rows[1]['costs']=deepcopy(rows[0]['costs'])
        with self.assertRaises(ValueError):aggregate(rows)
        rows=failed();rows[0]['costs']['whole_supplied_LP_seconds']=0
        with self.assertRaises(ValueError):aggregate(rows)
        rows=failed();rows[0]['costs']['native_calls']=0
        with self.assertRaises(ValueError):aggregate(rows)

    def test_failure_not_promoted_to_check(self):
        rows=failed();rows[0]['complete_independent_check']=True
        with self.assertRaises(ValueError):aggregate(rows)


if __name__=='__main__':unittest.main()
