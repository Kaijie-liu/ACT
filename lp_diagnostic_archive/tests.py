from copy import deepcopy
import unittest
from lp_diagnostic_archive.review import aggregate

class ArchiveControls(unittest.TestCase):
    def test_denominator_cost_and_inexact_witness_rejection(self):
        d={'network_SAFE':False,'network_UNSAFE':False,'primal_status':'NOT_EXACTLY_FEASIBLE',
            'upper_bound':None,'lower_bound':'-1','exact_optimality':False,
            'violation_counts':{'equality':1},'classification':'UNRESOLVED_CANDIDATE_VS_LP_RELAXATION'}
        c={'whole_diagnostic_seconds':6,'observed_phase_sum_seconds':5,'residual_clock_seconds':1,
            'phases':{p:{'seconds':x} for p,x in zip(('load','propose','package','check'),(1,2,1,1))},
            'native_calls':1,'native_seconds':.1}
        rows=[{'job_id':str(i),'status':'CHECKED_LP_DIAGNOSTIC','complete_independent_check':True,
            'diagnostic':deepcopy(d),'costs':deepcopy(c)} for i in range(4)]
        a=aggregate(rows);self.assertEqual(a['total_diagnostic_seconds'],24)
        self.assertEqual(a['checked_upper_bounds'],0);self.assertEqual(a['recorded_native_calls'],4)
        with self.assertRaises(ValueError):aggregate(rows[:-1])
        for key,value in [('upper_bound','-1'),('exact_optimality',True),('network_UNSAFE',True)]:
            bad=deepcopy(rows);bad[0]['diagnostic'][key]=value
            with self.assertRaises(ValueError):aggregate(bad)
        bad=deepcopy(rows);bad[0]['costs']['whole_diagnostic_seconds']=0
        with self.assertRaises(ValueError):aggregate(bad)

if __name__=='__main__':unittest.main()
