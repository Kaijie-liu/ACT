import copy
import unittest
from upstream_cost_analysis.analyze import windows,readiness


class AnalysisControls(unittest.TestCase):
    def test_partition_and_reject_bad_clocks(self):
        s={'start_seconds':10.,'end_seconds':30.,'elapsed_seconds':20.,'state':'COMPLETED'}
        q=[{'key':'s0_1_gate_lo','entered_seconds':12.,'seconds':3.,'status':'PROPOSED'},
           {'key':'s0_1_p0_weighted','entered_seconds':20.,'seconds':4.,'status':'PROPOSED'}]
        v=windows(s,q)
        self.assertEqual((v['query_window_seconds'],v['prefix_seconds'],v['inter_query_seconds'],v['suffix_seconds']),(7,2,5,6))
        self.assertEqual(v['outside_query_windows_seconds'],13)
        self.assertIsNone(v['native_solver_seconds'])
        for key,value in (('entered_seconds',14),('seconds',-1),('status','PENDING')):
            bad=copy.deepcopy(q);bad[1][key]=value
            with self.assertRaises(ValueError):windows(s,bad)
        with self.assertRaises(ValueError):windows(s,q+[q[0]])
        bad=copy.deepcopy(q);del bad[0]['seconds']
        with self.assertRaises(ValueError):windows(s,bad)

    def test_range_availability_not_weighted_proof(self):
        m={'supports':{k:{'status':'PROPOSED','certificate':{'file':k}} for k in
              ('s0_1_gate_lo','s0_1_gate_hi','s0_1_p0_lo','s0_1_p0_hi')},
           'obligations':[{'kind':'residual','pair':[0,1],'property_index':0,'weighted_status':'RANGE_UNAVAILABLE','certificate':None}]}
        r=readiness(m)['s0_1'][0]
        self.assertTrue(r['four_range_certificates_present']);self.assertFalse(r['weighted_certificate_present'])
        m['supports']['s0_1_gate_lo']['certificate']=None
        self.assertFalse(readiness(m)['s0_1'][0]['four_range_certificates_present'])


if __name__=='__main__':unittest.main()
