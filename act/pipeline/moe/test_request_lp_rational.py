import copy
import unittest
from unittest.mock import patch
from act.pipeline.moe.test_request_lp_order import OrderLPTests
from act.pipeline.moe.check_request_lp import aggregate,RATIONAL_TRUSTED
from act.back_end.solver.rational_mccormick import build
from act.back_end.solver.lp_certificate import propose


class RationalRequestTests(unittest.TestCase):
    def fixture(self):
        m,s=OrderLPTests().fixture()
        m.update(schema='request_lp_rational_v3',trusted_base=RATIONAL_TRUSTED)
        e=build(s['dle']['source'],[1,-1],0,[0,.5],[1,1]);s['oute']=e;s['outc']=propose(e['lp'])
        m['proofs']['out'].update(kind='rational_weighted',hz_sha256=e['source_sha256'])
        return m,s

    def test_complete_without_builder_or_solver(self):
        m,s=self.fixture()
        with patch('act.back_end.solver.rational_mccormick.build',side_effect=AssertionError('builder called')),patch('scipy.optimize.linprog',side_effect=AssertionError('solver called')):
            result=aggregate(m,s.__getitem__)
        self.assertEqual(result['counts']['residual'],1)
        self.assertEqual(result['trusted_base'],RATIONAL_TRUSTED)

    def test_scope_and_range_mutations(self):
        changes=[lambda m:m['proofs']['out'].update(request_id='other request'),
                 lambda m:m['proofs']['out'].update(scope={'pair':[1,0]}),
                 lambda m:m['proofs']['out'].update(property_index=1),
                 lambda m:m['proofs']['du'].update(hz_sha256='different factors'),
                 lambda m:m['obligations'][0].update(lambda_bounds=[0,.25]),
                 lambda m:m['obligations'][0].update(difference_bounds=[2,2]),
                 lambda m:m['obligations'].clear()]
        for change in changes:
            m,s=self.fixture();change(m)
            with self.assertRaises(ValueError):aggregate(m,s.__getitem__)

    def test_unknown_not_promoted(self):
        m,s=self.fixture();m['proofs']['out']['status']='UNKNOWN'
        self.assertEqual(aggregate(m,s.__getitem__)['status'],'UNKNOWN')


if __name__=='__main__':unittest.main()
