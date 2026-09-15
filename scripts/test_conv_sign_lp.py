import copy
import json
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp

from act.back_end.solver.hz_lp_export import export
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.solver.lp_certificate import identity
from scripts.check_conv_sign_lp import check_record
from scripts.conv_sign_lp_contract import jobs, parents


class SignEvidenceTests(unittest.TestCase):
    def example(self, center=1.):
        hz=SparseHZono(c=np.array([center]), Gc=sp.csr_matrix([[.25]]), Gb=sp.csr_matrix((1,0)),
            Ac=sp.csr_matrix((0,1)), Ab=sp.csr_matrix((0,0)), b=np.array([]),
            Auc=sp.csr_matrix((0,1)), Aub=sp.csr_matrix((0,0)), ub=np.array([]))
        record=export(hz,[1],sparse=True)
        cert={'lp_sha256':identity(record['lp']), 'inequality_dual':[], 'equality_dual':[],
              'claimed_lower_bound':str(center-.25)}
        scope={'pairs':[[0,3]], 'row':[ -1,0,0,0,0,1,0,0,0,0], 'constant':0}
        job={'expected_scope':scope,'parent_request_sha256':'parent'}
        capture={'scope':copy.deepcopy(scope),'parent_request_sha256':'parent','source_sha256':record['source_sha256']}
        return record,cert,capture,job

    def test_sign_without_optimal_status_or_solver_call(self):
        args=self.example()
        with patch('scipy.optimize.linprog',side_effect=AssertionError('checker called solver')):
            result=check_record(*args)
        self.assertEqual(result['status'],'CHECKED_POSITIVE_SUPPLIED_F0_LP')
        self.assertFalse(result['full_request_SAFE'])
        self.assertNotIn('solver_status',args[1])

    def test_nonpositive_bound_not_unsafe(self):
        self.assertEqual(check_record(*self.example(0.))['status'],'CHECKED_NONPOSITIVE_LOWER_BOUND')

    def test_forged_claim_or_identity_rejected(self):
        for key,value in [('claimed_lower_bound','10'),('lp_sha256','wrong')]:
            args=list(self.example());args[1][key]=value
            with self.assertRaises(ValueError):check_record(*args)

    def test_wrong_request_pair_property_rejected(self):
        for change in ('parent','pair','row'):
            args=list(self.example())
            if change=='parent':args[2]['parent_request_sha256']='another'
            elif change=='pair':args[2]['scope']['pairs']=[[1,2]]
            else:args[2]['scope']['row']=[1,-1]+[0]*8
            with self.assertRaises(ValueError):check_record(*args)

    def test_changed_center_coefficient_or_box_rejected(self):
        for field in ('offset','c','lower'):
            args=list(self.example());lp=args[0]['lp']
            lp[field]='7' if field=='offset' else ['7']
            args[1]['lp_sha256']=identity(lp)
            with self.assertRaises(ValueError):check_record(*args)

    def test_fixed_controls_and_no_gate_change(self):
        p=parents(); j=jobs()
        self.assertEqual([(v['case']['dataset_index'],v['case']['competitor']) for v in j],[(16,0),(98,1)])
        self.assertEqual(p['capture_seconds'],300);self.assertEqual(p['proposal_solver_seconds'],60)
        self.assertFalse(p['production_acceptance_changed']);self.assertFalse(p['retry'])

    def test_nonoptimal_valid_multipliers_are_accepted(self):
        # min x on [0,2], x>=1: y=0 proves0 even though optimum is1.
        from act.back_end.solver.lp_certificate import check
        lp={'c':[1], 'lower':[0], 'upper':[2], 'A':[[-1]], 'b':[-1]}
        cert={'lp_sha256':identity(lp),'inequality_dual':[0],'equality_dual':[],'claimed_lower_bound':0}
        self.assertEqual(check(lp,cert)['checked_lower_bound'],'0')
        cert['inequality_dual']=[1]
        with self.assertRaises(ValueError):check(lp,cert)

    def test_fresh_stdlib_only_checker_path(self):
        code = ('import json,sys; from scripts.check_conv_sign_lp import isolate,check_record; '
                'isolate(); result=check_record(*json.load(sys.stdin)); '
                'assert not any(k in sys.modules for k in ("torch","numpy","scipy")); '
                'print(json.dumps(result))')
        run=subprocess.run([sys.executable,'-S','-c',code],input=json.dumps(self.example()),
                           text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(run.stdout)['status'],'CHECKED_POSITIVE_SUPPLIED_F0_LP')

    def test_no_silent_binary_integrality_claim(self):
        hz=SparseHZono(c=np.array([1.]), Gc=sp.csr_matrix((1,0)), Gb=sp.csr_matrix([[.25]]),
            Ac=sp.csr_matrix((0,0)), Ab=sp.csr_matrix((0,1)), b=np.array([]),
            Auc=sp.csr_matrix((0,0)), Aub=sp.csr_matrix((0,1)), ub=np.array([]))
        record=export(hz,[1],sparse=True)
        _,_,capture,job=self.example();capture['source_sha256']=record['source_sha256']
        cert={'lp_sha256':identity(record['lp']),'inequality_dual':[],'equality_dual':[],'claimed_lower_bound':'.75'}
        self.assertEqual(check_record(record,cert,capture,job)['relaxed_binaries'],1)
        record['n_relaxed_binaries']=0
        with self.assertRaises(ValueError):check_record(record,cert,capture,job)


if __name__=='__main__':unittest.main()
