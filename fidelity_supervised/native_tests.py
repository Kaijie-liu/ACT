"""V2 fidelity controls; exact original-LP checker remains unchanged."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import math
import tempfile
import time
import unittest
from unittest.mock import patch
from single_check_portable.execution import ROOT,read
from lp_sandwich.check import identity,check
from exact_basis.tests import case
from sparse_basis.engine import propose, Limit
from fidelity_supervised import native


class Controls(unittest.TestCase):
    def setUp(self):
        self.base=Path(tempfile.mkdtemp(prefix='fidelity_native_',dir=ROOT/'data/moe/results'))

    def lp(self,t=3.6294188569593725e-10):
        lp,s,_,_=case();lp['c']=[-1,0]
        lp['A']={'shape':[1,2],'indptr':[0,2],'indices':[0,1],'data':[3,t]}
        s['lp_sha256']=identity(lp)
        return lp,s

    def test_tiny_entry_preserved_and_original_exact_feasibility(self):
        lp,s=self.lp();root=self.base/'captured';deadline=time.monotonic()+30
        r=native.capture(lp,s,root,deadline=deadline)
        self.assertEqual(r['readback_before'],r['submitted'])
        self.assertEqual(r['readback_after'],r['submitted'])
        self.assertEqual(r['options']['small_matrix_value'],1e-12)
        self.assertEqual(read(root/'import.json')['status'],'HighsStatus.kOk')
        m=native.map_capture(lp,s,r,identity(r),deadline=deadline)
        p=propose(lp,s,m['candidate'],m['hint'],identity(s),identity(m['hint']),deadline=deadline)
        out=check(p['bundle'],identity(s))
        self.assertEqual(out['primal_status'],'EXACT_FEASIBLE')
        self.assertFalse(out['network_SAFE'] or out['network_UNSAFE'])

    def test_fixed_floor_preflight_rejects_without_native_import(self):
        import highspy
        for i,t in enumerate([1e-12,math.nextafter(1e-12,0.),-1e-12,'1/'+str(2**1100)]):
            lp,s=self.lp(t)
            with patch.object(highspy.Highs,'passModel',side_effect=AssertionError('must not import')) as p:
                with self.assertRaises(Limit):native.capture(lp,s,self.base/str(i),deadline=time.monotonic()+10)
                p.assert_not_called()

    def test_immediately_above_floor_preserved(self):
        lp,s=self.lp(math.nextafter(1e-12,math.inf))
        r=native.capture(lp,s,self.base/'above',deadline=time.monotonic()+15)
        self.assertEqual(r['submitted'],r['readback_before'])

    def test_warning_rejected_and_readback_retained_before_solve(self):
        import highspy
        original=highspy.Highs.passModel
        def warning(h,model):
            original(h,model);return highspy.HighsStatus.kWarning
        lp,s=self.lp();root=self.base/'warning'
        with patch.object(highspy.Highs,'passModel',warning),patch.object(highspy.Highs,'run') as run:
            with self.assertRaises(ValueError):native.capture(lp,s,root,deadline=time.monotonic()+15)
            run.assert_not_called()
        self.assertEqual(read(root/'import_status.json')['status'],'HighsStatus.kWarning')
        self.assertTrue((root/'import.json').exists())
        self.assertFalse((root/'raw_native.json').exists())

    def test_changed_readback_rejected_before_solve(self):
        import highspy
        original=native.model_snapshot
        def wrong(*args):
            r=original(*args);r['rows'][1]['entries'].pop();return r
        lp,s=self.lp();root=self.base/'changed'
        with patch.object(native,'model_snapshot',wrong),patch.object(highspy.Highs,'run') as run:
            with self.assertRaises(ValueError):native.capture(lp,s,root,deadline=time.monotonic()+15)
            run.assert_not_called()
        self.assertTrue((root/'import.json').exists())

    def test_record_option_schema_and_matrix_mutation_rejected(self):
        lp,s=self.lp();r=native.capture(lp,s,self.base/'good',deadline=time.monotonic()+15)
        for what in ('threshold','matrix','schema'):
            bad=deepcopy(r)
            if what=='threshold':bad['options']['small_matrix_value']=1e-9
            elif what=='matrix':bad['readback_after']['rows'][1]['entries'].pop()
            else:bad['schema']='LARGE_NATIVE_BASIS_CAPTURE_V1'
            with self.assertRaises(ValueError):
                native.map_capture(lp,s,bad,identity(bad),deadline=time.monotonic()+15)

    def test_large_import_values_fail_closed(self):
        lp,s=self.lp();e=native.snapshot_input(lp,native.Budget(time.monotonic()+10))
        for key in ('matrix','cost','lower','rhs'):
            bad=deepcopy(e)
            if key=='matrix':bad['rows'][1]['entries'][0][1]=1e15
            elif key=='rhs':bad['rows'][0]['upper']=1e20
            else:bad[key][0]=1e20
            with self.assertRaises(Limit):native.preflight(bad,lambda:None)


if __name__=='__main__':unittest.main()
