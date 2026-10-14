"""Small exact-value formula tests, not workload or memory benchmarks."""
from fractions import Fraction as F
from itertools import product
import json
import sys
import unittest
from unittest.mock import patch

from scripts import assess_exact_domain as a


def representation(q):
    d=q.denominator
    if d & (d-1):
        raise ValueError('not dyadic')
    return abs(q.numerator).bit_length(), d.bit_length()-1


class ExactDomain(unittest.TestCase):
    def assertEnvelope(self, value, envelope):
        n,e=representation(value)
        self.assertLessEqual(n,envelope[0]); self.assertLessEqual(e,envelope[1])

    def test_small_binary_operations(self):
        for b in range(1,5):
            values={F(n,2**e) for n in range(-2**b+1,2**b) for e in range(b)}
            bounds=a.dyadic_bounds(b)
            for x,y in product(values,repeat=2):
                self.assertEnvelope(x+y,bounds['add_sub'])
                self.assertEnvelope(x-y,bounds['add_sub'])
                self.assertEnvelope(x*y,bounds['product'])

    def test_small_fused_expression(self):
        b=3; bounds=a.dyadic_bounds(b)
        values={F(n,2**e) for n in (-7,-1,0,1,7) for e in range(b)}
        for x,y,z in product(values,repeat=3):
            self.assertEnvelope(x+y*z,bounds['fused_add_product'])

    def test_four_differences_and_signs(self):
        for b in (1,3,8,32):
            m=2**b-1
            vals=(F(m),F(-m,2**(b-1)),F(m,2**(b-1)),F(-m))
            value=sum(abs(vals[i]-vals[(i+1)%4]) for i in range(4))
            self.assertEnvelope(value,a.dyadic_bounds(b)['four_absolute_differences'])

    def test_domain_excludes_margin_and_generic_candidate(self):
        for q in (F(1,3),F(1,10000000)):
            with self.assertRaises(ValueError): representation(q)
        self.assertEqual(representation(F(.5)),(1,1))
        self.assertEqual(representation(F(0)),(0,0))

    def test_cancellation_cannot_move_result_gate_inward(self):
        # Fixed expression: product is too large for B; final exact result fits.
        b=8; value=F(2**7)
        result=value*value-value*value
        self.assertGreater((value*value).numerator.bit_length(),b)
        self.assertEqual(result,0)

    def test_registered_bounds_and_partial_slots(self):
        self.assertEqual(a.dyadic_bounds(8192),dict(add_sub=[16384,8191],
            product=[16384,16382],fused_add_product=[24576,16382],
            four_absolute_differences=[16386,8191]))
        one,eight=a.named_payload(1),a.named_payload(8)
        self.assertEqual(one['exact_endpoint_value_slots'],103088)
        self.assertEqual(eight['exact_endpoint_value_slots'],824704)
        self.assertEqual(one['exact_other_slots_without_object_correspondence'],237328)
        self.assertEqual(one['native_constructor_isfinite_temporaries_bytes'],74059920)
        self.assertIsNone(one['full_rss_upper'])

    def test_bad_static_inputs(self):
        for v in (True,0,-1,1.5,None):
            with self.assertRaises(ValueError): a.dyadic_bounds(v)
        for v in (True,0,9,-1,None):
            with self.assertRaises(ValueError): a.named_payload(v)

    def test_anchor_drift_rejected(self):
        a.verify_files()
        with patch.dict(a.ANCHORS,{'scoped_source/scaled_endpoints.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'anchor'): a.verify_files()
        with patch.dict(a.ENV_ANCHORS,{'lib/python3.12/_pylong.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'anchor'): a.verify_files()

    def test_report_preserves_stop_and_all_gates(self):
        r=a.analyze()
        self.assertEqual(r['invariant'],dict(request_seconds=300,visits=8000000,
            work_result_bits=8192,stored_component_bits=4096))
        for key in ('mathematical_checks_removed','check_points_moved','production_changed',
                    'implementation_authorized','execution_admitted','complete_cost_upper_closed'):
            self.assertFalse(r[key])
        self.assertIsNone(r['new_quota']); self.assertIsNone(r['full_peak_memory_upper'])
        self.assertTrue({'torch','numpy','scipy','scoped_source'}.isdisjoint(sys.modules))
        self.assertEqual(r,json.loads((a.ROOT/'docs/hz_exact_domain_20261014.json').read_bytes()))


if __name__=='__main__': unittest.main()
