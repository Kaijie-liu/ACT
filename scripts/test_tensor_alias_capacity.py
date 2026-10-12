"""Finite static arithmetic tests; never instantiate a network or source."""
import unittest
from math import comb

from scripts.analyze_tensor_alias_capacity import analyze, consumer_floor, partitions, anchors, ANCHORS, ROOT


class Capacity(unittest.TestCase):
    def test_current_source_floor_and_actual_result_retained(self):
        r=analyze()
        self.assertEqual(r['current_source_unconditional_first_visits'],13369344)
        self.assertEqual(r['recipe']['first_weight_positions'],6684672)
        self.assertEqual(r['recipe']['later_weight_positions'],273408)
        self.assertFalse(r['real_execution_admitted'])
        self.assertIsNone(r['actual_nondegenerate_pairs'])

    def test_full_consumer_counts(self):
        cases=[(0,252,56,6894720),(7,315,63,8524656),(28,504,84,13414464)]
        for p,ep,tiles,lower in cases:
            r=consumer_floor(3072,8,10,3288,p)
            self.assertEqual((r['endpoints'],r['tiles'],r['complete_consumer_work_floor']),(ep,tiles,lower))
            self.assertEqual(r['node_floor'],6696)
        self.assertEqual(consumer_floor(3072,8,10,3288,28)['unconditional_factor_work'],9289728)

    def test_equal_router_is_not_claimed_impossible_from_consumer_alone(self):
        r=consumer_floor(3072,8,10,3288,0)
        self.assertLess(r['complete_consumer_work_floor'],8000000)
        self.assertGreater(r['complete_consumer_work_floor']+6684672,8000000)

    def test_all_equivalence_partitions_no_assumed_observation(self):
        values=list(partitions(8)); self.assertEqual(len(values),22)
        self.assertEqual(len(set(values)),22)
        counts=[]
        for sizes in values:
            labels=[i for i,n in enumerate(sizes) for _ in range(n)]
            enumeration=sum(labels[a]!=labels[b] for a in range(8) for b in range(a+1,8))
            formula=comb(8,2)-sum(comb(n,2) for n in sizes)
            self.assertEqual(enumeration,formula); counts.append(formula)
        self.assertEqual(min(p for p in counts if p),7)
        self.assertEqual(counts.count(0),1)

    def test_pair_local_tiling_and_obligation_denominator(self):
        # Splitting into smaller tiles can only add mandatory node scans.
        previous=None
        for width in (8,4,2,1):
            r=consumer_floor(3072,8,10,3288,7,width=width)
            self.assertEqual(r['required_properties'],252)
            if previous is not None: self.assertGreaterEqual(r['complete_consumer_work_floor'],previous)
            previous=r['complete_consumer_work_floor']
        with self.assertRaises(ValueError): consumer_floor(3072,8,10,3288,29)
        with self.assertRaises(ValueError): consumer_floor(3072,8,10,3288,0,width=9)

    def test_identity_mutation_and_missing_inventory_rejected(self):
        b={name:(ROOT/name).read_bytes() for name in ANCHORS}; anchors(b)
        changed=dict(b); k=next(iter(changed)); changed[k]+=b'\n'
        with self.assertRaises(ValueError): anchors(changed)
        changed=dict(b); changed.pop(k)
        with self.assertRaises(ValueError): anchors(changed)


if __name__=='__main__': unittest.main()
