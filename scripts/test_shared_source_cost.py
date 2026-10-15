"""Saved-record and arithmetic controls, no source or candidate construction."""
import copy
import unittest
from scripts import assess_shared_source_cost as a


class Controls(unittest.TestCase):
    def test_shape_bank_lower_bound_not_admission(self):
        r=a.analyze()
        self.assertEqual(r['snapshot_only_work'],dict(row_visits=3288,coefficient_visits=13922736,
                                                     arithmetic=0,total=13926024))
        self.assertEqual(r['three_snapshot_only_work'],41778072)
        self.assertTrue(r['barriers']['one_snapshot_exceeds']); self.assertFalse(r['full_capacity_admitted'])
        self.assertEqual(r['barriers']['expanded_memberships'],126632)
        self.assertFalse(r['memory']['full_rss_upper_closed']); self.assertFalse(r['memory']['full_work_upper_closed'])

    def test_complete_logical_io_and_parent(self):
        r=a.io_schedule()
        self.assertEqual(len(r['events']),17)
        self.assertEqual(r['coefficients'],dict(T=17,X=2,S=9,G=9,C=9))
        self.assertEqual(r['parent_coefficients'],dict(T=4,X=0,S=4,G=4,C=4))

    def test_saved_roster_and_mutation_refusal(self):
        original=a.load(a.CONTROLS)
        self.assertEqual(len(a.audit_controls(original)),16)
        for kind in ('omit','duplicate','result','logical','hash'):
            r=copy.deepcopy(original)
            if kind=='omit': r['differentials'].pop()
            elif kind=='duplicate': r['differentials'].append(copy.deepcopy(r['differentials'][0]))
            elif kind=='result': r['differentials'][0]['result']['status']='FORGED'
            elif kind=='logical': r['differentials'][1]['stats']['memberships']+=1
            else: r['bindings']['scoped_source/shared_source.py']='0'*64
            with self.assertRaises(ValueError): a.audit_controls(r)

    def test_counts_never_negative_or_relabelled(self):
        for p,r in ((True,1),(-1,1),(0,0),(1,2)):
            with self.assertRaises(ValueError): a.snapshot_work(p,r)
        self.assertEqual(a.snapshot_work(9,3)['total'],21)


if __name__=='__main__': unittest.main()
