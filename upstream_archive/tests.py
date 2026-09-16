import copy
import unittest
from upstream_archive.review import aggregates


class ArchiveControls(unittest.TestCase):
    def test_counts_pairs_cost_and_mutations(self):
        rows=[dict(job_id=f'{i}_{a}',dataset_index=i,arm=a,status='UNKNOWN_NONPOSITIVE',
              complete_independent_check=True,wall_seconds=10 if a=='single_check' else 20)
              for i in (207,209,211,214) for a in ('single_check','double_check')]
        value=aggregates(rows)
        self.assertEqual(value['by_arm']['single_check']['complete_checks'],4)
        self.assertEqual(value['paired_time_difference_median'],-10)
        self.assertEqual(value['whole_request_seconds_sum'],120)
        for changed in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):aggregates(changed)
        for k,v in (('dataset_index',98),('wall_seconds',-1),('status','TIMEOUT')):
            bad=copy.deepcopy(rows);bad[0][k]=v
            with self.assertRaises(ValueError):aggregates(bad)


if __name__=='__main__':unittest.main()
