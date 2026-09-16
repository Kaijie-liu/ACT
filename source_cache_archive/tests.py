import copy
import unittest
from source_cache_archive.review import aggregates


class ArchiveTests(unittest.TestCase):
    def test_costs_denominators_and_no_partial_positive(self):
        rows=[]
        for i in range(4):
            for arm in ('matrix_only','both'):
                on=arm=='both'
                rows.append({'job_id':f'{i}_{arm}','dataset_index':i,'arm':arm,
                    'status':'UNKNOWN_NONPOSITIVE' if on else 'UNKNOWN_MISSING_EVIDENCE',
                    'complete_independent_check':True,'wall_seconds':100 if on else 200,
                    'checked_result':{'required_obligations':9,'positive_obligations':2 if on else 1,
                        'missing_obligations':0 if on else 2,'nonpositive_obligations':7 if on else 6},
                    'metrics':{'timings':{}},'phases':{k:{'seconds':1} for k in ('capture','propose','package','check')},
                    'query_count':29 if on else 27})
        v=aggregates(rows,list(range(4)))
        self.assertEqual(v['paired_cost_median_seconds'],-100)
        self.assertEqual(v['by_arm']['both']['conditional_positive_requests'],0)
        self.assertEqual(v['by_arm']['matrix_only']['checked_obligations']['missing_obligations'],8)
        for bad in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):aggregates(bad,list(range(4)))
        for key,value in (('dataset_index',98),('wall_seconds',-1),('status','CHECKED_CONDITIONAL'),('status','TIMEOUT')):
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.assertRaises(ValueError):aggregates(bad,list(range(4)))


if __name__=='__main__':unittest.main()
