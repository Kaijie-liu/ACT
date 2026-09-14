import copy
import json
from pathlib import Path
import unittest
from act.pipeline.moe.review_external_static_pair import validate


class StaticExternalAuditTests(unittest.TestCase):
    def record(self):
        return json.loads((Path(__file__).parent/'results/external_static_pair_review_20260914_r2.json').read_text())

    def test_complete_real_obligations_are_numerical_only(self):
        result=validate(self.record())
        self.assertEqual(result['positive_rows'],18)
        self.assertEqual(result['completed_rows'],18)

    def test_fail_closed_inventory_properties_counts_and_identity(self):
        for mutate in (lambda r:r['results'].pop(),
                       lambda r:r['results'][0]['C'][0][0].__setitem__(5,0),
                       lambda r:r['results'][0]['bounds']['lower'].__setitem__(0,1000),
                       lambda r:r.__setitem__('positive_rows',19),
                       lambda r:r['results'][0].__setitem__('formal_SAFE',True),
                       lambda r:r['results'][0].__setitem__('dtype','float32'),
                       lambda r:r['launch']['config'].__setitem__('method','CROWN-Optimized')):
            record=self.record();mutate(record)
            with self.assertRaises(ValueError):validate(record)

    def test_timeout_remains_in_denominator(self):
        r=self.record()
        r['results'][0]={'pair':[2,4],'status':'TIMEOUT','formal_SAFE':False}
        r['status']='NOT_ALL_STATIC_OBLIGATIONS_POSITIVE';r['positive_rows']=9
        self.assertEqual(validate(r),{'completed_rows':9,'positive_rows':9,'required_rows':18,
                                     'status':'NOT_ALL_STATIC_OBLIGATIONS_POSITIVE'})


if __name__=='__main__':unittest.main()
