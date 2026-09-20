import json
from pathlib import Path
import tempfile
import unittest
from analyze_property_ranges import analyze

ROOT=Path(__file__).resolve().parents[1]


class SavedPropertyAnalysisTests(unittest.TestCase):
    def original(self):
        return json.loads((ROOT/'docs/property_ranges_v1_review.json').read_bytes())

    def with_record(self,record):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            path=Path(tmp)/'review.json';path.write_text(json.dumps(record))
            return analyze(path)

    def test_exact_rebuild_from_committed_review(self):
        self.assertEqual(analyze(ROOT/'docs/property_ranges_v1_review.json'),
                         json.loads((ROOT/'docs/property_ranges_v1_analysis.json').read_bytes()))

    def test_missing_duplicate_and_false_positive_reject(self):
        for mode in ('missing','duplicate','count','request'):
            record=self.original();r=record['arms'][0]['result']
            if mode=='missing':r['rows'].pop()
            elif mode=='duplicate':r['rows'][-1]=r['rows'][0]
            elif mode=='count':r['positive_bounds']+=1
            else:r['complete_declared_real_output_proof']=not r['complete_declared_real_output_proof']
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.with_record(record)

    def test_range_and_roster_mutations_reject(self):
        for mode in ('widen','old','row','calls'):
            record=self.original();a=record['arms'][1];r=a['generation']['range_rows'][0]
            if mode=='widen':r['checked_range']=['-99999','99999']
            elif mode=='old':r['generator_box']=['-99999','99999']
            elif mode=='row':r['row']=999
            else:r['calls']=3
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.with_record(record)

    def test_incomplete_check_is_not_nine_nonpositive_or_zero_cost(self):
        record=self.original();record['arms'][1]['complete']=False
        result=self.with_record(record);arm=result['arms'][1]
        self.assertIsNone(arm['nonpositive']);self.assertIsNone(arm['missing'])
        self.assertEqual(arm['unchecked_obligations'],9)
        self.assertGreater(arm['seconds'],0);self.assertEqual(result['common_bound_changes'],[])


if __name__=='__main__':unittest.main()
