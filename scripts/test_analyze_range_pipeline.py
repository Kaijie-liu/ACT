import copy
import json
from pathlib import Path
import tempfile
import unittest
from analyze_range_pipeline import analyze

ROOT=Path(__file__).resolve().parents[1]


class SavedAnalysisTests(unittest.TestCase):
    def test_rebuild_from_committed_review_only(self):
        self.assertEqual(analyze(ROOT/'docs/range_pipeline_v1_review.json'),
            json.loads((ROOT/'docs/range_pipeline_v1_analysis.json').read_bytes()))

    def test_missing_duplicate_and_false_positive_counts_rejected(self):
        original=json.loads((ROOT/'docs/range_pipeline_v1_review.json').read_bytes())
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            p=Path(tmp)/'review.json'
            for mode in ('missing','duplicate','count'):
                d=copy.deepcopy(original);r=d['arms'][0]['result']
                if mode=='missing':r['rows'].pop()
                elif mode=='duplicate':r['rows'][-1]=r['rows'][0]
                else:r['positive_bounds']=1
                p.write_text(json.dumps(d))
                with self.subTest(mode=mode),self.assertRaises(ValueError):analyze(p)

    def test_widened_range_not_reported_as_tightening(self):
        d=json.loads((ROOT/'docs/range_pipeline_v1_review.json').read_bytes())
        d['arms'][1]['generation']['range_rows'][0]['checked_range']=['-999999','999999']
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            p=Path(tmp)/'review.json';p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'range widened'):analyze(p)


if __name__=='__main__':unittest.main()
