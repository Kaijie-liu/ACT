from pathlib import Path
import tempfile
import unittest
from soplex_compat.controls import lp
from soplex_fidelity.review import compare,ROOT


class ReviewControls(unittest.TestCase):
    def test_separate_parser_mapping_and_exactness(self):
        p=lp(c=[1,2],lo=[0,0],hi=[1,1],e=[[3,4]],h=[1],offset='1/7')
        # Column and within-row native order need not equal input order.
        text='SOPLEX_RATIONAL_READBACK_V1\nCOLUMNS 2\nx1 2 0 1\nx0 1 0 1\nROWS 1\ne0 1 1 2 x1 4 x0 3\nEND\n'
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as d:
            f=Path(d)/'readback';f.write_text(text)
            self.assertEqual(compare(p,f)['nonzero_entries'],2)
            for bad in [text.replace('x0 3','x0 -3'),text.replace('e0 1 1','e0 -inf 1'),
                        text.replace('x0 1 0 1','x0 1 0 2'),text.replace('x1 4','x0 4'),
                        text.replace('END\n','')]:
                f.write_text(bad)
                with self.assertRaises((ValueError,StopIteration)):compare(p,f)

