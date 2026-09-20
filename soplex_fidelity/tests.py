import copy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest

from soplex_compat.controls import cases,csr,lp
from soplex_compat.io import verify_readback as small_verify
from soplex_fidelity.io import bounded,load_job,save,write_lp,verify_readback
from soplex_fidelity.run import ROOT,native_import,roster


RAW=Path(tempfile.mkdtemp(prefix='soplex_large_controls_',dir=ROOT/'data/moe/results'))


class Controls(unittest.TestCase):
    def test_native_read_only_regression(self):
        for name,p,_ in cases():
            with self.subTest(name=name):
                root=RAW/name;root.mkdir();write_lp(p,root/'input.lp')
                native_import(root/'input.lp',root/'readback.txt',root)
                r=verify_readback(p,root/'readback.txt');self.assertTrue(r['all_fields_equal'])
                small_verify(p,(root/'readback.txt').read_text())

    def test_streaming_wrap_and_large_dimensions(self):
        n=512;root=RAW/'wide';root.mkdir()
        p=lp(c=[0]*n,lo=[-1]*n,hi=[1]*n,e=[['1/7']*n],h=['1/3'])
        write_lp(p,root/'input.lp')
        self.assertLessEqual(max(map(len,(root/'input.lp').read_text().splitlines())),4096)
        native_import(root/'input.lp',root/'readback.txt',root)
        self.assertEqual(verify_readback(p,root/'readback.txt')['nonzero_entries'],n)

    def test_corruption_and_partial_rejected(self):
        p=lp(e=[[3]],h=[1]);root=RAW/'mutations';root.mkdir()
        original='SOPLEX_RATIONAL_READBACK_V1\nCOLUMNS 1\nx0 1 -1 1\nROWS 1\ne0 1 1 1 x0 3\nEND\n'
        bads=[original.replace('x0 3','x0 0'),original.replace('x0 1 -1 1','x0 1 0 1'),
              original.replace('e0 1 1','e0 -inf 1'),original.replace('END\n',''),
              original.replace('e0','e1'),original.replace('1 x0 3','2 x0 1 x0 2'),original+'junk\n']
        for i,text in enumerate(bads):
            path=root/f'{i}.txt';path.write_text(text)
            with self.assertRaises(ValueError):verify_readback(p,path)

    def test_deadline_and_shape_and_bits(self):
        def stop():raise TimeoutError('control cutoff')
        with self.assertRaises(TimeoutError):write_lp(lp(),RAW/'late.lp',stop)
        with self.assertRaises(ValueError):bounded(str(2**4096))
        p=lp();p['A']['indices']=[0,0];p['A']['data']=[1,1];p['A']['indptr']=[0,2];p['A']['shape']=[1,1];p['b']=[0]
        with self.assertRaises(ValueError):write_lp(p,RAW/'invalid.lp')

    def test_invalid_source_binding_before_import(self):
        for key,val in [('export',{'path':'absent','sha256':'0'*64}),('statement_sha256','0'*64)]:
            j=copy.deepcopy(roster()[0]);j[key]=val
            with self.assertRaises((ValueError,FileNotFoundError)):load_job(j)


if __name__=='__main__':unittest.main()
