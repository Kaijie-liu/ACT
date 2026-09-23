import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_bn_expansion_edges import observe_bn


class KnownConverterDefectControls(unittest.TestCase):
    """Assert diagnosis of sealed source; not assertions that converter is sound.

    When repairing the converter, replace these historical expectations with
    source-conformance regressions in the new version; preserve this evidence.
    """
    def test_nonunit_scale_bypassed_and_toy_edge_repairs(self):
        r=observe_bn();self.assertEqual(r['source'],[20.,20.]);self.assertEqual(r['converted'],[12.,12.])
        self.assertTrue(r['edges'][0]['scale_bypassed']);self.assertEqual(r['source_vs_toy_rewired'],0.)
    def test_negative_scale(self):
        r=observe_bn(scale=-2.);self.assertGreater(r['source_vs_converted'],0.)
        self.assertEqual(r['source_vs_toy_rewired'],0.)
    def test_unit_scale_masks_defect(self):
        r=observe_bn(scale=1.);self.assertTrue(r['edges'][0]['scale_bypassed'])
        self.assertEqual(r['source_vs_converted'],0.)
    def test_first_layer_bn(self):
        r=observe_bn(insert_conv=False);self.assertEqual(r['source_vs_toy_rewired'],0.)
        self.assertGreater(r['source_vs_converted'],0.)
    def test_spatial_one(self):
        r=observe_bn(spatial=1);self.assertEqual(r['source_vs_toy_rewired'],0.)
        self.assertGreater(r['source_vs_converted'],0.)


if __name__=='__main__':unittest.main()
