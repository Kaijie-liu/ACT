import sys
import json
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_bn_expansion_edges import observe_bn


class RepairedConverterControls(unittest.TestCase):
    """Current source conformance; old defective outcomes remain frozen in JSON."""
    def test_nonunit_scale_no_longer_bypassed(self):
        r=observe_bn();self.assertEqual(r['source'],[20.,20.]);self.assertEqual(r['converted'],[20.,20.])
        self.assertFalse(r['edges'][0]['scale_bypassed']);self.assertEqual(r['source_vs_toy_rewired'],0.)
    def test_negative_scale(self):
        r=observe_bn(scale=-2.);self.assertEqual(r['source_vs_converted'],0.)
        self.assertEqual(r['source_vs_toy_rewired'],0.)
    def test_unit_scale_masks_defect(self):
        r=observe_bn(scale=1.);self.assertFalse(r['edges'][0]['scale_bypassed'])
        self.assertEqual(r['source_vs_converted'],0.)
    def test_first_layer_bn(self):
        r=observe_bn(insert_conv=False);self.assertEqual(r['source_vs_toy_rewired'],0.)
        self.assertEqual(r['source_vs_converted'],0.)
    def test_spatial_one(self):
        r=observe_bn(spatial=1);self.assertEqual(r['source_vs_toy_rewired'],0.)
        self.assertEqual(r['source_vs_converted'],0.)

    def test_old_findings_not_relabelled(self):
        old=json.loads((Path(__file__).resolve().parents[1]/'docs/metamoe_bn_edge_controls_20260923_r1.json').read_text())
        self.assertEqual(old['cases'][0]['converted'],[12.,12.])
        self.assertTrue(old['cases'][0]['edges'][0]['scale_bypassed'])


if __name__=='__main__':unittest.main()
