"""Only the new descriptive layer capture; no retraining or test-data access."""
import copy
import unittest
import numpy as np
import torch
from torch import nn
from act.back_end.moe.multilayer import RoutedLayer
from act.back_end.moe.multilayer_external import SequentialTop2
from act.pipeline.moe.architecture_s2_evaluation import (
    layer_width_diagnostics, check_layer_diagnostics, summarize_layer_diagnostics,
    primary_batch,
)


def network(outputs=3):
    return nn.Sequential(nn.Flatten(), nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, outputs))


class LayerWidthControls(unittest.TestCase):
    def test_hand_computed_affine_relu_widths(self):
        m = network().double()
        with torch.no_grad():
            m[1].weight.copy_(torch.tensor([[1., 0.], [0., -2.]])); m[1].bias.copy_(torch.tensor([0., 1.]))
            m[3].weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., -1.]])); m[3].bias.zero_()
        d = layer_width_diagnostics(m, torch.tensor([[-1., 0.]], dtype=torch.double),
                                   torch.tensor([[2., 1.]], dtype=torch.double))
        check_layer_diagnostics(d, family='dense', count=1)
        self.assertEqual([r['mean_width'] for r in d['rows']], [2.5, 1.5, 2.])
        self.assertEqual(d['rows'][1]['relu_unstable_fraction'], 1.)

    def test_dense_moe_bitwise_bound_and_weight_differential(self):
        torch.manual_seed(20261031)
        models = [network(), SequentialTop2([RoutedLayer(network(8), [network() for _ in range(8)],
                         k=2, mode='selected_softmax')])]
        pixels = torch.tensor([[0, 255], [123, 89], [255, 0]], dtype=torch.uint8)
        labels = torch.tensor([0, 1, 2])
        for model, family in zip(models, ('dense', 'moe')):
            before = {k:v.clone() for k,v in model.state_dict().items()}
            off, _ = primary_batch(model, pixels, labels)
            on, metrics = primary_batch(model, pixels, labels, layer_diagnostics=True)
            self.assertEqual(set(off), set(on))
            for k in off: self.assertEqual(off[k].tobytes(), on[k].tobytes())
            self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k,v in before.items()))
            check_layer_diagnostics(metrics['layer_diagnostics'], family=family, count=3)
            self.assertGreaterEqual(metrics['seconds'], metrics['layer_diagnostics']['seconds'])

    def test_weighted_aggregation_not_mean_of_unequal_batches(self):
        d = dict(schema='IBP_LAYER_WIDTHS_V1', count=2, seconds=.2,
                 rows=[dict(network='dense',layer='1',operator='Linear',neurons=2,
                            mean_width=3.,relu_unstable_fraction=None)])
        d2 = copy.deepcopy(d); d2.update(count=1,seconds=.1);d2['rows'][0]['mean_width']=9.
        rows = [dict(start=0,stop=2,seconds=.3,layer_diagnostics=d),
                dict(start=2,stop=3,seconds=.2,layer_diagnostics=d2)]
        summary = summarize_layer_diagnostics(rows, family='dense')
        self.assertEqual(summary['rows'][0]['mean_width'], 5.)
        self.assertEqual(summary['count'], 3);self.assertAlmostEqual(summary['seconds'], .3)
        bad = copy.deepcopy(rows);bad[1]['layer_diagnostics']['rows'][0]['neurons']=3
        with self.assertRaises(ValueError):summarize_layer_diagnostics(bad,family='dense')
        bad = copy.deepcopy(rows);bad[0]['seconds']=.1
        with self.assertRaises(ValueError):summarize_layer_diagnostics(bad,family='dense')

    def test_missing_network_and_nonfinite_measurement_rejected(self):
        model = SequentialTop2([RoutedLayer(network(8), [network() for _ in range(8)],
                              k=2, mode='selected_softmax')])
        d = layer_width_diagnostics(model, torch.zeros(2,2), torch.ones(2,2))
        check_layer_diagnostics(d, family='moe', count=2)
        missing = copy.deepcopy(d);missing['rows']=[r for r in missing['rows'] if r['network']!='expert_7']
        with self.assertRaises(ValueError):check_layer_diagnostics(missing, family='moe', count=2)
        bad = copy.deepcopy(d);bad['rows'][0]['mean_width']=np.nan
        with self.assertRaises(ValueError):check_layer_diagnostics(bad, family='moe', count=2)


if __name__ == '__main__': unittest.main()
