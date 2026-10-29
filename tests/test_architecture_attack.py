"""New S1 attack/domain controls, executed in the pinned evaluator environment."""
import base64
import copy
import math
from pathlib import Path
import struct
import time
import unittest
import torch
from act.pipeline.moe.architecture_attack import attack, encode_point, decode_point
from act.pipeline.moe.multilayer_study import public_pgd
from multilayer_stage0 import affine
from architecture_s1_execution import pgd_catalog, bound_input
from scoped_proof.io import load

DEADLINE = None


def domain():
    x = torch.zeros((1, 1), dtype=torch.float64)
    return dict(center=x, lower=x-1, upper=x+1, rows=torch.tensor([[1., -1.]]), thresholds=torch.zeros(1))


class Controls(unittest.TestCase):
    def test_lossless_mixed_endpoints_and_interior(self):
        lo = torch.tensor([[-1., 0., 1., 0., 0.]], dtype=torch.float64)
        hi = torch.tensor([[1., 1., 1., 1., 1.]], dtype=torch.float64)
        point = torch.tensor([[-1., 1., 1., math.nextafter(.5, 1.), math.nextafter(0., 1.)]])
        packed = encode_point(point, lo, hi)
        self.assertTrue(torch.equal(point, decode_point(packed, lo, hi)))
        self.assertEqual(len(base64.b64decode(packed['interior'])), 16)
        for field, wrong in [('shape',[5]), ('mask',base64.b64encode(b'\x03'*5).decode()),
                ('interior',base64.b64encode(struct.pack('<d',2.)).decode())]:
            bad = dict(packed, **{field:wrong})
            with self.assertRaises(ValueError):
                decode_point(bad, lo, hi)
        with self.assertRaises(ValueError):
            encode_point(hi+1, lo, hi)

    def test_cifar_shape_roundtrip_without_point_quantization(self):
        rng = torch.Generator().manual_seed(29)
        center = torch.rand((1,3,32,32), generator=rng, dtype=torch.float64)
        lo = (center-8/255).clamp_min(0); hi = (center+8/255).clamp_max(1)
        point = lo+(hi-lo)*torch.rand(center.shape, generator=rng, dtype=torch.float64)
        point.flatten()[::3] = lo.flatten()[::3]
        point.flatten()[1::3] = hi.flatten()[1::3]
        packed = encode_point(point, lo, hi)
        self.assertTrue(torch.equal(point, decode_point(packed, lo, hi)))

    def test_zero_failure_is_not_safety(self):
        model = affine([[0.],[0.]], [0.,0.])
        result = attack(model, domain(), radius=8, seed=4, deadline=DEADLINE)
        self.assertEqual(result['status'], 'NO_REPLAYED_COUNTEREXAMPLE')
        self.assertEqual(result['iterations'], 40)
        self.assertEqual(result['step_size'], 8/255/4)
        self.assertFalse(result['replayed'])
        self.assertNotIn('witness', result)

    def test_same_witness_seed_and_old_default(self):
        model = affine([[1.],[0.]], [.25,0.]); tensors = domain()
        a = public_pgd(model, tensors, DEADLINE, 4)
        b = public_pgd(model, tensors, DEADLINE, 4, step_size=2/255/4)
        self.assertEqual(a['status'], b['status'])
        self.assertEqual(a['witness'], b['witness'])
        result = attack(model, tensors, radius=2, seed=4, deadline=DEADLINE)
        # A finite attack need not find the known negative part of this box.
        # Preserve the R1 failed expectation; compare actual identical searches.
        self.assertEqual(result['status'], a['status'])
        if a['status'] == 'UNSAFE_REPLAYED':
            restored = decode_point(result['witness'], tensors['lower'], tensors['upper'])
            self.assertTrue(torch.equal(restored, torch.tensor(a['witness'], dtype=torch.float64)))
            self.assertTrue(result['replayed'])
        else:
            self.assertNotIn('witness', result)
            self.assertFalse(result['replayed'])
            self.assertTrue(result['completed_fixed_search'])

    def test_clean_wrong_reference_and_deadline(self):
        model = affine([[0.],[0.]], [-1.,0.])
        r = attack(model, domain(), radius=2, seed=0, deadline=DEADLINE)
        self.assertEqual(r['witness'], {'reference':'bound input center'})
        self.assertEqual(r['iterations'], 0)
        self.assertTrue(r['replayed'])
        with self.assertRaises(TimeoutError):
            attack(model, domain(), radius=2, seed=0, deadline=time.monotonic()-1)
        with self.assertRaises(ValueError):
            attack(model, domain(), radius=4, seed=0, deadline=DEADLINE)

    def test_auxiliary_roster_is_all_inputs_and_not_verification_slots(self):
        models = [dict(id=str(i)) for i in range(12)]
        rows = pgd_catalog(models)
        self.assertEqual(len(rows), 2400)
        items = [(r['model'],i,r['radius']) for r in rows for i in r['indices']]
        self.assertEqual(len(items), 24000)
        self.assertEqual(len(set(items)), 24000)
        self.assertEqual({i for _,i,_ in items}, set(range(1000)))

    def test_materialized_input_helper_identity(self):
        root = Path('/data1/Kane/MOE/results/architecture_s1_inputs_20261029_r1')
        inputs = torch.load(root/'inputs.pt', map_location='cpu', weights_only=True)
        catalog = load(root/'catalog.json')
        for index in (0,98,199,999):
            for radius in (2,8):
                tensors, label = bound_input(inputs,catalog,index,radius)
                self.assertEqual(tensors['rows'].shape,(9,10))
                self.assertEqual(label,int(inputs['labels'][index]))
        bad = copy.deepcopy(catalog); bad['samples'][0]['domains']['2']['rows']='wrong'
        with self.assertRaises(ValueError):
            bound_input(inputs,bad,0,2)
