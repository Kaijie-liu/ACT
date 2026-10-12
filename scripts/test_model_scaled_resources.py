"""Arithmetic and identity controls for a proposal, not verifier experiments."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from scripts import assess_model_scaled_resources as a


class Resources(unittest.TestCase):
    def test_registered_shape_and_unconditional_roots(self):
        v = a.shape_inventory(3072,8,10,[128],[256,128])
        expected = dict(first_weight_positions=6684672, later_weight_positions=273408,
            affine_outputs=3288, relu_units=3200, parameter_values=6961368,
            source_values_with_center=6964440, raw_binary64_tensor_bytes=55715520,
            pair_output_roots=86016, mandatory_roots=126632, roots_upper=136232,
            factors_upper=32048, nodes_upper=19496, local_entries_upper=6732720,
            references_upper=277280, views=82, guards=336)
        for k, n in expected.items(): self.assertEqual(v[k], n, k)
        self.assertGreater(v['pair_output_roots'],65536)

    def test_full_obligations_both_candidates_and_pair_local_tiles(self):
        v = a.shape_inventory(3072,8,10,[128],[256,128])
        self.assertEqual((v['pairs'],v['property_duties'],v['endpoints_upper'],v['candidates_upper']),
                         (28,252,504,1008))
        self.assertEqual(v['pair_dual_rows_upper'],2700)
        self.assertEqual(v['dual_scalar_entries_upper'],2721600)
        self.assertEqual(v['pair_local_tiles_upper'],84)
        scalar = a.shape_inventory(3072,8,10,[128],[256,128],1)
        self.assertEqual(scalar['pair_local_tiles_upper'],504)
        self.assertEqual(scalar['candidates_upper'],1008)

    def test_small_shape_root_roster_by_explicit_views(self):
        for n,e,c,rh,xh in ((2,2,3,[4],[3]),(3,4,2,[2,3],[4,2])):
            graphs = [[n,*rh,e]]+[[n,*xh,c] for _ in range(e)]
            outputs = [n,n]  # input and common
            relu = 0; affine = 0
            for graph in graphs:
                outputs.append(n)  # Flatten
                for k,size in enumerate(graph[1:]):
                    outputs.append(size); affine += size
                    if k < len(graph)-2: outputs.append(size); relu += size
            pairs = [(i,j) for i in range(e) for j in range(i+1,e)]
            outputs.extend(n for _ in pairs)
            guards = sum(1 for pair in pairs for i in pair for j in range(e) if j not in pair)
            v=a.shape_inventory(n,e,c,rh,xh)
            self.assertEqual(v['mandatory_roots'],sum(outputs)+guards)
            self.assertEqual(v['roots_upper'],sum(outputs)+guards+3*relu)
            self.assertEqual(v['affine_outputs'],affine)

    def test_unknown_not_confused_with_proven_refusal_or_capacity(self):
        self.assertEqual(a.capacity_classification(10,lower=11),'SHAPE_PROVES_CURRENT_REFUSAL')
        self.assertEqual(a.capacity_classification(10,upper=11),'UPPER_EXCEEDS_CAP_ACTUAL_UNKNOWN')
        self.assertEqual(a.capacity_classification(10,upper=10),'STRUCTURAL_UPPER_FITS_ONLY')
        self.assertEqual(a.capacity_classification(10),'COMPLETE_BOUND_MISSING')
        for kwargs in ({'lower':True},{'upper':-1},{'lower':3,'upper':2}):
            with self.assertRaises(ValueError): a.capacity_classification(10,**kwargs)

    def ledgers(self):
        return {m:{p:dict(kind='UPPER',counts=dict.fromkeys(a.COUNTERS,1),
                         derivation='TEST_ARITHMETIC_ONLY') for p in a.PHASES}
                for m in ('reference','proposed')}

    def test_missing_bound_never_gets_a_quota(self):
        l=self.ledgers(); l['proposed']['source_build']=None
        r=a.shared_work_proposal(l,ceiling=20_000_000)
        self.assertIsNone(r['candidate_work_cap']); self.assertFalse(r['execution_admitted'])
        self.assertIn('proposed/source_build',r['missing_phases'])
        del l['reference']['source_check']
        with self.assertRaisesRegex(ValueError,'phase roster'): a.shared_work_proposal(l,ceiling=None)

    def test_floors_measurements_and_invalid_counters_are_not_uppers(self):
        for kind in ('LOWER','MEASURED','PARTIAL'):
            l=self.ledgers(); l['proposed']['source_build']['kind']=kind
            with self.assertRaisesRegex(ValueError,'upper-bound'): a.shared_work_proposal(l,ceiling=None)
        for bad in (-1,True,1.5):
            l=self.ledgers(); l['reference']['source_build']['counts']['arithmetic']=bad
            with self.assertRaises(ValueError): a.shared_work_proposal(l,ceiling=None)

    def test_common_policy_rounding_and_absolute_ceiling(self):
        l=self.ledgers(); l['reference']['source_build']['counts']['arithmetic']=9_100_000
        r=a.shared_work_proposal(l,ceiling=20_000_000)
        self.assertEqual(r['candidate_work_cap'],10_000_000)
        self.assertEqual(r['status'],'ARITHMETIC_PROPOSAL_ONLY')
        self.assertFalse(r['execution_admitted']); self.assertFalse(r['bound_derivations_independently_verified'])
        self.assertEqual(a.shared_work_proposal(l,ceiling=8_000_000)['status'],'ABOVE_ABSOLUTE_CEILING')
        self.assertEqual(a.shared_work_proposal(l,ceiling=None)['status'],'ABSOLUTE_CEILING_NOT_FROZEN')
        switched={'reference':deepcopy(l['proposed']),'proposed':deepcopy(l['reference'])}
        self.assertEqual(a.shared_work_proposal(switched,ceiling=20_000_000)['candidate_work_cap'],10_000_000)

    def test_report_is_design_only_and_old_limits_remain(self):
        r=a.analyze(); p=a.current_policy()
        self.assertEqual((p['visits'],p['stored_bits'],p['work_bits']),(8000000,4096,8192))
        self.assertIsNone(r['proposal']['candidate_work_cap'])
        for key in ('production_policy_changed','implementation_admitted','real_execution_admitted'):
            self.assertFalse(r[key])
        for key in ('new_source_runs','new_bound_queries','checkpoint_loads','gpu_calls'):
            self.assertEqual(r[key],0)
        resources={v['resource']:v for v in r['current_resource_vector']}
        self.assertEqual(resources['view_root_entries']['status'],'SHAPE_PROVES_CURRENT_REFUSAL')
        self.assertEqual(resources['production_work']['structural_lower'],13369344)
        self.assertEqual(resources['local_entries']['status'],'UPPER_EXCEEDS_CAP_ACTUAL_UNKNOWN')
        self.assertTrue(all(v['observed_actual'] is None for v in resources.values()))

    def test_wrong_binding_and_dimensions_refused(self):
        with patch.dict(a.EXTRA_ANCHORS,{'act/back_end/moe/factory.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'reviewed code changed'): a.analyze()
        for params in ((1,9,2,[1],[1]),(True,2,2,[1],[1]),(1,2,2,[],[1])):
            with self.assertRaises(ValueError): a.shape_inventory(*params)


if __name__ == '__main__': unittest.main()
