"""Controls for the new hypothesis; no trained model or frozen B4 query."""
import copy
import math
import os
import time
import unittest

import torch

from act.back_end.moe.guard_dual_refinement import CandidateEnvelope, refined_conditional_bounds
from act.back_end.moe.moe_aware_bab import conditional_bounds, GuardObjectives, history_objectives
from test_moe_aware_bab import AnalyticBody, SOURCE
import guard_dual_refinement as study


class EnvelopeTests(unittest.TestCase):
    def test_negative_rows_and_per_property_retention(self):
        bank = CandidateEnvelope(2, 2)
        original = torch.tensor([-2., 3.], dtype=torch.float64)
        bank.add(original, torch.zeros((2, 2), dtype=torch.float64), 'baseline:0')
        original[0] = 100.
        bank.add(torch.tensor([-1., -5.], dtype=torch.float64),
                 torch.ones((2, 2), dtype=torch.float64), 'refinement:0')
        self.assertEqual(bank.lower.tolist(), [-1., 3.])
        self.assertEqual(bank.beta.tolist(), [[1., 1.], [0., 0.]])
        self.assertEqual(bank.origin, ['refinement:0', 'baseline:0'])
        for beta in (torch.tensor([[-1., 0.], [0., 0.]]),
                     torch.tensor([[math.nan, 0.], [0., 0.]])):
            with self.assertRaises(ValueError):
                bank.add(torch.zeros(2), beta, 'bad')

    def test_receiver_recomputes_full_envelope_and_rejects_mutations(self):
        base = [[-2., 1.], [-1., .5]] + [[-2., .5]] * 18
        extra = [[-.5, .75]] * 41
        trace = [dict(phase='baseline', iteration=i, lower=v, beta_nonzero=0)
                 for i, v in enumerate(base)]
        trace += [dict(phase='refinement', iteration=i, lower=v, beta_nonzero=2)
                  for i, v in enumerate(extra)]
        trace += [dict(phase='probe', iteration=0, lower=[-1., .5], beta_nonzero=4)]
        row = dict(source_id=SOURCE, history=[[0, 1]], cells=[], objective_count=2,
            guard_count=2, use_guards=True, lower=[-.5, 1.], zero_lower=[-1., 1.],
            baseline_lower=[-1., 1.], refinement_lower=[-.5, .75], improvement=[.5, 0.],
            selected_beta=[[1., 0.], [0., 0.]], beta_count=4, beta_nonzero=1,
            formal_SAFE=False, complete_MoE=False,
            engine_adapter='baseline_preserving_CROWN_guard_refinement_v1',
            baseline_limit=20, refinement_update_limit=40, beta_start=0., beta_probe=1e-4,
            evaluations=dict(baseline=20, refinement=41, probe=1), candidate_evaluations=62,
            alpha_star_iteration=1, alpha_star_sha256='b'*64, trace=trace)
        # Earliest largest sum is iteration1 (-.5), versus iteration0 (-1).
        args = (SOURCE, [[0, 1]], 'GUARD_REFINE', 2, 2, time.monotonic()+30)
        result = study.aggregate(row, *args)
        self.assertEqual(result['status'], 'UNKNOWN')
        for field, value in [('lower', [-.5, .75]), ('baseline_lower', [-2., 1.]),
                ('source_id', 'c'*64), ('history', [[0, 2]]), ('guard_count', 1),
                ('beta_start', .1), ('candidate_evaluations', 61),
                ('selected_beta', [[-1., 0.], [0., 0.]]), ('trace', trace[:-1])]:
            bad = copy.deepcopy(row); bad[field] = value
            with self.assertRaises((ValueError, KeyError)):
                study.aggregate(bad, *args)
        with self.assertRaises(TimeoutError):
            study.aggregate(row, *args[:-1], time.monotonic()-1)


@unittest.skipUnless('GATE_CONTROL_DEADLINE' in os.environ, 'isolated native control batch')
class NativeRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import genbab_cpu_ready
        genbab_cpu_ready.probe()
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)
        cls.deadline = float(os.environ['GATE_CONTROL_DEADLINE'])

    def record(self, kind, result):
        from test_gate_decoupling import API_RESULTS
        API_RESULTS.append(dict(kind=kind, lower=result['lower'],
            evaluations=result.get('evaluations', result.get('candidate_evaluations'))))

    def compare(self, graph, low, high):
        center = (low+high)/2
        old = conditional_bounds(graph, center, low, high, deadline=self.deadline, iterations=20)
        self.record('old_alpha20', old)
        on = refined_conditional_bounds(graph, center, low, high, deadline=self.deadline)
        self.record('warm_alpha_beta', on)
        alpha = refined_conditional_bounds(graph, center, low, high,
                                          deadline=self.deadline, use_guards=False)
        self.record('equal_extra_alpha', alpha)
        self.assertEqual(old['lower'], on['baseline_lower'])
        self.assertEqual(old['lower'], alpha['baseline_lower'])
        self.assertEqual(on['alpha_star_sha256'], alpha['alpha_star_sha256'])
        self.assertTrue(all(a >= b for a, b in zip(on['lower'], old['lower'])))
        self.assertTrue(all(a >= b for a, b in zip(alpha['lower'], old['lower'])))
        self.assertEqual(on['trace'][on['evaluations']['baseline']]['beta_nonzero'], 0)
        study.aggregate(on, SOURCE, [list(p) for p in graph.history], 'GUARD_REFINE',
                        graph.objective_count, graph.guard_count, self.deadline)
        return on

    def test_native_affine_relu_guard_and_properties(self):
        for nonlinear in (False, True):
            graph = GuardObjectives(AnalyticBody(nonlinear).eval(), ((0, 1),), (3,),
                torch.tensor([[1.], [-2.]]), torch.tensor([.1, 3.]), source_id=SOURCE).eval()
            on = self.compare(graph, -torch.ones((1, 1)), torch.ones((1, 1)))
            # For g=ReLU(x)-1/4 the scalar Lagrangian has a genuine gap:
            # max_beta min(-.9+.25*beta, 1.1-.75*beta) = -.4 at beta=2.
            # The constrained minimum .35 is NOT attainable by this dual.
            optimum = -.4 if nonlinear else .1
            self.assertAlmostEqual(on['lower'][0], optimum, places=6)
            self.assertLessEqual(on['lower'][0], optimum+1e-7)
            self.assertLessEqual(on['lower'][1], 1.+1e-7)
            self.assertGreater(on['beta_nonzero'], 0)

    def test_native_shared_gate_nine_properties(self):
        from test_gate_decoupling import fixture
        model, rows, thresholds = fixture(2, classes=10)
        graph = history_objectives(model, ((0, 1), (0, 1)), rows, thresholds, source_id=SOURCE)
        on = self.compare(graph, -torch.ones((1, 1))*.25, torch.ones((1, 1))*.25)
        self.assertEqual(len(on['lower']), 9)
        self.assertFalse(on['formal_SAFE'])

    def test_native_partial_prefix_never_returns_final(self):
        graph = GuardObjectives(AnalyticBody().eval(), ((0, 1),), (3,),
            torch.ones((1, 1)), torch.tensor([.1]), source_id=SOURCE).eval()
        saved = []
        def stop(name, value):
            saved.append(value)
            raise TimeoutError('fixed interruption after complete baseline')
        with self.assertRaisesRegex(TimeoutError, 'fixed interruption'):
            refined_conditional_bounds(graph, torch.zeros((1, 1)), -torch.ones((1, 1)),
                torch.ones((1, 1)), deadline=self.deadline, report=stop)
        self.assertEqual(len(saved), 1)
        self.record('intentional_partial_timeout', saved[0])


if __name__ == '__main__':
    unittest.main()
