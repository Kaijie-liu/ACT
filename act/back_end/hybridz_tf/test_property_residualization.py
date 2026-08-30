#!/usr/bin/env python3
"""Exact toy audits for property-aware shared residual factors.

Run from the repository root:

    python -m act.back_end.hybridz_tf.test_property_residualization
"""

from __future__ import annotations

from fractions import Fraction
import random
import unittest

from act.back_end.hybridz_tf.property_residualization import (
    FractionScalarDAG,
    ScalarNode,
    audit_fraction_phase_containment,
    encode_fraction_relaxation,
    exact_fraction_lp_range,
    fraction_phase_oracle,
    nested_residual_plan,
    property_residual_candidates,
    relu_residual_envelope,
    residual_budget_frontier,
    strict_stop_loss_prefix,
)


Q = Fraction


def _two_shared_residual_diamonds() -> FractionScalarDAG:
    """Two ReLU values fan out and cancel after distinct residual branches."""

    return FractionScalarDAG.make(
        [
            ScalarNode.input("x"),
            ScalarNode.relu("h_pos", "x"),
            ScalarNode.affine("neg_x", {"x": -1}),
            ScalarNode.relu("h_neg", "neg_x"),
            ScalarNode.affine("pos_left", {"h_pos": 2, "x": 1}),
            ScalarNode.affine("pos_right", {"h_pos": 2, "x": -1}),
            ScalarNode.affine("neg_left", {"h_neg": 3, "x": 2}),
            ScalarNode.affine("neg_right", {"h_neg": 3, "x": -2}),
            ScalarNode.affine(
                "out",
                {
                    "pos_left": 1,
                    "pos_right": -1,
                    "neg_left": 1,
                    "neg_right": -1,
                },
            ),
        ],
        input_lower=-1,
        input_upper=1,
        output="out",
    )


def _nonlinear_residual_identity() -> FractionScalarDAG:
    """h1-h3 is identically zero but ordinary independent hulls have a gap."""

    return FractionScalarDAG.make(
        [
            ScalarNode.input("x"),
            ScalarNode.relu("h1", "x"),
            ScalarNode.affine("neg_x", {"x": -1}),
            ScalarNode.relu("h2", "neg_x"),
            ScalarNode.affine("difference", {"h1": 1, "h2": -1}),
            ScalarNode.relu("h3", "difference"),
            ScalarNode.affine("out", {"h1": 1, "h3": -1}),
        ],
        input_lower=-1,
        input_upper=1,
        output="out",
    )


class PropertyResidualizationAudit(unittest.TestCase):
    def test_fraction_residual_envelope_endpoint_proof(self) -> None:
        for lower, upper in (
            (Q(-1), Q(1)),
            (Q(-7, 3), Q(5, 2)),
            (Q(-1, 17), Q(19, 11)),
        ):
            for slope in (Q(0), Q(1), upper / (upper - lower)):
                envelope = relu_residual_envelope(
                    lower, upper, slope=slope
                )
                endpoint_residuals = (
                    -slope * lower,
                    Q(0),
                    (Q(1) - slope) * upper,
                )
                self.assertEqual(
                    envelope.residual_lower, min(endpoint_residuals)
                )
                self.assertEqual(
                    envelope.residual_upper, max(endpoint_residuals)
                )
                self.assertTrue(
                    all(
                        envelope.residual_lower
                        <= value
                        <= envelope.residual_upper
                        for value in endpoint_residuals
                    )
                )

            secant = relu_residual_envelope(lower, upper)
            for slope in (Q(0), Q(1), Q(1, 4), Q(3, 4)):
                trial = relu_residual_envelope(
                    lower, upper, slope=slope
                )
                self.assertLessEqual(
                    secant.residual_upper, trial.residual_upper
                )

    def test_shared_factor_survives_fanout_and_cancels(self) -> None:
        dag = _two_shared_residual_diamonds()
        oracle = fraction_phase_oracle(dag)
        self.assertEqual((oracle.true_lower, oracle.true_upper), (Q(-6), Q(6)))

        candidates = property_residual_candidates(dag, oracle)
        by_name = {candidate.node: candidate for candidate in candidates}
        self.assertEqual(set(by_name), {"h_pos", "h_neg"})
        for candidate in by_name.values():
            self.assertEqual(candidate.signed_property_sensitivity, 0)
            self.assertEqual(candidate.shared_radius_impact, 0)
            self.assertGreater(candidate.duplicated_radius_impact, 0)
            self.assertGreaterEqual(candidate.fanout_count, 2)

        plan = nested_residual_plan(candidates, 2)
        relaxation = encode_fraction_relaxation(dag, plan, oracle=oracle)
        self.assertEqual(relaxation.residual_factor_count, 2)
        # The node-level factors are shared across both consumers and cancel
        # algebraically from the final property.
        self.assertEqual(relaxation.objective_residual_nnz, 0)
        audit = audit_fraction_phase_containment(relaxation, oracle=oracle)
        self.assertGreater(audit.coupling_checks, -1)
        self.assertGreaterEqual(audit.endpoint_assignments, 2)

    def test_budget_frontier_has_monotone_size_and_exact_tightness(self) -> None:
        frontier = residual_budget_frontier(
            _two_shared_residual_diamonds(),
            max_budget=2,
        )
        self.assertEqual([metric.requested_budget for metric in frontier], [0, 1, 2])
        self.assertEqual([len(metric.selected) for metric in frontier], [0, 1, 2])
        self.assertEqual([metric.coupling_rows for metric in frontier], [4, 2, 0])
        self.assertTrue(
            all(
                right.coupling_nnz <= left.coupling_nnz
                for left, right in zip(frontier, frontier[1:])
            )
        )
        self.assertEqual(
            [metric.relaxation_upper for metric in frontier],
            [Q(6), Q(6), Q(6)],
        )
        self.assertEqual(
            [metric.upper_gap for metric in frontier],
            [Q(0), Q(0), Q(0)],
        )
        decision = strict_stop_loss_prefix(frontier)
        self.assertEqual(decision.reason, "all_prefixes_promoted")
        self.assertEqual(len(decision.accepted), 3)
        self.assertIsNone(decision.rejected)

    def test_fraction_phase_oracle_proves_nonlinear_residual_identity(self) -> None:
        dag = _nonlinear_residual_identity()
        oracle = fraction_phase_oracle(dag)
        self.assertEqual(oracle.true_lower, 0)
        self.assertEqual(oracle.true_upper, 0)
        self.assertEqual(oracle.enumerated_phase_patterns, 8)
        self.assertGreaterEqual(oracle.feasible_phase_regions, 2)

        # Every proposed nested encoding remains sound on all exact phases.
        frontier = residual_budget_frontier(dag, max_budget=3)
        self.assertTrue(
            all(metric.relaxation_lower <= 0 <= metric.relaxation_upper
                for metric in frontier)
        )
        self.assertTrue(
            all(metric.containment_assignments > 0 for metric in frontier)
        )

    def test_stop_loss_halts_at_first_plateau_or_tightness_regression(self) -> None:
        frontier = residual_budget_frontier(
            _nonlinear_residual_identity(),
            max_budget=3,
        )
        decision = strict_stop_loss_prefix(frontier)
        self.assertIn(
            decision.reason,
            {
                "property_upper_regressed",
                "coupling_nnz_increased",
                "budget_plateau_no_row_saving_candidate",
            },
        )
        self.assertIsNotNone(decision.rejected)
        self.assertLess(len(decision.accepted), len(frontier))

    def test_bounded_fraction_fuzz_all_budgets_contain_true_graph(self) -> None:
        rng = random.Random(73191)
        encodings = endpoint_assignments = 0
        for _case in range(64):
            nodes = [ScalarNode.input("x")]
            available = ["x"]
            for layer in range(3):
                picked = rng.sample(
                    available,
                    k=min(len(available), rng.randint(1, 2)),
                )
                terms = {
                    name: Q(
                        rng.choice((-3, -2, -1, 1, 2, 3)),
                        rng.choice((1, 2, 3)),
                    )
                    for name in picked
                }
                pre, relu = f"p{layer}", f"h{layer}"
                nodes.append(
                    ScalarNode.affine(
                        pre,
                        terms,
                        Q(rng.randint(-2, 2), 2),
                    )
                )
                nodes.append(ScalarNode.relu(relu, pre))
                available.append(relu)
            output_terms = {
                name: Q(rng.choice((-2, -1, 1, 2)))
                for name in rng.sample(available, k=min(3, len(available)))
            }
            nodes.append(
                ScalarNode.affine(
                    "out",
                    output_terms,
                    Q(rng.randint(-1, 1), 2),
                )
            )
            dag = FractionScalarDAG.make(
                nodes,
                input_lower=-1,
                input_upper=1,
                output="out",
            )
            oracle = fraction_phase_oracle(dag)
            candidates = property_residual_candidates(dag, oracle)
            for budget in range(len(candidates) + 1):
                relaxation = encode_fraction_relaxation(
                    dag,
                    nested_residual_plan(candidates, budget),
                    oracle=oracle,
                )
                audit = audit_fraction_phase_containment(
                    relaxation, oracle=oracle
                )
                lp = exact_fraction_lp_range(relaxation)
                self.assertLessEqual(lp.lower, oracle.true_lower)
                self.assertGreaterEqual(lp.upper, oracle.true_upper)
                endpoint_assignments += audit.endpoint_assignments
                encodings += 1
        self.assertGreaterEqual(encodings, 128)
        self.assertGreaterEqual(endpoint_assignments, encodings)


if __name__ == "__main__":
    unittest.main(verbosity=2)
