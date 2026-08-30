"""Controlled gates for layer-diverse, gain-tested joint ReLU branching."""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from act.back_end.bab.bab import (
    _gain_tested_multi_split,
    _survival_controlled_split_depth,
)
from act.back_end.bab.branching.branching import (
    _collect_neuron_candidates,
    _multi_split_from_groups,
    _propose_joint_split_groups,
)
from act.back_end.bab.node import SubproblemBatch
from act.back_end.config import BaBConfig
from act.back_end.core import Bounds


def _layer(layer_id: int, width: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=layer_id,
        out_vars=(0, width - 1),
    )


class JointGainBranchingTests(unittest.TestCase):
    def test_default_is_off_and_nonpositive_group_count_fails(self):
        self.assertEqual(BaBConfig().joint_gain_groups, 1)
        with self.assertRaisesRegex(ValueError, "joint_gain_groups"):
            BaBConfig(joint_gain_groups=0)
        with self.assertRaisesRegex(ValueError, "property_branch_focus"):
            BaBConfig(property_branch_focus="unknown")
        with self.assertRaisesRegex(ValueError, "frontier_contraction_target"):
            BaBConfig(frontier_contraction_target=1.01)

    def test_survivor_rate_caps_split_depth_by_frontier_equation(self):
        self.assertEqual(
            _survival_controlled_split_depth(3, 0.10, 0.90),
            3,
        )
        self.assertEqual(
            _survival_controlled_split_depth(3, 0.22, 0.90),
            2,
        )
        self.assertEqual(
            _survival_controlled_split_depth(3, 0.30, 0.90),
            1,
        )
        self.assertEqual(
            _survival_controlled_split_depth(3, 1.00, 0.90),
            1,
        )

    def test_worst_property_focus_avoids_easy_row_vote_dilution(self):
        batch = SubproblemBatch(
            lb=torch.tensor([[-1.0]]),
            ub=torch.tensor([[1.0]]),
            depths=torch.zeros(1, dtype=torch.long),
        )
        bounds = {
            10: Bounds(
                torch.tensor([[-2.0, -2.0]]),
                torch.tensor([[2.0, 2.0]]),
            )
        }
        # Row zero is the current worst rival and strongly prefers neuron 1.
        # The easier row's large neuron-0 sensitivity flips the old sum vote.
        nu = {10: torch.tensor([[[1.0, 10.0], [20.0, 0.0]]])}
        summed = _collect_neuron_candidates(batch, bounds, nu)
        focused = _collect_neuron_candidates(
            batch,
            bounds,
            nu,
            spec_row_index=torch.tensor([0]),
        )
        self.assertIsNotNone(summed)
        self.assertIsNotNone(focused)
        self.assertEqual(int(summed[0].argmax(dim=1)[0]), 0)
        self.assertEqual(int(focused[0].argmax(dim=1)[0]), 1)

    def test_proposals_keep_global_topk_and_add_layer_diversity(self):
        scores = torch.tensor([[10.0, 9.0, 8.0, 7.0]])
        layers = torch.tensor([[10, 10, 20, 30]])
        neurons = torch.tensor([[0, 1, 0, 0]])
        proposed = _propose_joint_split_groups(
            scores,
            layers,
            neurons,
            k_levels=2,
            max_groups=4,
            pool_size=4,
        )
        self.assertIsNotNone(proposed)
        group_layers, group_neurons = proposed
        self.assertEqual(group_layers.shape, (1, 4, 2))
        self.assertEqual(group_layers[0, 0].tolist(), [10, 10])
        self.assertEqual(group_neurons[0, 0].tolist(), [0, 1])
        self.assertTrue(
            any(
                len(set(group_layers[0, group].tolist())) == 2
                for group in range(1, group_layers.shape[1])
            )
        )

    def test_complete_joint_partition_covers_fraction_phase_oracle(self):
        net = SimpleNamespace(by_id={10: _layer(10, 1), 20: _layer(20, 1)})
        parent = SubproblemBatch(
            lb=torch.tensor([[-1.0]]),
            ub=torch.tensor([[1.0]]),
            depths=torch.zeros(1, dtype=torch.long),
        )
        children, _ = _multi_split_from_groups(
            parent,
            net,
            torch.tensor([[10, 20]]),
            torch.tensor([[0, 0]]),
            2,
        )
        patterns = {
            (
                int(children.split_signs[10][row, 0, 0].item()),
                int(children.split_signs[20][row, 0, 0].item()),
            )
            for row in range(children.batch_size)
        }
        self.assertEqual(
            patterns,
            {(1, 1), (1, -1), (-1, 1), (-1, -1)},
        )

        # Exact rational preactivations at each point select at least one
        # child.  Zero lies on both closed phase halfspaces, so overlap at the
        # boundary is allowed while the union remains complete.
        for first, second in (
            (Fraction(-3, 4), Fraction(1, 5)),
            (Fraction(0), Fraction(-2, 3)),
            (Fraction(7, 9), Fraction(0)),
            (Fraction(1, 3), Fraction(5, 8)),
        ):
            self.assertTrue(
                any(
                    (first >= 0 if sign_first > 0 else first <= 0)
                    and (second >= 0 if sign_second > 0 else second <= 0)
                    for sign_first, sign_second in patterns
                )
            )

    def test_measured_worst_child_selects_diverse_group(self):
        net = SimpleNamespace(
            by_id={10: _layer(10, 2), 20: _layer(20, 2)}
        )
        batch = SubproblemBatch(
            lb=torch.tensor([[-1.0]]),
            ub=torch.tensor([[1.0]]),
            depths=torch.zeros(1, dtype=torch.long),
        )
        bounds = {
            10: Bounds(
                torch.tensor([[-2.0, -2.0]]),
                torch.tensor([[2.0, 2.0]]),
            ),
            20: Bounds(
                torch.tensor([[-2.0, -2.0]]),
                torch.tensor([[2.0, 2.0]]),
            ),
        }
        nu = {
            10: torch.tensor([[[10.0, 9.0]]]),
            20: torch.tensor([[[8.0, 7.0]]]),
        }

        def measured_dispatch(**kwargs):
            probe = kwargs["batch"]
            lower = torch.zeros(probe.batch_size)
            for row in range(probe.batch_size):
                uses_first = bool(
                    torch.any(probe.split_signs[10][row] != 0).item()
                )
                uses_second = bool(
                    torch.any(probe.split_signs[20][row] != 0).item()
                )
                # This is a frozen exact-oracle fixture: every complete
                # cross-layer partition has certified child LB 1, while the
                # redundant same-layer partition has LB 0.
                lower[row] = 1.0 if uses_first and uses_second else 0.0
            return SimpleNamespace(
                solution=SimpleNamespace(max_viol=-lower)
            )

        with patch(
            "act.back_end.bab.bab._dispatch_dual_solve",
            side_effect=measured_dispatch,
        ):
            audit = {}
            result = _gain_tested_multi_split(
                batch,
                net,
                SimpleNamespace(),
                BaBConfig(
                    solver_tier="dual_alpha_eta",
                    branching_method="gain",
                    multi_split_levels=2,
                    joint_gain_groups=4,
                ),
                None,
                bounds,
                bounds,
                nu,
                (),
                k_levels=2,
                max_groups=4,
                max_probe_batch=16,
                audit=audit,
            )
        self.assertIsNotNone(result)
        self.assertEqual(audit["probe_nodes"], 16)
        self.assertEqual(audit["selected_nonbaseline_lanes"], 1)
        self.assertEqual(audit["selected_more_diverse_lanes"], 1)
        self.assertEqual(audit["baseline_worst_child_lb"], [0.0])
        self.assertEqual(audit["selected_worst_child_lb"], [1.0])
        children, _ = result
        self.assertEqual(children.batch_size, 4)
        for row in range(children.batch_size):
            self.assertTrue(
                torch.any(children.split_signs[10][row] != 0).item()
            )
            self.assertTrue(
                torch.any(children.split_signs[20][row] != 0).item()
            )


if __name__ == "__main__":
    unittest.main()
