"""Soundness/tightness gates for child-local backward bound refinement."""

from __future__ import annotations

import unittest

import torch

from act.back_end.bab.bab import (
    _branch_layers_with_unstable_successors,
    _filter_branching_state_to_unstable_successors,
)
from act.back_end.bab.node import SubproblemBatch
from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf.tf_forward import compute_forward_bounds
from act.back_end.solver.solver_dual import DualSolver
from act.front_end.specs import OutKind, OutputSpec
from act.util.device_manager import initialize_device


DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _opposite_relu_residual_net() -> Net:
    """x -> [x,-x] -> ReLU -> sum - 3/4 -> ReLU."""

    assertion = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.tensor([1.0], dtype=DTYPE),
        d=torch.tensor([1.0], dtype=DTYPE),
    ).encode_linear(B=1, n_out=1, device=DEVICE, dtype=DTYPE)
    layers = [
        Layer(
            id=0,
            kind="INPUT",
            params={"shape": (1, 1), "dtype": "torch.float64"},
            in_vars=[],
            out_vars=[0],
        ),
        Layer(
            id=1,
            kind="INPUT_SPEC",
            params={
                "kind": "BOX",
                "lb": torch.tensor([[-1.0]], dtype=DTYPE),
                "ub": torch.tensor([[1.0]], dtype=DTYPE),
            },
            in_vars=[0],
            out_vars=[0],
        ),
        Layer(
            id=2,
            kind="DENSE",
            params={
                "weight": torch.tensor([[1.0], [-1.0]], dtype=DTYPE),
                "bias": torch.zeros(2, dtype=DTYPE),
                "in_features": 1,
                "out_features": 2,
            },
            in_vars=[0],
            out_vars=[1, 2],
        ),
        Layer(
            id=3,
            kind="RELU",
            params={},
            in_vars=[1, 2],
            out_vars=[3, 4],
        ),
        Layer(
            id=4,
            kind="DENSE",
            params={
                "weight": torch.tensor([[1.0, 1.0]], dtype=DTYPE),
                "bias": torch.tensor([-0.75], dtype=DTYPE),
                "in_features": 2,
                "out_features": 1,
            },
            in_vars=[3, 4],
            out_vars=[5],
        ),
        Layer(
            id=5,
            kind="RELU",
            params={},
            in_vars=[5],
            out_vars=[6],
        ),
        Layer(
            id=6,
            kind="ASSERT",
            params=assertion,
            in_vars=[6],
            out_vars=[6],
        ),
    ]
    return Net(
        layers=layers,
        preds={
            0: [],
            1: [0],
            2: [1],
            3: [2],
            4: [3],
            5: [4],
            6: [5],
        },
        succs={
            0: [1],
            1: [2],
            2: [3],
            3: [4],
            4: [5],
            5: [6],
            6: [],
        },
    )


class PerSubproblemRefineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_device("cpu", "float64")

    def test_split_hardened_refine_is_monotone_and_grid_sound(self):
        net = _opposite_relu_residual_net()
        lower = torch.full((2, 1), -1.0, dtype=DTYPE)
        upper = torch.full((2, 1), 1.0, dtype=DTYPE)
        before = compute_forward_bounds(net, lower, upper)
        split_signs = {
            3: torch.tensor(
                [
                    [[1.0, 0.0]],
                    [[-1.0, 0.0]],
                ],
                dtype=DTYPE,
            )
        }
        audit = {}
        after = DualSolver().refine_intermediate_bounds_batched(
            net,
            before,
            split_signs=split_signs,
            mode="split_successors",
            rows_cap=4,
            optimize_iters=0,
            lane_chunk=2,
            layer_cap=2,
            audit=audit,
        )
        self.assertEqual(audit["selected_layer_ids"], [5])
        self.assertEqual(audit["queried_objective_rows"], 4)

        for layer_id, old in before.items():
            new = after[layer_id]
            self.assertTrue(torch.all(new.lb >= old.lb - 1e-12))
            self.assertTrue(torch.all(new.ub <= old.ub + 1e-12))
            self.assertTrue(torch.all(new.lb <= new.ub + 1e-12))

        # Layer 3 stores preactivation bounds.  The requested phase must be
        # explicit in each lane and the downstream residual preactivation
        # (layer 5) has exact upper 1/4 on both branches.
        self.assertGreaterEqual(float(after[3].lb[0, 0]), 0.0)
        self.assertLessEqual(float(after[3].ub[1, 0]), 0.0)
        self.assertLessEqual(float(after[5].ub.max()), 0.250000000001)

        grid = torch.linspace(-1.0, 1.0, 4001, dtype=DTYPE)
        for lane in range(2):
            phase_points = grid[grid >= 0] if lane == 0 else grid[grid <= 0]
            first = torch.stack([phase_points, -phase_points], dim=1)
            first_relu = torch.relu(first)
            residual = first_relu.sum(dim=1) - 0.75
            final_relu = torch.relu(residual)
            exact_by_layer = {
                3: first,
                5: residual.unsqueeze(1),
                6: final_relu.unsqueeze(1),
            }
            for layer_id, exact in exact_by_layer.items():
                bound = after[layer_id]
                self.assertTrue(
                    torch.all(exact >= bound.lb[lane] - 1e-12)
                )
                self.assertTrue(
                    torch.all(exact <= bound.ub[lane] + 1e-12)
                )

    def test_terminal_relu_is_excluded_from_long_horizon_candidates(self):
        net = _opposite_relu_residual_net()
        bounds = compute_forward_bounds(
            net,
            torch.tensor([[-1.0]], dtype=DTYPE),
            torch.tensor([[1.0]], dtype=DTYPE),
        )
        nu = {
            3: torch.ones(1, 1, 2, dtype=DTYPE),
            5: torch.ones(1, 1, 1, dtype=DTYPE),
        }
        self.assertEqual(
            _branch_layers_with_unstable_successors(net, bounds, nu),
            {3},
        )
        batch = SubproblemBatch(
            lb=torch.tensor([[-1.0]], dtype=DTYPE),
            ub=torch.tensor([[1.0]], dtype=DTYPE),
            depths=torch.zeros(1, dtype=torch.long),
        )
        filtered_bounds, filtered_nu, eligible, applied = (
            _filter_branching_state_to_unstable_successors(
                batch, net, bounds, nu
            )
        )
        self.assertTrue(applied)
        self.assertEqual(eligible, {3})
        self.assertEqual(set(filtered_bounds or {}), {3})
        self.assertEqual(set(filtered_nu or {}), {3})

        # When one lane has already exhausted every nonterminal candidate,
        # the batch-wide filter must fail safe to the complete original
        # branching state instead of letting top-k select an -inf entry.
        two_lane_bounds = compute_forward_bounds(
            net,
            torch.full((2, 1), -1.0, dtype=DTYPE),
            torch.full((2, 1), 1.0, dtype=DTYPE),
        )
        two_lane_nu = {
            3: torch.ones(2, 1, 2, dtype=DTYPE),
            5: torch.ones(2, 1, 1, dtype=DTYPE),
        }
        two_lane_batch = SubproblemBatch(
            lb=torch.full((2, 1), -1.0, dtype=DTYPE),
            ub=torch.full((2, 1), 1.0, dtype=DTYPE),
            depths=torch.zeros(2, dtype=torch.long),
            split_signs={
                3: torch.tensor(
                    [
                        [[0.0, 0.0]],
                        [[1.0, -1.0]],
                    ],
                    dtype=DTYPE,
                )
            },
        )
        fallback_bounds, fallback_nu, eligible, applied = (
            _filter_branching_state_to_unstable_successors(
                two_lane_batch,
                net,
                two_lane_bounds,
                two_lane_nu,
            )
        )
        self.assertFalse(applied)
        self.assertEqual(eligible, {3})
        self.assertIs(fallback_bounds, two_lane_bounds)
        self.assertIs(fallback_nu, two_lane_nu)

    def test_split_phase_must_be_shared_by_every_property_row(self):
        net = _opposite_relu_residual_net()
        bounds = compute_forward_bounds(
            net,
            torch.tensor([[-1.0]], dtype=DTYPE),
            torch.tensor([[1.0]], dtype=DTYPE),
        )
        mismatched = {
            3: torch.tensor(
                [[[1.0, 0.0], [-1.0, 0.0]]],
                dtype=DTYPE,
            )
        }
        with self.assertRaisesRegex(ValueError, "shared"):
            DualSolver()._harden_split_bounds(bounds, mismatched)


if __name__ == "__main__":
    unittest.main()
