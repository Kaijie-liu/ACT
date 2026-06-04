"""ImageHZ-lite Phase 0 unit tests.

Per §9R-9 of the prototype plan, these tests are the entry condition for
running any sentinel. They cover:

  - test_conv2d_equiv_dense_hz
  - test_relu_triangle_soundness_small_random
  - test_maxpool_stable_exact
  - test_maxpool_unstable_containment
  - test_flatten_column_order
  - test_budget_fail_closed
  - test_structural_gate_matches_only_conv_relu_maxpool_flatten

All tests run on CPU with seeded randomness. Anything failing here
blocks Phase 0 sentinel runs.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from research.imagehz_lite.budget import Budget, BudgetExceeded
from research.imagehz_lite.domain import (
    ImageHZLite,
    Phase0FlattenSnapshot,
    TileBlock,
)
from research.imagehz_lite.ops import (
    apply_conv2d,
    apply_flatten,
    apply_maxpool2d,
    apply_relu_triangle,
    structural_gate_passes,
)


def _full_root_hz(center: torch.Tensor, radius: torch.Tensor) -> ImageHZLite:
    """Build an ImageHZLite where each (c, h, w) position carries one
    root generator with the given radius. Useful as a starting point
    for tests."""
    C, H, W = center.shape
    tiles = []
    fid = 0
    for ci in range(C):
        for hi in range(H):
            for wi in range(W):
                r = float(radius[ci, hi, wi].item())
                if r == 0.0:
                    fid += 1
                    continue
                G = torch.tensor(
                    [[[[r]]]],
                    dtype=torch.float64,
                )
                tiles.append(TileBlock(
                    origin_chw=(ci, hi, wi),
                    shape=(1, 1, 1),
                    G_tile=G,
                    factor_ids=(fid,),
                    aux_meta={
                        "kind": "root",
                        "spawn_layer": 0,
                        "spawn_op": "input",
                        "parent_block": None,
                    },
                ))
                fid += 1
    return ImageHZLite(c=center.clone(), tiles=tiles)


class TestImageHZLite(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(20260604)
        np.random.seed(20260604)

    # ── Conv2D equivalence ────────────────────────────────────────

    def test_conv2d_equiv_dense_hz(self) -> None:
        """Apply Conv2D to ImageHZ-lite and to a brute-force enumeration
        of the input set; compare bounds at each output position."""
        C_in, H, W = 2, 4, 4
        C_out = 3
        k = 3
        center = torch.zeros((C_in, H, W), dtype=torch.float64)
        radius = torch.full_like(center, 0.1)
        hz = _full_root_hz(center, radius)

        weight = torch.randn((C_out, C_in, k, k), dtype=torch.float64)
        bias = torch.randn((C_out,), dtype=torch.float64)
        hz_out = apply_conv2d(hz, weight, bias=bias, stride=1, padding=0)
        lb_lite, ub_lite = hz_out.bounds()

        # Brute force: sample 200 corners of [-r, +r]^N input and compute
        # actual conv output range.
        N = C_in * H * W
        # Only use 200 sampled corners (full enumeration is 2^32 here);
        # the lite bounds must over-approximate every realized point.
        for _ in range(200):
            sign = torch.from_numpy(
                np.random.choice([-1.0, 1.0], size=N)
            ).to(torch.float64).reshape(C_in, H, W)
            x = center + radius * sign
            y = F.conv2d(
                x.unsqueeze(0), weight, bias=bias,
                stride=1, padding=0,
            ).squeeze(0)
            self.assertTrue(
                torch.all(y >= lb_lite - 1e-9),
                msg="conv2d output below lite lb",
            )
            self.assertTrue(
                torch.all(y <= ub_lite + 1e-9),
                msg="conv2d output above lite ub",
            )

    # ── ReLU soundness ────────────────────────────────────────────

    def test_relu_triangle_soundness_small_random(self) -> None:
        """ReLU output set must contain every realized output of the
        underlying box."""
        for trial in range(50):
            C, H, W = 2, 2, 2
            center = (torch.rand((C, H, W), dtype=torch.float64) - 0.5) * 2.0
            radius = torch.rand((C, H, W), dtype=torch.float64) * 0.8 + 0.1
            hz = _full_root_hz(center, radius)
            budget = Budget(max_relu_aux_per_image=10_000)
            hz_out, _ = apply_relu_triangle(
                hz, budget, layer_id=1, next_aux_id=10_000,
            )
            lb_o, ub_o = hz_out.bounds()
            # sample 30 corners
            N = C * H * W
            for _ in range(30):
                sign = torch.from_numpy(
                    np.random.choice([-1.0, 1.0], size=N)
                ).to(torch.float64).reshape(C, H, W)
                x = center + radius * sign
                y = torch.relu(x)
                self.assertTrue(
                    torch.all(y >= lb_o - 1e-9),
                    msg=f"trial {trial}: y below lb",
                )
                self.assertTrue(
                    torch.all(y <= ub_o + 1e-9),
                    msg=f"trial {trial}: y above ub",
                )

    # ── MaxPool stable-exact ──────────────────────────────────────

    def test_maxpool_stable_exact(self) -> None:
        """When one input position dominates its pool window, that
        window's output equals the dominant input position's center
        (exact). When no dominant position exists, the output is the
        unstable over-approximation.

        Setup: position (0, 1, 1) dominates the first pool window
        (0..1, 0..1) (lb=9.95 vs max_others ub=0.01). The other three
        windows contain no dominant cell (all positions have identical
        ±0.01 bounds), so those are unstable.
        """
        C, H, W = 1, 4, 4
        center = torch.zeros((C, H, W), dtype=torch.float64)
        center[0, 1, 1] = 10.0
        radius = torch.full_like(center, 0.01)
        radius[0, 1, 1] = 0.05
        hz = _full_root_hz(center, radius)
        budget = Budget(max_relu_aux_per_image=10_000)
        hz_out, _, stats = apply_maxpool2d(
            hz, kernel_size=2, stride=2,
            budget=budget, layer_id=2, next_aux_id=10_000,
        )
        # 1 stable winner (containing (0,1,1)); 3 unstable windows.
        self.assertEqual(stats["n_stable"], 1)
        self.assertEqual(stats["n_unstable"], 3)
        self.assertEqual(stats["n_output_positions"], 4)
        # Stable window's output center is the dominant input's center.
        self.assertAlmostEqual(
            float(hz_out.c[0, 0, 0].item()), 10.0, places=12,
        )
        # And that output position has root provenance from the
        # winner's TileBlock.
        self.assertGreaterEqual(
            stats["n_output_positions_with_root_provenance"], 1,
        )

    # ── MaxPool unstable containment ──────────────────────────────

    def test_maxpool_unstable_containment(self) -> None:
        """When no stable winner, the lite output set must contain
        every realized max over the input."""
        C, H, W = 1, 4, 4
        center = torch.zeros((C, H, W), dtype=torch.float64)
        center[0, 0, 0] = 0.5
        center[0, 0, 1] = 0.6
        center[0, 1, 0] = 0.55
        center[0, 1, 1] = 0.4
        radius = torch.full_like(center, 0.3)
        hz = _full_root_hz(center, radius)
        budget = Budget(max_relu_aux_per_image=10_000)
        hz_out, _, stats = apply_maxpool2d(
            hz, kernel_size=2, stride=2,
            budget=budget, layer_id=2, next_aux_id=10_000,
        )
        # No stable winner in the (0..1, 0..1) window.
        self.assertGreater(stats["n_unstable"], 0)
        lb_o, ub_o = hz_out.bounds()
        # Sample some realizations and confirm containment.
        N = C * H * W
        for _ in range(100):
            sign = torch.from_numpy(
                np.random.choice([-1.0, 1.0], size=N)
            ).to(torch.float64).reshape(C, H, W)
            x = center + radius * sign
            y = F.max_pool2d(x.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
            self.assertTrue(
                torch.all(y >= lb_o - 1e-9),
                msg="unstable maxpool y below lb",
            )
            self.assertTrue(
                torch.all(y <= ub_o + 1e-9),
                msg="unstable maxpool y above ub",
            )

    # ── Flatten column order ──────────────────────────────────────

    def test_flatten_column_order(self) -> None:
        """The snapshot's blocks_meta must be ordered:
           1) root before relu_aux,
           2) by spawn_layer,
           3) by origin_chw.
        """
        # Build an HZ with a mix of root + aux tiles, intentionally
        # out of order.
        center = torch.zeros((1, 2, 2), dtype=torch.float64)
        tiles = [
            TileBlock(
                origin_chw=(0, 1, 0),
                shape=(1, 1, 1),
                G_tile=torch.tensor([[[[0.5]]]], dtype=torch.float64),
                factor_ids=(50,),
                aux_meta={"kind": "relu_aux", "spawn_layer": 3,
                          "spawn_op": "x", "parent_block": None},
            ),
            TileBlock(
                origin_chw=(0, 0, 1),
                shape=(1, 1, 1),
                G_tile=torch.tensor([[[[0.5]]]], dtype=torch.float64),
                factor_ids=(2,),
                aux_meta={"kind": "root", "spawn_layer": 0,
                          "spawn_op": "input", "parent_block": None},
            ),
            TileBlock(
                origin_chw=(0, 0, 0),
                shape=(1, 1, 1),
                G_tile=torch.tensor([[[[0.5]]]], dtype=torch.float64),
                factor_ids=(1,),
                aux_meta={"kind": "root", "spawn_layer": 0,
                          "spawn_op": "input", "parent_block": None},
            ),
            TileBlock(
                origin_chw=(0, 0, 0),
                shape=(1, 1, 1),
                G_tile=torch.tensor([[[[0.5]]]], dtype=torch.float64),
                factor_ids=(40,),
                aux_meta={"kind": "relu_aux", "spawn_layer": 2,
                          "spawn_op": "x", "parent_block": None},
            ),
        ]
        hz = ImageHZLite(c=center, tiles=tiles)
        snap = apply_flatten(hz, girard_fires=[], peak_memory_bytes=0,
                              wall_s=0.0)
        order = [
            (b["aux_kind"], b["spawn_layer"], b["origin_chw"])
            for b in snap.blocks_meta
        ]
        self.assertEqual(order, [
            ("root", 0, (0, 0, 0)),
            ("root", 0, (0, 0, 1)),
            ("relu_aux", 2, (0, 0, 0)),
            ("relu_aux", 3, (0, 1, 0)),
        ])
        self.assertEqual(snap.root_ng_at_flatten, 2)
        self.assertEqual(snap.total_aux_count, 2)

    # ── Budget fail-closed ────────────────────────────────────────

    def test_budget_fail_closed(self) -> None:
        """When ReLU aux count would exceed the budget,
        BudgetExceeded is raised and no partial state leaks."""
        C, H, W = 1, 4, 4
        center = (torch.rand((C, H, W), dtype=torch.float64) - 0.5) * 2.0
        radius = torch.full_like(center, 0.8)
        hz = _full_root_hz(center, radius)
        # All 16 positions are likely unstable. Set the cap to 3.
        budget = Budget(max_relu_aux_per_image=3)
        with self.assertRaises(BudgetExceeded) as cm:
            apply_relu_triangle(
                hz, budget, layer_id=99, next_aux_id=10_000,
            )
        self.assertEqual(cm.exception.kind, "relu_aux")
        self.assertEqual(cm.exception.layer_id, 99)
        # budget should be in a state where the event was recorded.
        self.assertEqual(len(budget.fail_closed_events), 1)
        self.assertEqual(budget.fail_closed_events[0]["kind"], "relu_aux")

    # ── Deterministic exhaustive corner enumeration ────────────────

    def test_relu_exhaustive_2x2_containment(self) -> None:
        """On a 2x2 single-channel toy box, enumerate ALL 2^4 = 16
        corner inputs and verify each realized ReLU output is contained
        in the ImageHZ-lite bounds. Deterministic; no random sampling.
        """
        from itertools import product
        C, H, W = 1, 2, 2
        center = torch.tensor(
            [[[[0.0, 0.3], [-0.2, 0.4]]]], dtype=torch.float64,
        ).squeeze(0)
        radius = torch.tensor(
            [[[[0.5, 0.4], [0.6, 0.3]]]], dtype=torch.float64,
        ).squeeze(0)
        hz = _full_root_hz(center, radius)
        budget = Budget(max_relu_aux_per_image=10_000)
        hz_out, _ = apply_relu_triangle(
            hz, budget, layer_id=11, next_aux_id=10_000,
        )
        lb_o, ub_o = hz_out.bounds()
        N = C * H * W
        for sign_pattern in product([-1.0, 1.0], repeat=N):
            sign = torch.tensor(sign_pattern, dtype=torch.float64).reshape(C, H, W)
            x = center + radius * sign
            y = torch.relu(x)
            self.assertTrue(
                torch.all(y >= lb_o - 1e-12),
                msg=f"ReLU sign={sign_pattern}: y below lb",
            )
            self.assertTrue(
                torch.all(y <= ub_o + 1e-12),
                msg=f"ReLU sign={sign_pattern}: y above ub",
            )

    def test_maxpool_exhaustive_2x2_unstable_containment(self) -> None:
        """2x2 single-window MaxPool, all 16 corners enumerated.
        Verify true max ∈ lite output bounds for every realization.
        """
        from itertools import product
        C, H, W = 1, 2, 2
        center = torch.tensor(
            [[[[0.2, 0.3], [0.1, 0.25]]]], dtype=torch.float64,
        ).squeeze(0)
        radius = torch.tensor(
            [[[[0.4, 0.4], [0.4, 0.4]]]], dtype=torch.float64,
        ).squeeze(0)
        hz = _full_root_hz(center, radius)
        budget = Budget(max_relu_aux_per_image=10_000)
        hz_out, _, stats = apply_maxpool2d(
            hz, kernel_size=2, stride=2,
            budget=budget, layer_id=12, next_aux_id=10_000,
        )
        lb_o, ub_o = hz_out.bounds()
        N = C * H * W
        for sign_pattern in product([-1.0, 1.0], repeat=N):
            sign = torch.tensor(sign_pattern, dtype=torch.float64).reshape(C, H, W)
            x = center + radius * sign
            y = F.max_pool2d(x.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
            self.assertTrue(
                torch.all(y >= lb_o - 1e-12),
                msg=f"MaxPool sign={sign_pattern}: y below lb (y={y}, lb={lb_o})",
            )
            self.assertTrue(
                torch.all(y <= ub_o + 1e-12),
                msg=f"MaxPool sign={sign_pattern}: y above ub (y={y}, ub={ub_o})",
            )
        # Window is unstable (all 4 cells in [-0.2, 0.7], no dominant
        # winner; 0.7 < 0.7 fails the stable check).
        self.assertGreaterEqual(stats["n_unstable"], 1)

    # ── Structural gate ───────────────────────────────────────────

    def test_structural_gate_matches_only_conv_relu_maxpool_flatten(self) -> None:
        """The structural gate must accept VGG-like prefixes and reject
        ResNet (Add), CIFAR (no MaxPool), and LSNC (Mul/Sub/Div)."""
        vgg_prefix = ["Conv2D", "Relu", "Conv2D", "Relu", "MaxPool2D",
                       "Conv2D", "Relu", "MaxPool2D"]
        self.assertTrue(structural_gate_passes(
            vgg_prefix + ["Flatten", "Gemm"],
            flatten_index=len(vgg_prefix),
            trace_has_girard_root_loss_at_maxpool_or_relu=True,
        ))
        # No trace evidence → reject (per §9R-5 second conjunct).
        self.assertFalse(structural_gate_passes(
            vgg_prefix + ["Flatten", "Gemm"],
            flatten_index=len(vgg_prefix),
            trace_has_girard_root_loss_at_maxpool_or_relu=False,
        ))
        # ResNet (Add in prefix) → reject.
        self.assertFalse(structural_gate_passes(
            ["Conv2D", "Relu", "Conv2D", "Add", "Relu",
             "Flatten", "Gemm"],
            flatten_index=5,
            trace_has_girard_root_loss_at_maxpool_or_relu=True,
        ))
        # CIFAR (no MaxPool, BatchNorm in prefix) → reject.
        self.assertFalse(structural_gate_passes(
            ["Conv2D", "BatchNormalization", "Relu", "Flatten", "Gemm"],
            flatten_index=3,
            trace_has_girard_root_loss_at_maxpool_or_relu=True,
        ))
        # LSNC (Mul/Sub) → reject.
        self.assertFalse(structural_gate_passes(
            ["Mul", "Sub", "Conv2D", "Relu", "Flatten"],
            flatten_index=4,
            trace_has_girard_root_loss_at_maxpool_or_relu=True,
        ))


if __name__ == "__main__":
    unittest.main()
