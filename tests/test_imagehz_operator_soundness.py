"""ImageHZ operator soundness — 4 mandatory tests per advisor rev-2.

1. Conv2D toy: ImageHZ → Flatten → bounds covers reference dense-HZ
   → Conv2D → bounds (small toy).
2. ReLU toy: ImageHZ → triangle → output bounds cover bounds of all
   randomly sampled exact points in the input box.
3. ADD toy: same `xi_id` merges; distinct `xi_id` produce union.
4. Flatten bridge: post-bridge SparseGcZ bounds are NOT narrower
   than ImageHZ bounds at the bridge point.

Run: PYTHONPATH=/data1/Kane/ACT python tests/test_imagehz_operator_soundness.py
"""
from __future__ import annotations

import sys
import unittest

import torch

sys.path.insert(0, ".")

from act.back_end.imagehz import (  # noqa: E402
    BoundingBox,
    ImageHZ,
    SpatialGenerator,
    apply_add,
    apply_avgpool2d,
    apply_conv2d,
    apply_maxpool2d,
    apply_relu_triangle,
    flatten_to_sparsegcz,
)


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


def _full_image_generator(C: int, H: int, W: int,
                          values: torch.Tensor, xi_id: int
                          ) -> SpatialGenerator:
    return SpatialGenerator(
        region=BoundingBox(0, C, 0, H, 0, W),
        values=values.contiguous(),
        xi_id=xi_id,
    )


def _box_bounds_via_sampling(
    c: torch.Tensor,
    gens: list[SpatialGenerator],
    n_samples: int = 5000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample xi ∈ [-1, 1]^ng uniformly; return min/max per cell."""
    # Group by xi_id so shared xis use a single sample.
    xi_ids = sorted({g.xi_id for g in gens})
    id_to_idx = {x: i for i, x in enumerate(xi_ids)}
    ng_u = len(xi_ids)
    xi = (torch.rand(n_samples, ng_u, dtype=c.dtype, device=c.device)
          * 2.0 - 1.0)
    # Add corner samples to stress the extremes.
    corners = torch.stack([
        torch.full((ng_u,), -1.0, dtype=c.dtype, device=c.device),
        torch.full((ng_u,),  1.0, dtype=c.dtype, device=c.device),
    ])
    xi = torch.cat([xi, corners], dim=0)

    out_min = c.clone()
    out_max = c.clone()
    for s_idx in range(xi.shape[0]):
        sample = c.clone()
        for g in gens:
            r = g.region
            xi_val = xi[s_idx, id_to_idx[g.xi_id]]
            sample[
                r.c_lo:r.c_hi, r.h_lo:r.h_hi, r.w_lo:r.w_hi,
            ] += xi_val * g.values
        out_min = torch.minimum(out_min, sample)
        out_max = torch.maximum(out_max, sample)
    return out_min, out_max


class ImageHZSoundnessTests(unittest.TestCase):

    # ── Test 1 — Conv2D bounds ───────────────────────────────────

    def test_conv2d_bounds_cover_dense_reference(self):
        """The flatten of ImageHZ-after-Conv must have bounds that
        cover the dense reference HZ-after-Conv bounds."""
        _seed(1)
        C_in, H, W = 3, 4, 4
        C_out, kH, kW = 8, 3, 3
        ng = 4

        c = torch.randn(C_in, H, W, dtype=torch.float64)
        # Build 4 full-image generators (so we have a directly
        # comparable dense HZ).
        gens = []
        for i in range(ng):
            vals = torch.randn(C_in, H, W, dtype=torch.float64) * 0.1
            gens.append(_full_image_generator(C_in, H, W, vals, xi_id=i))
        hz = ImageHZ(c=c, generators=gens)

        weight = torch.randn(C_out, C_in, kH, kW, dtype=torch.float64) * 0.1
        bias = torch.randn(C_out, dtype=torch.float64) * 0.1

        # ImageHZ path.
        hz_out = apply_conv2d(hz, weight, bias, stride=1, padding=1)
        lb_img, ub_img = hz_out.bounds()

        # Reference dense path: flatten center+each generator, apply
        # conv to each, compute box bounds.
        c_full = c
        # Reference compute: for each xi sample at ±1 per generator,
        # we can compute the dense bounds analytically:
        # lb_ref = conv(c) - sum_i |conv(g_i)| (per output position)
        # ub_ref = conv(c) + sum_i |conv(g_i)|
        c_conv = torch.nn.functional.conv2d(
            c.unsqueeze(0), weight, bias=bias,
            stride=1, padding=1,
        ).squeeze(0)
        rad_ref = torch.zeros_like(c_conv)
        for g in gens:
            g_full = torch.zeros_like(c)
            r = g.region
            g_full[r.c_lo:r.c_hi, r.h_lo:r.h_hi, r.w_lo:r.w_hi] = g.values
            g_conv = torch.nn.functional.conv2d(
                g_full.unsqueeze(0), weight, bias=None,
                stride=1, padding=1,
            ).squeeze(0)
            rad_ref += g_conv.abs()
        lb_ref = c_conv - rad_ref
        ub_ref = c_conv + rad_ref

        # ImageHZ bounds must cover the reference (lb_img ≤ lb_ref,
        # ub_img ≥ ub_ref).
        tol = 1e-9
        self.assertTrue(
            (lb_img <= lb_ref + tol).all(),
            f"lb violation: max(lb_img - lb_ref) = "
            f"{(lb_img - lb_ref).max().item()}",
        )
        self.assertTrue(
            (ub_img >= ub_ref - tol).all(),
            f"ub violation: max(ub_ref - ub_img) = "
            f"{(ub_ref - ub_img).max().item()}",
        )

    # ── Test 2 — ReLU triangle bounds cover sampled exact ─────

    def test_relu_triangle_bounds_cover_sampled_exact(self):
        """ReLU triangle output bounds must cover any sampled exact
        point's ReLU output."""
        _seed(2)
        C, H, W = 2, 3, 3
        ng = 3
        c = torch.randn(C, H, W, dtype=torch.float64) * 0.3
        gens = []
        for i in range(ng):
            vals = torch.randn(C, H, W, dtype=torch.float64) * 0.2
            gens.append(_full_image_generator(C, H, W, vals, xi_id=i))
        hz = ImageHZ(c=c, generators=gens)

        # Apply triangle (root_ng = ng, so next aux id = ng).
        hz_relu, _ = apply_relu_triangle(hz, next_aux_id=ng)
        lb_out, ub_out = hz_relu.bounds()

        # Sample exact ReLU outputs over the input box.
        n_samples = 4000
        xi_in = (torch.rand(n_samples, ng, dtype=torch.float64) * 2.0 - 1.0)
        # Add the corners.
        for sign_mask in range(1 << ng):
            corner = torch.tensor(
                [1.0 if (sign_mask >> b) & 1 else -1.0
                 for b in range(ng)],
                dtype=torch.float64,
            ).unsqueeze(0)
            xi_in = torch.cat([xi_in, corner], dim=0)

        # For each sample, compute the actual pre-activation and
        # ReLU output; verify it lies inside [lb_out, ub_out].
        violations = 0
        max_violation_lb = 0.0
        max_violation_ub = 0.0
        for s_idx in range(xi_in.shape[0]):
            pre = c.clone()
            for g_idx, g in enumerate(gens):
                pre += xi_in[s_idx, g_idx] * g.values
            relu = torch.clamp(pre, min=0.0)
            v_lb = (lb_out - relu).max().item()
            v_ub = (relu - ub_out).max().item()
            if v_lb > 1e-9 or v_ub > 1e-9:
                violations += 1
                max_violation_lb = max(max_violation_lb, v_lb)
                max_violation_ub = max(max_violation_ub, v_ub)
        self.assertEqual(
            violations, 0,
            f"{violations} samples outside ReLU bounds; "
            f"max lb violation={max_violation_lb}, "
            f"max ub violation={max_violation_ub}",
        )

    # ── Test 3 — ADD merges by xi_id ─────────────────────────────

    def test_add_merges_same_xi_unions_different_xi(self):
        """Same xi_id → single combined generator; different
        xi_ids → preserved as separate generators."""
        C, H, W = 1, 2, 2
        ng_a = 2  # xi_ids 0, 1
        ng_b = 2  # xi_ids 1, 2 — id 1 should merge

        c_a = torch.zeros(C, H, W, dtype=torch.float64)
        c_b = torch.zeros(C, H, W, dtype=torch.float64)
        # hz_a generators: xi_id 0 (region top half), xi_id 1 (region bottom half)
        g_a0 = SpatialGenerator(
            region=BoundingBox(0, 1, 0, 1, 0, 2),
            values=torch.tensor([[[1.0, 0.5]]], dtype=torch.float64),
            xi_id=0,
        )
        g_a1 = SpatialGenerator(
            region=BoundingBox(0, 1, 1, 2, 0, 2),
            values=torch.tensor([[[0.3, -0.7]]], dtype=torch.float64),
            xi_id=1,
        )
        # hz_b: xi_id 1 (shifted region — region union test), xi_id 2
        g_b1 = SpatialGenerator(
            region=BoundingBox(0, 1, 0, 1, 0, 2),
            values=torch.tensor([[[2.0, 0.0]]], dtype=torch.float64),
            xi_id=1,
        )
        g_b2 = SpatialGenerator(
            region=BoundingBox(0, 1, 0, 2, 0, 2),
            values=torch.tensor([
                [[0.1, 0.2], [0.3, 0.4]],
            ], dtype=torch.float64),
            xi_id=2,
        )
        hz_a = ImageHZ(c=c_a, generators=[g_a0, g_a1])
        hz_b = ImageHZ(c=c_b, generators=[g_b1, g_b2])

        hz_sum = apply_add(hz_a, hz_b)

        # Expect 3 unique xi_ids in output: 0, 1, 2.
        out_ids = sorted([g.xi_id for g in hz_sum.generators])
        self.assertEqual(out_ids, [0, 1, 2])

        # xi_id 1's region should be the union of (h: 1..2, w: 0..2)
        # and (h: 0..1, w: 0..2) = (h: 0..2, w: 0..2).
        g_one = next(g for g in hz_sum.generators if g.xi_id == 1)
        self.assertEqual(
            (g_one.region.h_lo, g_one.region.h_hi), (0, 2)
        )
        self.assertEqual(
            (g_one.region.w_lo, g_one.region.w_hi), (0, 2)
        )
        # values at union: top row = g_b1's [2.0, 0.0]; bottom row =
        # g_a1's [0.3, -0.7]; no overlap in original h-coords, so no
        # element-wise sum (besides the implicit zeros).
        expected = torch.tensor([
            [[2.0, 0.0], [0.3, -0.7]],
        ], dtype=torch.float64)
        self.assertTrue(
            torch.allclose(g_one.values, expected, atol=1e-12),
            f"xi_id=1 merged values wrong:\n got {g_one.values}\n "
            f"expected {expected}",
        )

    def test_add_merges_same_xi_overlapping_regions_sums(self):
        """When two same-xi generators have OVERLAPPING regions,
        the merged values must be element-wise summed in the
        overlap."""
        C, H, W = 1, 2, 2
        c_z = torch.zeros(C, H, W, dtype=torch.float64)
        g_a = SpatialGenerator(
            region=BoundingBox(0, 1, 0, 1, 0, 2),
            values=torch.tensor([[[1.0, 1.0]]], dtype=torch.float64),
            xi_id=0,
        )
        g_b = SpatialGenerator(
            region=BoundingBox(0, 1, 0, 1, 0, 2),
            values=torch.tensor([[[2.0, -3.0]]], dtype=torch.float64),
            xi_id=0,
        )
        hz_a = ImageHZ(c=c_z, generators=[g_a])
        hz_b = ImageHZ(c=c_z, generators=[g_b])
        hz_sum = apply_add(hz_a, hz_b)
        # One generator with summed values.
        self.assertEqual(len(hz_sum.generators), 1)
        merged = hz_sum.generators[0]
        self.assertEqual(merged.xi_id, 0)
        self.assertTrue(torch.allclose(
            merged.values,
            torch.tensor([[[3.0, -2.0]]], dtype=torch.float64),
            atol=1e-12,
        ))

    # ── Test 4 — Flatten bridge soundness ───────────────────────

    def test_flatten_bridge_sparsegcz_bounds_match_imagehz(self):
        """Post-bridge SparseGcZ bounds must be NOT narrower than
        ImageHZ bounds at the bridge point. (Equal by construction
        when each xi_id appears at most once per generator.)"""
        _seed(4)
        C, H, W = 2, 3, 3
        ng = 4
        c = torch.randn(C, H, W, dtype=torch.float64)
        gens = []
        for i in range(ng):
            vals = torch.randn(C, H, W, dtype=torch.float64) * 0.3
            gens.append(_full_image_generator(C, H, W, vals, xi_id=i))
        hz = ImageHZ(c=c, generators=gens)
        lb_img, ub_img = hz.bounds()

        sparse = flatten_to_sparsegcz(hz)
        lb_sp, ub_sp = sparse.bounds()
        lb_sp = lb_sp.reshape(C, H, W)
        ub_sp = ub_sp.reshape(C, H, W)

        # NOT narrower: lb_sp ≤ lb_img + tol AND ub_sp ≥ ub_img - tol
        tol = 1e-9
        self.assertTrue(
            (lb_sp <= lb_img + tol).all(),
            f"Bridge narrowed lb by max {(lb_sp - lb_img).max().item()}",
        )
        self.assertTrue(
            (ub_sp >= ub_img - tol).all(),
            f"Bridge narrowed ub by max {(ub_img - ub_sp).max().item()}",
        )

    # ── Bonus — MaxPool forbidden ───────────────────────────────

    def test_maxpool_raises_not_implemented(self):
        """Advisor rev-2 guard 1: MaxPool must fail-closed."""
        c = torch.zeros(2, 4, 4, dtype=torch.float64)
        hz = ImageHZ(c=c, generators=[])
        with self.assertRaises(NotImplementedError):
            apply_maxpool2d(hz, kernel_size=2)

    # ── Step 2.0 — guards on bridge merging ───────────────────

    def test_bridge_merges_duplicate_xi_id(self):
        """Advisor Step 2.0 guard 1: when the bridge sees two
        generators with the same xi_id, default mode merges them
        into a single SparseGcZ column. Without this, the LP would
        treat shared-xi columns as independent and produce a
        spurious loose bound."""
        C, H, W = 1, 1, 2
        c = torch.zeros(C, H, W, dtype=torch.float64)
        # Two same-xi generators with disjoint positions.
        g0 = SpatialGenerator(
            region=BoundingBox(0, 1, 0, 1, 0, 1),
            values=torch.tensor([[[1.0]]], dtype=torch.float64),
            xi_id=7,
        )
        g1 = SpatialGenerator(
            region=BoundingBox(0, 1, 0, 1, 1, 2),
            values=torch.tensor([[[1.0]]], dtype=torch.float64),
            xi_id=7,
        )
        hz = ImageHZ(c=c, generators=[g0, g1])
        # The ImageHZ bounds for a single xi treat both positions
        # together: at xi=+1, both positions get +1; at xi=-1, both
        # get -1.
        # → lb = [-1, -1], ub = [+1, +1]
        lb_img, ub_img = hz.bounds()
        self.assertTrue(torch.allclose(
            lb_img.reshape(-1),
            torch.tensor([-1.0, -1.0], dtype=torch.float64),
        ))

        # Bridge default (merging) should produce ONE sparse column.
        sparse = flatten_to_sparsegcz(hz)
        # ng must be 1, not 2.
        self.assertEqual(int(sparse.Gc_sparse.shape[1]), 1)

    def test_bridge_strict_mode_rejects_duplicate_xi_id(self):
        """Strict mode signals upstream merging failures."""
        C, H, W = 1, 1, 2
        c = torch.zeros(C, H, W, dtype=torch.float64)
        g0 = SpatialGenerator(
            region=BoundingBox(0, 1, 0, 1, 0, 1),
            values=torch.tensor([[[1.0]]], dtype=torch.float64),
            xi_id=7,
        )
        g1 = SpatialGenerator(
            region=BoundingBox(0, 1, 0, 1, 1, 2),
            values=torch.tensor([[[1.0]]], dtype=torch.float64),
            xi_id=7,
        )
        hz = ImageHZ(c=c, generators=[g0, g1])
        with self.assertRaises(ValueError):
            flatten_to_sparsegcz(hz, strict_unique_xi_id=True)

    def test_stats_schema(self):
        """Advisor Step 2.0 guard 2: per-layer telemetry schema is
        stable and includes the required fields."""
        c = torch.zeros(3, 4, 5, dtype=torch.float64)
        g0 = SpatialGenerator(
            region=BoundingBox(0, 3, 0, 4, 0, 5),
            values=torch.zeros(3, 4, 5, dtype=torch.float64) + 0.1,
            xi_id=0,
        )
        g1 = SpatialGenerator(
            region=BoundingBox(0, 1, 0, 2, 0, 3),
            values=torch.zeros(1, 2, 3, dtype=torch.float64) + 0.2,
            xi_id=1,
        )
        hz = ImageHZ(c=c, generators=[g0, g1])
        s = hz.stats()
        # All required keys present.
        required = {
            "num_generators", "num_unique_xi_id", "total_nonzeros",
            "max_region_numel", "avg_region_numel",
            "dim_C", "dim_H", "dim_W",
        }
        self.assertEqual(set(s.keys()), required)
        self.assertEqual(s["num_generators"], 2)
        self.assertEqual(s["num_unique_xi_id"], 2)
        # Region sizes: g0 = 3*4*5 = 60, g1 = 1*2*3 = 6
        self.assertEqual(s["max_region_numel"], 60)
        self.assertEqual(s["total_nonzeros"], 60 + 6)
        self.assertEqual(s["dim_C"], 3)
        self.assertEqual(s["dim_H"], 4)
        self.assertEqual(s["dim_W"], 5)


def _run() -> int:
    suite = unittest.TestLoader().loadTestsFromTestCase(
        ImageHZSoundnessTests
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    n_pass = result.testsRun - len(result.failures) - len(result.errors)
    print(f"\nResult: {n_pass}/{result.testsRun} passed")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(_run())
