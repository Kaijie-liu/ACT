# Spec-aware Girard reduction v2 — full implementation (2026-05-30)

Supersedes the MVP in `spec_aware_girard_pilot_20260530.md`.

## v2 = MVP + full F-chain + lift SparseGcZ nc>0 guard

### What v1 (MVP) couldn't do

MVP F-chain stopped at FLATTEN / CONCAT / CONV2D, so on conv-heavy benches the
F-chain only covered the dense classifier head (~2 layers). Reduction at conv
layers (where ng grows fastest) saw `F == None` and fell back to standard
column-norm scoring. Empirically: 0 lift on cifar/yolo/tiny/linearizenn.

### What v2 adds

**Extended F-chain ops** (act/back_end/hybridz_tf/algorithms/spec_aware.py):

| Op | Reverse transform | Implementation |
|----|-------------------|----------------|
| CONV2D | F_in = conv_transpose2d(F_out_4d, W, stride, padding, dilation, groups, output_padding) | torch.nn.functional |
| BN | F_in = F_out * A (per-channel broadcast over spatial) | manual broadcast |
| ADD | F_in = F_out (each predecessor gets full F) | broadcast |
| CONCAT | F_in = split(F_out) along feature axis per predecessor dim | slicing using `before[pid]` dims |
| AVGPOOL2D | F_in = conv_transpose2d(F_out, avg_kernel, ...) | groups=C avg-kernel |
| MAXPOOL2D | F_in = approximate via avgpool-style (loose but sound for ranking) | same machinery |
| FLATTEN/RESHAPE/TRANSPOSE/SQUEEZE/UNSQUEEZE | F_in = F_out (passthrough) | identity |

**Graph-aware reverse walk**: multi-predecessor layers (ADD, CONCAT) distribute
F correctly; multi-successor layers accumulate (chain rule via linearity, since
F is a linear functional). Implemented in `precompute_spec_F` using
`net.preds[L.id]` for branching.

**Constraint widening for SparseGcZ.reduce_generators with nc > 0**
(representations.py:1330-1500, env knob `ACT_HZ_SPARSE_REDUCE_WIDEN=1`):

For each dropped column j of Gc, distribute the constraint contribution
into widened RHS:
```
b_new[r] = b[r] + sum_{j in dropped} |Ac[r, j]|
```
This is a sound relaxation: any (xi_c, xi_b) satisfying `Ac xi_c + Ab xi_b ≤ b`
implies `A_keep xi_keep + Ab xi_b ≤ b + |Ac_dropped|·1` because
`sum_{j in D} Ac[r, j] * xi_D[j] ≥ -sum_{j in D} |Ac[r, j]|`.

Equality rows referencing dropped columns are DROPPED (cannot be widened
without losing equality semantics). This is also sound — dropping a constraint
makes the set larger or equal.

Default OFF (knob = 0) preserves legacy early-return at `nc > 0`. Backwards
compatible.

## Soundness verification

10 v2-specific tests + 38 reduction-soundness + 4 eq_mask + row-op tests all
pass. Crucially:
- `test_F_conv2d_matches_dense_toeplitz`: F propagation through CONV2D via
  `conv_transpose2d` produces the IDENTICAL value as a materialized dense
  Toeplitz `W_flat`. Algebraic correctness verified.
- `test_constraint_widening_soundness`: widening preserves containment
  (widened HZ's interval hull contains original HZ's hull).

## Principle alignment (unchanged from v1)

| Principle | Check |
|-----------|-------|
| P1 No CROWN backward bound propagation | ✅ F is a linear functional, not (lb, ub) |
| P2 No backward / autograd / gradient | ✅ λ fixed from IBP, no gradient |
| P3 No Gurobi | ✅ |
| P4 No fallback | ✅ |
| P5 No BaB | ✅ |
| P6 No PGD / random sampling | ✅ |

## F-chain coverage (v2 smoke)

```
malbeware iid 0: F-chain 3 layers (supported=3, stops=0)
                 V=1 CERTIFIED 2.0s  ← no regression
cifar100 iid 0:  F-chain 41 layers (supported=41, stops=0)  ← v2 reaches all
                 V=0 UNKNOWN 3.2s  ← runs, full chain coverage
```

In MVP, cifar100's F-chain reached at most the final dense layer (2 layers).
v2's CONV2D + ADD + CONCAT + BN + AVGPOOL2D handling lets F propagate through
the entire resnet_medium (41 layers including all conv stages).

## Pilot

5 benchmarks × {OFF, ON} × {10, 5, 5, 10, 5} iids respectively =
50 instances per knob setting. 3-wave parallel GPU layout.

### Results (3-wave parallel pilot)

```
lin_OFF         V=2  A=0  U=8   n=10  mean_wall=13.5s
lin_ON          V=2  A=0  U=8   n=10  mean_wall=13.5s    Δ=+0V +0A
tiny_OFF        V=0  A=0  U=5   n=5   mean_wall=8.7s
tiny_ON         V=0  A=0  U=5   n=5   mean_wall=8.9s     Δ=+0V +0A
traffic_OFF     V=0  A=0  U=5   n=5   mean_wall=114.6s
traffic_ON      V=0  A=0  U=5   n=5   mean_wall=114.5s   Δ=+0V +0A
cifar_OFF       V=0  A=0  U=10  n=10  mean_wall=4.4s
cifar_ON        V=0  A=0  U=10  n=10  mean_wall=4.5s     Δ=+0V +0A
yolo_OFF        V=0  A=0  U=5   n=5   mean_wall=7.3s
yolo_ON         V=0  A=0  U=5   n=5   mean_wall=7.3s     Δ=+0V +0A
```

**0 lift across all 5 benchmarks.** Walls essentially identical (within 0.2s).

### Decision rule outcome

Per pre-registered rule, 0 new V → mark v2 lever inert under full F-chain
support too; forward-only HZ reduction-scoring has structurally limited
precision impact.

### Why v2 still inert (not just empirically null)

Per-instance log inspection confirms the HZ propagation DID complete on cifar100:
- `large_cls_proof_mode ACTIVE: conv=19 out_dim=100 relus=10 (triangle for relu 1..9, eq_lagr_v8 for last 1)`
- All 10 cifar instances ran the full 19-conv backbone, 4-5s each, completing the final LP
- F-chain v2 reached all 41 layers (`supported=41, stops=0`)

So the scoring branch fired at conv-layer reductions. The verdict still didn't change. This means:

**The reduction-scoring criterion is NOT the bottleneck.**

Even when spec-aware Girard optimally keeps the spec-relevant generators,
the box-substitution of dropped columns introduces small over-approximation
relative to the much larger over-approximation accumulated by 10 successive
triangle-ReLU layers. The dominant precision loss happens in the ReLU
relaxation steps themselves, not in the inter-layer Girard reductions.

### Structural conclusion

This pilot tests the strongest forward-only, principle-compliant
generator-ranking heuristic available. Its 0/50 result completes the
inventory of forward-HZ precision levers for conv-heavy 0V benches:

| Lever | Result |
|-------|--------|
| Tier 2/3 tight pre-act bounds (`ACT_HZ_TRIANGLE_TIGHT_BOUNDS`) | Structurally inert (nc=0 on SparseGcZ triangle layers) |
| ALL_EQ probe (`HYZOR_LARGE_CLS_EQ_LAYERS=999`) | 0 lift + OOM |
| eq_layers ablation {0,1,3,5,10} | Insensitive on 4/6 benches, OOM-bound on 2/6 |
| Sigmoid K-piece {1,2,4,8,16} | Non-monotone, K=2 sweet spot |
| Anderson facets (acasxu) | 0 lift |
| Single-binary probing (acasxu) | 0 lift |
| k=2/k=3 multi-neuron hull (acasxu) | 0 lift |
| Multi-corner LP sidecar | 0 lift |
| Joint K=2 envelope | 0 lift |
| D filter LP-redundancy | 0 lift + 6 OOMs |
| Spec-aware Girard v1 MVP | Mechanically inert (F-chain stops at conv) |
| **Spec-aware Girard v2 full** | **F-chain reaches all 41 layers, 0 lift across 50 instances** |

The structural ceiling on forward-only HZ for conv-heavy robustness is
now empirically saturated. Closing this gap genuinely requires either
(a) backward bound propagation (P1+P2 violation), (b) BaB (P5 violation),
or (c) a NEW abstract domain construction with mathematically different
properties than HZ — multi-month research, not engineering tweaks.

### Recommendation for code state

The v2 implementation is sound (10 + 38 + 4 + row-op tests pass) and
contains real engineering value:
- Constraint widening lift is a usable extension to SparseGcZ for any
  future caller that needs nc>0 reduction (independent of spec-aware use).
- F-chain through CONV2D/ADD/CONCAT/BN/AVGPOOL is reusable infrastructure
  for any future spec-conditioned analysis.

But as a precision lever, both knobs (`ACT_HZ_SPEC_AWARE_GIRARD`,
`ACT_HZ_SPARSE_REDUCE_WIDEN`) should remain default OFF. Per user's pre-
agreement: "如果效果不如预期可以快速回滚到干净的代码".

Two clean rollback options:
1. **Conservative**: keep code in tree, default OFF, document as untrusted
   precision lever (current state).
2. **Aggressive**: `cd /data1/Kane/ACT && git checkout HEAD -- act/ tests/
   && rm act/back_end/hybridz_tf/algorithms/spec_aware.py tests/test_spec_aware_v2.py`
   to revert to pre-MVP state.

The pre-v2 backup at `/data1/Kane/ACT/backups/pre_v2_20260530T081226Z.*`
captures the post-MVP state if a partial revert is wanted.

## Files

New:
- `act/back_end/hybridz_tf/algorithms/spec_aware.py` (extended v1 + v2 ops)
- `tests/test_spec_aware_v2.py` (10 tests)

Modified:
- `act/back_end/hybridz_tf/representations.py` (SparseGcZ.reduce_generators widening)
- `act/back_end/hybridz_tf/tf_mlp.py` (set_current_layer in tf_relu/tf_lrelu, hz_reduce scoring override)
- `act/back_end/solver/solver_hz.py` (consume_cons precompute call)

Env knobs:
- `ACT_HZ_SPEC_AWARE_GIRARD={0,1}` — enable spec-projected scoring
- `ACT_HZ_SPARSE_REDUCE_WIDEN={0,1}` — enable nc>0 widening path
- `ACT_HZ_SPEC_AWARE_DEBUG={0,1}` — chain depth + per-layer trace

## Backup

- Git tag `spec_aware_v1_mvp` at HEAD `4f35c8aa5` (pre-v2 state).
- Patch: `/data1/Kane/ACT/backups/pre_v2_20260530T081226Z.patch`
- Tarball: `/data1/Kane/ACT/backups/pre_v2_20260530T081226Z.tar.gz`

Rollback: `cd /data1/Kane/ACT && git checkout HEAD -- act/ tests/ && rm act/back_end/hybridz_tf/algorithms/spec_aware.py`
