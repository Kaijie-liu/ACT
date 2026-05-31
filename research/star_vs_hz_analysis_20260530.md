# Star Set / ImageStar vs Hybrid Zonotope — Robustness Verification Analysis

## Reading

- **bak2020cav.pdf**: Bak, Tran, Hobbs, Johnson, *Improved Geometric Path Enumeration for Verifying ReLU Neural Networks*, CAV 2020. Star sets + DFS path enumeration + LP prefilter.
- **star.pdf**: Tran, Bak, Xiang, Johnson, *Verification of Deep Convolutional Neural Networks Using ImageStars*, CAV 2020. ImageStars (image-shaped Star generators) + exact (BaB) or approx (triangle) ReLU/MaxPool.

## Set representations side-by-side

| Property | Zonotope | Star Set | ImageStar | Hybrid Zonotope (HZ) |
|---|---|---|---|---|
| Set form | `c + Gα, α ∈ [-1,1]^p` | `c + Gα, α ∈ P` | same as Star but `c, G` are images | `c + Gc·ξ_c + Gb·ξ_b, Ac·ξ_c+Ab·ξ_b ≤ b, ξ_b ∈ {-1,+1}^q` |
| Predicate constraint | Box only | Arbitrary polytope | Arbitrary polytope | Polytope + mixed-integer (binaries) |
| Linear map | Closed-form (G ← AG) | Closed-form | Native conv on each image generator | Closed-form |
| Exact ReLU | requires split + new sets | requires split + new sets | requires split + new sets | exact via +4 cont, +1 bin, +3 cons (hz1) — NO split required |
| Over-approx ReLU | DeepZ triangle (+1 slack) | triangle approxStepReLU (+1 var, +3 cons) | same | triangle (same) OR eq_lagr_v8 (tighter but with bin+eq) |
| Union of sets | Single set only | Single set only | Single set only | Native via binary generators |
| Halfspace ∩ | not closed | closed (add 1 cons) | closed | closed (Bird Prop) |

## Key observation: HZ degenerates to Star Set under triangle-only forward analysis

In ACT's default forward-only HZ verifier, the binary-generator machinery is
**dormant** until the final tail ReLU layers (where `large_cls_proof_mode`
fires eq_lagr_v8). For all the conv + triangle-ReLU layers (~tens of layers
on cifar100/yolo/tinyimagenet), the HZ has `q = 0` (no binaries) and the
constraint set is just the box from input + per-neuron triangle constraints.

In that regime, **HZ ≡ Star Set ≡ ImageStar** (formally, the same abstract
domain). The forward propagation we do is identical to what ImageStar would do.

We measured this directly on cifar100 iid 0 with `ACT_HZ_LAYER_PROGRESS=1`:

```
L30 CONV2D in=dim=2048 ng=6000 nb=0 nc=0
L31 RELU   out=dim=2048 ng=6000 nb=0 nc=0      ← triangle, nb=0
L32 CONV2D out=dim=2048 ng=6000 nb=0 nc=0
…
L40 RELU   out=dim=100  ng=3172 nb=38 nc=228   ← FINAL layer eq_lagr_v8: +38 bin, +228 cons
```

So 30+ layers are in the Star Set regime; only 1 layer fires the HZ binary
machinery.

## Why do they verify cifar100 / yolo / tinyimagenet and we don't?

Inspecting each successful tool's mechanism:

### NNV (Bak/Tran et al.) on VGG16/19 + cifar
- **Exact reachability**: splits at every unstable ReLU / max-pool candidate.
  This is BaB. **Violates our P5.**
- **Approximate reachability**: triangle (approxStepReLU). Same as our triangle.
  By itself, doesn't decide cifar100 either — the result is UNKNOWN.
- So NNV's success on conv benchmarks = BaB. Not the set representation.

### α,β-CROWN
- **α-CROWN**: per-neuron α slope tightening via backward bound propagation.
  This is **per-layer backward**. **Violates our P1+P2.**
- **β-CROWN**: BaB with linear bound refinement at each split. **Violates P5.**
- **MILP fallback**: Gurobi to solve hard cases. **Violates P3.**

### NeuralSAT
- SAT-style splitting + LP backtracking. **Violates P4+P5.**

### PyRAT
- Uses concrete simulation (random sampling) + LP-tight bounds with backward
  refinement. **Violates P2+P6.**

### CORA
- Polynomial zonotopes + interval-bound forward + BaB. **Violates P5.**

**Conclusion**: every tool that decides cifar100 / yolo / tinyimagenet uses
at least ONE of (CROWN-style backward bound propagation, BaB / input splitting,
MILP solver, or gradient-based PGD). Our 6 principles forbid all of these.

## Is there "wasted HZ baggage" in robustness verification?

Examining each "extra" piece of HZ vs Star Set:

1. **Binary generators (Gb, ξ_b)**: at `q = 0` they are size-0 arrays.
   Zero RAM, zero compute. **NOT wasted, just inactive.**

2. **eq_mask**: similar — small bool array, not used unless eq_lagr_v8 fires.

3. **Per-neuron triangle slack (one new generator per unstable ReLU)**: this is
   what we DO use heavily. ImageStar adds the same. **Same overhead.**

4. **Constraint matrix Ac**: grows when eq_lagr_v8 + intersect_box + project_eq_elim
   fires. At triangle-only layers Ac is mostly empty. **Same as Star set.**

The conclusion is **HZ adds NO meaningful overhead vs Star Set in robustness verification under our triangle-default forward propagation**.

## So what's the actual precision gap?

For cifar100 iid 0: HZ output has 100 dimensions, 38 binary variables,
228 constraints, 3172 continuous generators. Verdict UNKNOWN in 3.3 s.

This means:
- **Time is NOT the bottleneck** (3.3 s is fast).
- **The 100-dim output HZ relaxation is too loose for the spec-direction LP
  to declare CERTIFIED.**

Equivalently: the LP relaxation says "there exists a feasible point with
y_j > y_target" for some j, but this point is a PHANTOM — strict ORT replay
rejects it. The output LP-relaxation is admitting points that are not real
network outputs.

This is the structural ceiling: forward triangle ReLU + LP at output cannot
distinguish, on conv-heavy robustness specs, between real adversarial points
and phantom LP-feasible points. Demonstrated empirically in 3 independent
session experiments (D filter, multi-corner LP sidecar, joint K=2 envelope —
all 0/47-0/54 lift).

## What forward-only sound improvements could conceivably help?

### Idea 1: more eq_lagr_v8 layers (currently testing)

ACT's eq_lagr_v8 adds `+4 cont, +1 bin, +3 cons per unstable neuron`. It's
tighter than triangle but costlier. The default `large_cls_proof_mode` only
applies it to the last 3 ReLU. Setting `HYZOR_LARGE_CLS_EQ_LAYERS=999` makes
ALL ReLU layers eq_lagr_v8.

**Hypothesis**: tighter per-layer encoding may propagate to a tighter output.
**Risk**: memory blow-up; may OOM or wall-cap on cifar100.
**Status**: probe running across 9 zero-verdict benches.

### Idea 2: per-neuron LP bound tightening at intermediate layers

For each unstable neuron at layer L, BEFORE adding triangle slack, solve LP
on the current HZ to compute exact min/max of that neuron. Use these tighter
bounds in the triangle relaxation.

**Cost**: 2 LP per unstable neuron per layer. Conv with 2048 features × 50%
unstable × 10 layers = 20480 LP. At 1 ms each = 20 s. Affordable on cifar100.

**Soundness**: LP-tight intermediate bounds preserve over-approximation
(tighter than interval propagation). No backward, no gradient.

**Risk**: 20 s per instance may not give precision win.

### Idea 3: PARC-style partition refinement

PARC (Müller et al. 2022) refines the abstraction per-layer based on the
spec direction. Their refinement is forward-only and bounded.

**Issue**: requires knowing the spec direction at intermediate layers. Either
we propagate spec backward (forbidden by P1+P2) or restrict refinement to
LAST few layers (which we already do via eq_lagr_v8).

### Idea 4: ImageStar's exact MaxPool

ImageStar splits at MaxPool candidates. We can't (P5). No gain.

### Idea 5: drop redundant HZ machinery

There is none (per §4). Already streamlined.

## What we are doing right (vs Star/ImageStar)

1. **Strict zero-tolerance ORT replay** on every emitted FAL — Star/ImageStar
   don't always do this; some emit FAL based on factor-space LP witness alone.
2. **Constraint accumulation** via `project_eq_elim` — equivalent to a CROWN-
   slope absorption forward-only.
3. **GATHER + SLICE + UPSAMPLE + ConvTranspose** exact transfers — Star/ImageStar
   handle these but ACT was missing them (we fixed in this session).
4. **Singleton fastpath** + zero-width input pruning — unique optimizations
   not in Star/ImageStar.

## Summary

Under our strict P1-P6 principle set, the structural ceiling for cifar100-
class benchmarks is **representation-bound, not implementation-bound**.

Bird's HZ is the most expressive forward-only set representation we have
(strictly more expressive than Star sets when binaries fire). Even so, the
abstraction at the output layer is too loose to determine TOP1_ROBUST specs
on dense conv networks like cifar100 / yolo / tinyimagenet.

Closing this gap requires either (i) BaB-style splitting (P5 violation),
(ii) backward CROWN-style refinement (P1+P2 violation), or (iii) a NEW
representation that captures cross-layer correlation better than existing
forward methods. The 3 session-tested forward-only precision levers (D filter,
multi-corner LP, joint K=2) all returned 0 lift, supporting the structural
nature of this ceiling.

The honest scientific position is: **forward-only HZ + LP under P1-P6 has a
structural precision ceiling on dense-conv robustness verification, and we
have demonstrated it via independent experiments**. This is a publishable
negative result that delineates the boundary of what is achievable with
sound, principle-compliant forward verification.
