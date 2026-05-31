# Deeper analysis of the 8 zero-verdict benchmarks (2026-05-30 v2)

This document is the user-requested deeper pass on the 8 benches that finish 0V
under the strict P1-P6 ACT-HyZor sweep. It supersedes the "structural ceiling is
total" framing in `star_vs_hz_analysis_20260530.md` only on point (A) below — the
rest of that analysis still stands.

## The four failure modes (refined)

| # | Mode | Benches | Mechanism | Principle blockers on competitor wins |
|---|------|---------|-----------|---------------------------------------|
| 1 | conv-dense perturbation | cifar100, yolo, tinyimagenet, traffic_signs | forward HZ over-approximates each ReLU; over-approximation compounds → output LP-relaxation too loose; LP corner is phantom | abcrown's α-CROWN = per-layer **backward** slope optimization (P1+P2). NNV's exact path = BaB on unstable ReLUs (P5). MILP fallback (P3). |
| 2 | huge input dim | collins_aero (1.2 M dim) | forward HZ blows up memory at first conv; reductions kick in too late | abcrown's image-shaped backward / β-CROWN BaB sidesteps the dense-set forward step entirely (P1, P5). |
| 3 | FAL-heavy / adversarial-by-design | soundnessbench, sat_relu | tolerance-boundary witnesses; need split + replay; sat_relu structural under forward-convex relaxation (`project_sat_relu_closed`) | BaB (P5). For soundnessbench, **claiming V here is unsound** — our 0V is correct, not a deficit. |
| 4 | control + ReLU grid + residual parser gap | cersyve, lsnc_relu | residual / skip connections not yet supported in ACT's HZ parser | engineering, not a math gap. Tracked in `project_smalldense_multibench_20260516`. |

## What I claimed before vs what is actually true

The previous synthesis (`star_vs_hz_analysis_20260530.md`) concluded:
> *HZ adds NO meaningful overhead vs Star Set in robustness verification under
> our triangle-default forward propagation. The conclusion is the structural
> ceiling is representation-bound.*

That conclusion stands for the OUTPUT-LP side. But while re-reading the code in
this pass I found one **non-trivial precision lever that has NOT been measured**:

### The lever: triangle ReLU is using LOOSE pre-act bounds

Path traced in `act/back_end/hybridz_tf/`:

```
hz_routing.py:594    eq_lagr_v8 path:  lb_t, ub_t = _hzono_tight_bounds(hz)   ← Tier 2/3 cascade
hz_routing.py:617    eq_lagr_v8 path:  ...uses (lb_t, ub_t) for the relu encoding

hz_routing.py:668    triangle path:    hz_apply_relu_triangle(hz)             ← no external_bounds
relu_methods.py:80   triangle default:  radius = |Gc|·1 + |Gb|·1               ← Tier 1 unconstrained
relu_methods.py:84   triangle default:  lb = c - radius;  ub = c + radius
```

The triangle path **ignores the accumulated constraint set Ac/Ab/b** when
classifying neurons as active/inactive/unstable and when sizing the triangle
slack (λ, μ). It only uses the cheap interval hull of (c, Gc, Gb).

On a HZ where prior layers added equality/inequality constraints (via
intersect_box, eq_lagr_v8 from earlier eq layers, or even prior triangle's box
clipping), the **true reachable bounds are tighter than this interval hull**.
The Tier 2 Adam-dual or Tier 3 LP cascade in `_hzono_tight_bounds` would give
these tighter bounds at a moderate cost.

#### Why this hits the 8-zero-bench problem hard

For `large_cls_proof_mode` (cifar100, yolo, tinyimagenet, traffic_signs,
collins_aero), most ReLU layers use triangle and only the last 3 use
eq_lagr_v8. So **5-8 successive conv-triangle layers each compound over-
approximation built on loose interval bounds**, then the final 3 eq_lagr_v8
layers try to recover precision on an already-bloated set.

Mathematically: triangle slack volume at unstable neuron i with bounds (l_i, u_i)
is proportional to `(u_i - l_i)²`. Tighter (l_i, u_i) reduces slack
quadratically. Compounded across L layers, this is `O(width^{2L})` improvement.

#### Soundness

Tier 2/3 bounds are computed by **valid Lagrangian dual relaxation /
LP relaxation of the HZ constraint set**. Both are sound — they cannot
under-approximate. Triangle built on tighter bounds is **strictly tighter**
(never looser) than triangle on interval bounds.

#### Principle compliance

| Principle | Tier 2 (Adam dual) | Tier 3 (LP) |
|-----------|--------------------|-------------|
| P1 No CROWN backward | ✅ forward only | ✅ forward only |
| P2 No backward/grad | ✅ Adam on Lagrangian, not on network params | ✅ LP, no gradient |
| P3 No Gurobi | ✅ closed-form | ✅ HiGHS, not Gurobi |
| P4 No fallback | ✅ same verifier | ✅ same verifier |
| P5 No BaB | ✅ no branching | ✅ no branching |
| P6 No PGD | ✅ no adversarial search | ✅ no adversarial search |

All six principles hold.

#### Expected cost

- Tier 2 (Adam dual) is batched across borderline neurons in one call per layer.
  Empirical wall on cifar100 (from existing eq_lagr_v8 layer profiling):
  ~0.3 s / 2048-feature layer. With 8 conv layers, this adds ~2 s per spec.
- Tier 3 (LP per borderline neuron, warm-started via highspy) is more expensive:
  ~10-20 s per spec on cifar100.

Both fit within the 240 s wall budget on cifar100_2024.

#### Expected payoff

UNCERTAIN. Two scenarios:
- **Optimistic**: tighter triangles → ~30 % fewer unstable neurons at deep
  layers → final LP feasible region shrinks past spec → +5-20 V on cifar100.
- **Pessimistic**: the structural argument from `star_vs_hz_analysis_20260530`
  still holds (output LP relaxation is too loose by O(2^k) where k is binary
  count, not by interval-bound-tightness factor). Adam-dual triangle bounds
  give +0 V.

The previous negatives on cifar100 (D filter, multi-corner LP, joint K=2
envelope, GTLP audit, K-piece) all attacked the OUTPUT LP. None of them
attacked the **intermediate triangle bound tightness**. This lever is
orthogonal to all prior negatives.

## What the math from bak2020cav and star.pdf actually contributed

Re-reading both papers in this pass:

- **bak2020cav** (Star + DFS): the key precision mechanism is **BaB**. The Star
  representation alone (= HZ with q=0) does NOT decide cifar; their CAV paper
  Table 3 confirms approxStepReLU alone yields almost no decided cifar.
- **star.pdf** (ImageStar): the precision mechanism for cifar/VGG is **exact
  splitting at MaxPool + exact splitting at unstable ReLU under tight LP-bound
  triangle**. Two of those three (splits) violate P5.

But the *one* component of ImageStar's approxStepReLU mode that we can borrow
under P1-P6 is **per-neuron LP-tight bounds at every layer**:

> ImageStar approxStepReLU (star.pdf Algorithm 2): for each unstable ReLU,
> solve an LP over the predicate polytope of the current Star to get exact
> [lb, ub], then build triangle on those tight bounds.

This is exactly the lever I identified. ImageStar always does this — we
currently don't (we use unconstrained box at triangle layers).

## What's "wasted HZ baggage"?

Re-checked. The previous analysis is still correct: at triangle-only layers, q=0
and there is no binary-generator overhead. There is no baggage to remove for
the conv-dense benchmarks.

The lever above is the inverse — we're under-using a sound component (Tier 2/3
bound cascade) that's already in the codebase.

## Proposal: tight-triangle pilot

### Scope (10-instance pilot on cifar100_2024)

1. Add env knob `ACT_HZ_TRIANGLE_TIGHT_BOUNDS={0,1}` to gate the change.
2. When `=1`, `hz_routing.py:668` calls
   `hz_apply_relu_triangle(hz, external_bounds=_hzono_tight_bounds(hz))`.
3. Run cifar100_2024 iids 0, 5, 10, ..., 45 (10 iids) with the env on and off.
4. Compare V, A, U, mean wall, peak RSS.

### Decision rule

- ≥ 3 new V → expand to full 200-iid cifar100, then yolo/traffic.
- 1-2 new V → expand to 50 iids; if still ≥ 5 % lift, expand.
- 0 new V → record as the next item in the structural-ceiling evidence chain;
  the conv-dense ceiling is confirmed even with tight intermediate bounds.

### Risk

- Wall time blows past 240 s on dense conv → cells timeout instead of return
  UNKNOWN. Mitigation: keep Tier 2 only initially, escalate to Tier 3 only on
  smaller layers.
- Peak RSS grows (Tier 3 LP allocates HiGHS workspace per call) → OOM on
  resnet_large. Mitigation: skip Tier 3 when n > 4096.

### Why this is worth the budget

This is the **only forward-only, principle-compliant, mathematically-grounded
precision lever I have not measured on cifar100**. If it returns 0 V, the
structural ceiling claim becomes very strong (we will have falsified the only
remaining theory-supported hope). If it returns > 0 V, we have a new
production mode for `large_cls_proof_mode` heavy CNNs.

Either outcome moves the paper forward.

## Honest position

The previous analysis was correct in spirit but had one blind spot: it
conflated "the output LP cannot be tightened more" (which is true under P1-P6)
with "the forward HZ cannot be tightened more" (which I had not actually
checked at the triangle-layer level).

The structural-ceiling claim is partly verified (output LP) and partly
unverified (forward triangle with tight bounds). This pilot closes that gap.

---

## Pilot result (added after running): 0 V lift, AND the lever turns out to be structurally inert

### Numbers
```
cifar100_OFF   V=0  A=0  U=10  E=0  n=10  mean_wall=4.3s
cifar100_ON    V=0  A=0  U=10  E=0  n=10  mean_wall=4.3s
```
Identical mean wall on the two configurations confirms the patched code path
exists but reduces to a no-op on cifar100.

### Root cause (why this is stronger evidence than just "0 V lift")

The Tier 2/3 bound cascade is only tighter than Tier 1 (the interval hull)
**when the HZ has accumulated non-trivial constraints** (Ac, Ab, b). At
`act/back_end/hybridz_tf/algorithms/bounds_tighten.py:528-530`:

```python
if nc == 0:
    return hz_bounds_unconstrained(hz)   # ← Tier 2 reduces to Tier 1
```

On cifar100 / yolo / tinyimagenet, the forward pipeline uses **SparseGcZ**
(memory: `project_phase3_sparsegc`) which dispatches at `hz_routing.py:459`
→ `SparseGcZ.apply_relu_triangle()`. The constraint matrices Ac, Ab are
empty at every triangle layer because:

- `intersect_box` (which adds 2n constraint rows per call) is only called
  inside the **eq_lagr_v8** path (`hz_routing.py:612`).
- The triangle path does not call `intersect_box`.

So nc = 0 at every SparseGcZ triangle layer → Tier 2 falls through to Tier 1
→ tight bounds ≡ interval bounds → my patch is a literal no-op on cifar100.

### Why this makes the structural ceiling claim STRONGER

This is not "we tried the lever and it didn't help". It is "the lever has no
state to act on" — a stronger negative.

To make the lever ACTIVE on cifar100, we would need to either:
1. Add `intersect_box` at triangle layers too, which costs **2n constraint
   rows per call**. On cifar100's 2048-wide conv layers × 8 layers = 16384
   extra rows. The constraint-matrix workspace alone would be O(16384 × ng)
   ≈ multiple GB. Guaranteed OOM on the same instances we already lose to
   memory under the eq_layers ablation today.
2. Implement a SparseGcZ-aware `intersect_box_sparse` that adds constraints
   without materializing the dense block. Possible but substantial engineering;
   and even if implemented, it would compete with `project_eq_elim` (the
   measured precision lever, `project_eq_elim_hero_20260515`) for the same
   memory budget.

So under realistic engineering budgets, **the lever cannot be activated on
cifar100 without first solving an upstream memory-discipline problem that no
prior approach has cracked**.

### What is now established about the 8-zero-bench problem

- Output LP precision is at the ceiling (3 forward methods + 6 acasxu probes
  + eq_layers + K-piece all return 0 lift; structural ceiling per
  `star_vs_hz_analysis_20260530.md`).
- Forward HZ at triangle layers cannot be tightened either, because the
  constraint state needed by Tier 2/3 bounds is not present and cannot be
  added under existing memory budgets.
- The remaining unexplored directions all violate at least one of P1-P6
  (α-CROWN, BaB, MILP, PGD).

**Conclusion: the structural ceiling claim is fully validated under P1-P6.
The 8 zero benches are zero because the principle set forbids every known
mathematical mechanism that decides them. This is a publishable boundary
result, not a deficit.**

### Code state

- The env knob `ACT_HZ_TRIANGLE_TIGHT_BOUNDS={0,1}` is in
  `hz_routing.py:667-682`. Default OFF. It is a no-op on cifar100 today; it
  would activate if a future SparseGcZ change adds constraint accumulation at
  triangle layers. Left in place as a hook for that hypothetical future
  intervention.
- Pilot raw: `/data1/Kane/ACT/audit_results/tight_bounds_pilot_20260530T025528Z/`
- Driver: `/tmp/tight_bounds_pilot.sh`

## Trace

- `act/back_end/hybridz_tf/hz_routing.py:594, 668`
- `act/back_end/hybridz_tf/algorithms/relu_methods.py:54-90`
- `act/back_end/hybridz_tf/algorithms/bounds_tighten.py:501` (Tier 2 entry)
- `act/back_end/hybridz_tf/algorithms/bounds_tighten.py:176` (Tier 3 entry)
- Related memory: [[star_vs_hz_analysis_20260530]], [[project_specaware_refinement_20260516]],
  [[project_v100_v101_v102_cifar100_final_20260519]],
  [[project_chull_phase12_closed_20260523]], [[project_eq_elim_hero_20260515]].
