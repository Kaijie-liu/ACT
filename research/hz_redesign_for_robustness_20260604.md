# HZ Redesign for Robustness Verification — Spec-Conditioned HZ Toward Competitive Totals

**Date**: 2026-06-04 night
**Goal**: identify the structural HZ changes that could plausibly move ACT
from 924 V/A toward tool-competitive totals, **while staying forward-only**.
**Status**: design analysis — NOT implementation. This document supersedes
the earlier "direct 2000+" framing with a staged, auditable roadmap.

This document builds on (not replaces) the existing memos:
- `star_vs_hz_analysis_20260530.md` — HZ ≡ Star under triangle-only forward; binaries dormant on conv layers.
- `hz_zero_benches_deeper_analysis_20260530.md` — 4 failure modes already classified.
- `dense_conv_forward_hz_plan_20260531.md` — conv-dense compounding as the main per-layer gap.
- `spec_aware_girard_v2_20260530.md` — F-chain reverse Girard tried; limited conv lift.
- `tool_comparison_20260604.md` — current cross-tool numbers.

The reader is assumed to have those open.

---

## 0. 2026-06-04 review update — do not overclaim the 2000+ path

The earlier draft treated `2000+ V/A` as a plausible endpoint of one
large redesign. That is too optimistic for a design memo. The right
position is:

1. **924 V/A is the current verified ACT row.**
2. **2000+ is a long-horizon target**, not an expected outcome of a
   first prototype.
3. The next real research mechanism is **Spec-Conditioned HZ (SC-HZ)**:
   a query-local forward HZ abstraction that spends generator budget
   on the rival/spec directions that matter.
4. The first gate is not "reach 2000"; the first gate is:

```text
Phase A target: prove the SC-HZ signal exists.
Pass if 80 sentinels show >=5 new V/A OR median LP UB reduction >=25%.
Fail if 0 new V/A AND median LP UB reduction <10%.
```

If Phase A passes, a realistic intermediate target is **924 -> 1200/1300**.
Only after that should we discuss 1600+ or 2000+. This keeps the project
from turning into another unbounded patch cycle.

---

## 1. The positioning problem

ACT today is **#1 pure-forward** but **#5 overall** (924 vs abcrown 2460). The honest weakness is this: if the user / reviewer cares only about "how many instances does this verifier decide?", we lose by **1536 V/A** to abcrown. The narrow "we are pure-forward and the strongest in our class" claim is correct but is also the kind of claim a reviewer can dismiss as "you defined a small class so you could win it."

**To make the forward-only claim load-bearing, ACT needs to be competitive in
absolute terms — not just within its principle class.** The long-horizon target
is 2000+ V/A, but the next engineering/research milestone is lower and
falsifiable: get from 924 to **1200/1300** using a benchmark-independent HZ
change. If that cannot be done, 2000+ is not credible under the current
principles.

The +1100 V/A gap to close decomposes as follows (using competitor ceilings as headroom estimates):

| Benchmark | ACT now | abcrown | NeuralSAT | PyRAT[con_z] | Realistic ACT target | Δ |
|---|---:|---:|---:|---:|---:|---:|
| cifar100_2024 | 0 | 101 | 96 | 15 | 80 | +80 |
| tinyimagenet_2024 | 1 | 140 | 134 | 0 | 80 | +79 |
| vggnet16_2022 | 1 | 14 | 6 | 15 | 10 | +9 |
| yolo_2023 | 0 | 62 | 58 | 40 | 35 | +35 |
| safenlp_2024 | 345 | 1080 | 858 | 472 | 750 | +405 |
| relusplitter | 7 | 113 | 62 | 41 | 60 | +53 |
| linearizenn_2024 | 17 | 59 | 59 | 60 | 55 | +38 |
| acasxu_2023 | 88 | 139 | 137 | 178 | 150 | +62 |
| metaroom_2023 | 15 | 94 | 94 | 60 | 80 | +65 |
| ml4acopf_2024 | 19 | 59 | 0 | 40 | 50 | +31 |
| nn4sys | 86 | 69 | 86 | 50 | 130 | +44 |
| malbeware | 136 | 149 | 128 | 125 | 145 | +9 |
| collins_rul_cnn_2022 | 51 | 39 | 39 | 58 | 58 | +7 |
| tllverifybench_2023 | 3 | 15 | 15 | 30 | 25 | +22 |
| **Total reachable** | — | — | — | — | — | **+939** |

So the headroom exists, but it is not automatically harvestable. The table is
a ceiling estimate from other engines, many of which use BaB/backward/splitting
that ACT forbids. It should be read as "where to look", not as promised lift.

The question is **what structural HZ change could move several large gaps at
once**, without becoming a per-benchmark profile collection. The candidate is
SC-HZ: query-local, spec-conditioned generator budgeting.

---

## 2. The pain points of standard HZ (experimental record)

From the existing memos and our 2026-04 → 2026-06 experiments, the four observed structural pain points of HZ in robustness verification are:

### 2.1 PP1 — HZ degenerates to Star Set on conv layers

Per `star_vs_hz_analysis_20260530`: under triangle-only forward propagation, the binary-generator machinery is **dormant**. For the entire conv body of cifar/tiny/vgg/yolo (tens of layers), the HZ has `nb = 0` and the constraint set `(Ac, Ab, b)` is empty. **We are running Star Set, not HZ.** HZ's exact-ReLU binary mechanism only activates at the tail (large_cls_proof_mode / eq_lagr_v8).

This is the central waste: we pay the HZ representation cost (carrying Ac, Ab, b around) but get no HZ benefit because the binaries are never used during conv propagation.

### 2.2 PP2 — Triangle slack compounds multiplicatively

Per `dense_conv_forward_hz_plan_20260531`: each unstable ReLU adds one aux generator with magnitude `μ_i = -l_i u_i / (2(u_i - l_i))`. After K stacked ReLUs, the cumulative aux mass is `Σ_k μ_k · |W_subsequent|`. For typical CIFAR conv body (~30 unstable ReLU positions × 10 layers = 300 unstable neurons, scaled through Conv weights), this is **the dominant source of LP UB looseness** at the final-tail. §7 confirmed: production endcap LP is **already at the per-neuron triangle math ceiling** — no per-neuron lever left.

So the compounding is fundamental to triangle relaxation. Joint relaxation (k-ReLU) is the only single-layer lever and was closed-negative on acasxu (`project_pairwise_hull_negative_20260516`).

### 2.3 PP3 — Generator count explodes; reduction loses information

Standard Girard reduction merges generators by row-norm. After Girard at MaxPool (e.g. VGG L11 fan-out 1→298→127599 in our ImageHZ-lite Phase 0 trace), the surviving generators are a heuristic-chosen subset. They preserve the column-norm structure but **not the directional structure relevant to the output query**.

Per `spec_aware_girard_v2`: a reverse-F-chain Girard that scores by output relevance was tried and showed limited lift on conv. The F-chain has to propagate through CONV, BN, ADD, CONCAT, AVGPOOL — all implemented in v2 — but the underlying issue is that **on conv layers, ALL generators are roughly equally output-relevant** (because Conv mixes spatial channels). So the directional ranking is flat and Girard's row-norm ranking ≈ directional ranking.

This is a deeper observation than "Girard could be smarter": for conv-heavy networks, generator pruning has fundamentally low gain because Conv homogenizes generator directions.

### 2.4 PP4 — Full-output HZ is computed when only the rival projection matters

For robustness, the ANSWER we want is `max over x in box of (W_out[r] - W_out[t]) · y(x)` for each rival `r`. We need ONE scalar per rival. Yet we compute the FULL output HZ `y` (1000-dim for ImageNet) and project at the end.

Carrying around the full HZ through the network is **expensive and largely wasted**: most of the per-position information is irrelevant to any specific rival.

PyRAT [con_z] does the same wasteful thing and gets to 40.3%. abcrown's α-CROWN essentially propagates the PER-RIVAL projection backward, getting bound tightness on the dimension that matters. We can't go backward (P1), but we should investigate whether we can propagate the PROJECTED HZ forward.

---

## 3. Five proposed redesigns

I rank them by **expected gain × principle-compliance**. The top two are the
only ones with enough structural reach to justify a new research track; they
are not yet evidence for a 2000+ result.

### 3.1 R1 — Spec-Conditioned HZ (SC-HZ) ⭐ TOP PROPOSAL

**Idea**: For each rival/spec row, propagate a query-local HZ whose generator
budget is spent on the directions that matter for that row. The relevance
signal may use a fixed linear functional derived from network weights, but
**soundness must not depend on that functional being correct**. If the
functional is wrong, the reduction is merely less informative; the set remains
an over-approximation.

**Mechanism**:

```text
Pre-compute (once per (model, rival)):
    d_N = W_out[y_true] - W_out[r]                # scalar functional at output
    d_L = W_{L+1}^T d_{L+1}    for L = N-1, ..., 1   # linear backward thru weights only

Forward propagate the per-rival HZ:
    1. Standard HZ ops (Conv, Dense, ReLU)
    2. At each layer L:
        - Compute relevance score per generator j: s_j = |d_L^T G_L[:, j]|
        - Keep top-K by s_j (e.g. K = 256)
        - Merge the remaining into a single "tail box" generator with row-sum mass
    3. ReLU triangle (or eq_lagr_v8 for tail) as normal
    4. At output: LP UB on (d_N^T y) directly  (1-D, very cheap)

Verdict aggregation:
    If LP UB(d_r^T y) < 0 for all r:  CERT
    Else: decode xi* and ORT replay → FAL or UNK
```

**Why this can remain forward-only**:
- The `d_L` pre-computation is a fixed model-structural functional, not a
  bound. It uses no input box, no layer bounds, no dual variables, no slopes,
  no autograd, and no optimization.
- The forward pass uses `d_L` only as a SCORE for generator reduction.
- The reduction is sound for **any** score ordering because dropped generators
  are over-approximated by a tail box.
- The LP solved at the end is on the same forward-propagated HZ; no bound at
  layer `L' > L` refines a bound at layer `L`.

**Principle compliance** (drawing the line carefully):
- P1 (forward-only): ACCEPTABLE ONLY under the explicit definition in §5:
  backward-flowing **bound information** is forbidden; fixed architecture-only
  scoring metadata is allowed if soundness is independent of it.
- P2 (no gradients): COMPLIANT. `d_L` is a fixed weight product, NOT an autograd-derived quantity.
- P3 (continuous LP only): COMPLIANT.
- P4 (no BaB, no input split): COMPLIANT.
- P5 (no random falsification): COMPLIANT.

If the advisor rejects this distinction, SC-HZ must be re-scoped to a weaker
forward-only score that does not use transposed weight products. Under that
strict reading, 2000+ is very unlikely.

**Expected gains (hypotheses, not claims)**:
- Phase A signal target: `+5 V/A` over 80 sentinels OR median LP UB reduction
  >= 25%.
- Phase B/C intermediate target: `+100 .. +250` full-sweep V/A if Phase A
  signal generalizes.
- Long-horizon optimistic target with SC-HZ + selective exact-HZ ReLU +
  batching: `+500 .. +900`. This is what would make 1600+ plausible.

The earlier `+675` single-mechanism estimate was too speculative and should
not be cited as an expected result.

**Open design questions**:
1. What K? Pilot at K ∈ {128, 256, 512, 1024} per layer.
2. How to merge low-relevance generators soundly? Probably row-sum bound (Girard-style) but applied to the pruned subset.
3. Does the per-rival forward parallelize on GPU? Probably yes; this is THE engineering win that makes R1 tractable.
4. How does SC-HZ interact with eq_lagr_v8 at the tail? Probably the
   per-rival pre-pruning makes eq_lagr_v8 cheaper at the tail. Test
   side-by-side.

**Stop gate**:
- 20 sentinels each on cifar / tinyimagenet / safenlp / acasxu.
- PASS iff cumulative new V/A >= 5 OR median LP UB reduction >= 25%.
- FAIL iff new V/A = 0 AND median LP UB reduction < 10%.

---

### 3.2 R2 — Selective binary activation: SBA-HZ ⭐ SECOND PROPOSAL

**Idea**: HZ's binary generators are dormant during conv body propagation (PP1). Activate them STRATEGICALLY, per-layer, on a SMALL number of "high-impact" unstable neurons. For these neurons, encode the ReLU **EXACTLY** via the HZ binary mechanism (hz1 paper Prop. 4: +1 continuous, +1 binary, +3 constraints per neuron). For the remaining unstable neurons, keep the triangle.

**Mechanism**:

```text
Per ReLU layer L with unstable neurons U_L:
    1. Score each i ∈ U_L by output relevance × triangle slack:
           score_i = |d_L[i]| · μ_i
       where μ_i is the triangle aux magnitude and d_L is the rival direction.
    2. Select top-K_L exact neurons (K_L = 4-8 per layer).
    3. For each exact neuron i: emit HZ binary
           y_i = z_i · (1 + ξ_b^i) / 2
       with ξ_b^i ∈ {-1, +1} and exact ReLU constraints
           (Ac, Ab, b) extended per hz1 Prop. 4
    4. For the remaining U_L \ {top-K}: standard triangle.
```

The LP solved at the end is on the **continuous relaxation** of the binary generators (ξ_b ∈ [-1, +1] instead of {-1, +1}). This is a STRICTLY TIGHTER LP than the all-triangle baseline, because the exact-ReLU encoding includes the bilinear constraint that triangle relaxes.

**Why this is forward-only**:
- Per-layer, the selection is local. The score may reuse SC-HZ relevance
  metadata, but the exact-HZ ReLU encoding is emitted during forward
  propagation and remains an over-approximation after continuous relaxation.
- The binary variables are continuous-relaxed for LP purposes (no MILP).
- The forward pass adds binary generators but the LP at the end is still a normal continuous LP.

**Principle compliance**:
- P1: same boundary as SC-HZ if the score uses `d_L`; clean if the score uses
  only local forward quantities such as triangle slack `μ_i`.
- P2: COMPLIANT
- P3: COMPLIANT (continuous relaxation of binary generators; not MILP)
- P4: COMPLIANT
- P5: COMPLIANT

The "continuous relaxation of HZ binary" point is critical: hz1 Prop. 4's exact ReLU adds an integer variable, but the resulting LP relaxation is provably tighter than triangle because the exact constraint's relaxation hull includes the bilinear constraint. This is the "structured exact HZ" path that has NOT been tested before in our experiments.

**Expected gains (hypotheses)**:
- Phase after SC-HZ signal: `+50 .. +200` if a small number of exact-HZ ReLU
  binaries materially tightens acasxu / linearizenn / safenlp / relusplitter.
- If selective exact-HZ ReLU produces 0 lift on the Phase-B sentinels, close it;
  do not carry dormant binaries into production.

**Combined with SC-HZ**: this is a second-stage precision lever, not a
guaranteed path to 2000. It should only be implemented if Phase A shows that
query-local generator budgeting has real signal.

---

### 3.3 R3 — Batched per-rival GPU forward (engineering, not algorithmic)

**Idea**: SC-HZ's per-rival forward is N_rivals separate forwards per iid. For N_rivals = 99 and forward cost ~2 s per rival, that's 200 s per iid; for 200 iids, 40,000 s = 11 hours per benchmark. Too slow.

The engineering fix: **GPU-batched** per-rival forward. Each rival is a separate "batch dimension"; the HZ ops (Conv, Dense, ReLU) all batch naturally. The K-cap pruning is per-batch (each rival has its own K generators).

Memory budget per iid: K × n_layers × dim_per_layer × n_rivals × 8 B = 256 × 30 × 100k × 99 × 8 ≈ 600 GB. Too much. Need to cap n_rivals per batch and stream.

Realistic scheme: batch in groups of 10 rivals at a time. Memory per group: 60 GB (fits in 96 GB H100). Time per group: ~3 s. Total per iid: ~30 s. Total per benchmark (200 iids): ~100 min. **Tractable.**

**Expected gain**: not a V/A lift on its own, but it makes R1 viable at scale. Without R3, R1 is single-iid-only research; with R3, R1 is a production lift.

**Principle compliance**: COMPLIANT (pure engineering).

---

### 3.4 R4 — Stable-fastpath everywhere (lift the existing nn4sys profile to all benchmarks)

**Idea**: The `STABLE_AFFINE_FASTPATH` env knob (in `_nn4sys_lindex_profile`) exactly closes ReLU when interval propagation fixes both sides. It currently fires only on nn4sys + lindex. The mechanism is benchmark-name-gated, but the underlying check is structural: "if >X% of ReLUs are stable, take the linear fastpath".

Promote the gate to STRUCTURAL: any iid with stable_ratio ≥ 90% takes fastpath at the relevant layer. Specifically:
- For small ε robustness queries on well-trained networks, most ReLUs are stable.
- The cheap interval pass that detects "stable" is the gate; the fastpath drops the triangle relaxation entirely for stable neurons (no aux generator).

**Expected gain**:
- malbeware: **+5** (more iids stable-dominant)
- collins_rul: **+3**
- metaroom_2023: **+10** (singleton fastpath extension)
- linearizenn: **+5**
- cersyve / lsnc_relu: gated by parser support
- subtotal: **+23**

**Principle compliance**: COMPLIANT — structural gate, sound fastpath, no backward refinement.

---

### 3.5 R5 — Dedicated MaxPool: locality-preserving exact MaxPool

**Idea**: Our ImageHZ-lite Phase 0 trace showed MaxPool fans out 1→298→127599 tile blocks. The current dense-HZ MaxPool similarly compounds. A dedicated MaxPool that:
1. Identifies stable max windows (one position dominates) → exact pass-through
2. For unstable windows, uses HZ binary generators (one binary per window) — exact MaxPool via hz1-style construction
3. Memory-budget-gated: if binary count exceeds budget, fall back to triangle (sound, looser)

**Expected gain**:
- vggnet16_2022: **+5** (MaxPool fires at L11/L18/L25/L32 are exactly where we lose information)
- yolo_2023: **+15** (detector arch has many MaxPools)
- cifar100_2024: **+10** (modest; many CIFAR backbones use MaxPool sparingly)
- tinyimagenet_2024: **+15**
- subtotal: **+45**

**Principle compliance**: COMPLIANT (forward HZ binary generators are the mechanism).

---

## 4. Estimated total lift, ranked combination

| Milestone | Compliance risk | Evidence needed | Plausible total if successful |
|---|---|---|---:|
| Current | clean | existing full sweep | 924 |
| Engineering cleanup only (R3/R4/R5) | clean | no-lost sweep | 1000 .. 1100 |
| SC-HZ Phase A/B | boundary must be accepted | sentinel LP UB reduction + new V/A | 1100 .. 1300 |
| SC-HZ + selective exact-HZ ReLU | same boundary if using `d_L`; otherwise clean local scoring | expanded sentinel sweep | 1300 .. 1600 |
| Mature SC-HZ portfolio | same boundary + substantial engineering | full canonical sweep | 1600+ possible |
| 2000+ | unproven | would require SC-HZ to move safenlp + dense-conv + small-dense simultaneously | long-horizon only |

This table deliberately replaces the earlier `~1977` point estimate. The
old number was a useful ambition but not a defensible expected outcome.

---

## 5. The principle question — is SC-HZ's d_L "backward propagation"?

This is the central decision. Two readings:

### 5.1 Strict reading: "any non-forward dataflow violates P1"

Under this reading, computing `d_L = W^T d_{L+1}` is a backward pass. Forbidden. R1 closes; lift is limited to R3+R4+R5 (+68).

This reading is **internally consistent** but it also forbids:
- Pre-computing the rival classifier matrix `(W_out[t] - W_out[r])` since that's a "backward" operation on the output layer.
- Any per-rival or per-spec preprocessing.
- Realistically, this is the reading that makes the forward-only claim narrowest.

### 5.2 Distinction-based reading: "forward-only means no bound refinement; functional pre-computation is fine"

Under this reading, `d_L` is a linear functional fixed by the network architecture, not a refined bound. Computing it is "looking at the weights to know which generators to keep" — analogous to "pre-computing the network's Lipschitz constant before propagating". This makes SC-HZ a legitimate research candidate, but it does **not** by itself imply any score target.

This reading is also internally consistent. It permits:
- Fixed per-spec precomputation that is chosen before propagation and whose
  correctness is independent of the computed bounds.
- Per-rival forward propagation where the rival direction is a known constant.

The cleanest analogy is: a forward-HZ that runs on the CONCATENATED MODEL `(model, rival_classifier)` is still forward-HZ. R1 just computes that concatenation's structure efficiently.

### 5.3 My recommendation

**Reading 5.2 is the principled one**. Forward-only should mean **no bound information flows from output back to bound updates**, not **no architectural information flows backward at all**. Under 5.2, R1 is forward-only.

I think the cleanest way to defend this in the paper is to define forward-only as:

> A verifier is **forward-only** iff for every layer L, the bound information at L is determined SOLELY by the bound information at L-1 and the operator at L. No bound at L' > L can refine the bound at L.

Under this definition, SC-HZ's per-rival pre-computation does NOT change ANY bound at any layer. It only changes WHICH generators are kept during forward propagation. The bounds are still propagated forward and are still sound; we just prune the representation per rival.

If this definition is accepted, SC-HZ is compliant enough to prototype. If the
stricter reading is required, SC-HZ must use weaker purely-local forward scores
and the 2000+ long-horizon target becomes unrealistic under the current rules.

---

## 6. Implementation plan (if approved)

### 6.1 Phasing

**Phase A — SC-HZ design lock + signal pilot** (1 week)
- Write design plan analogous to `cifar_finaltail_hull_plan.md`.
- Specify the K-cap policy, generator merge semantics, batched-GPU memory bounds.
- Implement SC-HZ generator budgeting only (no R2 yet) as a research module
  under `research/sc_hz/`.
- 80 sentinels total: 20 each on cifar100, tinyimagenet, safenlp, and acasxu.

**Phase B — Phase-A gate evaluation** (1 day)
- PASS iff cumulative new V/A >= 5 OR median LP UB reduction >= 25%.
- FAIL iff new V/A = 0 AND median LP UB reduction < 10%.
- Otherwise mark inconclusive, widen K on the worst sentinels once, and rerun
  only those sentinels.

**Phase C — full sentinel run** (3 days)
- If Phase A passes, expand to 6-8 benchmarks selected by headroom and
  low-error coverage: cifar, tiny, safenlp, acasxu, linearizenn, relusplitter,
  ml4acopf, and metaroom.
- Gate: clear no-lost audit plus either +50 aggregate V/A or median LP UB
  reduction >= 25% on the still-UNKNOWN subset.

**Phase D — combine with R2** (1 week)
- Add selective exact-HZ ReLU only if Phase C shows SC-HZ signal.
- Gate: measurable improvement over SC-HZ alone on the same sentinels; close
  immediately if it only adds rows/variables without reducing LP UB.

**Phase E — full canonical sweep** (3 days)
- Run the surviving SC-HZ variants on the full 22-benchmark sweep.
- Re-verify provenance contract (62/62 FAL receipt strict ORT replay).
- Cross-check against the 924 baseline; explicitly document the V/A delta per benchmark.

**Phase F — paper update** (1 week)
- Update `paper_skeleton_20260604.md` headline to whatever the full sweep produces.
- Add R1 design lock as a new appendix.
- Update Table 3 (cross-tool comparison) with the new ACT row.

**Total timeline**: ~3 weeks to know whether SC-HZ is real; longer if it
passes and needs productionization. Do not present this as a guaranteed path to
2000+.

### 6.2 Risk register

| Risk | Mitigation |
|---|---|
| Project rejects SC-HZ's `d_L` as backward | Re-scope SC-HZ to local forward scores only; likely lower ceiling. |
| Per-rival forward too slow even with GPU batching | Cluster rivals by direction similarity; do 10 cluster-forwards instead of 99 per-rival |
| K-cap too small loses precision | Per-iid K-cap adaptive: start K=128, expand if LP UB doesn't tighten |
| R2 binary generators trigger MILP-equivalent LP | Continuous relaxation only; cap the number of binary generators per iid (e.g. 200 total) |
| Engineering effort exceeds 3 weeks | Phase B/C gates allow early termination if the lift doesn't materialize |

---

## 7. What this does NOT do

This proposal does NOT:
- Re-open ImageHZ (CIFAR-ImageHZ closed by atlas v3, VGG-ImageHZ closed by §6b/§6c).
- Re-open Phase 2 pair-hull (CIFAR final-tail closed by §7).
- Introduce CROWN backward refinement (R1 is a linear functional, not a bound).
- Introduce MILP / BaB / random falsification.
- Affect the existing 924 V/A baseline (R1-R5 are NEW code paths, gated by env knobs; default behavior unchanged).

**The 924 V/A canonical claim stays sound regardless of the SC-HZ outcome.**
SC-HZ is an addition, not a replacement.

---

## 8. Comparison to existing tools' mechanisms

To be honest about where the redesign sits:

| Tool's mechanism | Status under R1 | Comment |
|---|---|---|
| abcrown α-CROWN backward | NOT used | α-CROWN propagates backward bounds; R1 only pre-computes linear functionals |
| abcrown β-CROWN BaB | NOT used | We forbid BaB |
| NeuralSAT BaB | NOT used | We forbid BaB |
| nnenum exact-star splitting | NOT used | We forbid splitting |
| PyRAT [hyb_z] | Same domain, less engineering | Our R3 (batched GPU) is what they don't have |
| Singh PRIMA k-ReLU | NOT used | Closed-negative on acasxu |
| Anderson facets | NOT used | Closed-negative on acasxu |

SC-HZ is a **new mechanism** in the verifier-design space: forward-only
propagation with **per-rival generator pruning conditioned on a pre-computed
linear functional**. The closest precedent is α-CROWN's spec-locality, but
α-CROWN optimizes bounds backward; SC-HZ only uses a fixed functional to budget
forward representation capacity, and the pruning step remains sound for any
ordering.

This is the kind of mechanism that, if it works, becomes a paper section on its own:

> "**Spec-Conditioned Hybrid Zonotopes**: a forward-only abstraction that
> budgets generator capacity per output query while preserving soundness by
> over-approximating every discarded direction."

---

## 9. Decision needed before implementation

Before any code lands, the project must explicitly record:

1. **Is SC-HZ's `d_L` pre-computation within the project's forward-only
   definition?** (Reading 5.1 vs 5.2.)
2. **Is R2's selective binary activation acceptable** under "no MILP" (since the LP is continuous relaxation)?
3. **What Phase A stop rule is binding?** The default in this document is the
   80-sentinel gate in §6.

Recommended project definition:

> A verifier is **forward-only** iff for every layer L, the bound information
> at L is determined SOLELY by the bound information at L−1 and the operator
> at L. No bound at L′ > L can refine the bound at L.

Under this definition, SC-HZ is compliant **only if** `d_L` is used as a
non-semantic reduction score and the PRUNE operation is sound for arbitrary
scores. The proof obligation is therefore stronger than "the direction is
useful"; it must show that even a useless direction cannot invalidate the
over-approximation.

Proceed to `research/dc_hz_phase_a_plan.md` only as a research-only Phase A
prototype. Do not change production defaults or paper headline until the gate
passes.

---

## 10. Files cited

- `research/star_vs_hz_analysis_20260530.md` — HZ ≡ Star degeneracy
- `research/hz_zero_benches_deeper_analysis_20260530.md` — 4 failure modes
- `research/dense_conv_forward_hz_plan_20260531.md` — conv-dense gap
- `research/spec_aware_girard_v2_20260530.md` — F-chain reverse Girard
- `research/tool_comparison_20260604.md` — current 924 V/A position
- `research/paper_skeleton_20260604.md` — paper-current state
- `/data1/Kane/HyZor/HZ/hz1.pdf` — Bird's HZ Prop. 4 (exact ReLU via +1 binary)
- `/data1/Kane/HyZor/HZ/PhD_Trevor_Bird_2022.pdf` — Bird PhD thesis on HZ
