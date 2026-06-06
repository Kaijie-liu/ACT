# Forward Plan — Principle-Internal Levers After Phase F1/F2b/FC-HZ Closure

**Date**: 2026-06-06 morning (REVISED for advisor's 5 corrections)
**Revision note**: this version ingests advisor feedback on:
1. Lever 4 (activation walk) elevated to `requires advisor principle ruling BEFORE code`
2. Lever 3 (exact-arith LP) downgraded to `audit-only until solver-policy ruling`
3. "D2/D3 CERT unreachable" qualifier added everywhere: `by current pipeline`
4. Lever 2 (low-dim) gate lowered: 50% → "≥5 NEW V/A or ≥20% yield"
5. Lever 1 (parser) explicit "same iid, same timeout, same profile" attribution rule
**Companion to**:
- `research/INNOVATION_BRIEF_sc_hz_20260604.md` (original hypothesis)
- `research/RETROSPECTIVE_sc_hz_what_did_not_work_20260606.md` (post-experiment)
- `research/PAPER_1472_CHARACTERIZATION_20260606.md` (1472 honest scope)

**Purpose**: After the Phase F1/F2b/FC-HZ closures and the retrospective,
synthesize the **forward-looking levers that remain principle-compliant**
and are NOT empirically closed. Written as an action plan with hard gates,
intended to drive Phase H0 (measurement) and possibly H1-H3 (sprints).

**Headline rule**: This document supersedes the section on "next steps" in
the retrospective. Anything from F1/F2b/FC-HZ/F3 that is empirically closed
is NOT a lever and is documented as closed below to prevent revival.

---

## 1. The corrected diagnosis: 4 distinct diseases, not one

The retrospective lumped the weak benchmarks together. They are actually
four structurally different problems, each with a different lever set.

| Disease | Benchmarks | Real failure cause | Principle-internal lever availability |
|---|---|---|---|
| **D1. Parser / architecture gaps** | `cctsdb_yolo_2023`, `linearizenn_2024` (Slice), `yolo_2023` (detection head), `metaroom_2023` parser residual, `nn4sys` / `ml4acopf` remaining ops | The verifier cannot even run the model correctly. NOT a precision problem. | **Engineering only**, zero principle cost |
| **D2. Dense-conv robust CERT** | `cifar100_2024`, `tinyimagenet_2024`, `vggnet16_2022`, `traffic_signs` | Convex relaxation barrier: triangle slack is aggregate-diffuse across many unstable ReLUs; per-neuron / per-pair / per-layer cuts cannot bridge the gap | CERT main mass **unreachable by current SC-HZ + DeepZ + LP-sidecar pipeline** (may be reachable via Phase H new abstraction; not by patches to current pipeline); FAL subset + low-effective-input-dim subset are reachable |
| **D3. Case-reasoning CERT** | `acasxu_2023`, `relusplitter`, `linearizenn_2024` dense parts | Needs activation case-split / exact-star (forbidden by P4) | CERT main mass **unreachable by current SC-HZ + DeepZ + LP-sidecar pipeline** (may be reachable via Phase H new abstraction; not by patches to current pipeline); FAL subset + boundary exact-arithmetic subset + ReLU motif simplification reachable |
| **D4. Universal hard** | `lsnc_relu` | Lyapunov/nonlinear, ALL tools ~0 | Not our weakness specifically. Should be flagged in paper as "universal hard" and removed from self-evaluation scorecard |

**Implication for the V/A target**: Of the ~528 V/A gap to 2000, an
unknown but probably-large fraction is in D2 + D3 robust-CERT — which is
**mechanism-blocked** within our principle set. The achievable fraction
within principles is **tens to low-hundreds**, not ~528. The exact
number requires Step 0 measurement (§3 below).

---

## 2. Closed levers (do not revive without new evidence)

To prevent re-entry, the following are documented closed:

| Closed lever | Closure evidence | Re-entry condition |
|---|---|---|
| **PRUNE** (`d_L^r`-driven generator selection) | 40/40 dense sentinels: LP UB monotonically grows as K shrinks | New ranking theorem AND independent monotonicity gate |
| **F1 (single-neuron triangle LP at last ReLU only)** as a V/A lever | 17% real cifar tightening, 0 NEW V/A on 8 sentinels, 0 NEW V/A on 218 small-dense instances | None — infra kept, but no further scoring runs |
| **F2b (same-layer pairwise joint hull)** | top_k=4/10/20/25=ALL-pair on cifar 113: 0% additional over F1 across 300 cuts | None — empirically dead on diffuse-slack networks |
| **FC-HZ (multi-layer triangle history LP)** | 8.1% median additional vs 40% gate | None — diminishing returns per layer, hard math ceiling |
| **Multi-neuron k≥3 LP cuts** | F2b 300-cut zero gain + Phase A PRIMA k=2/k=3 historical 0 lift | Pre-registered theoretical advance only |

These are NOT in the lever ROI list below. Any plan that proposes them
without new theoretical evidence is rejected by reference to this table.

---

## 3. Step 0 — Mandatory before any sprint: measure the harvestable subset

**Problem**: every sprint plan below assumes a per-bench estimate of
"how many UNK iids fall into which disease category". We have NEVER
measured this. Running sprints without it is gambling.

**Action**: build `research/sc_hz/phase_h0_harvestable_subset.py` that for
every (bench, UNK iid) emits a 5-way classification:

| Tag | Definition | Recovery mechanism |
|---|---|---|
| `parseable` | Walker fails with `not implemented` on a finite list of ops | D1 parser sprint |
| `fal_able` | LP rival max excess > 0 AND ORT replay of LP-corner candidate is "close" (within configurable threshold) | Lever 4 (FAL walker) |
| `low_dim` | Spec has effective input dimension `<<` raw shape (sparse perturbation; patch; subset of channels) | Lever 2 (low-dim profile) |
| `boundary_numeric` | F1 LP UB ≤ +ε for ε ∈ {1e-3, 1e-2} | Lever 3 (exact-arithmetic LP) |
| `robust_mechanism_blocked` | None of the above; F1 LP excess > +0.1 | Phase H new abstraction OR principle relaxation |

Output per bench:
```
{bench: {n_unk, parseable, fal_able, low_dim, boundary_numeric, robust_blocked}}
```

Across all 22 benchmarks, this gives a **measurable ceiling**:
- `Σ parseable + fal_able + low_dim + boundary_numeric` = principle-internal harvestable upper bound
- `Σ robust_blocked` = honest "out of scope under this principle set"

**Time**: 0.5 - 1 day implementation + 0.5 day run.
**Gate**: Step 0 output IS the data behind every subsequent sprint's
expected yield. No sprint runs without this.

This measurement is also **the strongest possible paper claim**:

> "Under our principle set, X / Y instances are mechanism-reachable.
>  We harvested X out of X. The remaining Y - X requires mechanisms
>  outside our principle scope. This is the effective ceiling of
>  forward-only / continuous-LP / no-BaB / no-gradient verification."

---

## 4. The 5 principle-internal levers, ranked by ROI

After Step 0, these are the levers actually available. Each has a
principle-compliance line, an expected yield band, and a hard gate.

### Lever 1 — Parser sprint (D1 disease, zero principle cost)

**Mechanism**: implement missing ONNX ops (Slice with dynamic shape,
Gather, Reshape with computed shape, variable-shape Conv head) so that
parseable-tagged UNK iids can actually run through the existing pipeline.

**Principle audit**: pure software engineering. No new math, no new
abstraction, no principle relaxation. Pass.

**Targets** (in priority order):
1. `linearizenn_2024` Slice (blocks all 47 UNK)
2. `cctsdb_yolo_2023` variable-shape Slice
3. `nn4sys` remaining Reshape/Slice/Gather
4. `ml4acopf_2024` remaining ops
5. `metaroom_2023` profile residual
6. `yolo_2023` detection head normalization

**Expected yield**: highly dependent on Step 0. If Step 0 shows
`linearizenn parseable=40`, ceiling on this bench is +40 V/A if all 40
also flip via existing mechanisms after parsing.

**Hard gate**: timebox 3-5 days. Stop conditions:
- < 30 NEW V/A after 5 days → close sprint, return to paper or H2-H3
- Any parser fix that "fixes" the parse but doesn't change V/A
   counts as 0 progress (parser fix without V/A lift is engineering
   plumbing, not a research contribution).

**ATTRIBUTION DISCIPLINE (advisor add 2026-06-06)**:
- Parser-fix-NEW counts ONLY if compared under **same iid, same
  timeout, same profile** as production baseline. If parser fix is
  paired with a wider timeout or different profile, the lift is
  attributed to timeout/profile, NOT to parser.
- Re-running parser-fix-iids under production baseline timeout and
  profile to confirm attribution is MANDATORY before counting.

### Lever 2 — Low-effective-input-dim profile (D2/D3 subset, principle-pure)

**Mechanism**: triangle slack and unstable ReLU count both scale with
the **effective input dimension** of the spec, NOT with the network size.

For:
- A cifar100 instance with L∞ ε over all 3072 input dims, slack accumulates
   across thousands of unstable ReLUs → barrier hits
- A cifar100 instance with sparse patch perturbation (k=20 pixels), the
   walker sees a 20-dim input, slack accumulates over a small number of
   neurons → forward HZ may CERT directly

Evidence: vggnet16's single FAL came from sparse-input spec. This is the
shadow of the same mechanism — but as a CERT lever, not just a FAL one.

**Action**: profile every UNK iid by effective input dimension. Run the
low-dim subset through the **existing** forward HZ pipeline. No new
mechanism. The forward propagation is just smaller.

**Principle audit**: zero principle cost. Just running existing pipeline
on a subset selected by spec geometry. Pass.

**Expected yield**: depends on how many UNK specs are actually low-dim.
For cifar100 / tiny / vgg / yolo, this could be 0 (if all specs are full
L∞ ε-balls) OR significant (if some specs are patch/sparse). Step 0
must measure this.

**Hard gate** (revised per advisor: aggressive 50% gate would kill small
but real signal):
- Step 0 reports `low_dim count per bench`
- Run low-dim subset; PILOT gate `≥ 5 NEW V/A OR ≥ 20% yield`
- Failure: < 5 NEW V/A AND < 10% yield → close.
- Pass: continue and re-evaluate at 40-iid scale.

### Lever 3 — Boundary exact-arithmetic LP (D3 boundary, **audit-only until solver-policy ruling**)

**Status correction**: advisor flagged that the project's principle set may
operationally restrict to `scipy.linprog` / `HiGHS` only. Rational LP /
self-implemented simplex would be mathematically continuous-LP, but a
DIFFERENT solver. Until the solver-policy ruling lands, Lever 3 is
downgraded to **audit-only numeric diagnosis** — its output cannot be
counted as NEW V/A.

**Smaller-step alternative to try first**: use HiGHS with tighter
tolerances (`primal_feasibility_tolerance` / `dual_feasibility_tolerance`
set to 1e-12 or `presolve='off'` + `time_limit=large`) + redundancy
removal on F3's boundary subset. This stays within current solver and
could resolve the +0.001 → +0.000 boundary cases without touching
solver policy. If HiGHS-tighter resolves, Lever 3 rational variant is
unnecessary. If HiGHS-tighter does NOT resolve, escalate Lever 3 to
solver-policy ruling.

**Mechanism**: F3 day 1 showed iids where F1 LP returns excess **at the
boundary**:
```
acasxu iid 107: HZ=+0.001 → F1=+0.000 (could be float noise around true ≤0)
acasxu iid 98:  HZ=+0.002 → F1=+0.002
acasxu iid 143: HZ=+0.004 → F1=+0.004
acasxu iid 102: HZ=+0.084 → F1=+0.008
```

For excess at +0.000 to +0.01 range, float LP solvers can return positive
values even when the true mathematical value is ≤ 0 (numerical conditioning,
roundoff). Switching to a **rational / exact-arithmetic LP** can resolve
the true sign.

The exact LP is still:
- continuous LP (no integers, no MILP) → P3 pass
- forward-only (uses the same constraint structure F1 builds) → P1 pass
- no BaB, no split → P4 pass
- no gradient → P2 pass

**Principle audit**: pass on all P1-P5. Pure numerical hardening.

**Expected yield**: limited to boundary-tagged iids only. Step 0
should report `boundary_numeric` count per bench. From F3 day-1 acasxu
65 near-CERT subset, perhaps 5-15 have true ≤ 0 and would flip.

**Hard gate**:
- Implement rational LP (`fractions.Fraction` or mpmath) on F3's
   boundary-tagged subset
- Gate: ≥ 50% of boundary-tagged iids flip to true CERT
- Failure: < 20% flip → close lever

### Lever 4 — Activation-mode walk falsifier (BARRIER-IMMUNE FAL, **REQUIRES ADVISOR PRINCIPLE RULING BEFORE ANY CODE**)

**Risk classification**: HIGHEST risk in this plan. Advisor's framing:
even without autograd, repeated concrete-replay + re-linearization
DIRECTION-UPDATE can be attacked as "deterministic whitebox attack /
gradient-free attack". The mathematical content is forward-structural,
but the EXTERNAL APPEARANCE is DeepFool-like. The framing matters for
audit and paper claims.

**HARD RULES that must be locked before code is written**:

1. **x_init source restriction**: walker x_init MUST come from a HZ/LP
   structured candidate (e.g., F1 LP rival-corner). x_init MAY NOT come
   from a natural data point, random sample, or corner pool.
2. **Per-step structured provenance**: every step records (mask hash,
   linear region id, W_eff generation chain). Without this provenance,
   the step is not counted.
3. **Naming**: this is NOT a "verifier proof path". It is a "structured
   FAL sidecar". Paper / reports MUST use the latter name. Misnaming is
   a project violation.
4. **Sign-off timing**: advisor MUST give explicit written principle
   ruling on `W_eff = W_{N+1}·diag(m_N)·...·diag(m_1)·W_1` BEFORE any
   code is written. The ruling must address whether fixed-activation-mask
   affine direction is "forward-structural" (same epistemic class as
   the brief's `d_L^r`) or "gradient-equivalent forbidden attack".

**Mechanism**: convex relaxation barrier blocks CERT, **not FAL**. FAL
only needs one ORT-validated counterexample; tightness of the relaxation
is irrelevant for FAL.

A deterministic, gradient-free, no-PGD falsifier:

```
Algorithm: ActivationModeWalk(x_init, model, spec)
  x ← x_init
  for k = 1 .. K_max:
      Run forward(model, x) to get y and ReLU mask m
      Build W_eff = W_{N+1} · diag(m_N) · ... · diag(m_1) · W_1
      Pick rival r with largest current spec violation
      d ← W_eff^T (e_r - e_y_true)
      x' ← box_corner(x, d, lb, ub)
      Run ORT(model, x') → y'
      if violates(spec, y'): return FAL with witness x'
      if y' == y (cycle): return UNK
      x ← x'
  return UNK
```

**Principle audit** (CRITICAL — must pass before any deployment):
- P1 (no backward bound refinement): `W_eff` is a structural coefficient
   from a FIXED activation mask, not a backward-propagated bound. Same
   epistemic status as the original brief's `d_L^r`. NEEDS EXPLICIT
   ADVISOR SIGN-OFF.
- P2 (no gradient): the `diag(m_i)` are activation masks observed at a
   concrete x, NOT `∇relu`. We never call autograd. Pass.
- P3 (continuous LP / no integer): no LP needed; closed-form box-corner.
   Pass.
- P4 (no BaB, no split): single-candidate iteration with cycle break.
   No tree, no input split. Pass.
- P5 (no random / corner search): every step is deterministic from
   `W_eff` and the current `x`. The seed is the LP-derived `x_init`.
   No random sampling. Pass.

**ADVISOR DECISION REQUIRED**: is `W_eff` (mask-as-observed) acceptable
under the project's principles? The original SC-HZ brief made the same
call for `d_L^r` (weight-only linearization, NOT bound info). The same
ruling should extend here, but it must be explicit, not assumed.

**Expected yield**: barrier-immune for the FAL-tagged subset.
- cifar100 instances at moderate ε have known non-robust ones — these
   are FAL candidates
- vggnet16's existing 1 FAL is the existence proof
- acasxu's 20/20 phantoms (single-corner-no-replay) are prime targets;
   multi-step walk converts some to true A
- For acasxu, spec is region→output-constraint not classification margin,
   so the walker maximizes `max(A·y - b)` instead of rival margin.
   1-2 days walker variant.

**Hard gate**:
- After advisor sign-off on principle audit
- Pilot on 20 cifar100 + 20 tiny + 20 acasxu FAL-tagged
- Gate: ≥ 10 NEW A across 60 instances → expand to all FAL-tagged
- Failure: ≤ 3 NEW A → close

### Lever 5 — ReLU motif structural simplification (D3, principle-pure)

**Mechanism**: `relusplitter` literally splits ReLUs and tests whether
the verifier preserves the implied identities. Add a forward-graph
canonicalization step that detects:

#### Duplicated ReLU
```
if z_a == z_b:
    y_a == y_b   (sound linear identity, not new info per neuron but
                  removes a generator pair)
```

#### Opposite ReLU pair
```
if y_pos = ReLU(z), y_neg = ReLU(-z):
    y_pos - y_neg = z    (linear equality, replaces 2 triangles)
```

#### Constant-fold affine chains
- `Dense + Add + Mul const` collapses
- Duplicate affine node merging
- Dead ReLU removal (provably stable from forward interval)

**Principle audit**: pure graph rewriting. The result is a forward HZ
on a smaller, mathematically equivalent graph. No backward, no split,
no LP relaxation change. Pass on all P1-P5.

**Expected yield**: most concentrated on `relusplitter` (by design) and
`linearizenn`. Could be high on these specific benches, near zero
elsewhere.

**Hard gate**:
- Implement motif detector
- Pilot on 20 relusplitter UNK + 20 linearizenn (post-Slice-parser) UNK
- Gate: ≥ 5 NEW V or median margin drop ≥ 50% on these 40
- Failure: 0 NEW V AND drop < 25% → close

---

## 5. Per-benchmark recommended strategy (after Step 0)

This table is conditional on Step 0 outputs. Numbers in parens are
the relevant Step 0 tag counts that gate the strategy.

| Benchmark | Step 0 tags to read | Strategy | Expected NEW V/A band |
|---|---|---|---|
| `safenlp_2024` residual | fal_able, low_dim | Lever 4 walker variants + multi-rival consensus | +10 to +50 |
| `cifar100_2024` | fal_able, low_dim, robust_blocked | Lever 2 (low_dim subset) + Lever 4 (fal_able subset) | +5 to +30 |
| `tinyimagenet_2024` | fal_able, low_dim, robust_blocked | Same as cifar100, smaller scale | +0 to +15 |
| `vggnet16_2022` | fal_able, low_dim, robust_blocked | Lever 2 + Lever 4 sparse-input focus | +0 to +5 |
| `yolo_2023` | parseable, fal_able | Lever 1 (parser) + Lever 4 if any parseable iids show FAL signal | +0 to +20 |
| `traffic_signs` | parseable, fal_able | Same as yolo | +0 to +10 |
| `cctsdb_yolo_2023` | parseable, fal_able | Lever 1 (Slice dynamic) priority | +0 to +20 |
| `linearizenn_2024` | parseable, boundary_numeric | Lever 1 (Slice) then Lever 3 boundary | +5 to +30 |
| `acasxu_2023` | boundary_numeric, fal_able | Lever 3 (exact LP) + Lever 4 walker (region→constraint variant) + Lever 5 motif | +10 to +30 |
| `relusplitter` | fal_able, motif-friendly | Lever 5 (motif detector) priority + Lever 4 | +20 to +60 (by design) |
| `metaroom_2023` | parseable, boundary_numeric | Lever 1 + Lever 3 | +0 to +5 |
| `nn4sys` | parseable | Lever 1 (Reshape/Slice/Gather) | +0 to +20 |
| `ml4acopf_2024` | parseable | Lever 1 (parser residual) | +0 to +10 |
| `lsnc_relu` | universal_hard | **EXCLUDE from scorecard** (all tools ~0) | (0, scorecard adjustment) |

Sum of lower bounds: ~50. Sum of upper bounds: ~305. Realistic midpoint:
**~150 NEW V/A**, taking 1472 to ~1622. This is well below 2000 and
this estimate is BEFORE Step 0 confirms the tag distributions.

---

## 6. The realistic V/A target ceiling

| Tier | Target | Routes | Time |
|---|---|---|---|
| **T0 (definite, NO sprint)** | 1472 | freeze + paper | 1-2 weeks writing |
| **T1 (likely, harvest the achievable)** | 1500 - 1620 | Step 0 + Levers 1-5 within principles | 2-3 weeks |
| **T2 (research-grade, uncertain)** | 1600 - 1800 | T1 + Phase H new abstraction (block-template / projected tail) | months |
| **T3 (principle-violating)** | 2000+ | T1 + relax P1 (backward bound) or P4 (BaB) | abandons project thesis |

**Honest recommendation**: T1 with paper writing in parallel. Step 0
should land within 1 day; if it shows < 50 harvestable, go straight to
T0 paper. If it shows > 150, T1 sprint is justified. T2 is research
investment, not "lift cifar100 from 0".

---

## 7. Concrete next-7-day action sequence

```
Day 1 morning:
  - Audit fix: FC-HZ xfail → strict=True
    (was @expectedFailure; strict=True catches future improvement as XPASS)
  - Test suite verify: 73 tests, OK (expected failures=1), no XPASS

Day 1 afternoon:
  - Implement research/sc_hz/phase_h0_harvestable_subset.py
  - Run on ALL 22 benchmarks, all UNK iids
  - Output: per-bench tag distribution, total harvestable ceiling

Day 2:
  - Advisor decision on Lever 4 principle audit
    (Is W_eff from fixed activation mask same epistemic status as d_L^r?)
    UNTIL ADVISOR RULES: NO Lever 4 code, no walker file created.
  - Read Step 0 output. If total harvestable < 50, GO TO PAPER.
  - If 50-150, run Lever 1 (parser) + Lever 2 (low-dim, pilot gate
    ≥5 NEW V/A or ≥20% yield) + Lever 3 audit-only (HiGHS tighter
    tolerance first, NOT rational LP).
  - If > 150, queue Lever 4 PENDING advisor ruling, not before.

Day 3-5:
  - Run selected sprints with hard gates
  - Daily aggregate report

Day 6-7:
  - Aggregate sprint result vs Step 0 predicted ceiling
  - If yield ≥ 80% of Step 0 ceiling: harvest succeeded, paper claim is
    "we captured all that is capturable within principles"
  - If yield < 50%: sprint failed; paper claim is 1472 baseline
```

---

## 8. What this plan does NOT do

- Does not propose continued F1/F2b/FC-HZ tightening on dense-conv.
   Empirically closed.
- Does not promise 2000+ V/A. The math says no within principles.
- Does not propose Phase H new abstraction as a sprint (it is research).
- Does not propose principle relaxation. The project's core thesis is
   strict forward-only.
- Does not lump all weak benchmarks together. The 4-disease decomposition
   is mandatory.

---

## 9. Test / regression discipline going forward

### 9.1 FC-HZ gate test
Change `@unittest.expectedFailure` to `@pytest.mark.xfail(strict=True)`
(or move file to `research/gates/`). The strict variant catches the case
where FC-HZ accidentally improves past 40% — that would be an XPASS
failure and signals revisit.

### 9.2 New gates for Phase H levers
Each Lever 1-5 sprint registers a gate file in `research/gates/` BEFORE
the sprint runs:
- expected yield range
- principle-audit checklist
- hard stop condition

A sprint that runs without a registered gate is rejected as "test theater".

### 9.3 Step 0 ceiling is canonical
`research/PHASE_H0_HARVESTABLE_SUBSET_RESULT.md` (to be produced by the
Step 0 run) is the canonical reference for "principle-internal achievable
V/A". Any document claiming a different number must reference this file
or it is overridden.

---

## 10. Files this plan refers to (existing + to-create)

### Existing
| File | Role |
|---|---|
| `INNOVATION_BRIEF_sc_hz_20260604.md` | original hypothesis (now with post-experiment correction header) |
| `RETROSPECTIVE_sc_hz_what_did_not_work_20260606.md` | what was tried and why it failed |
| `PAPER_1472_CHARACTERIZATION_20260606.md` | honest 1472 scope for paper |
| `phase_F1_*` / `phase_F2b_*` / `phase_F3_*` / `phase_G_*` | individual closure memos |

### To create (by Day 1 EOD)
| File | Role |
|---|---|
| `research/sc_hz/phase_h0_harvestable_subset.py` | Step 0 measurement script |
| `research/PHASE_H0_HARVESTABLE_SUBSET_RESULT.md` | output of Step 0; canonical ceiling |
| `research/gates/lever1_parser_sprint.md` | gate doc for parser sprint |
| `research/gates/lever3_exact_arithmetic_lp.md` | gate doc for exact-LP boundary lever |
| `research/gates/lever4_activation_walk_falsifier.md` | gate doc, requires advisor principle audit before sprint |
| `research/gates/lever5_motif_simplification.md` | gate doc for ReLU motif sprint |

### To create (Day 2+, only if Step 0 justifies)
| File | Role |
|---|---|
| `research/sc_hz/exact_arith_lp.py` | rational LP for boundary iids |
| `research/sc_hz/activation_walk_falsifier.py` | Lever 4 implementation |
| `research/sc_hz/relu_motif_detector.py` | Lever 5 graph canonicalization |
| `research/sc_hz/parsers/<op>.py` | Lever 1 missing ONNX op implementations |

---

## 11. Summary of the synthesis

| Synthesis point | What's new |
|---|---|
| 4 distinct diseases, not 1 | Replaces "strong/weak" with mechanism-classified categories |
| Step 0 mandatory before sprint | Replaces "let's try parser sprint" with "measure first" |
| Lever 2 (low-dim input) | New lever I had missed; principle-pure; scales with effective input dim |
| Lever 3 (exact-arith LP) | New lever for F3 boundary cases; rational LP zero principle cost |
| Lever 4 (activation walk FAL) | Barrier-immune lever; needs explicit advisor principle audit on W_eff |
| Lever 5 (motif simplification) | Forward graph canonicalization; principle-pure |
| FC-HZ test → strict=True xfail | Better than @expectedFailure: catches accidental improvement |
| Harvestable ceiling as paper claim | Replaces "1472 frozen" with measurable "we captured all reachable" |

**Bottom line**: 1472 → 1500-1620 in 2-3 weeks is realistic and
principle-compliant. 2000+ requires either Phase H research (months)
or principle relaxation (abandons project thesis).

The single most important action is **Step 0**: measure before sprinting.
