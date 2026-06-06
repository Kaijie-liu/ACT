# Phase H2 — New Forward Abstraction Design (Towards 2000+)

**Date**: 2026-06-06 (after Lever 1 sprint +60 candidate → **audited +20**,
1472 → **1492** per `SPRINT_AUDIT_RESULT_20260606.md`. collins_rul +39 was
100% double-counted vs r93 CERTIFIED baseline.)
**Plan reference**: `research/FORWARD_PLAN_principle_internal_levers_20260606.md`
**Step 0 reference**: `research/PHASE_H0_HARVESTABLE_SUBSET_RESULT_20260606.md`
**Sprint result**: `research/SPRINT_RESULTS_principle_internal_levers_20260606.md`

**Status**: DESIGN DOC. Not yet a build plan. Frames 4 candidate
forward abstractions, each with a toy gate and a cifar 113 gate, so
that we can decide which (if any) to invest engineering time in.

**Principle compliance**: every candidate respects P1-P5 by
construction. The bar for "principle-internal" is set higher than
Phase F: no single-layer cuts, no per-neuron triangle additions, no
backward iteration. Each candidate must demonstrate it captures
AGGREGATE block-level slack diffusion, not local per-pair correlation.

---

## 1. Why Phase H2 exists

Phase F (F1/F2b/FC-HZ) explored cuts at the SAME LP level as DeepZ
triangle and proved that local extensions yield ≤25-30% total drop.
Phase H1 (parser sprint) recovered ~60 V/A from engineering plumbing,
hitting the low end of the Step 0 forecast.

**The remaining ~468 V/A gap to 2000+ is structurally locked**:
- 735 robust_blocked (Step 0): no current-pipeline mechanism reaches them
- Concentrated in cifar100 (200), tinyimagenet (200), yolo (72),
   traffic_signs (45), tllverifybench (29), relusplitter (85), vgg (4)
- The shared property: DENSE-CONV CERT requires aggregate slack
   control, not local cuts

Phase H2 explores 4 new forward abstractions intended to capture
aggregate slack. Each is principle-pure; each comes with hard gates
to prevent endless extension.

---

## 2. The 4 candidate abstractions

### Candidate A — Output-Projected Constrained Forward Domain (OPC-FD)

**Idea**: instead of propagating a full HZ through the conv body and
projecting at the output, project EARLY and propagate a constrained
SUBSPACE of the HZ.

For each unsafe rival r, precompute the OUTPUT-direction `d_r =
W_out[r] - W_out[y_true]`. At each conv block L, instead of carrying
full state (n_L dim), carry:
- `q_L = T_L · h_L` where `T_L` is the basis of "directions relevant
   to rival r" derived from the FORWARD propagation of d_r alone (no
   bound info, no gradient).
- A residual interval `r_L` capturing what was projected away.

The constraint `T_L · h_L = q_L` is maintained as a linear constraint
in the LP at the output. The looseness then concentrates on the
PROJECTED dimensions, which are 1-3 instead of n_L (~512).

**Mechanism for breaking the barrier**: aggregate slack across the
1000+ unstable neurons at the final dense layer is dispersed across
the n_L dim, but the rival margin only sees the projection `d_r · h_L`.
By forcing the LP to ONLY freely use the projected directions (with
the rest constrained to interval), we cap the aggregate slack.

**Principle compliance**:
- P1 forward-only: T_L is computed from forward propagation of d_r
   ONLY (no bound info from layer > L)
- P2 no gradient: T_L = forward-prop of identity-skeleton (NO autograd)
- P3 continuous LP: HiGHS at output
- P4 no BaB / no split: single forward, single LP
- P5 no random / corner: T_L is deterministic from architecture + d_r

**Risk**: T_L is the same conceptual quantity as PRUNE's d_L^r, which
was empirically falsified on dense networks. The DIFFERENCE here is
that we use T_L as a SUBSPACE BASIS (multiple directions), not a
single direction, AND we RETAIN the residual interval as a sound
absorbing bucket.

**Toy gate**:
- 2-block ResNet toy with strongly aggregate slack
- DeepZ closed-form: loose by ≥3×
- F1 LP: loose by ≥2×
- OPC-FD: tight to within 20% of exact

**cifar 113 gate**:
- excess +0.146 → ≤ +0.05 OR CERT directly

---

### Candidate B — Residual-Block Template HZ (RB-T)

**Idea**: dense-conv networks are structured as residual blocks. For
each ResNet block, encode a SMALL number of fixed-direction template
constraints that capture block-level invariants:

1. `skip_path_output = block_input` (identity)
2. `block_output = skip + branch_output`
3. `sum_over_block_output_channels ≥ 0` (post-ReLU invariant)
4. `signed_channel_sum_in_direction_d ≤ block_input_norm × scale`
   (for d chosen from the rival direction's forward projection)

The block's reachable set is over-approximated by the intersection
of these template constraints + the HZ over generator coordinates.

**Mechanism for breaking the barrier**: cifar/tiny ReLU slack
accumulates across blocks. Adding block-level aggregate constraints
(sum of channel slack ≤ bound) directly attacks the diffusion.

**Principle compliance**:
- Same as Candidate A
- Templates are FIXED per architecture (not optimized per instance)

**Risk**: template count grows with depth. For a 10-block ResNet,
~30-40 template constraints per instance, ~3 per block. LP becomes
larger but manageable.

**Toy gate**:
- 2-block toy with explicit channel correlation
- F1 LP: loose
- RB-T: at least 40% tighter than F1

**cifar 113 gate**:
- Same as Candidate A

---

### Candidate C — Forward Constrained Zonotope with Retained Ac (FCZ-Ac)

**Idea**: don't pre-discard the equality constraint from earlier
ReLUs. Carry a SMALL number of these forward as polyhedral side
constraints in the output LP.

Existing pipeline: each ReLU layer adds slack auxiliary `s_i` with
constraint `y_i = lam_i z_i + mu_i + mu_i s_i, s_i ∈ [-1, +1]`. The
auxiliary aux is treated as independent at output LP.

FCZ-Ac: instead of independent aux, retain the IMPLICIT CONSTRAINT
`y_i ≥ z_i` (from ReLU). At output LP, this becomes:
`mu_i * (1 + s_i) ≥ (1 - lam_i) z_i` for each retained constraint.

But: doing this for ALL unstable neurons is expensive. The KEY is
TO PICK which subsets to retain. Forward heuristic: retain
constraints for neurons in the LAST 2-3 layers (post-final-block).

This is EXACTLY what F1 does for last layer, FC-HZ extended to all
layers, but FCZ-Ac would CHOOSE per-iid which subset to retain based
on forward LP-UB gap analysis (which neurons are diffuse contributors
vs concentrated).

**Mechanism for breaking the barrier**: forward-aware selective
constraint retention. Not all neurons need constraints; only the
"top loose" ones.

**Principle compliance**:
- The CHOICE of which constraints to retain uses LP UB gap data
  computed at THIS layer's forward pass (no later bound info).
- Selection is forward-deterministic; no backward, no gradient.

**Risk**: this is FC-HZ with smarter selection. FC-HZ gave 8% median
additional. Smarter selection might give 15-25%. Probably NOT enough
to flip cifar 113 (~17% needed).

**Toy gate**:
- F1 + FC-HZ: 8% additional
- FCZ-Ac on same toy: ≥25% additional

**cifar 113 gate**:
- Same

---

### Candidate D — Small Exact Tail Projected Hull (SETPH)

**Idea**: for the LAST 1-3 dense layers, the dimension is small
enough (~100-200) that we can compute the EXACT projected convex
hull of the rival margin for a FIXED activation pattern. We don't
do BaB / case split, but we do EXACT inside the last layers.

Specifically: at layer L (dense, ~200 neurons, last hidden), instead
of triangle relaxation we compute the EXACT polyhedral envelope of
`{(h_L, d_r · y_L) : h_L ∈ reachable_set_at_L}` using a small LP
per rival.

This is forward, continuous, no integer. The "exact" here refers to
the exact convex hull, computable in polynomial time because of the
small layer dimension at the tail.

**Mechanism for breaking the barrier**: cifar/tiny conv body has
many layers; but the LAST 1-3 dense layers are small enough for
exact projected hulls. The aggregate diffusion gets absorbed
earlier, and the tail does the projection cleanly.

**Principle compliance**:
- Polyhedral hull computation is continuous LP-based
- No BaB (no case split on activation)
- Triangle relaxation kept for conv body; only tail is exact

**Risk**: exact hull computation can be exponential in n. For n=100
it's still polynomial but slow. May not scale to all benches.

**Toy gate**:
- 2-block toy with 50-neuron tail
- F1 LP: loose
- SETPH: matches exact to within 5%

**cifar 113 gate**:
- Same. SETPH applied to last 2 dense layers.

---

## 3. Cross-candidate comparison

| Candidate | Principle compliance | Estimated F1 gap closing | Implementation effort | Generalization risk |
|---|---|---|---|---|
| A (OPC-FD) | strong | high (theoretically full barrier break) | 5-7 days | medium (PRUNE-like risk) |
| B (RB-T) | strong | high if templates well-chosen | 4-6 days | medium (architecture-specific) |
| C (FCZ-Ac) | strong | medium (extension of FC-HZ) | 3-5 days | LOW gain expected |
| D (SETPH) | strong | high on tail | 6-8 days | low (works where tail is small) |

**Ranking by likely-to-break-cifar-barrier**:
1. **A (OPC-FD)**: most promising; directly attacks projection-mismatch
2. **D (SETPH)**: most promising for benches with small tail dim
3. **B (RB-T)**: promising for ResNet structure
4. **C (FCZ-Ac)**: probably not enough (already extends F1/FC-HZ which failed)

---

## 4. Hard gates (binding before any sprint)

Each candidate MUST pass these before scaling beyond toy:

### Gate Z0 — Toy aggregate slack benchmark
- 2-block toy network with 32-64 neurons per block
- Aggregate slack accumulation (each block adds ~0.1 to LP UB)
- DeepZ: loose by ≥2×
- F1 LP: ≥+50% better than DeepZ
- Candidate must be ≥+50% better than F1 LP

If candidate gives <30% improvement over F1 LP on this toy, the
mechanism is fundamentally insufficient for dense-conv. CLOSE.

### Gate Z1 — cifar 113 worst rival
- Current F1 LP: +0.146 excess
- Candidate must achieve ≤ +0.05 excess
- OR achieve CERT directly

If candidate gives ≥+0.10 excess on cifar 113, CLOSE.

### Gate Z2 — 8-sentinel cifar/tiny dense-conv pilot
- cifar 113, 29, 180, 72, 168, 145 + tiny 99, 30
- Median excess drop ≥60% over F1
- OR ≥1 NEW CERT

If candidate gives <30% drop AND 0 NEW CERT, CLOSE.

### Gate Z3 — 40 sentinel pilot
- 40 dense-conv sentinels
- ≥5 NEW V/A OR median excess drop ≥40%

If candidate gives <2 NEW V/A AND drop < 25%, CLOSE.

### Gate Z4 — Full sweep
- All 22 benches × all UNK iids
- Target ≥100 NEW V/A across at least 3 different bench families
- Otherwise, the candidate is a single-benchmark hack and gets
   demoted to that bench's profile.

---

## 5. 2-month execution proposal

### Week 1: Candidate D (SETPH) — fastest principle check
- Implement small exact projected hull on dense tail
- Run on cersyve, dist_shift, vgg sparse subset
- If Gate Z0 passes on toy and Z2 passes on cifar 113, expand
- If fails: CLOSE D, move to A

### Week 2-3: Candidate A (OPC-FD) — main bet
- Implement output-projected constrained forward domain
- Run on cifar 113 + 8 sentinel
- Expected: hardest engineering, highest potential payoff
- Gate Z0 + Z1 must pass before scaling

### Week 4-5: If A passes Z2, scale to Z3 (40 sentinel)
- Run on cifar/tiny/yolo/traffic subset
- Hard target: ≥5 NEW V/A

### Week 6-8: Candidate B (RB-T) — if A fails or partially
- ResNet-specific templates
- Tests on cifar/tiny only
- Gate Z2 must pass

### Week 9+: Candidate C (FCZ-Ac) — only if A/B/D all fail
- Likely insufficient; included for completeness
- Gate Z0 must pass

### Throughout: maintain Lever 1 audit + Phase H1 cleanup
- Audit the +60 NEW V from Lever 1
- Cross-check dist_shift for double-count
- Cleanup parser fixes that didn't yield V/A (cgan ConvTranspose)

---

## 6. What we do NOT do in Phase H2

- Continue F1/F2b/FC-HZ tightening (empirically closed)
- Single-layer cuts on dense-conv (closure proved insufficient)
- "Lever 4 activation walk" until advisor principle ruling (separate path)
- "rational LP" until solver policy ruling (audit-only)
- Promise 2000+ on a specific timeline (research-grade uncertainty)

---

## 7. Realistic outcome scenarios

| Scenario | Phase H2 result | Final headline |
|---|---|---:|
| OPTIMISTIC | A passes Z3 + B passes; full Phase H2 succeeds | 1750-1900 |
| LIKELY | One of A/B passes Z2 but Z3 marginally; partial gains | 1600-1750 |
| PESSIMISTIC | All 4 candidates fail Z1/Z2; barrier holds | 1532 (current Lever 1) |
| WORST | New abstraction introduces soundness bug | <1472 (rollback) |

**Median expectation**: 1600-1750 by month-2. 2000+ is NOT included
because we have NO evidence that any of these candidates breaks the
barrier on real cifar at the magnitude needed.

---

## 8. Decision required before Phase H2 commits

1. Which candidate to start with: **A (OPC-FD)** or **D (SETPH)** first?
   - A is the bigger bet; D is the faster principle check
2. Lever 4 activation walker: principle ruling on `W_eff`?
   - Without this ruling, Lever 4 stays paused
3. Phase H1 audit timing: before or in parallel with Phase H2 W1?
   - Audit gives credibility to the +60; H2 design is independent

---

## 9. Files referenced

| File | Status |
|---|---|
| `research/INNOVATION_BRIEF_sc_hz_20260604.md` | Original brief (PRUNE hypothesis, falsified) |
| `research/RETROSPECTIVE_sc_hz_what_did_not_work_20260606.md` | Phase F closure evidence |
| `research/PHASE_H0_HARVESTABLE_SUBSET_RESULT_20260606.md` | Step 0 measurement |
| `research/SPRINT_RESULTS_principle_internal_levers_20260606.md` | Lever 1 sprint result |
| `research/FORWARD_PLAN_principle_internal_levers_20260606.md` | Forward plan |
| `research/PHASE_H2_new_abstraction_design_20260606.md` | this design doc |

---

## 10. Bottom line

Phase H2 is a 2-month research investment with the explicit aim of
breaking the dense-conv barrier within the principle set. It has 4
clean candidates, each with hard gates. The realistic outcome is
+100-250 NEW V/A (taking 1532 → 1600-1750). 2000+ is NOT a credible
target without further conceptual advances OR principle relaxation,
neither of which is on the table.

Phase H2 should start with Candidate D (SETPH, 1 week) as the fastest
principle check, then Candidate A (OPC-FD, 2-3 weeks) as the main bet.
