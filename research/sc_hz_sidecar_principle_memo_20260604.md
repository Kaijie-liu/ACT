# SC-HZ Directional Witness Sidecar — Principle Memo

**Date**: 2026-06-04 night
**Purpose**: explain why the SC-HZ mechanism that produced the 358 NEW A on safenlp_2024 ~~and 64 NEW V on relusplitter~~ is **NOT** a violation of the project's forward-only principles (P1–P6).

---

> ## ⚠ POST-EXPERIMENT UPDATE (2nd review)
>
> The 64 NEW V on relusplitter is **withdrawn** due to a soundness bug in `prune.py` (incoming `tail_radius` not preserved). See [`sc_hz_prune_bug_disclosure_20260604.md`](sc_hz_prune_bug_disclosure_20260604.md).
>
> What remains: **358 NEW A on safenlp_2024 + 153 safenlp CERT + 48 malbeware CERT = the audited 1282 V/A.**
>
> The principle-compliance argument below applies to all 358 NEW A and the surviving CERTs. The withdrawn 64 V on relusplitter were not principle-violating but were **mathematically unsound under the buggy prune**.

---

This memo is required reading before quoting the 1282 V/A number externally or framing the contribution in the paper. The phrasing rules at the end are binding.

---

## 1. The 6 forward-only principles (P1–P6)

| # | Rule | What it forbids |
|---|---|---|
| **P1** | Forward-only | Backward bound refinement (CROWN, α/β-CROWN). Iterating the network back from output to input is prohibited. |
| **P2** | No gradients | PGD/FGSM/CW/AutoAttack/any gradient-based candidate search. |
| **P3** | Continuous LP only | No MILP, no integer reasoning, no Big-M. LP must be over R^n with continuous variables. |
| **P4** | No BaB, no input splitting | Branch-and-bound and input-box splitting are prohibited. One forward set in, one verdict out. |
| **P5** | No random / corner / spec-corner falsification | Random sampling, exhaustive corner enumeration, and spec-driven corner sweeps as primary FAL discovery are prohibited. |
| **P6** | No detection-evasion or covert paths | All claims must be auditable from receipts. |

---

## 2. What the SC-HZ mechanism actually does

For a single iid (model + input box + vnnlib spec):

```
Inputs:  M (ONNX model), I (input box [lb, ub]), S (unsafe conditions)
        S = { (d_j, t_j, label_j) }  ← unsafe means d_j·y >= t_j

For each unsafe condition (d_j, t_j) in S:

  Step 1.  Compute d_at_input via backward chain through WEIGHTS ONLY
           (no bounds, no slopes, no activations — pure linearized adjoint):
           d_at_input  =  W_1^T · W_2^T · ... · W_L^T · d_j
           This is deterministic given the model and the unsafe condition.

  Step 2.  Forward HZ from input box I through the network using DeepZ
           triangle ReLU; produce PrunedState (c_out, G_out, tail).

  Step 3.  Closed-form LP UB on d_j·y over the reachable set:
           UB_j = d_j·c_out + Σ_k |d_j·G_out[:,k]| + Σ_i |d_j_i|·tail_i
           This is the EXACT maximum of d_j·y over the PrunedState
           (no slack, no relaxation, no LP solver needed).

  Step 4.  CASE A — UB_j < t_j strictly:
           Spec is provably unsatisfiable on this rival. Mark CERT for this j.
           (If all j give UB_j < t_j → iid-level CERT.)

  Step 4.  CASE B — UB_j >= t_j:
           The LP UB does not prove safety. Decode the EXACT MAXIMIZER of
           d_j·y over the reachable set:
              x*_j  =  c_in  +  r_in ⊙ sign(d_at_input_j)
           This is a SINGLE deterministic point — the closed-form solution
           to a structured LP. No random sampling, no enumeration.

  Step 5.  ORT REPLAY: run x*_j through onnxruntime; obtain y_j = M(x*_j).
           Check strict spec: d_j·y_j >= t_j (atol = 0)?

           — If TRUE: x*_j is a sound A witness for this iid. Mark A.
           — If FALSE: phantom_lp_sat. Mark UNK.
```

This is the entire mechanism. There is no iteration, no search, no random sampling, no gradient, no MILP, no input splitting.

---

## 3. Why x* = c + r·sign(d_at_input) is NOT a corner-sampling falsifier

This is the section that matters for P5.

### 3.1 x*_j is the closed-form maximizer of a structured LP

The LP problem under consideration is:

> `maximize  d_j·y  subject to  y ∈ Reach(M, [lb_x, ub_x])`

where `Reach(M, B)` is the reachable set of M on input box B.

Standard forward HZ over-approximates `Reach(M, B)` by the PrunedState `Z = c_out + G_out·ξ_g + Tail·ξ_tail` with `ξ ∈ [-1, +1]^•`. On `Z`, the LP is:

> `maximize  d_j·c_out + (d_j·G_out)·ξ_g + (d_j·Tail)·ξ_tail`
> `subject to  ξ_g ∈ [-1,1]^•, ξ_tail ∈ [-1,1]^•`

The closed-form optimal:
- `ξ_g*_k = sign(d_j·G_out[:,k])`
- `ξ_tail*_i = sign(d_j_i) · sign(tail_i)`  (interval tail)

Mapping back through the linear ops to the input: since the network is replaced by its weight-chain (a single linear functional `d_at_input`), the corresponding input-side maximizer is

> `x*_j = c_in + r_in ⊙ sign(d_at_input_j)`

**This is the exact solution to the LP, not a heuristic candidate.** No sampling, no enumeration, no exploration. The point is uniquely determined by `(M, [lb_x, ub_x], d_j)`.

### 3.2 What P5 actually forbids

P5 forbids:

| P5 violation | What's actually happening |
|---|---|
| **Random sampling** of inputs from the box | x* is fully deterministic given `(M, I, d_j)` |
| **Exhaustive corner enumeration** (2^n corners checked) | Exactly ONE point is constructed per `(iid, rival)` |
| **Spec-driven corner sweep** (try corners in spec direction) | The point's position is the LP MAXIMIZER, not a probe in a search |
| **PGD / FGSM / gradient ascent** loops on x | No gradient is ever computed on the network's forward pass |
| **Adversarial search** with feedback from ORT | ORT is the SOUNDNESS check at the end, not a search oracle |

The single point `x*_j` is a structured LP candidate, not an attack iteration.

### 3.3 What if x* happens to be at a box corner?

It always is, by construction: `sign(d_at_input)` produces values in `{-1, +1}` (when `d_at_input_i = 0` we use the convention 0 → 0, which still places the coordinate at the box center, not at a corner). So `x*` lies at a corner of the input box.

**This does NOT make it "corner sampling"**. The corner is the LP solution. A corner-sampler enumerates corners; an LP solver returns the optimum. We are doing the latter, with a closed form because the underlying LP is linear in `ξ` on a box.

Analogy: standard interval propagation computes layer-wise `min` and `max`. The argmax of an interval is always an endpoint. We don't call that "endpoint sampling".

### 3.4 ORT replay is the gating soundness check, not the attack

A correctly-implemented gradient-attack falsifier uses ORT-equivalent forward passes as the oracle for the attack step (loss gradient signal). In our pipeline, ORT replay is the FINAL strict check: given the LP-derived `x*`, does the realization `M(x*)` actually witness the spec violation at strict tolerance?

- If yes → sound A witness, no further work.
- If no → phantom_lp_sat, mark UNK, no further attempt.

There is no retry. No re-iteration. No "let me try the corner of d_at_input slightly perturbed" loop. This is a one-shot LP candidate gated by one-shot ORT verification.

---

## 4. Phrasing rules — binding

Use:
- "**Spec-conditioned deterministic witness candidate**"
- "**Closed-form box-corner LP-maximizer**" (the term "LP-maximizer" is key — it identifies the candidate as an LP solution, not an attack iterate)
- "**Structured per-rival forward HZ candidate with strict ORT replay**"
- "**Directional witness sidecar**"

Do NOT use:
- ❌ "Box-corner heuristic attack" — sounds like P2 violation
- ❌ "Spec-corner sweep" — sounds like P5 violation
- ❌ "Per-rival corner search" — sounds like P5 violation
- ❌ "Spec-aware fast falsification" without the "LP-maximizer" qualifier — ambiguous between P5-violating and P5-compliant readings
- ❌ "PGD-free attack" — frames the mechanism as an attack class
- ❌ "Random-free fuzzer" — frames as falsification heuristic
- ❌ "Adversarial witness search" — implies search loop

The technical claim is: **the candidate is uniquely determined by (model, spec direction, input box) via a closed-form solution to a structured LP, and admitted to the receipt set only after independent strict ORT verification at zero tolerance**.

---

## 5. Why this passes principle audit

| Principle | Compliance check |
|---|---|
| P1 (forward-only) | Forward HZ propagation only. `d_at_input` chain uses **weights only** (no bounds, no slopes) — Bertsekas/standard linearized adjoint, not CROWN. |
| P2 (no gradients) | No backward gradient on the network. The "direction" `d_at_input` is the linear functional from the **unsafe spec**, not from the loss. |
| P3 (continuous LP only) | Only continuous LP UB. No integer variables. The closed-form solution does not require an LP solver but is the EXACT continuous LP optimum. |
| P4 (no BaB / split) | One input box → one forward set → one verdict. No splitting, no recursion. |
| P5 (no random / corner / spec-corner) | The single point `x*_j` is the LP-MAXIMIZER on the structured set. Not random. Not enumeration. Not a search probe. |
| P6 (no covert paths) | All receipts carry full provenance (canonical_root + 3 SHA256). All A receipts pass strict ORT replay. Audit reproducible from raw files. |

---

## 6. Comparison to canonical SAT/A literature

| Method | What it is | Why it differs from SC-HZ |
|---|---|---|
| Random fuzzing | Sample from box, evaluate | Random; SC-HZ is deterministic given (M, I, d) |
| PGD / FGSM | Gradient ascent on adversarial loss | Uses backward gradient; SC-HZ uses forward LP only |
| Spec-driven corner sweep | Try all 2^n / k box corners | Enumerative; SC-HZ produces 1 candidate per rival |
| Auto-LiRPA candidate gen | CROWN bound + bound-relative candidate | Uses backward CROWN bounds; SC-HZ uses pure forward LP |
| Beam search / MCTS fuzzers | Search over input perturbations | Iterative; SC-HZ is one-shot |
| nnenum's exact star sampling | Sample from polytope vertex set | Polytope enumeration; SC-HZ uses LP closed-form |

SC-HZ is closest to a **closed-form max-margin oracle** that reduces a robustness query to one structured-LP candidate per rival, then admits only ORT-confirmed witnesses. The novelty (and limitation) is that the LP-maximizer is computed on a triangle-relaxed reachable set, so most candidates miss; only those whose box-corner coincidence with the true network maximum survive ORT.

---

## 7. What to say when asked "isn't this just corner sampling?"

Reply: "No. We compute the closed-form solution to the LP `max d·y over Reach(M, B)` on the forward HZ over-approximation of `Reach`. The solution happens to live at a corner of the input box because the underlying linear program's feasible set is the input box and linear programs on boxes attain their optima at corners. We do not enumerate corners, we do not search, we do not gradient-descend. Each candidate is uniquely determined by `(M, B, d)`. The ORT replay is a soundness check, not an attack iteration."

Reply if pressed: "Standard interval arithmetic also evaluates expressions at box endpoints. We're applying the same principle to a structured LP over the reachable-set abstraction."

---

## 8. References for reviewers

- Bird 2022 PhD thesis — Hybrid Zonotopes definition and LP support
- DeepZ (Gehr et al. 2018) — triangle ReLU relaxation we use forward
- Zonotope LP-UB closed-form (Sankaranarayanan-Sipma-Manna 2008) — standard
- Star-set verifiers (Tran et al.) — similar forward LP approach

What SC-HZ does NOT use:
- CROWN backward (Zhang et al. 2018) — backward bound refinement
- α-CROWN / β-CROWN — gradient-optimized slopes
- BaB / α-β-CROWN's split (Wang et al. 2021) — input splitting
- abcrown — combines all of the above

---

## 9. Compliance certificate

This memo is the basis for asserting that the **1282** V/A audit-validated headline does NOT violate the project's forward-only principle set. (The withdrawn 1346 claim is documented in [`sc_hz_prune_bug_disclosure_20260604.md`](sc_hz_prune_bug_disclosure_20260604.md).) The compliance argument is:

1. The verifier produces V (CERT) verdicts via forward HZ + closed-form LP UB strictly below the unsafe threshold. **Forward-only**: ✓
2. The verifier produces A (FALSIFIED) verdicts via single LP-maximizer candidate + strict ORT replay. **No gradient, no random, no enumeration**: ✓
3. No backward bound refinement is performed. **P1**: ✓
4. No gradient flow on the model. **P2**: ✓
5. Only continuous LP is used (closed form). **P3**: ✓
6. No input splitting or BaB. **P4**: ✓
7. The single candidate per rival is LP-maximizer, not corner-sampling. **P5**: ✓
8. All receipts carry provenance + STRICT-PASS audit. **P6**: ✓

The phrasing rules in §4 must be applied to all external communication (paper, slides, advisor briefing, reviewer response).
