# CIFAR Final-Tail Hull Prototype — Design Plan (Advisor-locked 2026-06-04 evening)

Authorized by: roadmap §7 (NEXT RESEARCH TARGET after ImageHZ closed-negative)
Sentinel selection: `audit_results/cifar_finaltail_hull_sentinels_20260604.json`

This document is the design lock **before any code begins**. It encodes
the exact mechanism, scope, principle invariants, prior-closure caveats,
and hard stop gate. Per the lesson from the §6b denominator burn,
nothing in the prototype deviates from this spec; if a deviation is
required, this file is edited first.

---

## 1. Mechanism — explicit forward LP over (ξ, z, h)

Per advisor 2026-06-04 evening directive. The mechanism is the
**per-neuron forward convex-hull (triangle) LP** applied to CIFAR's
final hidden ReLU, written from the snapshot's mathematical
description with no shortcuts. **No pair-hull cuts in Phase 1.**

### 1.1 Mathematical object

Given the FLATTEN snapshot exposing `(c, Gc)` at the input of the
tail (last Dense), the final hidden ReLU pre-activation is:

```text
z = c_z + G_z · ξ      ξ ∈ [-1, +1]^ng
h = ReLU(z)
y = W_out · h + b_out
```

where `c_z = W_h · c + b_h` and `G_z = W_h · Gc`, with `(W_h, b_h)`
being the last Dense before the hidden ReLU and `(W_out, b_out)`
being the final Dense.

### 1.2 Per-neuron classification

For each output index j of the hidden ReLU:

- **stable active**: `l_j ≥ 0`  →  encode `h_j = z_j` as equality
- **stable inactive**: `u_j ≤ 0`  →  encode `h_j = 0` as equality
- **unstable**: `l_j < 0 < u_j`  →  encode the forward convex hull
  (i.e. the per-neuron triangle):

  ```text
  h_j ≥ 0
  h_j ≥ z_j
  h_j ≤ u_j / (u_j - l_j) · (z_j - l_j)
  ```

This is the **tightest convex relaxation of ReLU on a single
neuron** (`star.pdf` Theorem 1). At the per-neuron level there is no
further hull to tighten.

### 1.3 Per-rival LP

For each rival class `r ≠ y_true`:

```text
maximize     y_r - y_true
subject to   ξ ∈ [-1, +1]^ng
             z = c_z + G_z · ξ
             per-neuron triangle / equality constraints (§1.2)
             y = W_out · h + b_out
```

This is a continuous LP. No CROWN, no backward, no MILP, no BaB,
no random / corner / PGD sampling. Solver: HiGHS via highspy.

### 1.4 Two output classes

After all `n_class - 1` per-rival LPs have been solved on an iid:

- **CERT**: if every rival's LP maximum is `< 0` (strictly), the
  iid is sound certified. Emit a CERT receipt with the LP UB per
  rival, the snapshot path, and the provenance bundle.
- **FAL candidate**: if some rival's LP maximum is `≥ 0`, the LP
  produces a `ξ*` realizing that maximum. Decode `ξ*` back to the
  original input and run **strict ORT replay**. If ORT confirms
  the spec violation under zero-tolerance, emit a FAL receipt.
  **If ORT replay fails for any reason, the iid stays UNKNOWN.**
  A LP-feasible candidate that doesn't realize is NOT a FAL.

The receipt schema mirrors the existing CIFAR endcap receipt format
exactly so the audit harness can ingest it without changes.

---

## 2. Why this is the right baseline (and not pair-hull)

### 2.1 Per-neuron triangle is provably tight

There is no tighter single-neuron convex hull than the 3-facet
triangle. So per-neuron there is no precision lever beyond what
production endcap LP already uses.

### 2.2 The prototype's value is correctness, not new theory

Production CIFAR endcap LP (`pilot_cifar_endcap_diagnose._solve_endcap_lp_with_solution`)
already implements the same mathematics — but it lives in a HyZor-side
pilot file that has gone through multiple revisions and shape-bug
patches. The §7 prototype is a clean-room reimplementation from this
spec, written in `research/cifar_finaltail_hull_lp.py`, that:

- reads the same FLATTEN snapshot,
- builds the same LP from scratch,
- compares LP UB per rival against the production endcap result,
- and emits its own receipt that the audit harness can verify
  independently.

If the clean LP matches production on every sentinel, the §7 stop
gate measures **production's** precision against the spec — and a
FAIL closes this line definitively (production is already at the
mathematical ceiling at the per-neuron level).

### 2.3 Pair-hull is a Phase-2 contingency only

A multi-neuron joint hull on the top-K unstable neurons in the
final hidden ReLU is a strictly tighter relaxation in 2-D (or
higher) than the product of triangles. But Singh PRIMA k=2 / k=3
and Anderson 2020 facets were already closed-negative on acasxu
(memory entries `project_pairwise_hull_negative_20260516`,
`project_triple_hull_negative_20260516`, `project_anderson_facets_negative_20260516`).

The CIFAR final-tail layer is geometrically different from acasxu
(dense `Gemm → ReLU → Gemm` over a SparseGcZ flatten vs sparse
mid-network ReLU), so pair-hull on CIFAR final-tail is NOT a
retread. But it is **strictly Phase-2 contingent**: it only runs
if the Phase-1 clean per-neuron triangle LP returns FAIL on the
20-sentinel gate while showing structural room for improvement
(e.g. high per-neuron coupling on unstable indices), and only after
explicit advisor sign-off.

---

## 3. Invariants

| ID | Invariant |
|---|---|
| I1 | Forward-only. No CROWN, no backward, no gradients. |
| I2 | No MILP, no integer reasoning. Continuous LP only. |
| I3 | No branch-and-bound, no input splitting. |
| I4 | No random / corner / PGD candidates. FAL must pass strict ORT replay. |
| I5 | No CIFAR-specific env knobs. Activation is structural: tail must be `Dense → ReLU → Dense` and final hidden ReLU must have ≥ 1 unstable neuron. |
| I6 | Fail-closed. Any shape or contract violation raises and the iid stays UNKNOWN. |
| I7 | No silent fallback. Production CIFAR endcap LP keeps owning every iid; the prototype writes its own per-iid receipt without modifying production. |
| I8 | Provenance bundle on every receipt: `canonical_root + instances_csv_sha256 + onnx_sha256 + vnnlib_sha256`. |

---

## 4. Scope (Phase 1)

| Component | Action |
|---|---|
| Final hidden ReLU | Per-neuron triangle constraints (§1.2) |
| All other layers | Untouched |
| LP solver | HiGHS via highspy |
| Snapshot consumer | Existing CIFAR endcap FLATTEN snapshots at `audit_results/clean_canonical_sweep_cifar_rerun_20260603T225458Z/cifar100_2024/L*_FLATTEN.pkl` (already on disk; no production rerun) |
| ORT replay | Use the existing strict-replay helper in `receipt_factor_aware_endcap_lp` — read-only consumer; no edits to that file |

What is **NOT** in scope (Phase 1):

- Multi-neuron pair-hull cuts (Phase 2 contingency).
- Witness search beyond decoding the LP `ξ*`.
- Productionization, env knob exposure, routing integration.
- Any change to `pilot_cifar_endcap_*.py`, `cli.py`, or `hz_routing.py`.

---

## 5. Hard stop gate

20 sentinels = `113, 29, 153, 72, 105, 102, 174, 180, 110, 116,
                 168, 75, 133, 92, 165, 86, 137, 15, 82, 93`
(saved at `audit_results/cifar_finaltail_hull_sentinels_20260604.json`).

ALL 20 must complete (no per-iid crash). Then:

```text
PASS iff
    (≥ 3 new V/A across the 20 sentinels)
    OR
    (median LP UB reduction across the 20 sentinels ≥ 30%
                                relative to the production baseline)

FAIL iff
    (0 new V/A)
    AND
    (median LP UB movement < 10%)

INCONCLUSIVE in between → advisor decides whether to escalate to
Phase-2 pair-hull or close anyway.
```

PASS → widen to other top-1-robust benchmarks (Tiny, malbeware)
under a separate gate. Not to CIFAR-only env knobs.

FAIL → close §7 definitively; pivot to §10 paper.

---

## 6. Implementation layout (Phase 1)

```text
research/cifar_finaltail_hull_lp.py     # standalone driver
audit_results/cifar_finaltail_hull_phase1_<STAMP>/
  per_iid/
    iid<NNN>.json         # baseline_lp_min[rival], hull_lp_min[rival],
                          # wall_s, verdict_change, candidate_ort_holds,
                          # full provenance bundle
  smoke_summary.json      # 3-iid smoke vs production baseline
  gate.json               # final §7 gate evaluation
```

The driver:

1. Loads the snapshot at `L0XX_FLATTEN.pkl`.
2. Loads the final two Gemms + the final ReLU from the ONNX graph.
3. For each sentinel iid:
   a. Resolves the rival set from the vnnlib query.
   b. Computes per-neuron `l_j, u_j` from the snapshot.
   c. Builds and solves the per-rival LP (§1.3).
   d. If any rival LP max ≥ 0, decodes `ξ*` to input, runs strict
      ORT replay; otherwise emits CERT.
   e. Writes the per-iid JSON.
4. Compares against the production baseline (read from the same
   sweep's per_instance.json) and writes `gate.json`.

The driver reads the existing `iidNNN_provenance.json` files for the
provenance bundle and writes them to every emitted receipt.

---

## 7. 3-iid smoke (entry condition for the 20-sentinel run)

Before running on 20 sentinels:

- Smoke on iids **113, 29, 153** (the 3 lowest LP-margin sentinels).
- Per-iid pass condition: the clean LP reproduces the production
  baseline LP UB within `|Δ| ≤ 1e-6` per rival (numerical-precision
  parity), AND the FAL candidate decoding stays sound (no shape
  errors, no silent fallback).
- If parity holds: production LP is at the spec ceiling. The 20-iid
  gate will measure precisely what the production LP already
  captures.
- If the clean LP is **strictly tighter** (parity violated in our
  favor): production has an unintended approximation. The 20-iid
  gate will show that as LP UB reduction.
- If the clean LP is **strictly looser** (parity violated against us):
  the prototype has a bug. Fix before the 20-iid run.

---

## 8. Receipt schema (Phase 1)

```json
{
  "iid": 113,
  "production_lp_baseline_per_rival": {"0": 0.5829, "...": "..."},
  "hull_lp_per_rival": {"0": 0.5829, "...": "..."},
  "production_verdict": "UNKNOWN",
  "hull_verdict": "UNKNOWN" | "CERT" | "FAL",
  "lp_ub_reduction_pct_per_rival": {"0": 0.0, "...": "..."},
  "median_lp_ub_reduction_pct": 0.0,
  "wall_s": 0.0,
  "candidate_ort_holds": true | false | null,
  "candidate_xi_star_sha256": "...",
  "canonical_root": "/data1/Kane/data/...",
  "instances_csv_sha256": "...",
  "onnx_sha256": "...",
  "vnnlib_sha256": "..."
}
```

The schema is open for one revision **only if** the smoke run shows
a measurement gap. Anything else is a §1-2 design change and goes
through the lock first.

---

## 9. Audit trail

- Sentinel selection: `audit_results/cifar_finaltail_hull_sentinels_20260604.json`
- Production endcap LP reference: `/data1/Kane/HyZor/pilot_cifar_endcap_diagnose.py::_solve_endcap_lp_with_solution`
- Source atlas: `audit_results/cifar_unknown_margin_atlas_canonical_20260603T121947Z/`
- Snapshot dir: `audit_results/clean_canonical_sweep_cifar_rerun_20260603T225458Z/cifar100_2024/`
- Prior closures (memory): `project_pairwise_hull_negative_20260516`,
  `project_anderson_facets_negative_20260516`,
  `project_triple_hull_negative_20260516`.

This plan supersedes the prior optional §7b text in the roadmap and
the original §7 (now §7-Hist). It is the authoritative scope for §7.
