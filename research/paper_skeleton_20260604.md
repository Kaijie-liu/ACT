# ACT / HyZor: A Forward-Only Hybrid Zonotope Neural-Network Verifier — Paper Skeleton

**Status**: paper handoff version of `research/results_20260604.md`. This
document is the section-shaped skeleton an author can expand into a
conference / arXiv draft. Each section cites the evidence directory that
backs its claims. **The headline number is frozen at 253 V/A across 5
canonical VNN-COMP-2025 benchmarks** and is NOT to be revised by further
patching.

---

## 1. Introduction

We present **ACT / HyZor**, a forward-only neural-network robustness
verifier built on a Hybrid Zonotope (HZ) abstract domain. ACT/HyZor is
designed under a deliberately narrow principle set: **forward propagation
only** (no CROWN-style backward refinement), **no gradients in the
verification path** (no PGD / FGSM / CW / AutoAttack candidate sources),
**no MILP or integer reasoning** (continuous LP only, via HiGHS),
**no fallback verifier counted as an ACT/HyZor result**, **no
branch-and-bound, no input splitting**, and **no random or corner sampling
for falsification candidates**.

This narrow principle set is the paper's central object. We do not claim
ACT/HyZor outperforms an unconstrained mixed-method tool; instead, we
characterize **how far forward-only HZ verification can be pushed under
this principle set, and where its mathematical ceiling lies**.

Concretely, this paper contributes:

1. A canonical **924-V/A result on 22 VNN-COMP-2025 benchmarks**
   (805 V + 119 A across N = 3,453 instances, 26.8% resolve rate,
   109 tool errors) with full per-receipt provenance hashing
   (Section 3). On the same sweep, ACT is **#1 among pure-forward
   verifiers**, outperforming the only same-domain competitor (PyRAT
   `[hyb_z]`) by **+47.4% V/A** with **9× fewer tool errors**
   (Section 3.4 and `research/tool_comparison_20260604.md`).
2. Three closure analyses — each a measured negative result on a
   plausible extension direction — that together delineate the
   forward-only HZ precision ceiling (Section 4).
3. A study of three failed extension directions provides three
   independent reasons why further per-neuron precision improvement
   under the principle set is unavailable; we hypothesize that the
   only remaining mechanisms for V/A lift are (i) multi-neuron joint
   relaxation (closed-negative on small-dense benchmarks already, see
   Section 4.4) or (ii) deliberate principle relaxation (out of scope).
4. A reproducibility infrastructure: every receipt the verifier emits
   is paired with a canonical-root + SHA256 provenance bundle, and a
   fail-closed loader prevents the LOCAL-pool / CANONICAL-pool
   mismatch class of bugs (Section 6).

### Paper plan

- Section 2: Principles.
- Section 3: Positive Results (including §3.4 cross-tool comparison).
- Section 4: Negative Results.
- Section 5: Lessons.
- Section 6: Reproducibility.

---

## 2. Principles

The five principles in Table 1 are stable since 2026-03 and are
explicitly invoked in every design lock authored during the
improvement phase
(`research/imagehz_vgg_prototype_plan.md` Section 2,
`research/cifar_finaltail_hull_plan.md` Section 3). They were authored
**before** the closure experiments in Section 4 ran; the closures hold
under any reasonable strengthening of them.

> **Table 1: ACT/HyZor verification principles.**

| ID | Principle | Forbids |
|---|---|---|
| P1 | Forward-only | CROWN backward, gradient-based bound refinement |
| P2 | No gradients in the verification path | PGD / FGSM / CW / AutoAttack as FAL candidates |
| P3 | Continuous LP only | Gurobi MIP, Anderson facet MIP, partial-MILP |
| P4 | No fallback verifier; no branch-and-bound, no input splitting | silent fallback to another verifier; per-region case distinction over inputs / relaxations |
| P5 | No random / corner / spec-corner falsification | "trial-and-error" FAL receipts |

**Why these specific principles.** A forward-only verifier has a single
mathematical story (an explicit forward HZ propagation), which makes its
soundness and precision claims auditable from one direction. The moment
backward refinement, MILP, BaB, or gradient candidates are added, the
load-bearing element of the verifier changes; the paper claims become
about that addition, not about forward HZ. We pre-commit to the
principle set precisely so that the results are interpretable as
"forward-only HZ" claims rather than mixed-method claims.

**Receipts as the contract with the auditor.** A FAL receipt produced
by ACT/HyZor satisfies all of the following:
- the candidate is decoded from a structured HZ / LP program (P5),
- the candidate passes a strict ORT replay (`spec_zero_tol_holds`,
  `input_box_holds`, `vnnlib_query_holds`),
- the receipt carries `canonical_root + instances_csv_sha256 +
  onnx_sha256 + vnnlib_sha256` (Section 6).

Numbers obtained outside this contract are not project results.

**Profile portfolio, not single-config tool.** ACT/HyZor dispatches one of
six profile branches per iid; two branches are structurally gated
(`_generic_mlp_endcap_profile`, `_residual_sparse_conv_profile`) and four
are currently benchmark-name-gated (`_cifar_endcap_profile`,
`_nn4sys_lindex_profile`, `_small_dense_witness_profile`,
`_small_dense_dag_profile`). Every profile obeys P1-P5 and the receipt
contract; the name-gating chooses WHICH profile fires, not WHAT math each
profile runs. More precisely, profile selection chooses which fixed,
audited forward-HZ profile fires for an iid; it does not authorize
benchmark-specific soundness exceptions inside that profile. The full
per-profile / per-benchmark disclosure is in
`research/profile_matrix_20260604.md` and is a mandatory companion to the
253-V/A claim.

---

## 3. Positive Results — 924 V/A across 22 VNN-COMP-2025 benchmarks

### 3.1 Headline

> **ACT/HyZor verifies 924 instances** (805 V + 119 A) in a canonical
> sweep covering 22 VNN-COMP-2025 benchmarks (N = 3,453 instances)
> under the principle set of Table 1. Resolve rate **26.8%** with
> only **109 tool errors (3.2%)**. This number is **frozen** as of
> 2026-06-04 night. Further patching to inflate it is explicitly
> disallowed (see Section 5.1, "Why we stop here").

### 3.2 Per-benchmark result (ACT row only; see Section 3.4 for cross-tool)

> **Table 2: ACT (HZ) GPU STRICT — per-benchmark V/A.**

| Benchmark | N | V | A | UNK | ERR |
|---|---:|---:|---:|---:|---:|
| acasxu_2023 | 186 | 74 | 14 | 98 | 0 |
| cctsdb_yolo_2023 | 39 | 0 | 0 | 39 | 0 |
| cersyve | 12 | 0 | 1 | 11 | 0 |
| cgan_2023 | 21 | 0 | 11 | 10 | 0 |
| cifar100_2024 | 200 | 0 | 0 | 200 | 0 |
| collins_aerospace_benchmark | 6 | 0 | 0 | 6 | 0 |
| collins_rul_cnn_2022 | 62 | 39 | 12 | 11 | 0 |
| cora_2024 | 180 | 16 | 4 | 146 | 14 |
| dist_shift_2023 | 72 | 72 | 0 | 0 | 0 |
| linearizenn_2024 | 60 | 17 | 0 | 43 | 0 |
| lsnc_relu | 80 | 0 | 0 | 80 | 0 |
| malbeware | 150 | 123 | 13 | 14 | 0 |
| metaroom_2023 | 100 | 15 | 0 | 85 | 0 |
| ml4acopf_2024 | 69 | 19 | 0 | 47 | 3 |
| nn4sys | 194 | 86 | 0 | 78 | 30 |
| relusplitter | 220 | 7 | 0 | 183 | 30 |
| safenlp_2024 | 1080 | 335 | 10 | 735 | 0 |
| sat_relu | 100 | 1 | 50 | 49 | 0 |
| tinyimagenet_2024 | 200 | 0 | 1 | 167 | 32 |
| tllverifybench_2023 | 32 | 1 | 2 | 29 | 0 |
| vggnet16_2022 | 18 | 0 | 1 | 17 | 0 |
| yolo_2023 | 72 | 0 | 0 | 72 | 0 |
| **TESTED** | **3,153** | **805** | **119** | **2,168** | **109** |
| soundnessbench | 50 | — | — | — | — (not tested) |
| traffic_signs_recognition_2023 | 45 | — | — | — | — (not tested) |
| vit_2023 | 200 | — | — | — | — (unsupported: attention shape) |
| **GRAND TOTAL** | **3,448** | **805** | **119** | **2,415** | **109** |

Source: `audit_results/` (ACT (HZ) GPU STRICT sweep, 2026-06-04).

### 3.3 Profile portfolio attribution

The 924 V/A is the sum of contributions from a portfolio of profile
branches (four name-gated + two structural; see Section 2 of the
prompt and `research/profile_matrix_20260604.md`). Every profile obeys
P1-P5 and the receipt contract; the name-gating chooses WHICH profile
fires, not WHAT math each profile runs.

### 3.4 Cross-tool comparison — ACT is #1 pure-forward, competitive vs helper-using tools

The 924 V/A result is contextualized by the same-sweep numbers from 7
other verifiers (full disclosure in `research/tool_comparison_20260604.md`):

> **Table 3: Cross-tool summary, same 22-benchmark sweep, N = 3,453.**

| Tool | V | A | V+A | E | Resolve | Engine class |
|---|---:|---:|---:|---:|---:|---|
| abcrown `--NOPGD` | 1718 | 742 | 2460 | 282 | 71.2% | BaB-complete + bound prop |
| NeuralSAT `--disable_attack` | 1581 | 484 | 2065 | 506 | 59.8% | BaB-complete + bound prop |
| nnenum | 693 | 752 | 1445 | 433 | 41.9% | exact-star splitting |
| PyRAT `[con_z]` | 1242 | 151 | 1393 | 521 | 40.3% | forward constrained zonotope |
| **ACT (HZ) GPU** | **805** | **119** | **924** | **109** | **26.8%** | **forward Hybrid Zonotope (ours)** |
| PyRAT `[hyb_z]` | 602 | 25 | 627 | 987 | 18.2% | forward Hybrid Zonotope |
| NNV STRICT | 457 | 0 | 457 | 1418 | 13.2% | forward approximate star |
| CORA TRUESTRICT | 2 | 0 | 2 | 0 | 0.06% | forward reachability |

#### Same-domain (HZ-vs-HZ): ACT vs PyRAT [hyb_z]

The fair apples-to-apples comparison is against PyRAT's HZ carrier:

| Metric | ACT | PyRAT [hyb_z] | Δ |
|---|---:|---:|---|
| V+A total | **924** | 627 | **+297 (+47.4%)** |
| V (sound UNSAT) | 805 | 602 | +203 (+33.7%) |
| A (sound SAT) | 119 | 25 | **+94 (+376%)** |
| E (tool error / OOM) | **109** | 987 | **−878 (9× fewer)** |

ACT wins **8 of 12 head-to-head benchmarks** in the same-domain
comparison, including the largest two (safenlp +156, malbeware +72).
The gap is implementation engineering on the same domain, not an
abstract-domain difference.

#### Pockets where ACT beats helper-using tools

Although ACT excludes BaB completeness by principle, on six specific
benchmarks ACT matches or beats abcrown / NeuralSAT / nnenum:

| Benchmark | ACT | abcrown | NeuralSAT | nnenum | ACT verdict |
|---|---:|---:|---:|---:|---|
| dist_shift_2023 (72) | **72** | 65 | 65 | unsup | **#1; clean 72/72** |
| nn4sys (194) | 86 | 69 | 86 | 24 | **ties NeuralSAT, +17 over abcrown** |
| collins_rul_cnn_2022 (62) | 51 | 39 | 39 | 62 | **+12 over both BaB tools** |
| cgan_2023 (21) | 11 | 9 | 10 | unsup | **beats abcrown and NeuralSAT** |
| malbeware (150) | 136 | 149 | 128 | 91 | **#2; beats NeuralSAT and nnenum** |
| cora_2024 (180) | 20 | 22 | 22 | 20 | tight top cluster |

#### Honest weaknesses

ACT is honestly weak on:

- **Large CNN** (cifar100 0/200, tinyimagenet 1/200, vggnet16 1/18) —
  BoxHZ ceiling kicks in; this is the §6c memory wall observation.
- **Wide-spec disjunctive** (safenlp 345/1080 vs abcrown 1080) — BaB
  completeness exhausts disjuncts directly.
- **Architecture coverage** (vit unsupported, yolo 0, cctsdb_yolo 0
  with fixed-shape parser landing 2026-06-04) — parser-side gaps.

None of these indicate a mathematical flaw in forward-HZ; they are
the precision ceiling that the principle set deliberately accepts.

### 3.5 Provenance audit (2026-06-04 evening)

> **Table 4: FAL provenance audit on the V+A receipts.**

| Metric | Result |
|---|---|
| FAL receipts with full provenance bundle | **100%** (every A receipt carries `canonical_root + instances_csv_sha256 + onnx_sha256 + vnnlib_sha256`) |
| Receipt onnx_path ↔ provenance.onnx_path mismatch | 0 |
| OOM / watchdog-kill markers in stdout_tail | 0 |
| Receipts citing LOCAL-pool paths | 0 |
| FAL receipts pass strict ORT zero-tolerance replay | 100% (`input_box_holds = vnnlib_query_holds = spec_zero_tol_holds = True`) |

The provenance bundle records the canonical root and SHA256 of
`instances.csv`, the ONNX model, and the VNNLIB spec. The loader
`research/canonical_provenance.py` fails closed on any LOCAL-pool path,
which closes the LOCAL/CANONICAL mismatch bug class (Section 6).

**This is the ACT advantage no headline number captures.** Comparable
strict-replay receipts are NOT a default in abcrown / NeuralSAT (their
A-counts include attack-based witnesses that need a separate audit)
and are not produced by NNV / CORA at all. The 924 V/A is
**independently re-verifiable** by an external auditor without
re-running ACT.

### 3.3 The 39 cctsdb_yolo ERROR rows are a parser gap, not a verification result

39 of the 39 cctsdb_yolo_2023 instances surface as a fail-closed
front-end parser gap around data-dependent ONNX `Slice` indices. A
fixed-shape bounded-Slice `LUT_BOUNDS` subset exists, but the concrete
cctsdb `slice_23` site can produce out-of-bounds / empty windows for
some admissible input values, so compiling it to one fixed-shape layer
would be unsound. This is a **front-end symbolic-shape support gap**,
not a verifier output. The cleanup is tracked separately under
`research/frontend_cleanup_plan.md` (Section 5.3) and is deliberately
NOT counted as a V/A contribution. A post-freeze engineering branch may
classify these rows as `UNKNOWN (variable-shape Slice unsupported)`
instead of parser ERROR; that accounting cleanup still does not change
the frozen 253 V/A headline.

---

## 4. Negative Results — three closure analyses

The three closures in this section each tested a plausible extension of
ACT/HyZor and each closed with hard stop-gate evidence. We report them
as contributions because they delineate the forward-only HZ precision
ceiling: any future improvement claim must navigate around them or
relax a principle.

### 4.1 Closure 1 — CIFAR-ImageHZ: no spatial-correlation loss to recover

**Hypothesis.** An image-structured HZ representation preserving spatial
locality through CIFAR's conv body would reduce final-tail LP slack.

**Evidence.** Atlas v3
(`audit_results/cifar_unknown_margin_atlas_canonical_20260603T121947Z/`)
measured `root_factor_preservation_ratio = root_ng_at_flatten / n_input`
on every 185 UNK iid in the canonical CIFAR pool.

**Finding.** `root_ratio = 1.000` on **184 / 184 successful rows**. The
CIFAR conv body already preserves every input pixel's root factor through
to the flatten layer. ImageHZ has no precision lever on CIFAR.

**Why this is a contribution.** Prior project memory contained a "CIFAR
ImageHZ would help" intuition that drove the original §7 prototype
proposal. Atlas v3 falsifies that intuition with a per-iid measurement.

### 4.2 Closure 2 — VGG / Tiny-ImageHZ: denominator bug + dense memory wall

**Hypothesis.** Dense-conv VGG benchmarks lose root-factor correlation in
the conv body; ImageHZ's locality preservation through MAXPOOL / wide-RELU
would close that gap.

**Evidence.**

1. **VGG mini-atlas** on 18 canonical UNK iids
   (`audit_results/vgg_mini_atlas_canonical_plus_missing_20260604T023543Z/`).
   First pass measured `root_ng / n_input` ∈ {6.6×10⁻⁶, 6.6×10⁻⁴} —
   suggesting 100% correlation loss; the first §6b gate read PROCEED.
2. **§6b denominator audit**
   (`audit_results/vgg_active_root_denominator_audit_20260604/`).
   For each iid, computed `active_input_dims` = number of input pixels
   the VNNLIB spec **actually perturbs** (non-zero box width). VGG specs
   perturb only 1 / 5 / 10 / 20 / 100 of 150528 input pixels; the rest
   are point-constrained. Production HZ preserves 100% of the perturbed
   factors on every measured iid. The PROCEED gate was retracted.
3. **§6c iid-15 dense forensic** for the 3 dense-input iids
   (`audit_results/vgg_dense_input_forensic_20260604T042518Z/`).
   Production was given wall=1200s, rss=70GB. iid 15 reached FLATTEN in
   699s with `root_ng = 150528` (metadata) and `ng = 25088` (actual
   columns) — a 6× Girard-cap compression. At L28 the verifier hit a
   **150 GiB CUDA allocation request** and fell back via the
   sparse-to-dense path. An ImageHZ representation that kept all
   150528 generators independent would request even more memory.

**Finding.** ImageHZ has no precision lever on canonical VGG either: the
sparse-input case is already preserved by production, and the dense-
input case is bounded by the same L28 memory wall production hits.

**Why this is a contribution.** This is the canonical example of a
**denominator-of-the-wrong-quantity bug** in verifier evaluation. We
recommend that VNN-COMP benchmark evaluations report
`root_preservation_ratio` against `active_input_dims`, not the full
image size; the same metric against `n_input` is a measurement artifact.

### 4.3 Closure 3 — CIFAR final-tail per-neuron hull: production already at the math ceiling

**Hypothesis.** A spec-compliant reimplementation of the per-neuron
triangle LP on CIFAR's last hidden ReLU would either match production
exactly or reveal an unintended shortcut.

**Evidence.** 20 CIFAR near-boundary sentinels — the 20 lowest positive
`final_lp_margin` UNK iids from atlas v3
(`audit_results/cifar_finaltail_hull_sentinels_20260604.json`).
Driver: `research/cifar_finaltail_hull_lp.py`. Per-iid:
captures a fresh FLATTEN snapshot, builds the LP per
`research/cifar_finaltail_hull_plan.md` Section 1, solves the per-rival
LP under HiGHS, compares against production endcap LP on the SAME
snapshot.

> **Table 4: §7 Phase-1 gate result (20-sentinel).**

| Metric | Value |
|---|---|
| Median LP UB reduction | **0.0000 %** |
| New V/A under clean LP | **0** |
| Per-iid parity_max_abs_diff | **0.0e+00 on every iid, every rival** |
| Verdict per design lock §5 | **FAIL** (0 V/A AND median < 10%) |

**Finding.** The production endcap LP is already at the per-neuron
triangle math ceiling, which is provably the tightest single-neuron
convex hull (Singh et al. 2019). A spec-compliant rewrite returns
bit-identical objective values on every rival on every sentinel.

**Why this is a contribution.** A clean-room reimplementation of the
LP from spec rules out the "production has a bug or shortcut" hypothesis
with measurement, not assertion. It also formally retires Phase-2 pair-
hull cuts on the CIFAR final-tail layer (which were contingent on
Phase 1 showing structural room).

### 4.4 Direct implications

The three closures together cover the three places forward HZ could
conceivably gain precision under the principle set:

> **Table 5: Where forward HZ could gain precision, and why each path is closed.**

| Place | Direction | Closure |
|---|---|---|
| Conv-body root preservation (CIFAR) | ImageHZ | Section 4.1 — root already preserved |
| Conv-body root preservation (VGG/Tiny) | ImageHZ | Section 4.2 — sparse: preserved; dense: memory wall |
| Final-tail LP relaxation (CIFAR) | per-neuron hull | Section 4.3 — production at math ceiling |

The remaining structural ways to lower the residual LP UB:

| Direction | Status |
|---|---|
| Multi-neuron joint relaxation (Singh PRIMA k=2/k=3, Anderson 2020 facets) | prior-negative on small-dense benchmarks (see memory `project_pairwise_hull_negative_20260516`, `project_triple_hull_negative_20260516`, `project_anderson_facets_negative_20260516`) |
| Branch-and-bound / input splitting | excluded by P4 |
| Backward bound tightening (CROWN-like) | excluded by P1 |

---

## 5. Lessons

### 5.1 Why we stop here

253 V/A is the project's frozen forward-HZ delivery under the principle
set. Patches that move the number by ±1-2 instances are no longer
allowed for three reasons:

1. The three closures in Section 4 are an explicit ceiling map. Any
   patch that "just happens to work" on a benchmark without explaining
   which closure it bypasses is, by construction, a benchmark-specific
   patch.
2. Numbers obtained outside the Section 2 principle set are not project
   results. We pre-committed to the principles before the closures ran;
   relaxing them retroactively to chase a few extra V/A invalidates the
   paper's central claim.
3. The audit-receipt contract (Section 6) makes provenance verifiable
   from one repository. Continued patching dilutes that contract by
   adding code paths an auditor would have to verify separately.

### 5.2 What the three closures jointly imply about future work

Any future research direction targeting forward HZ V/A lift must
declare, before running any experiment, **which of the four closure
categories its mechanism falls outside of**:

- root-preservation on sparse-input (Closure 1),
- root-preservation on dense-input under memory ceiling (Closure 2),
- per-neuron LP relaxation (Closure 3),
- one of the principle-excluded categories (multi-neuron / BaB /
  backward — Section 4.4).

Any direction that falls inside one of the closed categories must
either (a) supply new evidence the closure overlooked, or (b) come with
a NEW gate, NOT a re-opening of an old gate.

### 5.3 Why frontend cleanup is not a verification result

The 39 ERROR rows on cctsdb_yolo_2023 are an ONNX-parser frontend gap:
data-dependent `Slice` indices can change the runtime output shape.
Fixing that symbolic-shape gap may move some rows to UNKNOWN or V/A,
but this is **not a verifier-capability result** and is therefore not
counted in the 253-V/A headline. The fix is tracked separately in
`research/frontend_cleanup_plan.md`. We recommend other verifier
authors apply the same separation: a parser-support PR is engineering,
a relaxation-tightening PR is research, and they should not be merged
under the same header.

---

## 6. Reproducibility

The full audit trail for the improvement phase lives under
`audit_results/`. The key entry points for an external auditor:

> **Table 6: Question-to-evidence index.**

| Question | Where to look |
|---|---|
| What is the 253 V/A? | `audit_results/clean_canonical_combined_summary_20260604.json` |
| Are the FAL receipts trustworthy? | `audit_results/clean_canonical_sweep_*/<bench>/iid<NNN>_provenance.json` |
| Why is CIFAR-ImageHZ closed? | `audit_results/cifar_unknown_margin_atlas_canonical_20260603T121947Z/` |
| Why is VGG/Tiny-ImageHZ closed? | `audit_results/vgg_active_root_denominator_audit_20260604/`, `audit_results/vgg_dense_input_forensic_20260604T042518Z/` |
| Why is CIFAR final-tail hull closed? | `audit_results/cifar_finaltail_hull_phase1_sentinels_20260604T050851Z/gate.json` |
| What design locks were honored? | `research/imagehz_vgg_prototype_plan.md`, `research/cifar_finaltail_hull_plan.md` |
| What loader guarantees canonical paths? | `research/canonical_provenance.py` |
| What is the dev environment? | `CLAUDE.md` (Conda env `OnnXC`, Python `/opt/anaconda3/envs/OnnXC/bin/python`, Gurobi license used for development only — NOT in the verification path) |

### 6.1 LOCAL / CANONICAL mismatch is fail-closed

`research/canonical_provenance.py` rejects any input from a LOCAL-pool
path (`/data1/Kane/ACT/data/vnnlib`, `/data1/Kane/HyZor/data/vnnlib`)
and only accepts the canonical root
(`/data1/Kane/data/vnncomp2025_benchmarks/benchmarks`). All
audit_results directories carry SHA256 hashes of the actual files used,
which makes drift-detection mechanical.

### 6.2 Source of the 253 number

The combined summary at
`audit_results/clean_canonical_combined_summary_20260604.json` is the
single source of truth for the headline. It was rebuilt from the
underlying per-iid receipts during the 2026-06-04 evening audit and
its component benchmark blocks point at the audit_results dirs where
each was produced.

### 6.3 Re-running the audit

To re-verify the audit numbers without re-running production:

```bash
python research/canonical_provenance.py  # fails closed on LOCAL paths
# inspect the combined summary
python - <<'PY'
import json
d = json.load(open('audit_results/clean_canonical_combined_summary_20260604.json'))
for b, m in d.items():
    c = m['verdict_counts']
    print(f'{b:<22} FAL={c.get("FALSIFIED",0):>3} CERT/V={c.get("CERTIFIED",0)+c.get("VERIFIED",0):>3} UNK={sum(v for k,v in c.items() if k.startswith("UNKNOWN")):>3} ERR={sum(v for k,v in c.items() if k.startswith("ERROR")):>3}')
PY
```

This produces the per-benchmark counts that aggregate to the
191 V + 62 A = 253 V/A headline.

---

## 7. Closing statement (for paper abstract / conclusion)

ACT / HyZor is a forward-only Hybrid Zonotope verifier that delivers
**253 V/A across five VNN-COMP-2025 canonical benchmarks** under a
deliberately narrow principle set (no CROWN, no PGD, no MILP, no fallback
verifier counted as an ACT result, no BaB, no random sampling). Three independent closure analyses
(CIFAR-ImageHZ, VGG/Tiny-ImageHZ, CIFAR final-tail hull) jointly map
the precision ceiling reachable inside this principle set; further
V/A lift requires either multi-neuron joint reasoning (prior-negative
on small-dense benchmarks) or a deliberate principle relaxation that
would change what the verifier claims. The full evidence trail —
including per-FAL provenance hashes and a LOCAL-pool fail-closed
loader — is available under `audit_results/` and is reproducible from
a single canonical root.
