# ACT/HZ Verification Improvement Roadmap

Date: 2026-06-04 (latest update — full-result review and SC-HZ future track)
Scope: archived improvement roadmap plus forward research plan for ACT/HZ under the project design principles.

## 0. Current Status Superseding Note — 924 is the full row, 253 is a subset

This roadmap contains older sections written during the five-benchmark
stabilization cycle. Those sections are preserved for provenance, but the
current score context is now:

```text
Full cross-tool row:      ACT/HZ GPU STRICT = 924 V/A over the 3,453-instance comparison table.
Five-benchmark subset:    253 V/A across CIFAR / Tiny / nn4sys / malbeware / yolo.
```

The 253 result is **not** the current total capability claim. It is a
provenance-clean subset that explains specific profile mechanisms and receipt
discipline. The current full-row claim is the 924 V/A table in
`research/tool_comparison_20260604.md` and related result summaries.

The old improvement phase is closed in a narrow sense: do not keep reopening
ImageHZ, final-tail per-neuron hulls, or benchmark-name-gated sidecars. The new
open question is broader and benchmark-independent:

> Can ACT/HZ become competitive in absolute terms by redesigning HZ itself for
> robustness verification, without CROWN/backward bounds, BaB, MILP, fallback,
> or random/corner falsification?

The current proposed next research track for that question is **Spec-Conditioned HZ
(SC-HZ)**, documented in:

- `research/hz_redesign_for_robustness_20260604.md`
- `research/dc_hz_phase_a_plan.md` (historical filename; mechanism renamed SC-HZ)

SC-HZ is **not** another benchmark patch. It is a query-local forward HZ
abstraction: keep generators relevant to the current spec/rival, merge the rest
with a sound tail-box over-approximation, and solve the resulting continuous LP
with strict replay for any FAL. Production defaults and the 924 row do not
change unless Phase A passes.

### SC-HZ Phase A gate

Run 80 sentinels:

- 20 `cifar100_2024`
- 20 `tinyimagenet_2024`
- 20 `safenlp_2024`
- 20 `acasxu_2023`

PASS iff:

- cumulative new V/A >= 5, OR
- median LP upper-bound reduction >= 25%.

FAIL iff:

- new V/A = 0 AND median LP upper-bound reduction < 10%, or
- any claimed FAL fails strict ORT replay / provenance checks.

INCONCLUSIVE:

- widen K once on the 10 weakest sentinels, rerun only those, then decide.

This is the only current route that plausibly changes ACT from "best
pure-forward but far behind abcrown/NeuralSAT" into a stronger absolute
verifier. If Phase A fails, the realistic conclusion is that 2000+ is not
reachable under the current principles without a deeper abstraction idea.

## 0a. 2026-06-04 Atlas v3 STOP Update

The 2026-06-03 canonical recovery (see §0b) was followed on the same day by
two new runs that close out the ImageHZ-for-CIFAR decision gate:

1. **Atlas v3 — canonical CIFAR 185 UNK** at
   `audit_results/cifar_unknown_margin_atlas_canonical_20260603T121947Z/`.
   184/185 successfully diagnosed (1 ERROR — iid 63 had compressed root_ng).
   Of the 184 successes:

   ```text
   root_ng_to_input_ratio = 1.000           : 185 / 185   (full pixel correlation preserved; iid 63 was backfilled 2026-06-04)
   root_ng_to_input_ratio < 1.000           :   0 / 185
   constraint-free HZ (nc=0 AND nb=0)       : 185 / 185   (snapshot has no aux equality/binary constraints — NOT a BoxHZ collapse; Gc is still present and ng > 0)
   LP candidate replay produced FAL         :   0 / 185
   Box candidate replay produced FAL        :   0 / 185
   LP UB median / max               : +1.36 / +10.93
   Phantom_lp median / max          : +2.15 / +14.43
   Final-ReLU unstable median       : 39 of 100
   Final-ReLU mu sum median         : +7.61
   LP / box ratio median            : 0.89   (LP only 11% tighter than box)
   ```

   **spatial_correlation_fixable share = 0 / 185 = 0.0%**. Below the < 5%
   threshold from §5: **STOP CIFAR-ImageHZ** (NOT "stop dense-conv" in
   general — the negative finding is specifically that on CIFAR the
   conv-body root correlation is already preserved end-to-end, so
   ImageHZ has no precision lever to pull *for CIFAR*). The actual CIFAR
   bottleneck is tail-relaxation looseness — final-hidden ReLU triangle
   slack + dense Gemm magnification — see §3.3 and §7 (the latter
   was closed-negative 2026-06-04 night; per-neuron triangle
   is at the math ceiling). The VGG/Tiny dense-conv branch was later
   closed by §6b/§6c; see the superseding note below.

2. **VGG L29 forensic — canonical iid 1** at
   `audit_results/vgg_l29_forensic_canonical_20260603T121532Z/`. 88s wall, no
   OOM under `CONV_FALLBACK_SAFE=1 + GIRARD_PRESERVE_ROOT=1`. All layers
   propagate cleanly (nc=0, nb=0 throughout). The apparent precision-loss
   events were Girard cap reductions at L32 (MAXPOOL ng 71317→10240, −86%)
   and L35 (RELU 10240→4096, −60%). At that point VGG looked like the only
   remaining benchmark where conv-body correlation loss measurably fired;
   §6b/§6c later retracted that interpretation for VGG/Tiny.

Historical consequences at the time of this update:

- §5 (Task A: CIFAR atlas) is **DONE** with STOP verdict.
- §6 (Task B: VGG forensic) is **DONE**; the OOM concern is solved by the
  existing env flags, and the new question is whether the Girard cap losses
  could be reduced by ImageHZ — see §6b (VGG mini-atlas).
- §7 (Task C: ImageHZ prototype) **must not target CIFAR**. If ImageHZ is to
  be prototyped, the gate moves to a VGG-mini-atlas (§6b) measuring
  root-factor-preservation in the actual reduction sites.
- The canonical CIFAR FAL count under forward-only HZ remains **+15**. The
  local-pool "+16 / 15→31 / 9.2% lift" numbers are still retracted; see §0b.

**SUPERSEDED 2026-06-04 night**: §6b mini-atlas ran and was REOPENED as denominator-bug
(see §6b update). §6c iid-15 dense forensic closed VGG-ImageHZ. §7 final-tail per-neuron
hull pilot ran and FAILED its gate (clean LP bit-exact with production). All three
verification-improvement lines are now CLOSED NEGATIVE. The improvement phase is over;
next step is §10 paper writeup.

## 0b. 2026-06-03 Canonical-Root Recovery Update

This roadmap has been updated after a provenance bug was found in the P0 CIFAR
closed-form witness experiments.

The issue was not a soundness failure in strict replay. It was an experiment
bookkeeping failure: `/data1/Kane/ACT/data/vnnlib/cifar100_2024/instances.csv`
and `/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/instances.csv`
both contain 200 CIFAR rows, but have **zero VNNLIB-file overlap**. Some P0
research scripts used the local pool while production runs and the baseline
used the canonical VNN-COMP pool. Therefore, the earlier `CIFAR 15 -> 31`
claim is retracted as a canonical benchmark result.

Recovery work completed:

- Added `research/canonical_provenance.py`, a fail-closed canonical-root loader.
- Retired the local-pool CIFAR loader in `research/imagehz_cifar_prototype.py`.
- Updated `research/p0_batch_topk_dispatch.py` to emit provenance hashes in
  receipts: canonical root, `instances.csv` SHA256, ONNX SHA256, and VNNLIB
  SHA256.
- Re-ran the closed-form P0 dispatch on the canonical 185-CIFAR-UNKNOWN pool:

```text
audit_results/p0_canonical_unknown185_20260603/summary.csv

FALSIFIED strict replay       0
ALL_RIVALS_LP_SAFE watchlist  1   (not promoted to CERT)
UNKNOWN_NO_FAL_REPLAYED     184
```

Consequence:

- The existing canonical CIFAR gain remains `+15 FAL`.
- The P0/A++ closed-form support-vector candidate line is **closed negative for
  canonical CIFAR score improvement** unless new evidence appears.
- The next real improvement line must return to dense-conv representation
  quality, especially ImageHZ / conv-body correlation preservation.

## 0c. 2026-06-04 CIFAR Clean-Sweep Dispatch Recovery

The first clean canonical sweep on 2026-06-04 incorrectly reported
`cifar100_2024 = 0 FAL / 200 UNKNOWN`. This was **not** an algorithmic
regression and should not be used as a scientific result.

Diagnosis:

- A CIFAR-only rerun with `ACT_HZ_TOPK_RIVAL_WITNESS=5` was started, but it
  again returned UNKNOWN on known-FAL iid 2.
- The invalid rerun was stopped after 21 UNKNOWNs. Continuing it would only
  waste GPU time.
- Minimal iid-2 reproduction showed the witness path was entered and the L38
  FLATTEN snapshot existed, but the witness extractor raised:

```text
IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
```

Root cause:

- `/data1/Kane/HyZor/receipt_factor_aware_endcap_lp.py` assumed snapshot
  center `c` had shape `(dim,)`.
- Current production snapshots store `c` as `(dim, 1)`.
- The tail algebra computed `W @ c + b`, and NumPy broadcast the bias to
  `(dim, dim)`, making ReLU masks two-dimensional and breaking the CIFAR
  witness path before replay.

Fix:

- Normalize `c_in = reshape(-1)` at the entry of `_compute_tail_output_gc`.
- Validate that `Gc_in` is 2-D and that `Gc_in.shape[0] == c_in.shape[0]`.
- Add env/profile diagnostics in `act/pipeline/cli.py` so future silent skips
  record `witness_on`, `topk_K`, `legacy_on`, `snap_dir_env`, and
  `snap_glob_count`.

Validation after the fix:

```text
manual iid2 receipt replay:
  input_box_holds      = True
  vnnlib_query_holds   = True
  spec_zero_tol_holds  = True
  all_checks_pass      = True

ACT integrated iid2:
  FALSIFIED, receipt emitted

15 known CIFAR FAL gate:
  15 / 15 FALSIFIED
  15 / 15 receipts pass strict replay checks
  root: audit_results/cifar_gate15_after_shape_fix_20260603T224838Z/
```

Immediate implication:

- The canonical CIFAR score remains **+15 FAL** in the frozen §9 table.
- The full clean canonical summary has been regenerated and audited:
  `audit_results/clean_canonical_combined_summary_20260604.json`.
- §6b/§6c VGG-ImageHZ and §7 CIFAR final-tail hull have both closed negative.
  No further ImageHZ or §7 Phase-2 work is authorized without fresh evidence
  and a new written gate.

## 1. Executive Summary

The benchmark-specific improvement cycle is now closed. It should not continue
as an open-ended sequence of profile tweaks or sidecars. We extracted the main
gains from small-dense, sparse-input, residual-sparse, and end-cap witness
mechanisms, then ran bounded diagnostics on the remaining dense-conv hypotheses.
Those diagnostics closed negative:

1. CIFAR-ImageHZ: atlas v3 showed root factors are already preserved through
   the conv body; ImageHZ has no correlation loss to recover on canonical CIFAR.
2. VGG/Tiny-ImageHZ: the initial PROCEED signal was a denominator bug on sparse
   specs, and dense VGG specs hit a memory wall rather than a usable precision
   lever.
3. CIFAR final-tail per-neuron hull: the clean LP is bit-exact with production;
   there is no per-neuron hull tightening left to harvest.

The current full cross-tool endpoint is the **924 V/A ACT/HZ GPU STRICT** row.
The **253 V/A across 5 benchmarks** result remains important, but it is a
provenance-clean subset, not the full capability number.

Further verification improvement work requires a new benchmark-independent
hypothesis and a written gate; it should not be framed as continuing the old
patch cycle. As of this update, that new hypothesis is **Spec-Conditioned HZ
(SC-HZ)**: a query-local forward HZ abstraction that spends generator budget on
the current spec/rival and soundly boxes the discarded generator tail.

The decisive point is: **do not keep adding one-off sidecars per benchmark**.
Future work should be either a general abstraction improvement, a clear
front-end correctness fix, or a bounded diagnostic with a written stop rule.
SC-HZ qualifies as the first category; cctsdb/yolo Slice cleanup qualifies as
the second but must not be reported as V/A improvement.

## 2. Design Principles That Must Remain Fixed

All future work must obey the project principles:

| ID | Principle | Operational Meaning |
| --- | --- | --- |
| P1 | No CROWN-style backward propagation | No per-neuron backward linear bound propagation. |
| P2 | No backward / no gradients | No autograd, PGD, FGSM, CW, DeepFool, or any gradient-derived candidate. |
| P3 | No Gurobi / no MILP | Continuous LP through SciPy/HiGHS/highspy is acceptable; integer/MILP solvers are not. |
| P4 | No fallback verifier | UNKNOWN stays UNKNOWN unless ACT/HZ itself proves or finds a strict replay witness. |
| P5 | No B&B / no input splitting | Do not split the input box or run branch-and-bound. |
| P6 | No random/corner-sample-then-check | FAL candidates must be produced by structured HZ/LP programs and pass raw ONNX strict replay. |

Additional reporting constraints:

- FAL must include strict ORT replay against the original ONNX and raw VNNLIB.
- Candidate receipts must record input-box validity and zero-tolerance unsafe
  condition satisfaction.
- Candidate receipts must record canonical provenance:
  `canonical_root`, `instances_csv_sha256`, `onnx_sha256`, and
  `vnnlib_sha256`.
- CERT promotion from new LP sidecars must remain disabled until separately
  audited. FAL-only sidecars are safer because replay is the final authority.
- Any new mechanism must pass the existing regression suite and a no-lost audit
  on the benchmarks it touches.

## 3. Current State

### 3.1 Confirmed Positive Mechanisms

The current targeted improvements are real and should be preserved:

| Mechanism | Benchmarks Benefited | Current Confirmed Gain |
| --- | --- | --- |
| CIFAR narrow end-cap FAL witness | `cifar100_2024` | +15 FAL |
| Generic MLP end-cap witness | `tinyimagenet_2024` | +34 FAL net |
| Residual sparse-conv profile | `yolo_2023` | +40 CERT |
| nn4sys/lindex query batching and stable-affine path | `nn4sys` lindex subset | +13 CERT |
| Single-Dense end-cap FAL path | `malbeware` UNKNOWN subset | +6 FAL |

Total targeted gain currently attributed to these mechanisms:

```text
CIFAR       15
Tiny        34
YOLO        40
nn4sys      13
malbeware    6
----------------
Targeted   108 decisions
```

This is not yet the same thing as a clean full canonical 24-benchmark sweep.
The final paper/result table must be generated from a controlled full run with
consistent profiles, memory caps, and no polluted OOM due to excessive
parallelism.

### 3.2 Negative or Closed Directions

The following directions should not be repeated without new evidence:

| Direction | Reason to Close |
| --- | --- |
| D filter / multi-corner LP / joint K=2 ReLU cuts | Multiple GPU 0-lift sweeps; output-level cuts did not move dense-conv verdicts. |
| PGD / random / corner candidate search | Violates project principle P6. |
| PyRAT direct adoption | Uses `sound=False` and split-like settings in some runs; useful as inspiration only. |
| P0/A++ closed-form support candidate on CIFAR | Canonical 185-UNKNOWN rerun gave 0 strict FAL; local-pool +16 claim was a provenance mismatch. |
| VGG 3-Dense tail LP | VGG FLATTEN snapshot collapses to `BoxHZ`; tail LP would have no input-correlation provenance. |
| soundnessbench output-halfspace end-cap LP | Snapshot works, but 5 sentinel candidates are ORT phantom; conv body is too loose. |
| LSNC exact Concat-only fixes | Engineering debt cleanup, but all tools are 0; benchmark is Lyapunov/bilinear-hard, not a simple ReLU failure. |

### 3.3 Latest Diagnostic Findings

#### CIFAR atlas v3 (2026-06-04, canonical)

Per-iid diagnostics on the 185 canonical CIFAR atlas-UNK iids
(`audit_results/cifar_unknown_margin_atlas_canonical_20260603T121947Z/`):

```text
successful rows         : 184 / 185   (1 ERROR on iid 63, root_ng compression)
root_ng / n_input = 1.0           : 185 / 185   (full pixel correlation preserved)
constraint-free HZ (nc=nb=0; Gc present): 185 / 185   (NOT BoxHZ collapse — the snapshot still has full Gc and ng>0; only equality/binary aux constraints are absent because PEE absorbed them)
LP-cand replay → FAL    :   0 / 184
box-cand replay → FAL   :   0 / 184
LP UB             min   median   max  : +0.11 / +1.36 / +10.93
phantom_lp        min   median   max  : +0.17 / +2.15 / +14.43
LP / box ratio    median             : 0.89   (LP only 11% tighter than box)
final ReLU unstable     median       : 39 / 100
final ReLU mu sum       median       : +7.61
```

Interpretation: CIFAR conv-body preserves pixel correlation natively. The
remaining UNKs are gapped at the tail (median 39 unstable hidden neurons,
mu_sum 7.6 magnified by W41 dense). ImageHZ would not move this.

#### CIFAR P0/A++ Closed-Form Candidate

The support-vector / closed-form box-LP candidate is sound as a FAL-only
candidate source when replayed by raw ORT, but it does not improve canonical
CIFAR:

```text
canonical 185 UNKNOWN rerun:
  FALSIFIED strict replay       0
  ALL_RIVALS_LP_SAFE watchlist  1
  UNKNOWN_NO_FAL_REPLAYED     184
```

This closes the A++ CIFAR-score line. It is not worth running topK=10,
traffic/tiny extensions, or paper-number updates from this line until a new
canonical-root signal appears.

#### VGG L29 forensic (2026-06-04, canonical)

`audit_results/vgg_l29_forensic_canonical_20260603T121532Z/`. No OOM under
`CONV_FALLBACK_SAFE=1 + GIRARD_PRESERVE_ROOT=1`; 88s wall, UNKNOWN. The
precision-loss events are Girard cap reductions at L32 MAXPOOL (ng 71317 →
10240, −86%) and L35 RELU (ng 10240 → 4096, −60%). `nc = 0` throughout.
This is the only canonical benchmark where conv-body correlation-loss
measurably fires; it is the natural ImageHZ candidate target if anything is
prototyped.

#### soundnessbench

`soundnessbench` has a single-Dense affine tail and output constraints of the
form `Y_i >= c` / `Y_i <= c`, not top-1 `Y_i >= Y_j` robust classification.

The output-halfspace feasibility LP was implemented as a research-only,
FAL-only pilot:

- root-only mode: all 5 sentinel LPs remain infeasible for unsafe with negative
  `max_rho`.
- full mode: small positive `max_rho` appears, but all candidates fail strict
  ORT replay.

Interpretation: the conv prefix abstraction is the bottleneck; the tail LP is
not enough.

#### VGG

VGG was checked before implementing a costly 3-Dense/2-ReLU tail LP. The
snapshot at the final FLATTEN is:

```text
hz_type = BoxHZ
dim     = 25088
ng      = 10240
has_Gc  = false
has_lb_ub = true
```

The collapse happens after a deep-layer CUDA OOM around L29, followed by a box
fallback. Once the snapshot is `BoxHZ`, a tail LP has no preserved
input-correlation structure. It may optimize over feature-box corners, but it
cannot produce reliable input-space witnesses except by strict replay, and such
candidates are expected to be phantom-heavy.

Interpretation: VGG's problem is memory and conv-body preservation, not Dense
tail expressiveness.

## 4. What Still Needs To Be Done

The remaining tasks are deliberately bounded. Each task has a clear purpose,
expected duration, and stop rule.

## 5. Task A: CIFAR UNKNOWN Margin Atlas — DONE (2026-06-04, STOP-CIFAR-ImageHZ)

**Result:** completed 2026-06-04 at
`audit_results/cifar_unknown_margin_atlas_canonical_20260603T121947Z/`,
with iid 63 backfilled the same day after the broadcast guard fix.
**185 / 185 successfully diagnosed.** spatial_correlation_fixable =
0 / 185 = 0.0%, below the < 5% threshold.

**Decision: STOP CIFAR-ImageHZ.** This is NOT "stop dense-conv" in
general — the negative finding is that CIFAR's conv body preserves
root pixel correlation natively. ImageHZ would not buy anything on
CIFAR. Other dense-conv benchmarks remain candidates per §6b.

CIFAR's current canonical FAL count under forward-only HZ is **+15**
and stays at +15. A++ closed-form, topK rival witness, and any per-iid
sidecar work for CIFAR score lift are all closed-negative. The §7
final-tail per-neuron hull pilot was the last remaining sanctioned
CIFAR direction; it ran 2026-06-04 night and gate-FAILED (clean LP
bit-exact with production endcap LP across 20 sentinels). All CIFAR
verification-improvement lines are now closed-negative.

See §0a for the full atlas v3 numbers.

One small cleanup: the atlas v3 driver's broadcast assumed `root_ng == n_input`.
For completeness, that bug should be fixed and iid 63 rerun so the atlas table
has all 185 rows. This is bookkeeping, not a precision question.

The remainder of this section is preserved as the original task specification.

### Purpose

Decide whether ImageHZ or another spatial-locality-preserving abstraction is
worth building.

Dense-conv benchmarks, especially CIFAR, are the central remaining weakness.
Before implementing a large new domain, we need to know whether the UNKNOWNs
are actually close enough for a better forward abstraction to move them.

### Inputs

- `cifar100_2024` full 200-instance run after current profiles.
- The 185 remaining canonical UNKNOWN instances after the current +15 FAL
  result. Do not use `/data1/Kane/ACT/data/vnnlib`.
- Existing HZ snapshots / traces where available.
- Additional snapshot/tracing if needed at:
  - pre-final FLATTEN,
  - final ReLU / Dense tail,
  - selected residual blocks,
  - final LP candidate extraction point.

### Measurements

For each UNKNOWN instance:

1. Worst rival ID.
2. Final LP margin.
3. Final box margin.
4. LP/box ratio.
5. ORT margin of LP candidate if candidate exists.
6. Whether candidate is replay-valid or phantom.
7. Top unstable ReLU slack contribution near tail.
8. Whether the margin deficit is dominated by:
   - final-tail ReLU relaxation,
   - residual ADD factor duplication,
   - conv-body spatial correlation loss,
   - full box fallback or reduction collapse,
   - genuinely large robust margin gap.

### Output

A CSV/JSON summary:

```text
cifar_unknown_margin_atlas_canonical_<timestamp>.csv
cifar_unknown_margin_atlas_canonical_<timestamp>.json
```

Every row must include the canonical VNNLIB SHA256. Old atlas files under
`audit_results/cifar_unknown_margin_atlas_20260603/` should be treated as
historical diagnostics only unless explicitly regenerated with
`research/canonical_provenance.py`.

With categories:

| Category | Meaning |
| --- | --- |
| `tail_fixable` | Deficit dominated by final ReLU/tail slack. |
| `spatial_correlation_fixable` | Conv body phantom appears tied to loss of local spatial correlation. |
| `memory_collapse` | HZ collapses to BoxHZ or loses root factors due to memory. |
| `forward_irreducible` | Even strong local forward analysis is unlikely to move the margin. |
| `needs_frontend_fix` | Failure is parser/operator/support related, not abstraction precision. |

### Go / No-Go Rule

Proceed to ImageHZ prototype only if:

```text
spatial_correlation_fixable >= 20% of CIFAR UNKNOWNs
```

Do not proceed if:

```text
spatial_correlation_fixable < 5%
```

If the result is between 5% and 20%, run a smaller 20-instance prototype only;
do not productionize.

### Duration

Estimated: **1-2 days**.

### Stop Rule

Stop when the atlas classifies at least 150 of the 185 UNKNOWNs with enough
confidence to estimate whether ImageHZ can move the benchmark. Do not spend more
than 2 days on this without a written reason.

## 6. Task B: VGG L29 OOM Forensic — DONE (2026-06-04, REFRAMED)

**Result:** completed on 2026-06-04 at
`audit_results/vgg_l29_forensic_canonical_20260603T121532Z/`. 88s wall, no
OOM under `ACT_HZ_LAYER_PROGRESS=1 + CONV_FALLBACK_SAFE=1 +
GIRARD_PRESERVE_ROOT=1`. Verdict UNKNOWN. Full HZ propagation through 38
layers with `nc=0, nb=0` throughout. The actual precision-loss events are:

| Layer | Op | dim | ng (post) | ng change |
|---|---|---|---|---|
| L31 | RELU | 100352 | 71317 | +40k aux |
| **L32** | **MAXPOOL2D** | 25088 | **10240** | **−86% (Girard cap)** |
| L34 | DENSE | 4096 | 10240 | |
| **L35** | **RELU** | 4096 | **4096** | **−60% (Girard cap)** |
| L37 | RELU | 4096 | 6000 | |
| L38 | DENSE | 1000 | 6000 | |

The OOM concern is resolved by the existing env flags. The remaining question
is whether ImageHZ can help VGG by reducing the pressure on Girard cap at L32
and L35 (so fewer correlation losses happen). That question is split out to
the new §6b VGG mini-atlas.

The remainder of this section is preserved as the original task specification.

### Purpose

Determine whether VGG's box fallback is caused by a single fixable dense
materialization or by structural memory blow-up.

### Required Trace

Run one VGG UNKNOWN instance with layer-level memory instrumentation:

- layer ID,
- op kind,
- HZ type before/after,
- `dim`, `ng`, `nb`, `nc`,
- GPU allocated/reserved before/after,
- whether the op entered fallback,
- exact failing allocation if OOM.

### Questions To Answer

1. Is L29 OOM caused by conv, ReLU triangle, reduction, ADD, or another op?
2. Does `ACT_HZ_CONV_FALLBACK_SAFE=1` fail because the failing op is not conv?
3. Can this be fixed with chunking or sparse path routing?
4. If fixed, does the FLATTEN snapshot remain HZono/SparseGcZ instead of BoxHZ?

### Go / No-Go Rule

Proceed with a VGG-specific memory fix only if:

- the OOM source is a single operator class,
- the fix is general to conv nets,
- it preserves HZ correlation,
- and it reduces fallback to BoxHZ on at least two VGG sentinel instances.

Do not build a Dense-tail VGG sidecar until VGG produces a non-BoxHZ FLATTEN
snapshot.

### Duration

Estimated: **0.5-1 day**.

## 6b. Task B+: VGG / Tiny Mini-Atlas — REOPENED (2026-06-04 evening, **DENOMINATOR BUG**)

The 2026-06-04 morning PROCEED verdict (see "Original final state" below)
was **retracted** the same evening after a denominator audit. The
PROCEED gate compared `root_ng_at_flatten` to `n_input = 3·224·224 = 150528`,
but the correct denominator is the **number of input dimensions the
VNNLIB spec actually perturbs** (i.e. `active_input_dims`), not the
whole image size. VNN-COMP VGG specs are L∞ perturbations on a small
subset of pixels — 1, 5, 10, 20, or 100 pixels per spec for iids 0-14;
the remaining ~150k input dimensions are point-constrained
(`lb == ub`) and carry zero generator mass from the start.

### Active-root audit (2026-06-04, `audit_results/vgg_active_root_denominator_audit_20260604/`)

```text
iid n_in    active_dims root_ng ratio     girard_layers              verdict
0   150528           1       1 1.0000     {25, 32, 35}               PRESERVED_100PCT
1   150528           1       1 1.0000     {25, 30, 32, 35}           PRESERVED_100PCT
2   150528           1       1 1.0000     {25, 29, 32, 35}           PRESERVED_100PCT
3   150528           5       5 1.0000     {11, 17, 18, 25, 29, 32, 35} PRESERVED_100PCT
4   150528           5       5 1.0000     {11, 18, 25, 29, 32, 35}   PRESERVED_100PCT
5   150528           5       5 1.0000     {11, 18, 25, 29, 32, 35}   PRESERVED_100PCT
6   150528          10      10 1.0000     {11, 17, 18, 25, 29, 32, 35} PRESERVED_100PCT
7-8 150528          10      10 1.0000     L11/L18/L25/L29/L32/L35    PRESERVED_100PCT
9-11 150528         20      20 1.0000     L11/(L17)/L18/L25/L29/L32/L35 PRESERVED_100PCT
12-14 150528       100     100 1.0000     L11/(L17)/L18/L25/L29/L32/L35 PRESERVED_100PCT
15-17 150528    150528    None    -       {}                         NO_SNAPSHOT (production wall-timeout)
```

### What this means

1. **For iids 0-14, the production HZ pipeline already preserves 100%
   of the root factors that the spec actually perturbs.** The "100%
   correlation-loss" claim from the original §6b gate evaluation was a
   denominator artifact, not a true loss. There is nothing for an
   alternative representation to recover at the root-factor level.
2. **The Girard cap fires observed at L11 / L18 / L25 / L32 MAXPOOL
   and L17 / L29 / L35 RELU are eliminating ReLU AUX generators**,
   not root factors. ImageHZ's locality argument was about preserving
   root factors; preserving aux generators is a different (and weaker)
   goal that does NOT change the tail LP feasibility.
3. **For iids 15-17 the question stays open.** These specs perturb
   all 150528 input dimensions, but the production HZ pipeline could
   not reach FLATTEN within the 607s wall budget — so we have no
   snapshot evidence either way. A lightweight forensic at the
   L4 / L9 / L16 / L25 boundaries (i.e. just after each early MAXPOOL)
   is needed to decide whether ImageHZ might help on this subset.

### ImageHZ-lite Phase 0 implementation also stalled

Independent of the denominator finding, the cap=10 → cap=11 ladder run
on iid 0 showed that the current TileBlock representation produces

```text
L0-L3  ng = 1            (Conv/ReLU stable)
L4     ng = 1 → 298      (MaxPool fan-out: per-output TileBlock)
L9     ng = 298 → 127599 (second MaxPool fan-out, exploded)
```

127599 TileBlocks at L9 makes every subsequent operator iterate over
that many Python objects; cap=11 hit a 300s timeout. The fundamental
issue is the implementation duplicates the SAME root factor into
hundreds of thousands of per-output-position tiles. None of those
tiles are new root factors; they are pure spatial-copy bookkeeping.

A future representation that wanted to do better would need to be a
"FactorField" — one per-factor coefficient field over the feature
map, with MaxPool stable case applied via gather, never producing
per-output tile duplication. But this would only be worth building
**if there's actual root loss to recover**, which the audit
shows there is not (at least for iids 0-14).

### Decision

- **ImageHZ-lite Phase 0 is PAUSED.** No further sentinel runs on the
  current TileBlock implementation; no Phase 1 V/A gate work.
- **VGG iids 0-14 are CLOSED for ImageHZ.** Audit confirms ratio = 1.0
  across all 15 iids. No precision lever exists for ImageHZ here.
- **VGG iids 15-17 require a separate dense-input feasibility forensic**
  (Task §6c below) before any ImageHZ decision on that subset.
- The Phase 0 design lock (§9-resolved in `imagehz_vgg_prototype_plan.md`),
  unit-test matrix (9/9 PASS), and TileBlock code remain in `research/imagehz_lite/`
  as historical artifacts. They MUST NOT be wired into production. If
  ImageHZ work resumes later, a FactorField representation will replace
  the TileBlock prototype.

### Receipts

- `audit_results/vgg_active_root_denominator_audit_20260604/summary.csv`
- `audit_results/vgg_active_root_denominator_audit_20260604/audit.json`
- `audit_results/imagehz_lite_phase0_ladder_iid0_cap10_20260604/per_iid/iid000.json`
  (the cap=10 trace showing the L9 fan-out)

---

## 6c. VGG dense-input forensic — iid 15 (2026-06-04 evening)

The §6b denominator audit closed iids 0-14 (ratio = 1.0). The only
remaining theoretical ImageHZ target was iids 15-17 (active input dims
= 150528, full-image L∞ perturbations). A lightweight forensic on
iid 15 was run with `wall = 1200 s` and `rss_cap = 70 GB` to capture
the full per-layer trace under `ACT_HZ_LAYER_PROGRESS=1`.

Run: `audit_results/vgg_dense_input_forensic_20260604T042518Z/`.

### Trace

```text
input        dim=150528    ng=150528  (all root factors alive)
L2  CONV2D   dim=3211264   ng=150528
L3  RELU     dim=3211264   ng=150528  (stable; no aux)
L4  CONV2D   dim=3211264   ng=150528
L5  RELU                   ng=150532  (4 unstable; 4 aux)
L6  MAXPOOL  dim=802816    ng=509908  (+360k aux)
L7  CONV2D                 ng=509908
L8  RELU                   ng=647776  (+138k aux)
L9  CONV2D                 ng=647776
L10 RELU                   ng=484173  (Girard cap −163k)
L11 MAXPOOL  dim=401408    ng=230328  (Girard cap −254k)
L12-L17 Conv/ReLU          ng up to 334007
L18 MAXPOOL  dim=200704    ng=132731  (Girard cap −201k)
L19-L24 Conv/ReLU          ng up to 401408
L25 MAXPOOL  dim=100352    ng=100352  (cap to dim)
L26 CONV2D                 ng=100352
L27 RELU                   ng=200704
L28 CONV2D                 CUDA OOM (150 GiB allocation); sparse fallback
L29 RELU                   ng=100352
L30 CONV2D                 ng=100352
L31 RELU                   ng=200704
L32 MAXPOOL  dim=25088     ng=25088   (cap to dim)
L33 FLATTEN                ng=25088   <— snapshot captured here

Snapshot at L33 FLATTEN:
    root_ng = 150528    (metadata: all original input pixel IDs still tracked)
    ng      = 25088     (actual generator column count; 6× compression)
    nc = 0,  nb = 0
    c.shape = (25088, 1)

Tail: L34 Dense → L35 ReLU (cap to 4096) → L36-L38 Dense → output dim=1000, ng=8192

Verdict: UNKNOWN   wall=699.8 s (11.7 min)
```

### Why `root_ng = 150528` does NOT mean root preservation

The production HZ snapshot records `root_ng = 150528` because it
preserves the **metadata count** of original input factor IDs across
all reductions. But the actual generator column count is `ng = 25088`,
6× smaller than the metadata count. The 150528 root factor IDs are
**aliased** into 25088 mixed columns by Girard cap — they are NOT
150528 independent variables anymore. The encoding is sound but
strictly looser than a representation that kept them independent.

This is what the §6b "100% correlation-loss" claim was actually
pointing at, AFTER fixing the denominator: not loss of which input
pixels are perturbed, but loss of variable-independence among the
already-tracked perturbed pixels.

### Why ImageHZ still can't help iids 15-17

1. The Girard cap fires that compress `ng` (L11: 647k → 230k, L18:
   334k → 132k, L25: 401k → 100k, L32: 200k → 25k) are forced by
   **memory pressure**: at L28 the production verifier already hits a
   150 GiB CUDA allocation request and falls back. ImageHZ would have
   to navigate the same resource ceiling. Any representation that kept
   all 150528 generators independent through L28 would request even
   more memory and OOM harder.
2. The §9R-1 TileBlock representation already exhibited fan-out
   pathology on a much simpler case (iid 0, single perturbed pixel,
   blew up to 127599 TileBlocks at L9 in the Phase 0 ladder). On
   dense inputs with 150528 starting tiles, even the first MaxPool
   would force the same per-output-position fan-out and crash before
   reaching L11.
3. The verdict for iid 15 is UNKNOWN even with full production trace.
   For ImageHZ to **change** that to V or A, the LP tail would have
   to produce a tighter or witness-bearing bound. With the wall
   spent on the conv body, the production endcap LP runs but finds
   no FAL — and a tighter LP at the cost of OOM is not viable.

### Decision

**Close ImageHZ-for-VGG entirely.** No further work on iids 15-17
either — the resource ceiling at L28 is the binding constraint, not
representation. The TileBlock implementation in `research/imagehz_lite/`
stays as a historical artifact (with the §9R design lock and 9/9
unit-test matrix that confirms its soundness on toy cases). A
"FactorField" representation could in principle do better on
generator independence, but only if a different way to navigate
L28's memory wall is found; that is out of scope for the current
improvement phase.

The project's next-step roadmap therefore reverts to:

- §9 (stabilization) — already achieved on 2026-06-04 morning:
  CIFAR 15/15 + tinyimagenet 34 + nn4sys 101 + malbeware 13 +
  yolo Slice-blocked = 253 V/A across 5 benchmarks, with provenance
  hashes on every receipt.
- §10 (documentation + paper) — describe the closed-negative
  ImageHZ-for-VGG result honestly: PROCEED gate from §6b was a
  denominator artifact; on canonical VGG the production HZ already
  preserves root metadata, and the binding constraint is memory at
  the deeper conv layers, not representation choice.
- The deferred yolo Slice frontend cleanup remains optional.

### Receipts

- `audit_results/vgg_dense_input_forensic_20260604T042518Z/snap_iid15/L033_FLATTEN.pkl`
- `audit_results/vgg_dense_input_forensic_20260604T042518Z/iid15/`
  (per_instance JSON + watchdog log with the full HZ-PROGRESS trace)

---

## 6b-original. Original final state — RETRACTED 2026-06-04 evening

### Final state (2026-06-04 02:35 UTC)

`audit_results/vgg_mini_atlas_canonical_plus_missing_20260604T023543Z/`
is the merged evidence root (original 18-iid base + iid-2 successful
rerun + iids-15/16/17 timeout reruns). Final gate evaluation:

| Metric | Value | Threshold |
|---|---|---|
| n_total_iids | 18 | — |
| n_with_snapshot | 15 (iids 15/16/17 still timed out at 607s wall) | — |
| `root_ng / n_input < 0.95` (correlation-lost share) | **15 / 15 = 100.0%** | ≥ 5% ✓ |
| iids with Girard fire at L11/L18/L25/L32 MAXPOOL or L17/L29/L35 RELU | **15 / 18 = 83.3%** | ≥ 1 iid ✓ |
| L32 MAXPOOL2D fire | universal — converges ng to ≈25088 across all full-trace iids | confirmed |
| L35 RELU fire | universal — converges ng to ≈4096 across all full-trace iids | confirmed |
| iid 2 rerun (was OOM at RSS 42.67 GB) | succeeded with `rss=60GB`; reached L38; L32 79042→11830 + L35 11830→4096 | unblocked |
| iids 15/16/17 timeouts (607s @ 600s budget) | non-blocking per advisor 2026-06-04 — 15/18 evidence is enough | accepted |

**Historical gate decision, later retracted: PROCEED to ImageHZ-lite prototype on VGG/Tiny.**
The prototype scope, invariants, and hard stop gates are spelled out in
`research/imagehz_vgg_prototype_plan.md`; §7 in this document points at
that file as the canonical source.

### Why the gate is solid

1. Correlation loss is **universal** on VGG canonical UNK iids — every
   single iid with a usable snapshot showed `root_ng ∈ {1, 5, 10, 20, 100}`
   against `n_input = 150528`. This is the opposite of CIFAR (§5/§3.3),
   where 185/185 had `root_ng = n_input`.
2. The loss is **concentrated at the operators ImageHZ's locality
   representation could delay** — six layers fire Girard cap reductions:
   L11, L18, L25 (MAXPOOL2D), L17, L29, L35 (RELU). The forensic
   already established that L32 and L35 are universal; the reparse
   extended that to L11 / L17 / L18 / L25 / L29 with per-iid evidence.
3. The remaining 3 iids' timeouts are a memory-budget issue, not an
   evidence gap. Even excluding them entirely, the gate passes by a
   wide margin.

### Acceptance evidence (advisor-specified)

| Criterion | Result |
|---|---|
| iid 2 reaches L33 FLATTEN with `root_ng_at_flatten` | ✓ reached L38, root_ng = 1, ng_at_flatten = 11830 |
| iids 15/16/17 progress past L3 | ✗ still L2..L3 (advisor anticipated; doesn't block) |
| Girard fires at L32 MAXPOOL / L35 RELU still present | ✓ iid 2 confirmed L32 79042→11830 + L35 11830→4096 |
| No OOM/ERROR pollution | ✓ 0 ERROR; only timeouts on 15/16/17 |

### Reparsing follow-up

The original 2026-06-03 driver only saw 18 layer rows (1 per iid) because it
read `stdout_tail` from `per_instance.json`, which `watchdog_runner.py`
truncates to the last 2000 bytes. `research/vgg_mini_atlas_reparse.py`
reads the persisted `out_dir/watchdog_<bench>_<iid>.log` files directly
and yields **530 layer rows + 84 Girard fires** across the 18 iids.
The reparse overwrites each per-iid JSON + `summary.csv` so the
on-disk evidence is consistent. The merge script
`research/vgg_mini_atlas_merge.py` produces a side-by-side
base + rerun atlas with `_merge_origin` provenance per iid.

### Original task specification (kept for reference)

#### Purpose

Decide whether ImageHZ is worth prototyping for VGG-class benchmarks.
Replaces the prior CIFAR-centric gate in §7.

### Inputs

- Canonical-root `vggnet16_2022` UNK iids, 10–20 sentinels.
- Optional: `tinyimagenet_2024` 10 sentinels if time allows.
- Provenance hash bundle required on every row (csv + onnx + vnnlib sha256).

### Measurements (per iid)

1. Per-layer trace from `[HZ-PROGRESS]`: layer ID, op kind, dim, ng, nc, nb.
2. Girard cap fires: layers where `ng_post < ng_pre` and reduction was triggered.
3. `root_factor_preserved_ratio` at FLATTEN = `root_ng_at_flatten / n_input`.
4. Whether the FLATTEN HZ collapsed to a BoxHZ surrogate (`has_Gc == False`).
5. Final LP UB and box UB at the output Gemm.
6. LP / box candidate replay outcome under strict ORT.

### Stop / Go Gate

Proceed to ImageHZ prototype only if:

```text
root_factor_preserved_ratio < 0.95   in >= 5% of VGG iids
AND
the loss is concentrated at a Girard cap site that ImageHZ's locality
representation could plausibly delay (MAXPOOL or wide ReLU)
```

Stop ImageHZ entirely if VGG mini-atlas shows < 5% correlation-loss sites,
or if every Girard cap is fired by intrinsic dim/ng ratio (not by memory
pressure that ImageHZ would relieve).

### Duration

Estimated: **0.5–1 day**.

## 7. Task C: CIFAR Final-Tail Hull Prototype — CLOSED NEGATIVE (2026-06-04 night, **GATE = FAIL**)

### Phase 1 result

Pilot run: `audit_results/cifar_finaltail_hull_phase1_sentinels_20260604T050851Z/`
Gate JSON: same dir / `gate.json`

20 sentinels, all completed:

```text
median LP UB reduction:   0.0000%
new V/A under clean LP:   0
parity_max_abs_diff:      0.0e+00 on every iid, every rival
verdict (per design lock §5):  FAIL  (0 V/A AND median < 10%)
```

| iid | unstable | production_max_ub | clean_max_ub | reduction% |
|---|---|---|---|---|
| 113 | 25 | 0.111369 | 0.111369 | 0.000 |
| 29 | 15 | 0.124612 | 0.124612 | 0.000 |
| 153 | 6 | 0.157776 | 0.157776 | 0.000 |
| 72 | 26 | 0.184108 | 0.184108 | 0.000 |
| 105 | 13 | 0.198512 | 0.198512 | 0.000 |
| 102 | 23 | 0.222848 | 0.222848 | 0.000 |
| 174 | 21 | 0.245202 | 0.245202 | 0.000 |
| 180 | 14 | 0.257408 | 0.257408 | 0.000 |
| 110 | 8 | 0.263000 | 0.263000 | 0.000 |
| 116 | 18 | 0.271720 | 0.271720 | 0.000 |
| 168 | 26 | 0.284066 | 0.284066 | 0.000 |
| 75 | 25 | 0.304665 | 0.304665 | 0.000 |
| 133 | 12 | 0.314395 | 0.314395 | 0.000 |
| 92 | 22 | 0.316069 | 0.316069 | 0.000 |
| 165 | 17 | 0.334476 | 0.334476 | 0.000 |
| 86 | 19 | 0.342721 | 0.342721 | 0.000 |
| 137 | 13 | 0.343894 | 0.343894 | 0.000 |
| 15 | 23 | 0.351620 | 0.351620 | 0.000 |
| 82 | 35 | 0.354295 | 0.354295 | 0.000 |
| 93 | 31 | 0.368283 | 0.368283 | 0.000 |

### Interpretation

The clean LP, implemented from the advisor 2026-06-04 design lock §1
mathematical spec, reproduces production endcap LP **bit-exactly**
across every rival on every sentinel. Per design lock §2.1 this is
the expected outcome at the per-neuron level — the triangle is the
**tightest single-neuron convex hull** (`star.pdf` Theorem 1) and
both implementations realize it.

The 20-sentinel gate therefore measures the **mathematical
ceiling at the per-neuron level** and finds production already at
it. There is no per-neuron precision lever left in this prototype's
scope.

### Decision

**Close §7 definitively.** No Phase 2 pair-hull cuts will be
attempted: per the design plan §2.3, Phase 2 was contingent on
Phase 1 showing structural room for improvement, which it did not.
The CIFAR final-tail residual LP UB is a property of the per-neuron
triangle, not an implementation gap.

The project's next-step roadmap reverts to:

- §9 stabilization remains stable at 253 V/A across 5 benches (audit
  verified 2026-06-04 evening: 62/62 FAL receipts have full provenance
  bundles, 0 OOM markers, malbeware dedup clean).
- §10 paper-ready writeup begins now. The closed-negative results
  on (a) CIFAR-ImageHZ (atlas v3), (b) VGG-ImageHZ (§6b/§6c), (c)
  CIFAR final-tail per-neuron hull (this section) are reported honestly
  with their evidence trails.
- Yolo Slice frontend cleanup stays optional and parallel.

No further verification-improvement research lines are open. Any
future ImageHZ or multi-neuron hull work would require fresh
evidence and a NEW gate decision — not a re-opening of this closure.

### Receipts

- `audit_results/cifar_finaltail_hull_phase1_smoke_20260604T050627Z/` (3-iid smoke)
- `audit_results/cifar_finaltail_hull_phase1_sentinels_20260604T050851Z/` (20-iid gate)
- `audit_results/cifar_finaltail_hull_sentinels_20260604.json` (sentinel selection)
- `research/cifar_finaltail_hull_plan.md` (design lock)
- `research/cifar_finaltail_hull_lp.py` (prototype code)

---

## 7-Spec. (Historical) Original §7 spec preserved for traceability

Below is the original §7 spec authoring the prototype. It is
preserved verbatim because the closure above refers to it; do not
re-execute.

### Authorization (now historical)

Authorized by: roadmap §7 (was NEXT RESEARCH TARGET; now CLOSED).
Sentinel selection: `audit_results/cifar_finaltail_hull_sentinels_20260604.json`.

At the time this was written, it was treated as the project's only remaining
live research line for verification improvement. It superseded the original
"Task C: ImageHZ Prototype" content. This section is now historical only:
§7 ran, failed its gate, and is closed-negative.

### Purpose

Atlas v3 (§5/§3.3) placed the canonical-CIFAR precision deficit at the
**final-hidden ReLU triangle slack + dense Gemm magnification**, NOT
at conv-body root correlation. The right intervention is a forward-only
hull tightening of the LAST hidden ReLU only — no ImageHZ, no
benchmark-specific sidecars, no backward, no MILP, no BaB.

### Why this is the right next line

- CIFAR conv body already preserves root factors (atlas v3 ratio = 1.0
  on 185 / 185 iids).
- Failure point is the tail relaxation: 39 unstable hidden neurons
  (median) × `mu_sum = 7.6` (median), magnified by `W_out @ Gc`.
- Tightening the last hidden ReLU alone is small in scope and
  benchmark-independent.
- It stays inside the principles: forward-only, LP-only (HiGHS), no
  CROWN, no MILP, no BaB, no random / corner / PGD candidates.

### Scope

- Targets only the **last hidden ReLU** before the output Gemm.
- Uses forward preactivation bounds (already available from the
  existing CIFAR endcap snapshot path).
- Continuous LP only via HiGHS / SciPy.
- **No new ONNX operator support.** If a snapshot's tail shape
  doesn't match the expected `Gemm → ReLU → Gemm`, the prototype
  fails-closed and the production path keeps ownership.

### Pilot

**20 CIFAR near-boundary sentinels** selected from
`audit_results/cifar_unknown_margin_atlas_canonical_20260603T121947Z/`
as the 20 rows with the lowest positive LP UB that are still UNKNOWN
after the §9 clean canonical sweep. Strict ORT replay required for
any FAL claim; LP-only certificate required for any V claim. Provenance
hashes (canonical root, csv sha256, onnx sha256, vnnlib sha256) must
land on every receipt the prototype emits.

### Stop / Go Gate

Proceed to wider rollout only if either:

```text
>= 3 new V/A receipts on the 20 sentinels
OR
median LP UB reduces by >= 30% across the 20 sentinels
```

Close the line definitively if:

```text
0 new V/A AND median LP UB movement < 10%
```

Do NOT extend the prototype into a benchmark-specific patch under any
outcome. If it works, the productionization path is to widen the same
mechanism to other top-1 robust benchmarks that share the failure
mode (tail relaxation), NOT to wire CIFAR-only env knobs.

### Duration

Estimated: **2–3 days** for the pilot run + gate evaluation.

### Forbidden during this prototype

- No CIFAR-ImageHZ (atlas v3 closed it).
- No VGG-ImageHZ (§6b/§6c closed it).
- No new ONNX operator support.
- No backward, no MILP, no BaB, no random / corner / PGD candidates.
- No benchmark-name-gated env knobs; structural gating only.
- No silent fallback when shapes don't match; fail-closed to UNKNOWN.

---

## 7-Hist. (Historical) Task C: ImageHZ Prototype — CIFAR-STOPPED; conditional on §6b

> **NOTE — all claims in this historical section are superseded by §6b (denominator-bug REOPENED) and §6c (iid 15 dense forensic). Do not execute any of the steps below. They are kept only for traceability of the original plan that was retracted on 2026-06-04 evening.**

### Purpose

If Task A passes the approval gate, build a forward-only image-structured HZ
domain that preserves spatial locality through convolutional bodies without
exploding memory.

### 2026-06-04 status

**CIFAR is no longer the target.** Atlas v3 (§0a) shows CIFAR conv body
already preserves full pixel correlation (`root_ng_to_input_ratio = 1.000`
across 184/184 successful rows), so the ImageHZ value proposition does not
apply to CIFAR. The prototype, if launched at all, will be gated by §6b's
VGG/Tiny mini-atlas, not by §5's CIFAR atlas.

The remainder of this section is preserved as the original task specification
but should be re-scoped to VGG before any code is written.

### 2026-06-04 update — canonical scope lives in a separate plan

§6b gate fired PROCEED. The full prototype scope, invariants, operator
list, phase-by-phase representation-only gate, and the hard "don't do"
list now live in:

```text
research/imagehz_vgg_prototype_plan.md
```

That file is the **authoritative source** for the prototype. Edits to
the scope must be made there first; this §7 section is a pointer.

Summary of what changed vs the original §7 plan:

- The prototype targets **VGG / Tiny conv-body Girard cap reductions**
  (L11 / L18 / L25 / L32 MAXPOOL + L17 / L29 / L35 RELU per the §6b
  reparse), not CIFAR.
- Phase 0 is **representation-only**: chase `root_ng_at_flatten`
  improvement of ≥ 10× on 8 sentinel iids, with no V/A claims and no
  changes to the existing verifier or witness sidecar.
- Phase 1 is the V/A gate: ≥ 1 new V/A or median LP/box margin ≥ 30%
  improvement across 20 sentinels; otherwise the line closes.
- Phase 2 productionization is a separate spec.

The rest of this section is the **historical** task specification and
is preserved for context only.

### Motivation (historical)

Current HZ stores generator columns globally. Dense-conv networks create either:

- many global generator interactions,
- high memory pressure,
- or loose triangle relaxations that create phantom LP candidates.

ImageHZ should preserve locality and structured generator blocks rather than
flattening everything into a giant global matrix too early.

### Minimum Prototype Scope (re-scoped 2026-06-04 — VGG/Tiny, NOT CIFAR)

CIFAR is explicitly excluded — atlas v3 showed CIFAR's conv body already
preserves full pixel correlation, so ImageHZ has no precision lever there.
The scope below targets VGG/Tiny-style conv-body reduction sites instead:

- input image box (ImageNet shape 3×224×224 or Tiny 3×64×64),
- Conv2D,
- ReLU triangle,
- MaxPool2D — needed for VGG; the §6b VGG forensic showed Girard cap fires
  exactly at the MAXPOOL boundary (L32),
- flatten,
- final dense tail export to existing LP sidecar.

Do not support all ONNX operators initially. Residual ADD support is
deferred unless a Tiny-style ResNet variant in the mini-atlas shows
correlation loss at an ADD site.

### Candidate Representation

Potential representation:

```text
ImageHZ:
  c: C x H x W center
  local generators: block/tile/channel-local sparse basis
  root provenance: mapping from local factors back to input root factors
  optional aux factors for ReLU relaxations
  safe export:
    - to SparseGcZ/HZono at FLATTEN if memory allows
    - to root-only FAL candidate generator only when replay-gated
```

### Required Tests

1. Toy Conv2D equivalence against dense HZ for small images.
2. ReLU triangle soundness on random toy boxes.
3. MaxPool2D soundness — output radii must over-approximate the input
   radii on random unit-box tests.
4. Flatten export equivalence on small networks.
5. **No-lost smoke on VGG mini-atlas sentinels** that the §6b gate
   labelled correlation-lost. The prototype must NOT regress them to
   ERROR or BoxHZ.
6. **CIFAR +15 FAL set must remain untouched** — the existing production
   path (single-rival endcap sidecar) keeps owning CIFAR; ImageHZ never
   runs on CIFAR.

### Prototype Benchmark

Use 20 CIFAR UNKNOWN sentinels selected from the atlas:

- 10 `spatial_correlation_fixable`,
- 5 hard negative,
- 5 near-boundary mixed.

### Success Gate

Proceed to production only if at least one is true:

```text
>= 3 new FAL/CERT on 20 sentinels
or
median worst LP margin improves by >= 30%
or
phantom replay failure rate drops by >= 30%
```

No productionization if:

```text
0 V/A and median margin movement < 10%
```

### Duration

Prototype: **1-2 weeks**.

## 7b. Task C+: SUPERSEDED — see §7

This section originally proposed the CIFAR final-tail hull as an
"optional" follow-up, conditional on §7 (then ImageHZ) passing. After
the 2026-06-04 evening ImageHZ-for-VGG closure (§6b/§6c), the CIFAR
final-tail hull was temporarily treated as the project's only remaining live
verification-improvement line and was promoted to §7. §7 has since run and
closed negative; read §7 for the final result, not as an active work item.

## 8. Task D: ImageHZ Productionization — CLOSED / NOT AUTHORIZED (2026-06-04 evening)

This section originally read "Productionization If ImageHZ Works"
and assumed §6b would fire PROCEED for ImageHZ. After the 2026-06-04
denominator audit (§6b REOPENED) and the iid 15 dense-input forensic
(§6c), ImageHZ-for-VGG is closed-negative on all 18 canonical VGG
iids:

- iids 0-14: production HZ already preserves 100% of root factors;
  no precision lever exists for ImageHZ.
- iids 15-17: dense input would face the same L28 CUDA OOM ceiling
  production hits (150 GiB allocation request); even a perfect
  representation cannot navigate that memory wall without a different
  resource model.

**No ImageHZ productionization is authorized.** The
`research/imagehz_lite/` code stays as a historical artifact with
its 9/9 unit-test matrix as evidence the design is sound on toy
cases. It MUST NOT be wired into production. The
`ACT_HZ_IMAGEHZ_PROFILE`, `ACT_HZ_IMAGEHZ_TILE_SIZE`, and other
prefixed env knobs proposed earlier are NOT to be implemented.

If ImageHZ work resumes in a future cycle, it would require:

1. A different representation (FactorField, not TileBlock) that does
   not duplicate factors into per-output-position tiles.
2. A different memory model that avoids the L28 dense fallback.
3. A separate, NEW gate decision based on fresh evidence — not the
   denominator-broken §6b PROCEED that was retracted.

Until then, no productionization, no env knobs, no routing changes,
no CI presence.

## 9. Task E: Stabilize Existing Result — DONE (2026-06-04 night)

This task was completed before §10 handoff. It is retained here as the
checklist that produced the frozen 253 V/A table.

### Purpose

Turn targeted improvements into a clean, reproducible, paper-ready result.

### Completed Work

1. Profile defaults frozen for the canonical evidence run.
2. Canonical stabilization sweep audited and consolidated into
   `audit_results/clean_canonical_combined_summary_20260604.json`.
3. OOM pollution checked:
   - no OOM / watchdog-kill marker appears in FAL receipt stdout tails,
   - high-memory reruns are excluded from the frozen score unless provenance
     and strict replay pass.
4. Consolidated table generated:
   - V,
   - A,
   - UNKNOWN,
   - TIMEOUT,
   - OOM,
   - ERROR,
   - LOST vs baseline.
5. FAL receipts validated:
   - 62 / 62 FAL receipts include full provenance sidecars,
   - strict ORT replay discipline remains mandatory,
   - no receipt means no FAL promotion.
6. Regression / provenance sanity checks completed for the frozen evidence.
7. Clean result memo written at `research/results_20260604.md`.

### Duration

Completed in this improvement cycle. No further §9 rerun is authorized merely
to improve numbers.

## 10. Task F: Documentation and Paper Integration — IN PROGRESS (2026-06-04 night)

### Result document landed

The improvement-phase result memo is now at:

```text
research/results_20260604.md
```

It is the **authoritative source** for the §10 narrative. The structure:

1. One-sentence summary.
2. Positive result — 253 V/A canonical sweep, with the aggregate table,
   provenance audit, and a precise definition of what 253 V/A means
   under the project principles.
3. Negative results — three closed lines (CIFAR-ImageHZ, VGG/Tiny-ImageHZ,
   CIFAR final-tail per-neuron hull). Each subsection cites the
   audit_results directory that produced its evidence.
4. Design principles — why no CROWN, no BaB, no PGD, no MILP.
5. Tooling and reproducibility — a question-to-evidence index.
6. Optional parallel cleanup (cctsdb_yolo Slice frontend) noted as
   non-verification-capability work.
7. Closing statement.

Future paper deliverables (arXiv / conference submission) should be
authored against `results_20260604.md` rather than against this roadmap
section. If new evidence forces a change, edit the results memo first
and update this section to point at the revised version.

## 11. Proposed Timeline — Final State (2026-06-04 night)

CIFAR atlas (§5), VGG denominator audit / dense forensic (§6b/§6c), and CIFAR
final-tail hull (§7) are all DONE. The improvement phase has ended.

### Current Path

| Time | Work |
| --- | --- |
| Done | §9 stabilization: 253 V/A, 62/62 FAL provenance, no OOM/watchdog marker in FAL receipts |
| Done | §6b denominator audit + §6c iid 15 dense forensic — close ImageHZ |
| Done | §7 CIFAR final-tail hull prototype — gate FAIL, close line |
| Now | §10 paper-ready writeup from `research/results_20260604.md` |
| Optional | cctsdb/yolo Slice frontend cleanup, tracked separately from verification score |

Estimated end of improvement cycle: **reached on 2026-06-04 night**.

### Retired paths

The "Narrow Research Path" (ImageHZ prototype, VGG-only) and the
"Productionize per §8" sub-step are RETIRED. ImageHZ for VGG is
closed-negative on every canonical VGG iid (§6b for 0-14, §6c for
15-17). No ImageHZ work will be authorized without a NEW gate based
on different evidence.

## 12. Clear Stop Conditions — Updated 2026-06-04 evening

The improvement phase should stop when one of these is true:

1. **ImageHZ is closed-negative.** Done as of 2026-06-04 evening
   (§6b denominator audit + §6c iid 15 dense forensic). Both the
   CIFAR-ImageHZ direction (closed by atlas v3) and the VGG/Tiny-
   ImageHZ direction (closed by §6b/§6c) are terminated.
2. **CIFAR final-tail hull prototype (§7) is negative by its stop
   gate.** Done as of 2026-06-04 night: 20/20 clean LP parity with
   production, 0 V/A, 0 median LP UB movement.
3. **§9 canonical stabilization is complete.** Done: frozen 253 V/A
   across 5 benchmarks with provenance.
4. Any proposed new direction lacks a benchmark-independent mechanism
   and only targets one dataset with a special-case patch.

Avoid continuing simply because some benchmark remains weak. Weakness
alone is not a reason to keep patching; there must be a general
mechanism with a clear experiment and stop rule.

## 13. What Not To Do Next — Updated 2026-06-04

Do not spend time on:

1. VGG 3-Dense LP while VGG snapshot is BoxHZ.
2. soundnessbench tail LP without improving conv-body precision.
3. root-only LP CERT claims.
4. benchmark-name-based environment knobs without structural gating.
5. random or corner sampling candidates.
6. branch-and-bound or input splitting.
7. CROWN-like backward bound propagation.
8. another output-cut-only variant unless it is tied to a measured margin
   deficit and has a stop rule.
9. **ImageHZ targeting CIFAR.** Atlas v3 (§0a / §5) showed CIFAR conv body
   preserves full pixel correlation; ImageHZ has no precision lever there.
10. **A++ closed-form / topK / per-iid sidecars for CIFAR score lift.**
    Closed-negative on canonical 185 UNK; further patches are forbidden.
11. **Citation of the polluted memory entries** `p0-fal-diff-clean`,
    `p0-unknown185-done`, `corrected-atlas-v1`, or the LOCAL-pool 15→31 framing.
    Those are flagged RETRACTED in `MEMORY.md`.

## 14. Immediate Next Commands / Scripts To Prepare — Updated 2026-06-04 night

All three old verification-improvement directions are now closed-negative
(§5 CIFAR-ImageHZ, §6b/§6c VGG-ImageHZ, §7 CIFAR final-tail per-neuron hull).
Do not reopen them.

There is one new research track to prepare: **SC-HZ Phase A**. It must be
implemented under `research/sc_hz/` and evaluated by the 80-sentinel gate in
§0. It is not a production profile and does not modify the 924 row unless it
passes.

The §9 stabilization subset is already complete and audited (253 V/A, 62/62
FAL provenance, 0 OOM markers, malbeware dedup clean) — do not re-run it merely
to improve numbers.

```bash
# Provenance smoke — sanity check before any new audit-style work
python research/canonical_provenance.py

# SC-HZ Phase A — implement only after the design lock is reviewed
# Target layout:
#   research/sc_hz/precompute_direction.py
#   research/sc_hz/prune.py
#   research/sc_hz/pruned_forward.py
#   research/sc_hz/run_sentinels.py
# Gate:
#   80 sentinels, >=5 new V/A OR median LP UB reduction >=25%.

# §10 paper writeup continues in parallel; it must distinguish:
#   full row = 924 V/A
#   five-benchmark stabilized subset = 253 V/A

# Optional parallel: yolo / cctsdb Slice frontend cleanup
# Reduces 39 ERROR rows in cctsdb_yolo_2023; not a verification-capability
# improvement. Run only if the final table needs cleaning before paper.
# Do NOT package this as a new V/A result.
```

NOT on the next-step list (do not run these):

- Any §7 Phase-2 pair-hull / multi-neuron variant. Phase 1 gate FAIL
  established production is at the per-neuron triangle math ceiling;
  Phase 2 was contingent on Phase 1 showing room and is therefore
  forbidden without fresh coupling evidence.
- Any ImageHZ work (TileBlock or FactorField) on CIFAR or VGG. All
  three ImageHZ directions are closed-negative.
- Any A++ closed-form / topK / per-iid sidecar variant for CIFAR score.
- Any benchmark-name-gated env knob.
- Any rerun of the §9 canonical sweep "to improve numbers".
- Any SC-HZ production/default change before the Phase A gate passes.

## 15. Final Recommendation — Updated 2026-06-04 night

After atlas v3 (§0a/§5), §6b denominator audit, §6c iid 15 dense
forensic, and the §7 final-tail per-neuron hull gate FAIL, the
picture for the old patch-cycle improvement phase is final:

- **CIFAR conv body is not the bottleneck.** Full pixel correlation is
  preserved natively (`root_ratio = 1.000` in 184/184 successful atlas
  v3 rows).
- **The remaining CIFAR gap at the per-neuron level has no lever
  left.** §7 Phase 1 showed the clean LP from spec is bit-exact with
  production endcap LP across 20 near-boundary sentinels (median
  reduction = 0.0000%, 0 new V/A, all 20 parities = 0). Production
  is at the per-neuron triangle math ceiling, which is the
  provably-tight single-neuron convex hull.
- **ImageHZ has no precision lever on canonical CIFAR.** Closed by
  atlas v3.
- **ImageHZ has no precision lever on canonical VGG either.** Closed
  by §6b (iids 0-14 production preservation = 100%) and §6c (iids
  15-17 hit the same L28 CUDA OOM ceiling production hits).
- **A++ closed-form, topK, per-iid sidecars for CIFAR score are all
  closed-negative.**
- **The local-pool 15→31 framing remains retracted.** Canonical
  CIFAR FAL = +15.
- **Clean canonical sweep across 5 benches gives 253 V/A**:
  CIFAR 15 + Tiny 35 + nn4sys 101 + malbeware 102 + yolo 0 (parser-blocked).
  Audit-verified 2026-06-04 evening: 62/62 FAL receipts carry the
  full provenance bundle, 0 OOM markers, malbeware dedup clean.
- **Full cross-tool ACT/HZ row is 924 V/A.** This is the number to use when
  discussing absolute competitiveness against abcrown / NeuralSAT / nnenum /
  PyRAT. The 253 row is a subset, not the headline total.

The three remaining structural ways to lower the residual LP UB are
all either prior-negative or out of principle scope:

| Direction | Status | Why excluded |
|---|---|---|
| Multi-neuron / joint relaxation (Singh PRIMA k=2/k=3, Anderson 2020 facets) | prior-negative on acasxu; §7 Phase 2 contingent on Phase 1 PROCEED, which did NOT happen | no fresh coupling evidence |
| Branch-and-bound / input splitting | excluded by design principle | I3 in `research/cifar_finaltail_hull_plan.md`, `feedback_design_principles` |
| Backward bound tightening (CROWN-like) | excluded by design principle | I1, `feedback_design_principles` |

The recommended next steps, in order:

1. Keep `research/canonical_provenance.py` mandatory for all research
   scripts.
2. **Freeze the old patch-cycle results.** Do not re-run §9 or reopen ImageHZ /
   final-tail hull / CIFAR sidecars to improve numbers.
3. **Use 924 V/A as the current full-row baseline.** Use 253 V/A only as the
   five-benchmark provenance-stabilized subset.
4. **Start SC-HZ Phase A if the project wants further capability gains.**
   This is a new abstraction hypothesis, not a continuation of benchmark
   patching. Gate: 80 sentinels, PASS iff >=5 new V/A OR median LP UB reduction
   >=25%; FAIL iff 0 new V/A and median reduction <10%.
5. **Continue §10 paper writeup in parallel.** The contents:
   - positive: 924 V/A full row plus 253 V/A provenance subset,
   - negative: three closed lines (§5 CIFAR-ImageHZ, §6b/§6c
     VGG-ImageHZ, §7 final-tail per-neuron hull),
   - conclusion: the forward HZ + per-neuron triangle ceiling has
     been mapped for the old profile family; further lift requires a
     benchmark-independent abstraction change such as SC-HZ.
6. **Optional parallel**: yolo / cctsdb Slice frontend parser cleanup.
   Reduces 39 ERROR rows; not a verification-capability result. Do
   not package as a V/A improvement.
7. If another future research hypothesis emerges, it MUST come with NEW
   evidence and a NEW gate — never as a re-opening of any of the closures above.

Items NOT on the next-step list: A++ continuation, topK=10, atlas v2
citation, LOCAL-pool P0 receipts, benchmark-name-gated sidecars,
CIFAR-ImageHZ, VGG-ImageHZ, ImageHZ productionization, §7 Phase-2
pair-hull, CROWN, BaB, PGD, MILP.
