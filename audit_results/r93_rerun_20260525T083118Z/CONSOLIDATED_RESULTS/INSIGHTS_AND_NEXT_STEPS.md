# Insights & next-step directions

**Source**: 7-round experiment cycle (2026-05-25 → 27), 30000+ instances across
25 VNN-COMP 2025 scored benchmarks on r93 ACT snapshot. Every insight below
cites the specific empirical observation that inspired it.

The directions are organized into **theoretical** (where there's an underlying
mathematical or representational reason a class of cases is hard) and
**engineering** (where we know how to do it but haven't), with **expected
gain** and a concrete **first experiment** for each.

---

## Theoretical directions

### T1. Device-numerical diversity as a sound coverage amplifier

**Evidence (sat_relu CPU/GPU divergence audit, SAT_RELU_CPU_GPU_DIVERGENCE_AUDIT.md)**:
- CPU canonical: 1V / 18A / 81U
- GPU canonical: 1V / 21A / 78U
- 5 stable iid divergences (34, 50, 56, 86, 92) — same code, same input, same
  ReLU encoding; only the BLAS / CUDA reduction order differs.
- All 5 divergent witnesses pass independent `strict_replay_for_act` ORT
  zero-tolerance check (i.e. they ARE real counterexamples for the network).
- **CPU∪GPU = 22 sound FAL** vs either-alone max 21.

**Mechanism**: ACT's `check_unsafe_for_act` LP relaxes `xi_b ∈ {-1,+1}^nb`
to `[-1,1]^nb`. The LP feasible region (HZ ∩ unsafe) typically has multiple
vertices. The simplex / HiGHS solver picks ONE vertex per call. Float
summation order on CPU vs CUDA gives ε-different objective values → different
pivot path → different vertex → different `xi_star` → different back-projected
witness `x_star`. Both vertices are sound; both witnesses pass `strict_replay`.

**Why this is a free lunch**: no algorithmic change, no soundness risk
(the `strict_replay_for_act` step in §7 of HZ_VERIFICATION_FLOW.md filters
spurious LP-relaxation witnesses regardless of which device produced them).

**Inspiration framing**: similar to ensemble methods in classical ML, but
operating on numerical-precision diversity rather than model diversity.

**Expected gain**: 5-15% more FAL on benchmarks with multiple sound witnesses
(any benchmark with FAL count ≥ 5). Across our archive that's malbeware /
collins_rul / acasxu / sat_relu / safenlp / cora / tinyimagenet — maybe
+20-40 additional FAL.

**First experiment**: for every FAL produced by CPU, also run that iid on GPU
and vice versa; take union of strict-clean witnesses. Should be `act_run`
wrapped with a `device_diversity` flag.

---

### T2. RSS-cap is the true ceiling for convolutional benchmarks, NOT precision

**Evidence (metaroom +52 phenomenon, FINAL_RESULTS_TABLE.md §C)**:
- metaroom_2023 CPU (24 GiB RSS cap): 37 CERT + 60 RSS-cap + 2 TO + 1 UNK
- metaroom_2023 GPU (96 GiB available): **89 CERT + 10 UNK + 1 ERR**
- Same algorithm, same network, same ReLU encoding. The ONLY change is
  available memory. **+52 CERT** is the largest single-benchmark gain in
  the entire archive.
- Same pattern: tinyimagenet CPU all RSS-cap → GPU runs through and even
  finds 1 FAL; traffic_signs CPU 15 RSS-cap → GPU 100% runnable.

**Mechanism**: HZ representation grows generators with depth. For ResNet-style
metaroom (k convolution layers, d channels), generator-matrix `Gc` is
`O(num_vars × ng)` where `ng` grows with each new linear op. Skip connections
and equality-Lagrangian ReLU push `ng` further. By the time the HZ reaches the
classifier, dense `Gc` exceeds 24 GiB.

**Theoretical insight**: this is a *representational* limit, not a
*precision* limit. The algorithm IS capable of verifying these instances
(GPU proves it). The CPU just runs out of physical memory before reaching
the verdict.

**Two improvement paths**:

| Path | Type | Expected gain | Reference |
|---|---|---|---|
| Default GPU for large-CNN benchmarks | Engineering | already-validated +52 metaroom | E2 below |
| **Sparse-Gc HZ representation** | Theoretical | unlocks the same benchmarks on CPU; could squeeze 20-50 CERT out of cifar100 / vggnet16 / tinyimagenet | `project_phase3_sparsegc` memory note + Bird PhD §4.3 |

**First theoretical experiment**: re-implement HZ with CSR-sparse `Gc` storage,
threshold-prune generators with `|Gc_col|` below ε, measure RSS on cifar100
iid 0 with bound 24 GiB. If sparse drops RSS below cap → high-payoff direction.

---

### T3. Encoding-compatibility cliff — the cora 1-of-9 selectivity

**Evidence (cora_2024 model-family breakdown)**:
- cora 180 instances span 9 model variants: `{mnist, cifar10, svhn} × {point, trades, set}`.
- 15 CERT achieved by ACT come **exclusively from `mnist-set.onnx`** (15/20 = 75% decision rate on that model).
- Other 8 variants (mnist-point, mnist-trades, cifar10-{point/trades/set},
  svhn-{point/trades/set}): **0 CERT each across 160 instances**.

**Mechanism hypothesis**: the `-set` defended networks were trained with
SET-based adversarial training (loss optimised over an abstract set, not a
single perturbed point). The defense aligns the network's decision boundary
with abstract-domain natural boundaries. Verifier (operating in abstract
domain) finds the safe set boundary easily.
- `-point`: standard adversarial training (loss on single PGD point) → boundary
  not aligned to abstract set → verifier over-approximation falls outside.
- `-trades`: PGD + KL regularisation → similar story to point.

**Theoretical implication**: verifier capability is not just abstract-domain
expressiveness; it's the structural compatibility between (training-time
abstract domain) and (verification-time abstract domain). Defenders who
verify-aware-train get free verification.

**Inspiration**: classic verification-friendly training (Mirman, Cohen-Wong) is
known. What's NEW from cora is the EMPIRICAL CLIFF — within a single benchmark,
varying only the defense type yields 75% → 0% verification rate. This is a
strong publishable observation about defense → verifier coupling.

**First experiment**: run sat_relu / malbeware / metaroom with both `-set`
and `-point` training and quantify the cliff. If consistent, this becomes
a §X of the ACT paper.

---

### T4. TO-bound vs RSS-bound is an orthogonal 2D dimension — different research priorities

**Evidence (Round 5 separate-bucket reporting, MASTER_INDEX §E)**: before
the watchdog separation, all bounded outcomes were lumped as `UNKNOWN`. After
splitting:

| Benchmark | TO frac | RSS frac | Diagnosis |
|---|---|---|---|
| cora_2024 | 90% (162/180) | 0% | wall-bound (algorithm) |
| relusplitter | 71% (156/220) | 7% (15) | wall-bound + minor RAM |
| tinyimagenet_2024 | 0% | 100% (200/200) | RSS-bound (representation) |
| cifar100_2024 | 0.5% (1) | 50% (100) | mostly RSS-bound |
| ml4acopf_2024 | 23% (16/69) | 0% | wall-bound |
| nn4sys | 31% (61/194) | 8% (15) | mixed |

**Theoretical insight**: "verifier capability" decomposes into two orthogonal
dimensions:
1. **Algorithmic precision** — what the analyzer would conclude given infinite resources.
2. **Computational efficiency** — does the analyzer finish within wall + memory.

A wall-bound benchmark (cora) benefits from algorithmic acceleration (faster
LP, better warm-start, parallel ReLU branches). An RSS-bound benchmark
(tinyimagenet) needs sparser representation — adding wall would not help.

**Why this matters**: prior verifier-comparison reports treat all `UNKNOWN`
the same, obscuring which research direction would help. Our separated
reporting cleanly identifies the target.

**Inspiration**: classical complexity theory's time vs space distinction
applied to verifier benchmarking.

**First experiment**: for each TO-bound benchmark, rerun with wall=300s
(5× current). Quantify how many TO migrate to CERT. This calibrates how
much algorithmic-speedup work is worth on each.

---

### T5. ACT FAL + ORT replay > consensus of mature verifiers (collins_rul + tinyimagenet)

**Evidence (SOUNDNESS_VS_VNNCOMP_OFFICIAL.md)**:
- collins_rul_cnn_2022 iids 0 / 22 / 47: 6 official tools (α,β-CROWN, NSAT,
  PyRAT, CORA, NNV, NNenum, SB) all report UNSAT under zero_tol AND small_tol.
- ACT reports FAL with `spec_zero_tol_holds=True` + `input_box_holds=True`
  receipts.
- Independent ORT replay (`onnxruntime` evaluating the network at
  `x_star`): 1000/1000 random box samples produce Y_0 ≈ 165 (unsafe region
  threshold: Y_0 ≤ 196.977).
- tinyimagenet iid 6: same pattern — official zero_tol UNSAT, ACT FAL with
  strict-clean receipt, ORT confirms.

**Mechanism**: official tools likely use LP-relaxed unsafe checks with a
small numerical slack, OR their candidate inputs land at corners that miss
the actual unsafe region. ACT's pipeline:
1. HZ over-approximation guarantees output containment.
2. LP feasibility on (HZ ∩ unsafe) finds a candidate `xi_star`.
3. Back-projection to `x_star` and **strict_replay via ORT** with zero
   tolerance: only kept if `y_ort` strictly satisfies unsafe rows.
4. Receipt SHA-binds model + spec + x_star + y_ort.

This 4-step pipeline is TIGHTER than what individual abstract verifiers
provide. The ORT check is the network's own ground truth.

**Theoretical implication**: even when 6 abstract verifiers agree, the
concrete network may produce a counterexample they all missed. Trust the
network + concrete replay over abstract-tool consensus.

**Why this is publishable**: it suggests VNN-COMP scoring should mandate
witness replay (not just trust the verifier's verdict). Currently, FAL is
"accept on tool's word".

**Inspiration**: 4-color-theorem-style independent verification: a proof by
abstract argument should always be checkable by a constructive witness.

**First action**: propose to VNN-COMP organizers: SAT submissions must
include `x_witness.npy` + `y_replay.npy` with sha256 chain; harness re-runs
via ORT and only accepts if witness violates spec zero-tolerance. This
mechanically resolves the collins_rul-style disputes forever.

---

## Engineering directions

### E1. Multi-pass progressive wall + parallel ReLU encoding race

**Evidence (cora_2024 wall_s distribution)**:
- 14 mnist-set CERTs ran 7-50s each (median 16s).
- 162 wall-TO instances all hit 68s (60s wall + grace).
- Burning 60s on hard instances BEFORE finishing the quick ones is
  inefficient use of the total `wall × num_inst` budget.

**Proposed schema**:

| Pass | Wall | Targets | Captures |
|---|---|---|---|
| 1 | 10s | all instances | the "quick CERT" cluster (mnist-set on cora, easy iids elsewhere) |
| 2 | 60s | UNK from pass 1 | the "moderate decisions" |
| 3 | 300s | UNK from pass 2 | the last squeezable few |

Within each pass, run **multiple ReLU encodings in parallel** (`triangle`,
`eq_lagr_v8`, `chull` per the `large_cls_proof_mode` registry); first
decision wins.

**Mechanism**: different encodings have different precision-vs-runtime
tradeoffs. `triangle` is fast but loose; `eq_lagr_v8` is tight but slow.
For easy instances `triangle` already decides, no need to wait for `eq_lagr_v8`.

**Expected gain**: 2-3× faster wall to same coverage at same total budget,
OR 10-20% more CERT at same wall.

**Inspiration**: SAT-solver portfolio approach (Hutter et al.) — different
algorithms win on different instances; running a portfolio in parallel
maximizes net coverage.

**First experiment**: implement `act_run_progressive <bench> <iids>` that
runs 3-pass with race. Measure on cora + relusplitter + ml4acopf. Expected:
+10-20 CERT at unchanged total compute.

---

### E2. Default GPU for "structurally large CNN" benchmarks

**Evidence**: metaroom_2023 CPU 37 → GPU 89; tinyimagenet GPU finds 1 FAL CPU
can't; traffic_signs CPU 15 RSS-cap → GPU 0; cifar100 CPU 100 RSS-cap → GPU
0.

**Mechanism**: same as T2 — GPU has 80-96 GiB available; CPU has 24-32 GiB
cap. Large-CNN HZ representation exceeds 24 GiB consistently.

**Engineering action**: in `act_run` add a `--auto-device` flag that
inspects `onnx_model` size + layer count and routes to `cuda` if exceeds
threshold. Thresholds calibrated from our data:
- model file > 50 MB → cuda
- conv-layer count > 20 → cuda
- input numel > 100_000 → cuda

**Expected gain**: 30-60 more CERT across vggnet16 / cifar100 / tinyimagenet
if combined with T2 sparse-Gc work; otherwise marginal except metaroom-style
"capacity-bound" cases.

---

### E3. Universal LUT_ENVELOPE for dynamic-control-input ops

**Evidence (R17 LUT_BOUNDS scaffold + Fix #8 candidate)**:
- cctsdb_yolo_2023: `OnnxSlice at slice_23: cannot resolve starts/ends`.
  Starts/ends come from variable input `X_12288`, `X_12289` (each spans 0..62).
- cgan_2023 iids 18/19/20: `OnnxResize at resize: cannot resolve scales or
  sizes`. Same pattern: scales tensor produced by a Constant whose value
  flows through arithmetic on a variable.
- ml4acopf occasionally has Reshape with computed shapes.
- These are all currently treated as conversion-time hard errors.

**Mechanism**: each of these ops takes (data, control_param) where
control_param's value is constrained but not constant. For sound HZ
propagation we need a static envelope of the op's output across all
control_param values in the constraint set.

**Generalized LUT pattern** (extends R17 from Slice to all ops):

```
sound_op_envelope(op, data, control_lb, control_ub):
    candidate_outputs = []
    for control_value in integer_lattice(control_lb, control_ub):
        candidate_outputs.append(op(data, control_value))
    out_lb = elementwise_min(candidate_outputs)
    out_ub = elementwise_max(candidate_outputs)
    return Layer(kind=LUT_BOUNDS, params={'lb': out_lb, 'ub': out_ub})
```

**Engineering action**: wrap this as `act/back_end/lut_envelope_dispatch.py`
with a registry: `{OnnxSlice: ..., OnnxResize: ..., OnnxReshape: ...}`.
Each handler computes integer-lattice candidates and returns LUT_BOUNDS.

**Expected gain**:
- cctsdb_yolo_2023 (currently 0/39 supported) → fully unblocked.
- cgan_2023 small_transformer (3 GPU ERR) → cleared.
- 5-10 not-yet-attempted models in real-world VNN-COMP would also unblock.

**Inspiration**: precondition-strengthening in symbolic execution. Don't
specialize the path; specialize the envelope of all reachable paths.

---

### E4. Regression test pack — catch silent breakage between fixes

**Evidence (Round 4 raw-first regression)**: Round 4 ACT fix (raw-first
ONNX) was tested only on nn4sys (its target) and committed. It silently
broke ml4acopf / lsnc_relu / yolo / collins_aero — all 4 went 100% ERROR.
We only caught it during GPU full sweep ~20 hr later.

**Engineering action**: maintain a `regression_pack.sh` that runs 8-10
representative iids covering distinct op patterns + recent fix areas:

```
acasxu_2023      iid 0     # small dense
collins_rul_cnn  iid 0     # CNN with FAL (Conv1D / receipt path)
malbeware        iid 0     # CNN with FAL  
acasxu_2023      iid 22    # has FAL (LP witness path)
ml4acopf_2024    iid 0     # transpose-heavy (catches Fix #5 / #7 regressions)
lsnc_relu        iid 0     # simple ReLU (catches Fix #5)
nn4sys           iid 137   # mscn family (catches the original Fix #1-#4 territory)
collins_aero     iid 0     # LeakyReLU + Upsample (Fix #6 + #7)
safenlp          iid 0     # large LP
vit_2023         iid 0     # smoke for blocker tracking
```

Each pre-commit fix: `regression_pack.sh` < 5 min. Catches "fix A broke B"
before merge.

**Inspiration**: classical CI regression-test culture, applied to numerical
verifier code which has no obvious correctness oracle other than "did the
verdict change unexpectedly?".

**Expected gain**: prevents the type of silent regression that cost us 20 hr
+ 8 hr Round 5 rerun. Sustained engineering velocity.

---

### E5. SHA-bound receipts as a community standard

**Evidence (collins_rul + tinyimagenet disputes)**: when ACT and 6 other tools
disagree, our receipts let us PROVE which side is right via SHA-bound
artifacts. No other tool in VNN-COMP 2025 produces this.

**Action**: propose to VNN-COMP organizers a `vnncomp-receipts/` directory
standard:
- `<bench>_iid<i>_<verdict>.json`: sha256(model) + sha256(spec) + sha256(witness)
- `<bench>_iid<i>_<verdict>.x_star.npy`: the actual witness
- `<bench>_iid<i>_<verdict>.y_ort.npy`: y from ORT replay

Scoring harness then ORT-replays every SAT and rejects if violation
not zero-tolerance reproducible.

**Inspiration**: cryptographic commitments + replay attacks → analog: a
SAT claim must come with a proof that's externally checkable.

**Expected gain**: structural elimination of label disputes; sets ACT
apart from other tools as the only producer of replayable witnesses.

---

## Prioritized roadmap (next 3 items by ROI)

| # | Direction | Cost | Expected gain | Risk |
|---|---|---|---|---|
| 1 | **E1 progressive wall + portfolio** | 1 week | +10-20% CERT at same compute | Low (proven by SAT portfolio literature) |
| 2 | **T2 sparse-Gc HZ representation** | 2-4 weeks | +20-50 CERT on cifar100/vggnet16/tinyimagenet | Medium (re-validate soundness on existing benchmarks) |
| 3 | **E3 universal LUT_ENVELOPE** | 1-2 weeks | unblocks cctsdb + cgan_small_transformer | Low (extends existing R17 scaffold) |

Items 4-10 (T1/T3/T4/T5/E2/E4/E5) are all lower-effort follow-ups with
incremental gains; their value is more in narrative + reproducibility +
defending the soundness story for the paper.

---

## Summary of inspiration sources

| Direction | Inspired by which experiment / observation |
|---|---|
| T1 device diversity | sat_relu 5 stable iid diffs (Round 2 audit) |
| T2 RSS-cap ceiling | metaroom_2023 CPU 37 → GPU 89 (Round 4 GPU sweep) |
| T3 encoding-compatibility cliff | cora 15 CERT all on `mnist-set` (Round 3) |
| T4 TO vs RSS dichotomy | Watchdog separate-bucket reporting in Round 5 |
| T5 abstract + ORT > consensus | collins_rul iids 0/22/47 + tinyimagenet iid 6 audits |
| E1 progressive wall | cora wall_s distribution: 7-50s CERTs vs 68s TO |
| E2 default GPU large CNN | metaroom + tinyimagenet contrast |
| E3 LUT_ENVELOPE generalization | cctsdb + cgan Round 4 errors with the same pattern |
| E4 regression pack | Round 4 silent regression of 4 benchmarks |
| E5 SHA receipts | collins_rul dispute resolution |
