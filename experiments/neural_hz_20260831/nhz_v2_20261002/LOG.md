# Neural-HZ v2 research log (session started 2026-10-02)

Append-only. Every entry states: hypothesis, what ran, exact files, result,
decision. Scores: formal 1870/2413 and E0 61/400 are unchanged unless an entry
explicitly records a full single-path replay (none so far).

Provenance for every entry unless stated otherwise: branch `redu-hz`, HEAD
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`, tracked binary diff SHA-256
`29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5`
(unchanged; this series edits no production file). Python
`/data1/Kane/miniconda3/envs/act-py312/bin/python` (torch 2.9.1+cu128),
GPU NVIDIA RTX PRO 6000 Blackwell 96 GB (shared with an unrelated
alpha-beta-CROWN job of another session).

## 2026-10-02 N000 — orientation facts that change the plan

1. **Baseline intermediate bounds.** The frozen 1870 portfolio
   (`/data1/Kane/HyZor/ACT_scripts_live_backup_20260630/hz_full_driver.py`,
   `hz_full_worker.py`) runs with `tight_bounds=True` for every family
   (`NO_TIGHT = frozenset()` in `hz_profiles_local.py`): each unstable ReLU
   gets two CPU LPs over the current HZ relaxation. Its own comment states that
   for `cora_2024` and `relusplitter` "the bottleneck is the 2n tight-bounds
   LPs". The terminal is scipy LP + HiGHS MILP (Gurobi off; the local Gurobi is
   a size-limited license).
2. **Baseline is a multi-version composite.** The worker imported
   `REPO=/data1/Kane/ACT`, which is now on branch
   `hybridz-pr3-solver-20260828` (commit 7f8f847a2) and no longer contains
   `_relu_tight_bounds`. The 2,213-row overlay is dated 2026-08-22. Together
   with `BASELINE_LOCK.md` ("composite tuple, no single 2,413-row CSV") this
   means no single current code path is known to reproduce 1870. Any formal
   promotion first needs a single-path baseline reproduction (or a candidate
   path that dominates it on all 1,870 rows).
3. **Current ACT HZ ReLU encoding loses correlation in its free projection.**
   `tf_mlp.sparse_hz_apply_relu_exact` writes an unstable output as
   `u/2 - (u/2) xi_2` with a fresh factor; the input/output coupling lives only
   in an equality row. The constraint-free ("fast") bound of every downstream
   neuron therefore treats each unstable ReLU as an independent box `[0,u]`.
4. **"configure selective"** in the goal text matches the VMCAI paper's
   configurable `K`-seg selective nonconvexity for sigmoid/tanh
   (`/data1/Kane/HyZor/VMCAI_2027___Kaijie_Guanqin/main.tex`, Sec. on
   configurable smooth-activation enclosures). This resolves the D109 open
   question about the comparator.
5. **External reference points.** `/data1/Kane/HyZor/FINAL_RESULTS_SUMMARY.md`
   lists NNV 190/200 on cifar100 and alpha-beta-CROWN 175/175 on
   tinyimagenet (older universe sizes). A local alpha-beta-CROWN run on E0
   CIFAR row 0 (attack skipped, 100 s) gives alpha-CROWN root global lower
   bound -0.5584 with 98/99 OR specs verified, then BaB TIMEOUT. Diagnostic
   only; never a verdict source here.

## 2026-10-02 N001 — projection-aligned shadow census (diagnostic)

Hypothesis: re-parametrising the exact ReLU graph so that its free projection
is the DeepZ ReLU image (`y = lam x + mu + mu eta`) gives much tighter
constraint-free bounds than the current fresh-factor encoding, on GPU.

Code: `gpu_shadow.py`, `run_n001_shadow_census.py` (float32, no directed
rounding, no verdict authority). Rows: all E0 UNKNOWN rows, model/spec SHA-256
checked against the v2 evidence ledgers.

| family | rows | mode | shadow CERT | median min-margin LB | best | median unstable |
|---|---:|---|---:|---:|---:|---:|
| CIFAR100 | 175 | fresh (current ACT) | 0 | -6.473 | -0.526 | 1820 |
| CIFAR100 | 175 | aligned (N001) | 0 | -0.965 | -0.169 | 1747 |
| CIFAR100 | 175 | interval | 0 | -115.3 | -39.7 | 2771 |
| Tiny | 164 | fresh | 0 | -8.215 | -1.518 | 1818 |
| Tiny | 164 | aligned | 0 | -1.048 | -0.128 | 1764 |
| Tiny | 164 | interval | 0 | -409.3 | -169.9 | 2394 |

Results: `results/n001_census_{cifar,tiny}_unknown_v1.jsonl`. GPU wall about
0.03 s per instance and mode. Decision: alignment is a real 6-8x margin
improvement over the current fast bounds but certifies nothing by itself;
unstable counts barely move because they are dominated by layers whose bounds
the shadow cannot tighten.

## 2026-10-02 N002 — GPU-batched LP tightening on the aligned HZ (probe)

Hypothesis: the baseline's own LP-tight step, solved on GPU over the aligned
encoding, reaches alpha-CROWN-root quality in about a second per instance.

Construction (`gpu_aligned_lp.py`): in the aligned encoding the triangle's
upper facet is implied by `eta <= 1`, so the LP relaxation needs only the two
lower-facet rows per unstable ReLU (`-y <= 0`, `x - y <= 0`). The MILP rows
`y <= u d`, `y <= x - l(1-d)` complete exactness. For any `nu >= 0`,
`c + nu^T b + ||g - A^T nu||_1` bounds `max c + g^T w` (weak duality), so every
iterate of a batched projected-Adam ascent is a valid bound; `nu = 0` is the
aligned shadow bound. Labelled as GPU acceleration of the baseline's existing
LP query, not as a domain innovation.

Ground truth (`probe_n002_highs_check.py`, CIFAR row 0, 200-iteration
intermediate bounds): GPU dual 0.6120 vs HiGHS exact LP 0.5952 for the worst
objective (class 53), 0.29 s vs 1.58 s per objective. Unstable counts per ReLU
after tightening: 544, 326, 122, 35, 8, 53, 0, 0, 0, 29 (1117 total). The
terminal LP lower bound -0.595 matches alpha-CROWN root -0.558 within the
intermediate-bound optimisation gap: the remaining gap is the single-neuron
convex barrier, not solver inaccuracy.

Full E0 UNKNOWN sweep launched (`results/n002_lp_{cifar,tiny}_unknown_v1.jsonl`).

N002 sweep outcome: CIFAR 175/175 rows completed
(`results/n002_lp_cifar_unknown_v1.jsonl`): LP-certified probe 4/175
(rows 28, 29, 179, 187), median terminal LP margin -0.455, 2.5-4 s/row.
The TinyImageNet sweep (`results/n002_lp_tiny_unknown_v1.jsonl`) **crashed
after 16 rows with CUDA OOM** (dense 64x56x56 generator tensors, about 8.8 GB
each, while another session's vLLM job and three of my jobs shared the GPU).
The partial file is retained; Tiny is covered by N007 with engine n003.3.

## 2026-10-02 N003 — general GPU engine (`nhz_engine.py`)

Generalised ONNX support (MatMul, Gemm, Add/Sub/Mul/Div with constants, Conv,
BN, Flatten, Reshape, Relu), VNNLIB parser for box inputs and disjunctions of
conjunctions of linear output atoms, terminal LP per unsafe disjunct
(`certify_disjuncts`). Engine self-test against ONNX Runtime on a random input:
max abs error 4.2e-5 (Cora mnist-point, float32). Versions: n003.2 releases
intermediate states by consumer count; n003.3 adds the batched
`terminal_query`, which also returns the box maximiser at the best multipliers
(witness candidate).

Cora smoke (`results/n003_smoke_cora_v2.jsonl`): set-trained models are
LP-certified (already CERT in the baseline); point/trades models have all
250 neurons of layers 2-7 unstable under the input radius 0.1 and violation
upper bounds around 1e4. These rows are almost certainly SAT; certification is
hopeless there.

## 2026-10-02 N004 — engine probe over the formal 543 unsolved rows

`run_n004_unsolved_probe.py`, manifest-driven with SHA-256 checks. Partial
result (163 rows when read): no new LP-certified row in acasxu (66), metaroom
(5), safenlp (1) or the first relusplitter rows; linearizenn fails on the
unsupported `Slice` operator. Expected: the baseline already used LP-tight
bounds on every family, so LP-level precision alone cannot add formal solves.

## 2026-10-02 N005 — terminal-LP witness decoding (primal heuristic)

`probe_n005_lp_witness.py` decodes the input part of the terminal LP solution
(box maximiser at the dual multipliers) and validates it with ONNX Runtime
against the original VNNLIB property. One decode per disjunct, no iterative
input search, no gradients of the concrete network. Classified as a
terminal-query primal heuristic (same role as LP rounding inside a MILP
solver), default-off, never credited as a representation gain; whether it is
acceptable under the no-attack rule is flagged for the user.

- Cora smoke: 4/5 mnist-point rows yield ORT-validated counterexamples; 0/5
  mnist-trades rows.
- E0 CIFAR historical-ADV rows: 17/21 re-found before the run was stopped
  (stopped by me because it was slow and held 36 GB; partial file retained;
  superseded by N007).

## 2026-10-02 N006 — gap attribution (first attempt failed: CUDA OOM)

Rerun pending. N008b below gives the per-layer triangle slack at the exact LP
optimum instead.

## 2026-10-02 N007 — single-path E0 diagnostic over all 400 rows (running)

`run_n007_e0_pipeline.py`, fixed configuration for every row: aligned
propagation with 300-iteration LP tightening, batched terminal LP (1000
iterations), witness decoding of open disjuncts, ORT validation. After 125
CIFAR rows: UNKNOWN->OPEN 106, UNKNOWN->LP_CERT_PROBE 2,
VALIDATED_ADV->ADV_ORT 13, VALIDATED_ADV->OPEN 4. The four lost historical ADV
rows mean this configuration does not yet satisfy the E0 retention rule.

## 2026-10-02 N004-sound — rigorous rounding engine (`nhz_sound.py`, n004.1)

Implements THEORY.md Section 5: float64 value map, per-coordinate rounding
radius `e`, `gamma_n` pads valid for any summation order, upward-rounded
`mu`, rows relaxed by the radii, terminal objective enlarged by `|a|^T e`, and
float64 re-evaluation of every dual bound with an explicit pad. Multipliers
from a float32 optimiser (any `nu >= 0` is valid).

Test (`test_n004_sound_engine.py`, rows 28/29/179/187 of E0 CIFAR UNKNOWN):

| row | rigorous worst disjunct UB | probe UB | max output rounding radius | ORT containment (64 random inputs) |
|---:|---:|---:|---:|---|
| 28 | -0.05385 | -0.04979 | 8.0e-10 | pass |
| 29 | -0.03885 | -0.03914 | 6.5e-10 | pass |
| 179 | -0.07849 | -0.07963 | 6.6e-10 | pass |
| 187 | -0.05867 | -0.05447 | 6.8e-10 | pass |

All four rows are excluded by a rigorous LP certificate over the exact
Neural-HZ state: the **first CERTs in the E0 namespace** (historical E0 has
0 CERT). They are capability evidence only: E0 promotion requires one path
replaying all 400 rows while keeping all 61 ADV, which N007 does not yet do.
Cost: about 6 min per row in float64 (FP64 throughput on this GPU is 1/64 of
FP32); a float32-rigorous mode is the obvious next optimisation.

## 2026-10-02 N008 — zonogon pair hulls (beyond the single-neuron triangle)

`probe_n008_pair_hull.py`: for each unstable ReLU, its most-correlated partner
in the same layer (|cosine| of x-forms; structural, not LP-driven); the exact
2-D latent projection of the pair is over-approximated by a 16-direction
support polygon intersected with the valid bounds; the 4-D convex hull of
(x_i, x_j, ReLU x_i, ReLU x_j) over it gives valid rows.

- GPU dual with the 56,976 extra rows: no improvement even when warm-started
  (the projected-Adam dual does not exploit many new rows; solver issue).
- Exact check (`probe_n008b_violation.py`, `probe_n008c_highs_loop.py`,
  CIFAR row 0, worst disjunct): thousands of pair rows are violated at the
  base LP optimum; a HiGHS cut loop converges in 3 rounds from **0.59463 to
  0.54823** (2,819 active pair rows). Gap reduction 8 percent: real but far
  from enough for this row.
- Triangle slack at the base LP optimum is largest in the last ReLU layer
  (mean 0.121, max 0.519) versus 0.005-0.02 in earlier layers.

N007 final for CIFAR (200 rows, `results/n007_e0_cifar_all_v1.jsonl`, float
probe): UNKNOWN->OPEN 171, UNKNOWN->LP_CERT_PROBE 4 (rows 28, 29, 179, 187),
VALIDATED_ADV->ADV_ORT 21, VALIDATED_ADV->OPEN 4 (rows 31, 54, 83, 114).
Among open UNKNOWN rows the terminal LP bound is below 0.1 for 11 rows, below
0.2 for 37 and below 0.3 for 56; 49 of the latter have a single open disjunct.
The TinyImageNet half was stopped by me after 102 rows because N015 supersedes
it (partial file retained).

## 2026-10-02 N009 — selective exactness on the last ReLU layer

`probe_n009_partial_milp.py`, CIFAR row 0, worst disjunct (class 53): exact
LP 0.595 -> **0.343** with integer binaries only on the last ReLU layer (29
binaries, HiGHS optimal, 102 s). Disjunct 83: -1.013 -> -1.501. This motivated
THEORY.md Section 7 (selective-exactness lattice). Mechanism: positive
coefficients of head ReLUs in the violation objective are convex terms; the
triangle relaxation pays its full slack exactly there.

## 2026-10-02 N010 — PDHG versus projected Adam (solver note)

`lp_pdhg.py`, `probe_n010_pdhg_check.py`: plain batched PDHG reaches 0.616
after 5000 iterations (5.4 s) versus Adam 0.634 after 1000 (1.2 s) and HiGHS
0.595. Neither first-order method gets close to the exact LP on these
degenerate LPs; decision: keep Adam for screening and intermediate bounds,
polish near-boundary terminal queries with HiGHS.

## 2026-10-02 N011 — partial MILP on near-boundary E0 CIFAR rows (float probe)

`run_n011_partial_milp.py` (gamma = last layer, all LP rows, 120 s per
disjunct). 12 rows completed: MILP-excluded (probe) rows 76, 50, 120, 137, 10,
101, 113, 167, 132, 100 (upper bounds -0.03 to -0.39); still open rows 105
(+0.058) and 153 (+0.095). Several runs crashed with CUDA OOM under a 0.12
memory fraction and were relaunched with 0.2 (`*_v2` files; memory cap only,
same semantics).

## 2026-10-02 N012 — float32-rigorous mode is not usable

`probe_n012_sound_fp32.py`: with float32 unit roundoff in the gamma pads the
output rounding radius reaches about 0.35 and the worst bounds become +0.56 to
+0.64 on rows that are certified in float64. The pads compound through |W|
over 20 layers. Decision: rigorous mode stays float64. `probe_n013_fp64_conv_bench.py`:
cuDNN float64 conv runs at 0.51 TFLOPS (float32 8.8), so float64 propagation
is affordable (5-7 s per CIFAR row in N015).

## 2026-10-02 N014 — per-layer level plans

`probe_n014_level_plan.py`, CIFAR row 76: keeping LP rows only for the last
1/2/3 layers gives upper bounds +0.080 / -0.012 / -0.041 versus -0.252 with
all rows, and is not faster (9 / 31 / 72 s versus 65 s). Decision: gamma on
the last layer, lambda on all layers.

## 2026-10-02 N015 — rigorous single-path E0 replay (preregistered, running)

New code: `nhz_terminal.py` (rigorous objective pads, batched rigorous terminal
LP, level-plan MILP with HiGHS objective cutoff), `run_n015_e0_sound_pipeline.py`.
HiGHS `objective_bound` semantics checked on a toy MILP (Infeasible when no
solution beats the cutoff). Smoke (`results/n015_smoke_v1.jsonl`): row 28 CERT
in 9.6 s (rigorous LP), row 76 CERT in 19.6 s (rigorous LP + cutoff MILP;
the uncut MILP needed 66 s), row 2 ADV in 9.3 s. Preregistration and source
freeze: `n015_e0_replay_v1/PREREG.md`, `n015_e0_replay_v1/FREEZE_SHA256SUMS`.
Full 400-row replay launched (CIFAR then Tiny).

## 2026-10-02 N016 — rigorous pipeline on formal unsolved rows (v1, stopped)

`run_n016_formal_sound_pipeline.py` (manifest-driven, budget = official
timeout, multi-box specs, ORT centre self-check per model). Smoke
(`results/n016_smoke_relusplitter_v1.jsonl`, rigorous): relusplitter:144
(baseline TIMEOUT) CERT 3.7 s, :162 (TIMEOUT) CERT 3.8 s, :173 (UNKNOWN) CERT
16.9 s with rigorous LP upper bound -0.021; :117 UNKNOWN. The float probe N004
had flagged 11 LP-certifiable relusplitter `cifar_biasfield` rows (margins
-0.9 to -21.8) where the baseline lost or timed out.
The full v1 run reached 11 rows (MetaRoom :14, :26, :45, :95 ADV from the LP
box maximiser in about 4 s each; baseline TIMEOUT) before I stopped it (see
N022).

## 2026-10-02 N017 — CERT sampling audit

`audit_n017_cert_sampling.py`: 2000 uniform ORT samples per input box for every
CERT plus an independent regex reading of the input box. Smoke CERTs
(relusplitter 144/162/173, CIFAR 28/76): 0 violations; the smallest sampled
safety slack is always at least the certified bound (e.g. 144: certified 16.1,
sampled 19.1).

## 2026-10-02 N018-N020 — Transformer extension (engine n005, float probe)

`nhz_attn.py` adds shape folding, Transpose/Concat/ReduceMean on states,
bilinear MatMul (zonotope product with exact diagonal mean and cross-term
radius) and Softmax (tangent plane at the centre in score differences plus an
interval-Hessian remainder, with LP rows `p in [p_lo, p_hi]` and
`sum p = 1`). ORT centre self-check: 6.7e-5 (PGD model), 1.5e-6 (IBP model).
- PGD model rows 0-2: terminal violation bounds 96-1258; hopeless at this
  input radius.
- IBP model, all 15 formal unsolved rows (`results/n018_vit_ibp_unsolved_v1.jsonl`):
  LP bounds +0.039 to +0.155 (all open).
- N019: row 141 with all 84 ReLU binaries exact: +0.0388 -> +0.0124 at the
  300 s limit.
- N020 attribution (dual box terms): ReLU triangles 0.038, softmax remainders
  0.009, bilinear 0.002 on row 141; similar on rows 100 and 132. Even exact
  ReLUs leave about 0.01 of attention slack; these rows sit at the boundary.

## 2026-10-02 N021 — retention sample on baseline-solved rows (v1, stopped)

12 random solved rows per ReLU family (seed 20261002); stopped after 10 rows
together with N016 (see N022), relaunch pending.

## 2026-10-02 N022/N023 — witness input semantics (correction of my own error)

`audit_n022_witness_replay.py` replays witnesses with an evaluator that
shares no code with the pipeline. v1/v2 rejected all four MetaRoom witnesses
as "outside the box". Investigation (`probe_n023_repair_check.py`): 5,219 of
MetaRoom's 5,376 input coordinates are fixed (`lb = ub`) at 8-digit decimals
such as 0.61035156 that no float32 equals; under that strict reading (S2: the
float32 vector itself satisfies the input assertions) no MetaRoom
counterexample can exist at all, yet the frozen baseline counts one. The
baseline ADV gate (`hz_full_worker._is_cex`, documented in
`/data1/Kane/HyZor/VMCAI2027/artifact/README_witness_validation.md`) uses S1:
a real point `center + rad*xi` inside the box, evaluated by ONNX Runtime on its
float32 rounding, output check with tolerance 1e-9. My pipeline already
implements S1 (with output tolerance 0). I had stopped N015 v1, N016 v1 and
N021 v1 on a mistaken defect diagnosis; their partial files are retained.
N022 v3 reports both semantics: all four MetaRoom witnesses are S1-valid and
S2-invalid. Decision: count ADV under S1, exactly as the 1870 baseline, and
report S2 alongside. `witness_util.to_box_f32` (S2 helper) is kept but not
used for decisions.

## 2026-10-02 N015 v2 / N016 v2 — relaunched

Same decision logic; runners also save the float64 witness point. E0 replay:
`n015_e0_replay_v2/` (PREREG v2 + FREEZE). Formal probe:
`results/n016v2_formal.jsonl` (freeze `results/n016v2_n021v2_FREEZE_SHA256SUMS`).

## 2026-10-02 N024 — sign-aware selective exactness (THEORY.md Section 9)

`nhz_terminal_sign.py`, `probe_n024_sign_plan.py`: only last-layer units with
a positive objective coefficient on their eta keep binaries. CIFAR rows
120/105/76/113: 17/32, 5/12, 17/32, 11/24 binaries; identical exclusion
verdicts and, on row 105, the identical optimum 0.0265. HiGHS wall time:
68.3 vs 32.9 s, 49.9 vs 102.6 s, 10.6 vs 11.3 s, 6.1 vs 6.7 s (sign vs full).
Not adopted (no consistent speed gain); recorded as a proved, measured
equivalence. Side measurement: a float64 CIFAR100-large row peaks at about
25 GB of GPU memory.

## 2026-10-02 N015 v2 -> v3 -> v4

v2 stopped after 3 rows (memory risk on CIFAR100-large, see N024). v3 (memory
fix, float64 engine, CIFAR only) stopped after 4 rows once the mixed-precision
engine below made a single engine for all 400 rows possible. v4 is the
preregistered single-path E0 replay (`n015_e0_replay_v4/`).

## 2026-10-02 N025 — mixed-precision sound engine (`nhz_sound_mp.py`, n006.2)

Float32 generator storage, float64 arithmetic on factor chunks, exact storage
rounding `sum_k |G64 - G32|` added to the rounding radius, eta generators
rounded up. Test (`test_n025_mp_engine.py`):

| row | engine | rigorous worst LP UB | output radius | peak GPU | wall |
|---|---|---:|---:|---:|---:|
| CIFAR 28 | mp | -0.05288 | 2.9e-5 | 1.4 GB | 2.5 s |
| CIFAR 28 | f64 | -0.05393 | 8.0e-10 | 3.0 GB | 2.3 s |
| CIFAR 29 | mp | -0.03828 | 1.7e-5 | 1.1 GB | 1.7 s |
| CIFAR 29 | f64 | -0.03983 | 6.5e-10 | 2.5 GB | 1.6 s |
| Tiny 68 | mp | -0.22808 | 3.0e-5 | 9.5 GB | 11.9 s |
| Tiny 40 | mp | -0.09978 | 4.8e-5 | 10.0 GB | 12.7 s |
| Tiny 13 | mp | -0.01725 | 3.1e-5 | 9.6 GB | 12.7 s |
| Tiny 33 | mp | -0.01150 | 4.2e-5 | 10.4 GB | 13.4 s |

The storage error has no `n u` factor, which is why this works where the pure
float32 rigorous mode (N012) failed. The four TinyImageNet rows are the first
rigorous TinyImageNet certificates (capability evidence only).
The TinyImageNet float probe N007 (102 rows before it was stopped) had found
all 21 historical ADV rows in its range plus one ADV on an E0 UNKNOWN row.

## 2026-10-02 N026 — candidate path v2 (fixed cascade) on retention failures

`nhz_path_v2.py`: LP -> LP witness -> gamma=last-layer MILP (40 percent of the
remaining budget) -> gamma=all-layers MILP (remaining budget), same for every
instance. On rows path v1 failed to retain (`results/n026_path_v2_retention_failures_v1.jsonl`):
relusplitter:87, 98, 83, 82 recovered as CERT through the all-layers stage
(upper bounds -0.024, -0.015, -0.0004, -0.0005); relusplitter:38 TIMEOUT
(30 s budget); tllverify:6 recovered as CERT (65.7 s). Retention sample of
path v1 (`results/n021v2_retention.jsonl`, 66+ rows): MetaRoom 12/12, Cora
12/12, SAT-ReLU 11/12, Malbeware 8/12 (4 ERROR = GPU memory cap 0.15),
relusplitter 6/12, TLL 3/6 at the time of reading; no CERT/ADV conflict.

## 2026-10-02 N027-N031 — smooth activations in the tower (float probes)

Engine n005.3 (`nhz_attn.py`): Sigmoid/Tanh shadow = DeepZ (slope
`min(f'(l), f'(u))`, exact range of `f(x) - lam x`); lambda level = chord and
five tangents, each shifted to validity on `[l,u]` by a 64-point grid maximum
plus a Lipschitz remainder. ORT centre self-check 2e-5.

dist_shift (72 rows, baseline 63 CERT + 7 ADV + 2 TIMEOUT, K3 configurable
selective nonconvexity, median about 23 s):

| configuration | baseline CERT recovered | time |
|---|---:|---|
| lambda only (`results/n027_distshift_all_v1.jsonl`) | 38/63 | 0.27 s/row |
| lambda + all ReLU binaries exact, no sigmoid binaries (`results/n030_distshift_all_v1.log`) | 55/63 | about 2 s/row, 142 s total |
| + segment unions on 40 sigmoid units, about 4 segments each (N031, 4 of the 8 remaining rows) | rows 12, 58 recovered; 48 (+0.27 at 120 s) and 39 (+0.78 optimal) not | 54-120 s |

All 7 baseline ADV rows stay open (no false certificate). The two unsolved
rows have LP bounds +9.1 and +13.0 and full-ReLU MILP bounds +4.4 and +3.9.
Reading: with a tight multi-line convex hull at the lambda level, the
nonconvexity budget is better spent on the ReLUs (55/63 with zero sigmoid
binaries, roughly 10x faster than the baseline median); the remaining 8 rows
genuinely need sigmoid nonconvexity, and the approximate GPU intermediate bounds
(looser than the baseline's exact per-neuron LP) hurt the segment enclosures.
This is NOT yet a demonstration of beating configurable selective
nonconvexity; it is a measured trade-off. No rigorous (sound-rounding) smooth
transformer exists yet.

## 2026-10-02 N029 — path v2 over formal unsolved rows (running)

`run_n029_formal_path_v2.py`, engine n006.2, family order relusplitter,
metaroom, cora, tllverify, cersyve, safenlp, acasxu. First 27 relusplitter rows:
new ADV on relusplitter:10 (baseline UNKNOWN), :16, :17 (baseline TIMEOUT),
all from the terminal LP box maximiser in under 1 s; the rest TIMEOUT.

## 2026-10-02 N032 — path v2 retention with engine n007 (stopped, superseded by N037)

Engine n007 (`nhz_sound_mp2.py`): n006 + rigorous Sigmoid/Tanh (THEORY.md
Section 10) + Slice/Concat on states. First 25 rows: cersyve 6/11 kept,
dist_shift 8/12, linearizenn 2/2; no conflict.

## 2026-10-02 N033 — lost E0 ADV rows under path v2 (300 s diagnostic budget)

CIFAR rows 31 and 54: ADV from the last-layer MILP incumbent at 74 s and 68 s
(inside the 100 s E0 budget). Rows 83 and 114: not found in 300 s. E0 v4 uses
path v1 logic, which also decodes last-layer MILP incumbents, so the expected
E0 retention outcome is 59 or fewer of 61, i.e. a retention failure.

## 2026-10-02 N034-N036 — why ADV and CERT were lost; path v3

- N034 (`probe_n034_cersyve_witness.py`): on cersyve:8 the full MILP optimum
  satisfied atom 1 with slack 0.037 but sat exactly on atom 2's boundary, so
  the strict ORT check failed (HZ value vs ORT output differ by 5e-7 only).
- N035 (`nhz_terminal_v3.py`, `probe_n035_epigraph.py`): epigraph min-slack
  objective `max T tau s.t. T tau <= b_k - a_k^T Y + pad_k` for multi-atom
  disjuncts (exclusion iff the maximum is negative; interior witnesses
  otherwise). cersyve 8/5/1: ORT-valid witnesses for all three.
- N036 (`nhz_sound_mp3.py`, engine n008): exact-LP polishing of intermediate
  bounds for small LPs (`K * R <= 6e6`, at most 1500 unstable neurons; HiGHS
  duals used only as multipliers of the rigorous weak-duality evaluation). On
  sat_relu:63 it changes nothing because the network has one ReLU layer; the
  real cause was a cascade defect: path v2 gave the last-layer stage only 40
  percent of the budget and skipped stage E when the network has one ReLU layer.
- Path v3 (`nhz_path_v3.py`): epigraph terminal + full budget for stage D when
  it already covers all ReLU layers + engine n008. Smoke: sat_relu:63 CERT 6.1 s;
  cersyve:8 ADV 18.5 s, :1 ADV 50.2 s, :5 ADV at 100.1 s (over budget, counts
  TIMEOUT); relusplitter:38 still TIMEOUT (30 s budget).

## 2026-10-02 N037 — path v3 retention sample (running)

`run_n037_retention_path_v3.py`, seed 20261002, 12 rows per family for 11
families (ViT and cGAN excluded: no rigorous attention / unsupported ops).

N029 and N037 were stopped by me (subsumed by N039; partial files retained):
N029 reached 40 relusplitter rows (new ADV relusplitter:10, :16, :17);
N037 reached 5 cersyve rows (4 kept, 1 ADV->TIMEOUT).

## 2026-10-02 N038 — consolidated rigorous engine n009 (`nhz_sound_v9.py`)

Operator registry; rigorous versions of every operator used so far plus
ConvTranspose, nearest Upsample/Resize, Pad, Squeeze/Gather/Cast, ReduceMean,
bilinear MatMul (true-value error `|x|e_y + |y|e_x + e_x e_y` plus float64
pads) and Softmax (difference box widened by the radii, tangent-term error
moved into the output radius, box and sum-to-one rows relaxed by the radii);
exact-LP polishing as n008. Self-test (`test_n038_v9_engine.py`, 64 random ORT
samples each, all inside the rigorous output enclosure):

| row | centre error vs ORT | rigorous worst LP UB | output radius |
|---|---:|---:|---:|
| CIFAR100 28 | 5.2e-6 | -0.0585 | 2.9e-5 |
| dist_shift 1 | 1.7e-5 | -2.4689 | 1.7e-5 |
| ViT 141 (IBP) | 9.1e-7 | +0.0388 | 7.6e-7 |
| ViT 101 (IBP) | 9.1e-7 | +0.1134 | 1.1e-6 |
| cGAN 0 | 2.3e-7 | +0.0098 | 4.7e-5 |
| LinearizeNN 0 | 1.4e-6 | +1.0955 | 3.1e-5 |

The rigorous attention bounds equal the float probe's (0.0388 on ViT 141).

## 2026-10-02 N039 — first single-path replay of all 2413 formal rows (running)

`n039_full_replay_v1/` (PREREG + FREEZE). Path v3 + engine n009, one
configuration, three workers partitioned by family; universe verified to be
exactly CERT 1063 / ADV 807 / TIMEOUT 274 / UNKNOWN 269. Expected to fail
promotion (known dist_shift and cGAN gaps); its value is the first measured
single-path vector.

N039 v1 (path v3) was stopped by me after 786 rows (`summarize_n039.py`):
safenlp 711/755 solved rows kept (30 ADV and 14 CERT lost, all TIMEOUT at the
20 s budget), relusplitter 3/4, acasxu 2/2; no conflict, no gain yet.

## 2026-10-03 N040 — pooling (engine n010, `nhz_sound_v10.py`)

AveragePool (affine, padded) and exact MaxPool via `max(a,b) = a + ReLU(b - a)`
(each step uses the aligned exact ReLU, so the max keeps binary phases). The
only formal user, cGAN small_transformer (rows 19, 20, baseline UNKNOWN), does
not fit the 0.2 GPU memory cap; not pursued (no retention requirement).

## 2026-10-03 N041 — why SafeNLP ADV rows timed out

`probe_n041_seed.py` on safenlp:3, :6, :57 (one ReLU layer, about 125
binaries, 20 s budget). Without early stop the solver found violating points
but kept optimising until the time limit. With `objective_target` (stop at
violation >= 1e-6): row 6 ADV in 0.8-2.9 s for all four seeds; row 3 only with
seed 2 (13.7 s; the baseline's winning branch was also `normal_seed2`); row 57
none of four seeds within 20 s. HiGHS releases the GIL (two 4 s solves in
threads take 4.01 s); `cancelSolve` does not interrupt a running MIP, the
MIP-interrupt callback does (stopped 0.36 s after the flag).

## 2026-10-03 N042 — path v4 (`nhz_terminal_v4.py`, `nhz_path_v4.py`)

Seed portfolio 0-3 in threads, early stop at the violation target, callback
interrupt, minimum of the valid dual bounds. Smoke (v4.2):
safenlp:3 ADV 15.2 s, :6 ADV 2.3 s, sat_relu:63 CERT 24.8 s, cersyve:5 ADV
3.3 s, tllverify:4 CERT at 601.2 s (over its 600 s budget, counts TIMEOUT).
On sat_relu:63 one seed proved infeasibility while others found points inside
the 1e-4 tolerance band; the rule then reports not-excluded unless the minimum
dual bound clears the margin, which is the conservative reading.

## 2026-10-03 N039 v2 — full replay with path v4.2 (running)

`n039_full_replay_v2/` (PREREG + FREEZE), two workers.

## 2026-10-03 N043 — pair hulls added to the exact-head MILP (negative)

`probe_n043_pairs_plus_head.py`: last-layer binaries exact plus zonogon pair
rows (2 partners) on the three preceding ReLU layers. CIFAR row 153: 0.0946
(head only, 15 s) -> 0.0921 with 118,219 extra rows (207 s, time limit); row
105: 0.0304 (178 s) -> no dual bound within 200 s with 41,817 extra rows.
Closed: negligible tightening at a large MILP cost.

## 2026-10-03 N039 v2 stopped; engine n009.2 and path v4.3; N039 v3 launched

v2 stopped by me after 556 SafeNLP rows (541 kept; 5 ADV and 9 CERT lost
to TIMEOUT at the 20 s budget; no conflict; gate already FAIL). Reason for the
stop: worker B spent more than 13 minutes in the first Cora row. Engine
n009.1 polished every unstable neuron with an exact HiGHS LP and ignored the
row deadline (about 3,500 dense LPs on Cora's 7x250 network).

Fix, fixed before relaunch: n009.2 polishes only layers with at most 100
unstable neurons, at most 600 LPs per propagation, and only during the first
25 percent of the row budget. Path v4.3 hands the row deadline to the engine.
Smoke (`probe_n042_path_v4_smoke.py`): cora:0 TIMEOUT->ADV 1.7 s, cora:2 and
:5 CERT in about 1-2 s, tllverify:1 CERT 26 s.

`n039_full_replay_v3/` (PREREG + FREEZE) relaunched on both workers with the
same family split. v2 files kept, never merged with v3.

## 2026-10-03 N044 — sign-aware binaries on the SafeNLP rows lost by N039 v2

`nhz_terminal_v5.py` (terminal v4 portfolio + Proposition 1' of THEORY.md
12.1), `probe_n044_sign_safenlp.py`, `results/n044_sign_safenlp.jsonl`. Same
engine n009.2, 20 s per row, 4 seeds, machine load about 34 on 20 cores.

| rows (baseline) | full binaries (v4 plan) | sign-aware (v5) |
|---|---|---|
| 57, 110, 346, 409, 439 (ADV) | 0 of 5 (346 found at 20.0 s, over budget) | 5 of 5, 0.6-1.6 s, ORT-valid |
| 280, 337, 353, 404, 539 (CERT) | 0 of 5 | 5 of 5, 13.7-19.2 s |
| 271, 272, 290, 375 (CERT) | 0 of 4 | 0 of 4 |

Binaries drop from 106-126 to 48-57 per row. N024's "no consistent speed gain"
on CIFAR does not carry over to one-layer networks: here the plan goes from
always timing out to solving 10 of 14. Two of the CERTs (353, 539) finish
within one second of the budget, so they are fragile under load.

## 2026-10-03 N045 — ideal latent-box cuts at the root (negative within 20 s)

`nhz_ideal_cuts.py` (THEORY.md 12.2), `probe_n045_ideal_cuts.py`. Row 271:
root LP bound 5.69 -> 3.08 after 7 rounds (326 cuts, 0.2 s); 272: 7.77 ->
3.64; 290: 6.14 -> 2.66. The MILP after the cuts ends weaker at 20 s (271:
1.63 vs 1.15 without cuts). Not adopted.

## 2026-10-03 N046 — how far the four open SafeNLP CERTs are from the budget

`probe_n046_cert_options.py`, single HiGHS thread each, 90 s, sign-aware plan.
Time to exclusion: row 271 34-81 s (seed 2 fastest), 272 29-68 s, 290 29-71 s
(about 5,000-7,000 nodes). The baseline needed 13.6, 15.2 and (not listed) s
with its full-binary HZ MILP under its own portfolio and load. The option
values the baseline portfolio used (seed 2, pscost 1, heuristics) do not bring
any of them under 20 s here. Gap: about 1.5-2x on this loaded machine.

## 2026-10-03 N047 — monotone units on every layer (Proposition 2, test)

`nhz_monotone.py`, `test_n047_monotone.py`, `nhz_terminal_v6.py`. The rewrite
into unit coordinates reproduces exact forward differences (relative error
5e-14 to 5e-16); 1,950 finite-difference derivatives all lie inside the interval
enclosure. On ACAS Xu (6 layers, 18-50 unstable each) only last-layer units
qualify (11 and 27 of 50); the interval reverse pass is too wide for earlier
layers. On sat_relu (one layer) it equals Proposition 1 (60 of 98 dropped).
Correct but no practical gain beyond the last layer with interval slopes.

## 2026-10-03 N048 — ideal cuts with a 90 s limit

Row 271: nodes 4,988 -> 3,285 with 8 rounds (326 cuts), but time unchanged
(34.2 -> 34.4 s) because each node LP is larger. Closed for SafeNLP.

## 2026-10-03 Note — origin of the E0 baseline ADVs

All 61 E0 retention anchors are `LEGACY_SIDECAR_ATTACK_WITNESS` with
`clip_to_box` repair (`../evidence/*_evidence_baseline_v2.json`). CIFAR rows 83
and 114 have unsafe slack 1.3e-3 and 9.4e-4. A domain-only path (LP box
maximiser, MILP incumbents) has not found them in 300 s (N033). Retaining
attack-found anchors without an attack stage is therefore unlikely. Whether
the candidate path may carry the baseline's attack stage only for
retention, with no gain credit from it, is a decision for the user.

## 2026-10-03 N049-N051 — why the baseline's MILP was faster: encoding

- N050 (baseline worker run read-only from `HyZor/OPERATOR_ADVANTAGE_20260923/engine`,
  outputs in the session scratchpad only): the baseline HZ MILP with all
  117/124 binaries proves SafeNLP 271/272 CERT in 16.1-21.1 s (verify time)
  on this machine under today's load. Our dense sign-aware plan needs
  29-34 s, so load does not explain the gap.
- N049 (`nhz_terminal_sparse.py`, `probe_n049_sparse.py`): explicit
  pre-activation columns (sparse big-M, same projection) cut the time by 25-30
  percent at equal node counts (271: 28.9 -> 20.6 s).
- N051 (`probe_n051_encodings.py`): disaggregated rows per binary unit
  (`x = x0 + x1, y = x1, x0 >= l(1-d), x1 <= u d`, as in the baseline's HZ
  ReLU) halve the node counts (271: 4,731 -> 2,229) and give 14.1, 20.0, 19.6,
  15.7 s on rows 271, 272, 290, 375, single thread, seed 0. HiGHS default
  thread setting changes nothing. Adopted as terminal v7 (`nhz_terminal_v7.py`,
  THEORY.md 12.4), which also adds the rigorous float32 storage gap
  `delta_k` to the tolerance of the value-map relation.

## 2026-10-03 N052 — strict witnesses on Cora by inward rounding

`audit_n039_witness.py` (independent reader of audit_n022) on the 81 Cora
witnesses of N039 v3: all S1-valid, all S2-invalid, as on MetaRoom.
`probe_n052_interior_round.py`: rounding x64 to float32 toward the interior
(one extra inward float32 step) makes all 81 S2-valid; max shift 6.0e-8; every
unsafe output assertion still holds. Adopted in path v5 (`nhz_path_v5.py`,
THEORY.md 12.5) as the first witness check, with S1 as the fallback.

## 2026-10-03 N053 — path v5 on the rows N039 v3 lost (running)

`run_n053_targeted.py` (same per-row logic as `run_n039v4_full_replay.py`,
explicit row list). First results: SafeNLP 3, 57, 110, 122 ADV in 1.0-12.0 s;
271 CERT 13.6 s; 272 CERT 18.1 s (all lost by v3).

N053 SafeNLP final: 27 of the 28 rows lost by v3 are retained by path v5
(`results/n053/safenlp_lost_v3.jsonl`); row 844 (CERT) still times out at
20 s; several CERTs finish at 16-18 s, so SafeNLP stays load-sensitive.
ViT with path v5 and engine n009.2 (`results/n053/vit_cert_sample.jsonl`):
PGD rows 15, 25 TIMEOUT (LP bound 292 and 426); IBP rows 101-104 UNKNOWN
before the deadline, 105 and 106 CERT (16.4 s, 12.0 s).

## 2026-10-03 N054 — softmax enclosure: tangent remainder explodes on PGD ViT

`probe_n054_vit_softmax.py`, `results/n054_vit_softmax.jsonl`, terminal LP
bound (rigorous, 1000 iterations):

| row (model, baseline) | n009.2 | n011.0 radius rule | n011.1 + coupling rows | n011.2 remainder rule |
|---|---:|---:|---:|---:|
| 15 (pgd, CERT) | 292.25 | 1.97 | 1.68 | **0.42** |
| 25 (pgd, CERT) | 425.97 | 3.71 | - | **1.45** |
| 2 (pgd, TIMEOUT) | 94.75 | 2.82 | - | **1.99** |
| 101 (ibp, CERT) | 0.117 | 0.222 | 0.157 | **0.111** |
| 102 (ibp, CERT) | 0.101 | - | 0.132 | **0.094** |

Mechanism: the n009 softmax remainder `0.5 delta^T H delta` uses `exp` of the
score-difference upper bounds; for wide PGD score ranges it exceeds the
trivial output range by orders of magnitude and the attention product
multiplies it. n011.2 (`nhz_sound_v11b.py`) uses, per output coordinate, the
interval `[plo, phi]` with one fresh factor iff the remainder alone is at
least the interval radius; otherwise the n009 form. Both are sound per
coordinate and each coordinate keeps its own fresh factor. One rule, no
constants; adopted. The baseline (ACT HybridZ ViT branch) reports LP upper
bounds 0.25 (15), -0.11 (25), 0.016 (101), 0.007 (102): its relaxation is still
tighter on these rows.

Path v5.1: a MILP stage that stopped at the violation target with a
spurious incumbent (attention is only lambda-exact, so relaxation points can
be unreal) is re-run without the target in its remaining time. On ViT 101-104
the v5.0 stages stopped this way and the row ended UNKNOWN with 36-79 s left.

## 2026-10-03 N055 — terminal v7 equivalence test

`test_n055_terminal_v7.py`, `results/n055_terminal_v7_test.jsonl`. LP
relaxations of the disaggregated v7 plan and the dense v5 plan agree to
about 1e-7 relative on SafeNLP 280, ACAS Xu 3 (gamma = 1 and 6 layers),
sat_relu 5, CERSYVE 1 (1 and 8), TLL 1 (1 and 6). MILP verdicts agree (both
excluded or both open; TLL 1 all-layers Infeasible in both). The vectorised
v7 builder (terminal-v7.2) takes under 10 ms on these rows. Proposition 2 is
applied only when the dense unit-coordinate matrices have at most 2e7 cells;
otherwise only the last-layer rule.

## 2026-10-03 N056 — fused attention-mix transformer (negative)

`nhz_sound_v12.py` (n012.0/.1): z = p0 V + [f(s) - f(s0)] + cross term, with
f(s) = sum softmax(s)_j m_j linearised in the scores, remainder
span(m) sigma^2 / 8 (n012.0) or the softmax-box-weighted variance bound
(n012.1), cross term via the softmax box. LP bounds: row 15 13.3 / 11.1
(n011.2: 0.42), row 25 13.7 / 13.5 (1.45), row 101 0.131 / 0.128 (0.111),
row 102 0.104 / 0.102 (0.094). `probe_n056_fused_terms.py`: on row 15 the
Taylor remainder of the second attention layer dominates (mean 1.21 with value
spread 6.35 and score deviation 1.20). Closed; n011.2 kept. (Agent reading
of ACT HybridZ: the baseline uses this decomposition but also re-links the
score term to Q and K, clips the remainder with tangent/secant bounds on
log p and enumerates the cross term; not reproduced here.)

## 2026-10-03 N057 — sat_relu failures of v3 and the witness snap

v3 lost sat_relu CERT 15, 59, 65 (573/494/565 binaries, 100 s limit, bound
stuck at the cutoff) and ADV 36, 40, 72, 78, 88 (MILP optimum on the unsafe
boundary with violation 1-2e-6; the decoded point misses it by solver
tolerance). Path v5.2: candidates are also tried with latent coordinates
within 1e-6 of +-1 snapped onto the box bound (numeric clean-up). Targeted run
(`results/n053/satrelu_v52_*.jsonl`): 36, 40, 72 ADV in 0.2 s; 15 CERT in 7.7 s.
ViT sample with path v5.1 + n011.2 (`results/n053/vit_cert_sample_v51_n0112.jsonl`):
CERT 15 (24.5 s), 101 (53.6 s), 105 (35.9 s), 106 (22.7 s); TIMEOUT 25, 39,
62, 78 (pgd), 102, 103, 107; UNKNOWN 104. Baseline: all 12 CERT.

## 2026-10-03 N039 v4 stopped, N039 v5 launched

v4 (`n039_full_replay_v4/`, path v5.1) stopped after 235 rows to include the
snap; v5 (`n039_full_replay_v5/`, path v5.2 + engine n011.2 + terminal v7.2)
launched with the same family split. PREREG and FREEZE in the directory.

## 2026-10-03 N058 — sign/curvature exactness for sigmoid units (negative on dist_shift)

Idea (THEORY.md 13, statement only): if the objective is non-increasing in a
sigmoid output and the unit lies in the convex region (u <= 0), only the lower
side matters and the tangent family describes it exactly; symmetrically for
non-decreasing units in the concave region. Such units need no phases;
mismatched units need segments on one side only.
`probe_n058_sigmoid_signs.py` on dist_shift rows 0, 8, 12, 39, 48, 51, 58, 71
(the eight not retained) and 1, 2: interval reverse mode through the
classifier (784 -> 32 -> 32 -> 10) gives no unit with a determined derivative
sign (dec = inc = 0 on every row), with interval slopes (24-45 unstable
classifier ReLUs) and with the engine's tighter phase sets
(`results/n058_sigmoid_signs_engine_slopes.jsonl`). Only saturated units (61-198
per row) are trivially exact. Closed for dist_shift: its classifier mixes signs.

## 2026-10-03 N059 — HiGHS presolve ignores the time limit on large plans

Cora row 1 (7 ReLU layers, 1,517 unstable, all-layers plan 7,387 x 8,073,
2.18 M nonzeros) overran its 30 s budget by 115-210 s under paths v5.2-v5.4.
Neither the build (0.4 s), the multi-layer monotone analysis, nor the MIP
interrupt was the cause. HiGHS log: "Presolve: Time limit reached", then 127 s
before the first node. With presolve off the same solve returns after 5.6-7.8 s
on a 5 s limit. Terminal v7.3-v7.5: multi-layer monotone analysis dropped,
build time charged to the stage, a wall-clock watchdog through the
simplex/IPM/MIP interrupt callbacks, presolve off above 5e5 nonzeros. Cora 1
and 4 now end at 32.8 s and 30.7 s.

N039 v5 (path v5.2, stopped at 1,298 rows for this fix): SafeNLP 1079/1079,
sat_relu 100/100, malbeware 102/102 retained, Cora 5/5 plus 4 gains, no
conflict. N039 v6 (path v5.5) launched with the same split.

## 2026-10-03 N060 — simplex-aware cross term for P @ V (negative)

`nhz_sound_v13.py` (n013.0): the cross term sum_j dp_ij dv_jd bounded by a
fractional knapsack over the softmax box with sum_j dp_ij = 1 - sum p_hat
(greedy; `test_n060_knapsack.py`: equal to scipy linprog to 3e-16 on 300
random cases, no violation at random feasible points). On ViT rows 15, 25, 39,
62, 78, 101-104, 107 the knapsack enclosure was never smaller than the DeepZ
product with its shared-factor diagonal correction (0 of 960 / 4,896
coordinates), so the LP bounds are unchanged. Closed.

## 2026-10-03 N061 — E0 replay with the formal candidate (running)

E0 v4 stopped by me after 127 CIFAR rows (retention already failed at row 83;
older path). Its shell chain then started the TinyImageNet part, which I also
stopped after 0 rows (empty file kept). `n061_e0_replay_v5/` (PREREG + FREEZE):
all 400 E0 rows with path v5.5 + engine n011.2 (`run_n061_e0_path_v5.py`),
nice 19, next to N039 v6.

## 2026-10-03 N062-N063 — where the ViT relaxation loses

- N062 (`probe_n062_vit_lp_iters.py`): the GPU LP is converged on IBP row 101
  (1000 / 5000 / 20000 iterations: 0.1087 / 0.1087 / 0.1083), so the gap to the
  baseline (0.016) is the relaxation, not the solver. On PGD row 15 more
  iterations help (0.296 -> 0.259; baseline 0.249).
- N063 (`probe_n063_vit_widths.py`, `results/n063_vit101_widths.txt`): engine
  width / sampled width per node on row 101 (affine nodes sit at 2.0, the
  sampling deficit): Q/K/V 2.6, scores 2.8, **softmax 7.2, P@V 12.6**, output
  projection 13.7; the residual Add returns to 2.3. The attention branch is the
  loss: about 2.6x at the softmax and 1.75x at P@V, in every block.

## 2026-10-03 N064 — difference-aware score bounds (engine n014) and re-linked fused mix (n015)

- n014 (`nhz_sound_v14.py`): softmax bounds from score differences computed as
  scale * q_i . (k_b - k_a) (shared key generators cancel before the product
  remainder), min with the old bound; value map unchanged. LP bounds vs
  n011.2: PGD 15 0.42 -> 0.30, 25 1.45 -> 1.32, 39 0.83 -> 0.73, 62 0.52 -> 0.43,
  78 0.26 -> 0.18; IBP 101 0.111 -> 0.109, 102-107 unchanged. Kept as the best
  attention engine; NOT in the running N039 v6 (n011.2), because a restart
  would void the replay and n014 changes only rows with Softmax nodes.
- n015 (`nhz_sound_v15.py`): fused mix with the Taylor term re-linked to Q and
  K (one product q . sum_j a_j k_j), chosen per coordinate against the n014
  composite by radius. 101 0.121, 102 0.090, 15 0.99, 25 3.85. Mixed and
  mostly worse: the radius criterion does not track LP tightness. Closed.

## 2026-10-03 N065 — every MILP plan dropped the smooth rows; terminal v8

Finding: all plan builders (`nhz_terminal._build_plan`, terminal v5, v7) kept
only the LP rows of ReLU phases. The rigorous Sigmoid/Tanh chord and tangent
rows and the Softmax box and sum-to-one rows were in the GPU LP but not in
any MILP stage. That is sound, since it only relaxes, but in the MILP a smooth
unit was just its DeepZ parallelogram. Affected formal families: dist_shift
(Sigmoid), vit_2023 (Softmax) and cgan_2023 (Tanh, Sigmoid, Softmax)
(`scan of all formal models`).

Terminal v8 (`nhz_terminal_v8.py`) with engine n016 (`nhz_sound_v16.py`, which
records the smooth units). Every smooth unit gets an explicit `x_j` with one
equality row. Its rigorous lines are written in `(x_j, eta_j)` with the
tolerances `ye + 2^-24 lam sum|gx|` and `|a| ex`. Optional phase segments use
Balas rows, one binary per segment, split at the inflection point.
`probe_n065_distshift_smooth.py`, all 72 dist_shift rows, all ReLU layers gamma,
no segments, 300 s (`results/n065_distshift_v8_all.jsonl`):

| | baseline CERT 63 | baseline ADV 7 | baseline TIMEOUT 2 |
|---|---|---|---|
| excluded | **57** (55 without smooth rows) | 0 | 0 |

Max wall time 15.3 s. Row 12: v7 plan +1.86 (open), v8 -3.02 (excluded, 0.6 s;
the baseline needed 13 s with K3 selective nonconvexity). Still open: rows 0, 8,
39, 48, 51 and 71, with upper bounds 1.5 to 6.5.

## 2026-10-03 N066 — validity test of the v8 plan

`test_n066_plan_v8_validity.py dist_shift_2023 12 2`: 100 exact points from
random box inputs, built with the engine's own value maps, with every plan
column filled (Balas pieces, active segment of each of the 784 sigmoid
units). Worst relative row violation 4.0e-15, no column-bound violation. Plan:
34,561 x 6,346, 106k nonzeros, 1,581 binaries.

## 2026-10-03 N067 — path v6 (terminal v8 + stage F), engine n017; replays moved

- Engine n017 (`nhz_sound_v17.py`) = n011.2 + n014 + n016. Path v6.1
  (`nhz_path_v6.py`): terminal v8.1 in every MILP stage; stage F (all ReLU layers
  gamma + 2 segments on every smooth unit with output range > 1e-4) after a
  failed stage E; the violation-target early stop only on exact plans. That
  removes the spurious-target reruns seen in E0 and ViT stages. Terminal v8.1:
  the sign rule is off when a smooth layer follows the last ReLU layer
  (Proposition 1' premise).
- dist_shift with path v6.0 (`results/n053/distshift_v6_*.jsonl`): row 0 CERT in
  268 s (stage F: 959 binaries, Infeasible at 260 s; the baseline needed 19 s),
  row 8 TIMEOUT (F upper 0.55 at 294 s; the baseline needed 233 s), row 12 CERT
  in 2.1 s.
- Validity (`test_n066_plan_v8_validity.py`, tanh and sigmoid): cgan 16, 3,073
  segmented units, worst row violation 4.5e-16.
- Smoke (`results/n053/smoke_v61_*.jsonl`, path v6.1 + n017, one baseline row
  per family): acasxu 7, cersyve 0, cora 2, malbeware 0, metaroom 0,
  relusplitter 2, safenlp 0, sat_relu 0 and vit 15 all retained.
- N039 v6 stopped at 1,446 rows (1,319 of 1,320 kept; malbeware 103 lost as a
  CUDA OOM ERROR). N061 stopped at 18 rows. Both superseded by
  **N039 v7** (`n039_full_replay_v7/`) and **N068** (`n068_e0_replay_v6/`), which
  run the same code. The runners retry a row once after an OOM inside its own
  deadline. PREREG and FREEZE are in each directory.
- `audit_n039_cert_sampling.py`: CERT sampling audit adapted to replay records.

## 2026-10-03 N070 — fail-closed guard for non-finite bounds; replays restarted

Engine n018 (`nhz_sound_v18.py`: the n014 score-difference remainder also in
the softmax value map) gave IBP ViT LP bounds 0.1077 / 0.0908 / 0.0773 on
rows 101 / 102 / 104, about the same as n014. On PGD rows it broke: 18,536 on
row 15 and **NaN** on row 25. Closed.

The NaN exposed a soundness hazard in every path up to v6.1. Open disjuncts
are selected by `v >= -1e-4`, which is false for NaN, so a NaN LP bound would
have been read as an exclusion. Fix: path v6.2 returns UNKNOWN if the
propagated state or its rows are not finite, and maps non-finite LP bounds to
+inf (open). A MILP exclusion needs a finite upper bound. Terminal v7.6 never
excludes with non-finite plan data. Scan of all 2,687 CERT records in this
directory: none has a non-finite LP or MILP bound field. Because the
recorded `lp_worst` is a maximum, a NaN mixed with finite values could still
be hidden. No NaN was ever observed with the engines used in those runs, but
every earlier CERT claim is treated as unconfirmed until the guarded replays
re-establish it. The diagnostic retention sample started before the fix runs
path v6.1 and is used only to find failure modes.

N039 v7 (375 + 16 rows) and N068 (3 rows) stopped; **N039 v8**
(`n039_full_replay_v8/`) and **N070** (`n070_e0_replay_v7/`) launched with path
v6.2 + engine n017 + terminal v8.1/v7.6. PREREG and FREEZE are in each directory.

## 2026-10-03 N071 — ACAS Xu 60: exact MILP finds no witness (retention sample)

Retention sample (`results/n053/retention_sample_v61_*.jsonl`, path v6.1, 10
baseline-solved rows from each of 7 families not yet reached by a replay). First
loss: ACAS Xu 60 (prop_2, four-atom conjunction), baseline ADV in 2.6 s via its
family branch `acasxu_cuts_fbbt`, TIMEOUT here. LP bound 2398; unstable units per
layer 35/35/49/50/50/50. `probe_n071_acas_witness.py`: on the exact all-layers
plan (248 binaries), none of the epigraph objective with target, the same with
HiGHS heuristic options, or a pure feasibility form (tau >= tiny, zero
objective) finds a feasible point in 60 s. The relaxation gives B&B no guidance.
The baseline's winning branch adds valid cuts and feasibility-based bound
tightening. Open.

## 2026-10-03 N072 — the element centre as first witness candidate; replays restarted

On ACAS Xu 60, 74 and 88 (lost ADVs in the retention sample) the input-box
centre is itself a counterexample, with prop_2 slack -0.0038, -0.014 and -0.0107.
The baseline's winning branch starts HiGHS from its base point (`--mip-start
base-binary`) and found these in about 2.5 s. 25 of the 34 ACAS Xu baseline ADVs
came from that branch. Path v6.3 checks the element centre, the image of
latent w = 0, first. That is one deterministic evaluation and no search. It is
recorded as witness source `element_centre`, separately from the LP box
maximiser, for the user's admissibility decision.

N039 v8 (559 rows) and N070 (5 rows) stopped; **N039 v9**
(`n039_full_replay_v9/`) and **N072** (`n072_e0_replay_v8/`) launched with path
v6.3. Families no replay has reached now run first in each worker. PREREG and
FREEZE are in each directory.

## 2026-10-03 N069 — intermediate selective-exactness levels on E0 (negative so far)

`probe_n069_e0_gamma_levels.py`, CIFAR rows closest to exclusion, 60 s per
level, terminal v8.1, engine n017. Row 92 (unstable per layer
330/225/83/18/51/18): gamma=1 upper 0.017 (10 binaries), gamma=2 0.061 (61),
gamma=3 0.050 (79). In a fixed time, more exact layers give a weaker bound
because the MILP is harder. (Remaining rows: see the result file.)

## 2026-10-03 N073 — losses tied to per-family baseline solver branches

Retention sample (path v6.1): CERSYVE 7 (baseline CERT in 46.9 s with branch
`cersyve_scip_cuts_fbbt`) times out here. The all-layers MILP (111 binaries)
reaches upper 0.47 after 97 s. ACAS Xu ADVs came from `acasxu_cuts_fbbt` (fixed
in path v6.3 by the centre witness, N072). Both branches combine valid ReLU
cuts, `--fbbt-passes 5` feasibility-based bound tightening and MIP starts. The
family-level choice of these options is not allowed for the candidate (hard
limit 3: structure, not family identity). Bound tightening that uses the
unsafe-property constraints is backward reasoning, which hard limit 3 restricts as
a source of domain claims. Whether a uniform, property-conditioned bound
tightening may be used, only to retain the baseline's own solves, is a
decision for the user. Not implemented.

## 2026-10-03 N074-N075 — SCIP as a uniform portfolio member

- N074 (`nhz_scip_backend.py`, `probe_n074_scip.py`): same v8 plan, same
  cutoff. CERSYVE 7: four HiGHS seeds open after 100 s (upper 0.47); SCIP
  `infeasible` (excluded) in 70 s. CERSYVE 3: HiGHS 70 s, SCIP 79 s. SCIP model
  build in pyscipopt takes 0.01-0.04 s for plans up to 1.3e5 nonzeros.
- Terminal v7.7: the portfolio is HiGHS seeds 0-2 plus a SCIP member in a forked child
  process (no shared interpreter lock; killed when another member decides or at
  the deadline) whenever the plan has at most 5e5 nonzeros. Exclusion is decided by
  any member's Infeasible under the cutoff, or by the minimum valid dual bound.
- N075 (`probe_n075_portfolio.py`, `results/n075_portfolio_fixed.jsonl`): CERSYVE 7
  excluded by SCIP at 83 s; SafeNLP 271 by a HiGHS seed at 15.4 s, SCIP stopped;
  ACAS Xu 7 by a HiGHS seed at 36 s, SCIP stopped. A first version of the
  probe tested SCIP alone by mistake (name check); the file
  `results/n075_portfolio.jsonl` is that wrong run.
- Path v6.4 = v6.3 + terminal v7.7. N039 v9 (36 rows) and N072 (8 rows)
  stopped; **N039 v10** (`n039_full_replay_v10/`) and **N076**
  (`n076_e0_replay_v9/`) launched. The v10 PREREG text was rewritten two minutes
  after launch because a shell substitution had failed; FREEZE refreshed for
  PREREG.md only. Code hashes are those at launch.

## 2026-10-03 N077 — what limits the ViT softmax (diagnostic)

`probe_n077_softmax_split.py`, engine n017, mean radii per softmax node. Row 101
(ibp), first softmax: tangent-plane linear part 0.0057, remainder 0.0042,
interval form 0.0058, score-difference radius 0.112. Later softmaxes: linear part
about equal to the interval, remainder small. Row 15 (pgd), second softmax:
remainder 4.99 against interval 0.096, all coordinates interval. Reading: on IBP
rows the linear part alone is already as wide as the interval, and the interval
itself comes from score-difference bounds about 1.4x wider than an ideal
zonotope, amplified by exp. Softmax-level changes cannot close the 7x gap to
the baseline. The score bounds (Q K^T with LayerNorm/BatchNorm inputs) are the
next target. Not pursued further in this session.

N069 continued: row 101 excluded at gamma=1 (-0.191, 36.7 s) and gamma=3, not at
gamma=2 within 60 s; row 105 not excluded at any level; row 52 best at gamma=2/3
(0.040/0.036 vs 0.125). No single level dominates; the stage structure (D
gamma=1, E all) is kept.

## 2026-10-03 N078 — dist_shift: segment fewer, more influential units (diagnostic scan)

`probe_n078_distshift_selective.py`: all ReLU layers gamma + smooth rows + 2
segments on the top-N sigmoid units by influence (chord gap x interval
|d obj / d y|, as N058), 300 s.

| row | N=50 | N=100 | N=200 | all 504 (N067) |
|---|---|---|---|---|
| 8 | 1.40 (28 s) | 0.78 (39 s) | **excluded (77 s)** | 0.55 at 294 s |
| 39 | 0.54 (14 s) | 0.47 (39 s) | 0.35 optimal (61 s) | - |

Segmenting every non-saturated unit is worse than segmenting a well-chosen
subset. Row 39 needs more exactness (its plan optimum at N=200 is still positive).
This was a diagnostic scan. The preregistered rule is tested in N079 (cover 90
percent of a generic structural score).

relusplitter 39 (retention sample, path v6.1): baseline CERT in 20.7 s, TIMEOUT
here. LP 1.79 with 9 open disjuncts; worst disjunct D 0.93 (27 binaries), E 0.70
(382 binaries) after 62 s. Suspected cause: the polishing caps leave intermediate
bounds of the 256-wide layers at GPU-LP quality. Tested in N080.

## 2026-10-03 N079-N081 — preregistered segment-selection rule; polishing caps; relusplitter 39

- N079 (`probe_n079_distshift_cover.py`): generic score `s_j = |g[eta_j]|`, the
  objective coefficient on smooth unit j's own fresh factor (its linearised
  downstream sensitivity times its shadow half-width; for conjunctions the max
  over atom rows). Segment (2 per unit) the non-saturated units in decreasing
  s_j until they cover 90 percent of the total. 300 s:

  | row | units segmented / non-saturated | result |
  |---|---|---|
  | 0 | 153 / 473 | **excluded in 25 s** (268 s with all units, N067) |
  | 8 | 150 / 504 | **excluded in 36 s** |
  | 39 | 198 / 581 | plan optimum +0.30 |
  | 48 | 144 / 517 | plan optimum +0.024 |
  | 51 | 132 / 561 | plan optimum +0.32 |
  | 71 | 156 / 499 | plan optimum +1.03 |

  Where the plan optimum is positive, only finer exactness can help (N079b/c:
  4 segments at 90 percent coverage, 2 segments at 99 percent).
- N080 (`probe_n080_polish_caps.py`): lifting the polishing caps tightens
  relusplitter 39 from 1.79 to 1.55 (1,330 LPs) but it still times out. Cora 1
  then propagates for 50 s. Caps kept.
- N081 (baseline worker, read-only, scratchpad output): on today's machine the
  baseline itself needs 92.9 s for relusplitter 39 (budget 90 s; 20.7 s when
  the baseline ran), with 153 binaries against our 401 unstable units
  (30/248/66/11/46). The box is identical (the VNNLIB is already clipped to
  [0, 1]). Why the baseline sees far fewer unstable units in the second
  layer is not resolved.

## 2026-10-03 N082 — finer segments; first v10 gains audited

- N079b (4 segments, 90 percent coverage, 300 s): dist_shift 48 **excluded at
  230 s** (588 binaries). Row 39 plan optimum +0.021 (806 binaries), down from
  +0.30 with 2 segments. N079c (99 percent, 2 segments) and N079d (4 segments
  on rows 0 and 8) are in the result files. A single-rule stage F with 4 segments
  is the candidate change for the final replay (next candidate, not v10).
- N039 v10 gains so far: cgan 8 and 11 (baseline UNKNOWN, ADV from the LP box
  maximiser) and cgan 13 (baseline UNKNOWN, CERT). Witness audit
  (`results/audit_v10_cgan_witness_*.jsonl`): all ten cgan witnesses in v10 are
  S1-valid; only row 0 is S2-valid. cgan specs fix many inputs to decimals that
  no float32 value matches, so S2 is impossible, as on MetaRoom. CERT sampling
  audit of cgan 13 (`results/audit_v10_cgan13_cert_*.jsonl`): 2,000 samples, 0
  violations, min unsafe slack 0.0039, independent box reader agrees.
- Strategy from here: no further restarts of N039 v10 or N076 unless a soundness
  issue appears. Improvements are collected for one final candidate replay.

## 2026-10-03 N083 — retention sample (path v6.1) final; probe load

Retention sample (`results/n053/retention_sample_v61_*.jsonl`, 10 baseline-solved
rows per family, path v6.1, memory fraction 0.1; diagnostic only):

| family | kept | lost (baseline branch, baseline time) |
|---|---|---|
| acasxu_2023 | 7/10 | 60, 74, 88 ADV (`acasxu_cuts_fbbt`, about 2.5 s); fixed by the centre witness (v6.3) |
| cersyve | 9/10 | 7 CERT (`cersyve_scip_cuts_fbbt`, 46.9 s); fixed by the SCIP member (v6.4) |
| cgan_2023 | 10/10 | - |
| linearizenn_2024 | 9/10 | 39 CERT (`linear_portfolio_m360`, 676 s of 900) |
| metaroom_2023 | 10/10 | - |
| relusplitter | 8/10 | 39 CERT (`normal`, 20.7 s; the baseline itself needs 92.9 s today), 108 CERT (`normal`, 46.4 s of 60) |
| tllverifybench_2023 | 7/10 | 25, 29, 30 ADV: CUDA OOM at fraction 0.1 (baseline TLL branches with eq-substitution, cut rows, 52-102 s) |

The E0 replay N076 shows slower propagation than N061 (row 3: 5.6 s vs 2.1 s)
and loses rows 3 and 10 that N061 certified. The machine load from my parallel
probes is the likely cause. The E0 gamma-level probe was stopped after 9 of 10
rows. From here at most one diagnostic probe runs beside the replays.

## 2026-10-03 N085 — independent soundness review; fixes; replays restarted as v11 / N084

A read-only review agent (CPU probes only, scripts in the session scratchpad:
`t_port.py`, `t_small.py`, `t_inw.py`, `t_int*.py`, `t_lp.py`) reviewed path v6.4,
terminal v7.7 and v8.1, the SCIP backend and the sign rule. Findings:

- **F1, definite latent false CERT.** `run_portfolio_v7` read `mip_dual_bound`
  from every HiGHS seed without checking `info.valid` or the status. On an error
  status (model or load error, memory limit, Not Set) or an exception inside a
  seed thread, the bound is 0, giving upper = cc + pad, which can be below
  -1e-4 for single-atom disjuncts. The reviewer reproduced this on a model that
  HiGHS rejects: real max violation +0.5, portfolio `excluded=True`. A tally of
  about 10.7k statuses in all result files showed no error status, so no
  recorded CERT is known to be affected. Fix (terminal v7.8): bounds only from
  valid info blocks of Optimal, Time limit reached, Interrupted by user or
  Target reached; an exception in a seed is an error status. Rerun: `Not Set`
  seeds give no bound and the portfolio is not excluded.
- **F2, possible.** HiGHS and SCIP silently drop |a| <= 1e-9. Reproduced:
  10^4 coefficients of 5e-10 made a feasible plan "infeasible". Fix:
  `_fold_tiny` removes such entries and relaxes the row bounds by their
  worst-case contribution over the column bounds, in the portfolio and in the
  SCIP backend. Rerun: not excluded.
- **F3, definite, low likelihood.** Segment piece columns were bounded by
  +-1e3, so a unit with |x| > 1e3 could be cut off. Fix (terminal v8.2): each
  piece is bounded by its own segment hull. Validity test N066 still passes
  (S = 2 and 4).
- **F4, definite false-ADV risk.** `inward_f32` could return a point outside a
  degenerate box coordinate and label it S2. Fix (path v6.5): no S2
  candidate when some coordinate contains no float32 value; S1 is then the
  only check.
- F5: `parse_vnnlib` ignores top-level X bounds when an input `or` is present.
  CERT stays sound, since a superset is verified. An ADV could lie outside the
  real region, and the independent witness audit (own VNNLIB reader, zero
  tolerance) is the safeguard. Recorded; ADVs are counted only after that audit.
- F6, not soundness: HiGHS 1.14 simplex and IPM interrupt callbacks never fire
  during a MIP, so the watchdog works only through the MIP callback and the
  time limit. With nb = 0, `Bound on objective reached` is not used as an
  exclusion, which is conservative.

Verified correct by the reviewer: the v7 and v8 rows at real points, column
offsets, the cutoff semantics of HiGHS and SCIP, interrupt statuses (never
`Infeasible`), the epigraph, the sign rule, non-finite handling, and the ADV
check.

N039 v10 (about 240 rows: 152/152 then 1 loss and 9 gains; dist_shift 0, 8 and
48 CERT via stage F, 39 and 51 lost, ADV 42 lost) and N076 (41 rows) stopped,
diagnostics only. **N039 v11** (`n039_full_replay_v11/`) and **N084**
(`n084_e0_replay_v10/`) launched with path v6.5. PREREG and FREEZE are in each
directory.

## 2026-10-03 N086 — dist_shift ADV 42 (v10 diagnostic)

Baseline ADV 42 (`configurable_fused_K3`, 99.7 s). v10: the E stage reached its
plan optimum (violation 0.402), but the decoded input is not a counterexample
on the real network, because the sigmoid relaxation lets the plan's y deviate.
Stage F (1,309 binaries, 2 segments on all non-saturated units) ended at the
300 s limit with a best relaxation point of 0.140, also not real. Open gap,
together with CERT rows 39 and 51 (plan optimum positive even with 4 segments
at 90 percent coverage for row 39). Local search around relaxation points
would be an attack and is not used.

## 2026-10-03 N087 — v11 scheduling amendment

To shorten the run, a third worker wC (linearizenn, tll, cora) was added about
one hour after launch, with the same frozen runner and code. A watcher stops
wB, by PID, after it logs vit_2023 199, so wB never starts linearizenn. Each
family's rows are taken only from its assigned worker. Recorded in
`n039_full_replay_v11/AMENDMENT_1_scheduling.md`; PREREG and FREEZE unchanged
and verified (21/21 OK).

## 2026-10-03 N088 — linearizenn gap (v11 in progress)

v11 loses linearizenn 13 and 14 (baseline CERT in 494 s and 556 s via
`linear_portfolio_m360`). Ours: LP bound 12.3, 7 ReLU layers, about 283
unstable units. The all-layers MILP (280-285 binaries) reaches only an upper
bound of 3.8-4.1 after 899 s. About eleven linearizenn baseline CERTs took
476-840 s with that branch, so further losses are expected. The relaxation is far
weaker than the baseline's on these AllInOne networks. To investigate after the
replay (no CPU-heavy probes during it).

## 2026-10-03 N089 — v11 resource failures and missing pooling (worker A)

- malbeware 103, 108, 115 (2 CERT, 1 ADV) end in CUDA OOM even after the
  in-deadline retry. The worker's self-imposed cap is 0.2 x 95 GB, and the GPU
  had about 74 GB free. These are resource failures, not verdicts. Planned
  for the final candidate: the retry raises the per-process memory fraction
  (for example to 0.5) when free memory allows.
- cgan 19 and 20 (small_transformer, baseline UNKNOWN) raise `NotImplementedError:
  MaxPool`. AveragePool and MaxPool (n010, LOG N040) were never merged into the
  n011-n017 lineage. No retention loss. Planned merge for the final candidate.
- SafeNLP 704 (ADV) times out at 20 s. Path v5 found it in 14.6 s. The
  portfolio now has SCIP in place of HiGHS seed 3. Near-budget SafeNLP rows
  stay load-sensitive (844 CERT at 19.9 s).

## 2026-10-03 N090 — v11 worker A complete; ACAS Xu ADVs (worker B in progress)

Worker A final (cersyve, cgan, metaroom, dist_shift, safenlp, sat_relu, malbeware):
1,490 of 1,510 baseline solves before malbeware's end, then 146/150 malbeware.
Losses: dist_shift 4 (39, 42, 51, 71), safenlp 1 (704), malbeware 4 (OOM:
103, 108, 115 and one more). Gains: cgan 2 ADV + 2 CERT, metaroom 4 ADV (S1-only
witnesses).

ACAS Xu (worker B, 72 of 186 rows): the centre witness recovers 9 of 13 baseline
ADVs in 3.6-4.7 s. Still lost: 55 (baseline `normal`, 69 s), 57 and 70
(`acasxu_cuts_fbbt`, 2.5 s), 69 (`normal`, 54 s). The baseline branch starts
HiGHS from a base point (`--mip-start base-binary`). Planned for the final
candidate: a MIP start for exact plans built from the element centre's exact
values (input w = 0; ReLU phases, Balas pieces and binaries from the true
activations; tau from the centre's slack), constructed as in test N066. It is a
feasibility hint only, so soundness is unaffected.

## 2026-10-03 N091-N094 — MIP start, stage W, pooling; final replay v12

- N091 (`probe_n091_mipstart.py`): MIP start from the element centre. The true
  ReLU phases (and active smooth segments) at latent w = 0 are set on the
  integer columns only (terminal v8.3 `exact_preacts`, v7.9 `setSolution`).
  On the exact all-layers plan, ACAS Xu 57: real ADV in 0.55 s with the start,
  none in 60 s without; ACAS Xu 70: 0.72 s with, none without. The start is a
  feasibility hint; HiGHS checks and completes it.
- N092-N093 (`run_n093_targeted.py`, path v6.6 + engine n019 = n017 +
  AveragePool/MaxPool from n010): ACAS 57/70 ADV in 39/50 s, CERSYVE 7 CERT 68 s,
  dist_shift 12, relusplitter 2, sat_relu 15 and ViT 15 kept. malbeware 103 no
  longer runs out of memory (the retry raises the memory fraction) but times
  out after the restart. cgan 19 (MaxPool) still runs out of memory. SafeNLP 704
  still times out.
- N094: stage W (path v6.7). On ReLU-only networks with several phase layers,
  the exact all-layers plan runs with the centre start and the violation target
  for 5 percent of the remaining time before stage D. ACAS 57/70 ADV in 3.5/5.5 s.
  CERT rows: ACAS 7 13.6 s, ACAS 114 0.7 s, CERSYVE 7 76.9 s, relusplitter 2 2.0 s.
- N039 v11 stopped (1,743 rows, diagnostic only; LOG N087-N090) together with
  E0 N084 (91 rows). Launching v12 killed my shell once, because an awk pattern
  matched its own command line. Nothing had started, and the launch was
  re-done from a script file. **N039 v12** (`n039_full_replay_v12/`, path
  v6.7 + engine n019, 3 workers) and **N092** (`n092_e0_replay_v11/`) are
  running. PREREG and FREEZE (23/23 OK) are in each directory.

## 2026-10-03 N095 — v12 progress and scheduling amendment

At 1,664 rows, v12 kept 1,570 of 1,581 baseline solves, with 9 gains and 0 conflicts.
Losses: dist_shift 39, 42, 51, 71; safenlp 704 (ADV) and 844 (CERT), both at 20.2 s on a 20 s
budget; malbeware 103 (timeout after the OOM retry); acasxu 55 and 69 (baseline `normal`, 69 s and
54 s); linearizenn 13 and 14. ACAS 57 and 70 are now ADV (MIP start / stage W). Scheduling
amendment `n039_full_replay_v12/AMENDMENT_1_scheduling.md`: cora_2024 moves to a new worker wD
when wA finishes; wC is stopped by PID after its last TLL row. No code change.

## 2026-10-03 N096 — v12 witness sources and Cora audit

Witness audit of v12 worker D (Cora, `results/audit_v12_wD_witness_*.jsonl`): all 118
witnesses are S1- and S2-valid; 99 are gains (94 on baseline TIMEOUT, 5 on UNKNOWN).

Witness sources in v12 so far (ADV outcomes):

| source | gains | retained ADVs |
|---|---:|---:|
| LP box maximiser (terminal LP dual -> box corner) | cora 96, cgan 2, metaroom 4, relusplitter 2 | safenlp 554, cora 19, malbeware 15, sat_relu 17, cgan 6, dist_shift 4, metaroom 1, linearizenn 1 |
| element centre | 0 | acasxu 28, safenlp 19, malbeware 4, cgan 2, cora 1 |
| MILP incumbents (stages W, D, E) | cora 3, acasxu 1 | safenlp 73, sat_relu 33, acasxu 4, cersyve 6, dist_shift 2 |

The LP box maximiser is the main witness source for retention as well as for gains. The user's
admissibility decision for it (and for the element centre) therefore affects the retained-ADV
count as much as the gains. The baseline found SafeNLP ADVs mostly through MILP incumbents.

## 2026-10-03 N097 — v12 scheduling amendment 2

wD finished Cora (180 rows: 19 baseline ADVs kept, 99 ADV gains, all witnesses S1- and S2-valid).
Two workers were added: wE (vit_2023) and wF (tllverifybench_2023), with the same frozen code.
wB will be stopped by PID after relusplitter 219 and wC after linearizenn_2024 59. The earlier
watcher, which waited for TLL 31 in wC's log, was stopped by PID. Assignment and timestamps are
in `n039_full_replay_v12/AMENDMENT_1_scheduling.md`.

## 2026-10-03 N098 — v12 findings on relusplitter and ViT (for the next candidate)

- Propagation ignores the row deadline on large models. relusplitter 110, 116, 141, 153 and 165 spend
  134-389 s in `propagate`; 116 is a baseline CERT lost this way (budget 60 s). Only the start of
  exact-LP polishing checks the budget share; the per-layer GPU LP tightening (300 iterations)
  never checks it. Next candidate: skip LP tightening of later layers once a fixed share of the
  row budget is spent. Like the polishing caps, this only loosens bounds and is sound.
- Stage W's 5 percent was taken per open disjunct, so on 9-disjunct properties W used 30-40 s
  before stage D (relusplitter 27, 55, 57, 67, 121, 151, 159). Its violation target also stops on
  points that fail the float32 ORT check (tiny violations; plans with integrality skipped by
  Prop. 1' are relaxations off the optimum). Next candidate: one W budget per row, not per disjunct.
- ViT IBP model (rows 100-106 so far): 101, 105, 106 CERT; 102, 103 TIMEOUT; 104 UNKNOWN (plan
  optimum positive). This matches the N053/N057 sample, about half of the IBP CERTs. The
  attention relaxation gap (LOG N063, N077) is the cause.

## 2026-10-03 N099 — user rule: gains only from Neural-HZ itself; cleanup to path v7.0

User decision (2026-10-03): "收益不允许来自任何攻击或者采样helper，我们的收益必须来源于neural-hz本身".
That rules out every witness or verdict source that is not the domain's own query:

| source used before | status | where |
|---|---|---|
| LP box maximiser (sign of the LP dual gradient, a box corner) | **not allowed**: LP-dual-guided attack-like heuristic | stage C, paths v1-v6.7 |
| input-box centre (latent w = 0) checked on ORT | **not allowed**: fixed sample point | stage C, paths v6.3-v6.7 |
| MIP start seeded from the centre's phases | **not allowed**: seeded from a sample point | terminal v8.3/v7.9 via paths v6.6-v6.7 |
| stage W (exact plan + centre start) | **removed** with the start | path v6.7 |
| MILP incumbents of the element's own plans (D, E, F), validated on ORT | allowed: domain query + independent validation | all paths |
| CERT from rigorous LP / MILP bounds | allowed | all paths |
| deterministic float32 rounding of a MILP incumbent (inward step, snapping coordinates within 1e-6 of a bound) | kept: numeric rounding of the one domain point, no new candidates; reported for review | paths v5+ |
| sampling (`audit_n017`, `audit_n039_cert_sampling`) | audit only, never a verdict | - |

Consequences for earlier results:
- Every ADV whose `witness_source` is `lp_box_maximiser` or `element_centre` does not count, and
  that includes retention. In N039 v12 that is 104 ADV gains (Cora 96, cgan 2, MetaRoom 4,
  relusplitter 2) and 671 retained ADVs (617 box maximiser, 54 centre). Earlier capability
  claims of MetaRoom 14/26/45/95 and of the Cora gains of N039 v3-v12 are withdrawn for the same
  reason. CERTs are unaffected (they never used these sources).
- N039 v12 (2,277 rows) and E0 N092 were stopped. They stay as diagnostics only.
- Path v7.0 (`nhz_path_v7.py`): stages A, B (LP), D, E, F (MILP); ADV only from MILP incumbents
  validated on ORT; no LP-dual corners, no centre, no MIP start, no stage W. Old path files are
  kept unchanged as history.
- Engine n020 (`nhz_sound_v20.py`): LP tightening stops after 40 percent of the row budget
  (sound: looser bounds), for the propagation overruns of N098.
- Runners: `run_n039v13_full_replay.py`, `run_n100_e0_path_v7.py`, targeted
  `run_n099_targeted.py`.

## 2026-10-03 N100 — domain-only smoke; engine n020.1; final replay v13 launched

- Smoke of path v7.0 (`results/n053/smoke_v70_*.jsonl`, 36 rows: baseline ADVs that v12 had
  recovered through the box maximiser or the centre, plus CERT checks). Still ADV through domain
  MILP incumbents: cgan 12/14/17, dist_shift 15/20/46, malbeware 71/112/132, metaroom 35,
  linearizenn 0, acasxu 97, cora 140, relusplitter 132/133, safenlp 30/449/498, sat_relu
  8/44/66, tll 5/11/23. Lost without the helpers: acasxu 71 and 87, cora 62 and 134. CERTs
  unchanged (acasxu 7, safenlp 271, dist_shift 0).
- relusplitter 116 still spent 128 s in propagation: exact-LP polishing (up to 600 dense HiGHS
  LPs, about 3,700 columns) ran past the share check. Engine n020.1 stops a running polishing
  loop at the 40 percent share as well; unpolished units keep their bound. relusplitter 116 is
  now CERT in 22.6 s (budget 60 s); relusplitter 2, acasxu 7, safenlp 271 unchanged.
- **N039 v13** (`n039_full_replay_v13/`, path v7.0 + engine n020.1, 5 workers) and **E0 N100**
  (`n100_e0_replay_v12/`) launched. PREREG and FREEZE (23/23 OK) are in each directory.

## 2026-10-03 N101 — v13 stopped for concurrency; v14 launched with four workers

v13 ran five formal workers plus E0 N100. SafeNLP 272 and 290 (CERT; kept by v12 with three
workers) timed out at 20.1-20.2 s. Hard limit 6 measures capability at the baseline's four-way
concurrency, so v13 (384 rows) and N100 (few rows) were stopped by PID. **N039 v14**
(`n039_full_replay_v14/`) runs the identical frozen code (code hashes equal to v13's) with four
workers (wA safenlp, sat_relu, malbeware, cersyve, metaroom, cgan; wB acasxu, relusplitter; wC
linearizenn, dist_shift; wD vit, cora, tll) and nothing else in parallel. The E0 replay with
the same code runs after v14 finishes.

## 2026-10-03 N102 — E0 replay scheduled after v14

`n102_e0_replay_v13/` (PREREG written now; FREEZE is written by the launcher at start, over the
same frozen code as v14). A watcher waits for the four v14 worker PIDs to exit, then starts
`run_n100_e0_path_v7.py` alone at memory fraction 0.3. N100's few rows are not used.

## 2026-10-03 N103 — empty-plan crash (malbeware 21) and the large single-layer malbeware ADVs

- v14 recorded malbeware 21 (baseline ADV) as ERROR `blocks must be 2-D` at 0.03 s. The network is
  affine on the box (0 unstable units), the LP left the disjunct open, and the MILP plan had no
  rows at all (no phases, no lambda rows, single atom), so `sp.vstack([])` failed. Paths up to
  v6.7 never reached this point on such rows because a helper had already answered. The fix
  (plan = latent box alone) is in copies, so the frozen v14 code stays untouched:
  `nhz_terminal_v7_10.py` (terminal v7.10), `nhz_terminal_v8_4.py` (v8.4), `nhz_path_v7_1.py`
  (path v7.1), `run_n103_targeted.py`. Test: malbeware 21 ADV in 0.4 s from the LP optimum of
  the box plan (a domain MILP incumbent with zero binaries). cgan 9 and 10 (v12 had them only
  through the centre) are ADV through stage-D incumbents in 362 s and 43 s (budget 900 s).
- malbeware 121, 139, 143 (baseline ADV; one ReLU layer with 9,700-13,800 unstable units; v12 found
  them through the centre in 34-77 s) time out under path v7.1: the exact plan has about 10k
  binaries and dense x rows, and HiGHS/SCIP find no incumbent within 100 s. These are losses of
  the domain-only path at this scale. v14 will record them the same way (its code has the same
  plans). A possible domain query for such rows is the exact LP optimum of the lambda level
  (lattice level 0, no binaries), i.e. the canonical abstract counterexample; whether the user
  regards it as a helper is asked in the report. It is not in any candidate.

## 2026-10-03 N104 — v14 worker A complete; scheduling amendment 1

Worker A finished its six families (safenlp, sat_relu, malbeware, cersyve, metaroom, cgan).
Losses there: safenlp 704 (ADV) and 844 (CERT) at 20.1 s on a 20 s budget; malbeware 21 (ADV,
empty-plan crash, LOG N103) and malbeware 121, 139, 143 (ADV, about 10k-binary plans, no
incumbent in 100 s). cgan 20 (baseline UNKNOWN) ran out of GPU memory even after the retry
(6.3 GB request; small_transformer with MaxPool), no retention loss. Gains so far: cgan 8 (ADV
from a stage-D incumbent), to be audited. Scheduling: worker E (tll, dist_shift, cora) took
worker A's slot; C stops after linearizenn 59 and D after vit 199 (by PID); the E0 replay N102
starts when all four have exited (`n039_full_replay_v14/AMENDMENT_1_scheduling.md`).

## 2026-10-03 N105 — ACAS Xu ADV retention under the domain-only rule (for the user's decision)

v14 so far keeps 2 of 15 ACAS Xu baseline ADVs (rows 55-71). In v12 28 of 34 were found by the
input-box centre and 4 by the centre-seeded MIP start; both are removed in path v7.0. The
exact all-layers plan (about 250 binaries) finds no incumbent within the 116 s budget from a
cold start. The baseline's own winning branch (`acasxu_cuts_fbbt`, 25 of 34 ADVs) ran HiGHS with
`--mip-start base-binary`, i.e. a warm start from its HZ base point. Question for the user: a
MIP warm start derived from the element's own centre is a solver setting (the incumbent is
still a point of the exact plan, completed by the solver and validated on ORT), while a direct
network evaluation at the centre is a sample. Under the strict reading both are out, and that
is what v14 runs. If the warm start is admissible, ACAS Xu and the large malbeware rows are
expected to be retained again.

## 2026-10-03 N106 — v14 interim vector at 1,973 rows

Retained 1,618 of 1,702 baseline solves; 0 conflicts; 6 gains (cgan 2 ADV + 2 CERT, relusplitter 1
CERT, vit 1 CERT), all from the domain's own LP/MILP. Losses: acasxu 26 ADV (all found by the
removed centre check in v12; LOG N105), vit 41 CERT (29 timeouts, 12 plan optima above zero; the
attention relaxation gap of LOG N063-N077), malbeware 7 (N103), relusplitter 4, linearizenn 3,
safenlp 2 (budget edge), tll 1. Remaining: relusplitter 130 rows, linearizenn 23, vit 25,
tll 11, dist_shift 72, cora 180.

## 2026-10-03 N107 — N039 v14 complete: the domain-only single-path replay of all 2,413 rows

Path v7.0 + engine n020.1, four workers (scheduling amendment 1), one frozen code set
(`n039_full_replay_v14/FREEZE_SHA256SUMS`, 23/23 verified), budgets = baseline timeouts.
**Retained 1,750 of 1,870 baseline solves; 0 conflicts; 40 gains (30 CERT, 10 ADV), every ADV a
stage-D MILP incumbent validated on ORT. Formal promotion gate: FAIL (120 losses).** Formal score
stays 1870/2413. Per-family table: `results/final_n039_full_replay_v14_table.md`.

| family | base | kept | lost | gains |
|---|---:|---:|---|---|
| safenlp | 1079 | 1077 | 704 ADV, 844 CERT (20.1 s on a 20 s budget) | - |
| sat_relu | 100 | 100 | - | - |
| malbeware | 150 | 143 | 21 ADV (empty-plan crash, N103), 121/139/143 + 2 more ADV (10k-binary plans), 103 CERT | - |
| cersyve | 11 | 11 | - | - |
| metaroom | 95 | 95 | - | - |
| cgan | 13 | 13 | - | 2 CERT, 2 ADV |
| dist_shift | 70 | 66 | 39, 51, 71 CERT; 42 ADV | - |
| acasxu | 120 | 94 | 26 ADV (found in v12 only by the removed centre check; N105) | - |
| relusplitter | 45 | 41 | 4 CERT (3, 39, 108, 116-class budget/scale rows) | **27 CERT** |
| linearizenn | 40 | 30 | 10 CERT (baseline portfolio 476-840 s) | - |
| tll | 17 | 15 | 2 ADV | 1 ADV |
| cora | 40 | 24 | 16 ADV (v12 had them through the LP box maximiser) | 7 ADV |
| vit | 90 | 41 | 49 CERT (33 timeouts, 16 plan optima above zero; attention gap) | 1 CERT |

Audits (`finalize_replay.py`, witness S1/S2 and CERT sampling) are running; numbers above are
pre-audit. E0 replay N102 started automatically after the last worker exited
(`n102_e0_replay_v13/LAUNCH_NOTE.md`, FREEZE 23/23).

## 2026-10-03 N108 — v14 audits

`finalize_replay.py n039_full_replay_v14` (`results/final_n039_full_replay_v14_*`): all 767 ADV
witnesses audited with the independent VNNLIB reader in exact rationals; every one is S1-valid
(baseline semantics), none unaudited. CERT sampling audit over all 30 CERT gains plus 20
retained CERTs per family (245 rows, 2,000 uniform samples each): 0 violations, independent box
reader agrees on every row. The audited vector of N107 stands: 1,750/1,870 retained, 0
conflicts, 40 gains, gate FAIL.

## 2026-10-03 N109 — E0 replay N102 terminated externally; relaunched as N109

N102 (`n102_e0_replay_v13/`) exited at 21:21 after 20 CIFAR rows (5 CERT on baseline UNKNOWN rows,
1 anchor lost, 14 TIMEOUT). Its log ends with a normal row line: no traceback, no error, no kill
message. Treated as an external termination on the shared machine. N109 (`n109_e0_replay_v14/`)
reruns all 400 E0 rows with the identical frozen code (hashes equal to N102 and N039 v14),
started with `setsid` so it is detached from any controlling session. N102's rows are kept as a
record and not used.

## 2026-10-03 N110 — reproducibility check: N102 vs N109 on CIFAR rows 0-19

Same frozen code, two separate processes: all 20 outcomes identical (5 CERT on baseline-UNKNOWN
rows: 3, 9, 10, 15, 17; anchor row 2 lost as TIMEOUT; 14 TIMEOUT). The domain-only path finds
anchor 2 (v12 found it through a stage-D incumbent in 14.5 s with the centre MIP start) no
longer within 100 s. N109 continues alone.

## 2026-10-04 N111 — SOUNDNESS CONFLICT on E0 CIFAR row 164 (N109); replay stopped

N109 row 164 (CIFAR100_resnet_large, eps 0.0039; baseline VALIDATED_ADV with an ORT-replayed
witness) came out **CERT**. Check: the baseline witness lies strictly inside our parsed box
(margin 1.25e-3 on every coordinate) and ORT gives disjunct 3 slack -0.022, i.e. a real
counterexample. Our stage E (all 5 ReLU layers gamma, 2,693 binaries) reported "Infeasible" from
all three HiGHS seeds and SCIP within 1.3 s for disjuncts 3, 54 and 37. A plan that excludes a
real point is not a relaxation of the exact concretisation: this is a false CERT. N109 was
stopped (fail closed). Until the root cause is found, every CERT produced by the engine lineage
n009-n020 with terminal v7/v8 plans is unconfirmed, including the 30 CERT gains of N039 v14 and
all E0 CERTs. Diagnosis follows.

## 2026-10-04 N112 — root cause of the row-164 false CERT: solver infeasibility on near-degenerate rows

`probe_n111_row164.py`, `probe_n111b_row164.py`, `probe_n111c_row164.py` at the baseline witness:
- engine bounds are sound (the true pre-activation is inside [l, u] in all 5 phase layers, margin
  >= 2.2e-4) and the value maps match ORT to within ex + 9e-7;
- the exact latent point satisfies every plan row to 3.9e-7 (135 y-relation rows off by that much)
  and has objective f = -0.5367 against the cutoff -0.5146;
- the LP relaxation is feasible (f = -1.11); the LP with the binaries fixed to the witness's phases
  is feasible with f = -0.623; yet HiGHS (presolve off) and SCIP declare the MILP **infeasible in
  1.1 s even without any cutoff**. With every finite row bound relaxed by 1e-5 the MILP behaves
  normally (incumbent -0.563, dual -0.974 after 60 s).
Reading: the y-relation rows have ranges of about 2 (ey + delta) = 1e-6 and coefficients mu down
to the solver tolerance scale; MIP domain propagation on these near-degenerate rows produces a
false infeasibility proof in both solvers. Our exclusion rule trusts that status, so a numerically
spurious "Infeasible" became a CERT. The domain math is not at fault; the plan is numerically
ill-posed for the solvers.

Fix (copies; frozen v14 files untouched): terminal v7.11 (`nhz_terminal_v7_11.py`) adds a
numerical safety width NUM_TOL = 1e-5 to every unit row and piece bound (y-relation, x-piece,
sign-rule rows, x0/x1 bounds and their d-rows); terminal v8.5 widens the smooth rows the same
way; path v7.2 (`nhz_path_v7_2.py`) uses them. All changes relax the plans, so soundness is kept
and the solvers see rows of width >= 2e-5. The margin 1e-4 still dominates the widening.

Consequence: every MILP-based exclusion of the whole lineage is unconfirmed until re-verified
with well-posed plans. LP-only CERTs (stage B, our own rigorous arithmetic) are not affected.

## 2026-10-04 N113 — fix validated on row 164; re-verification of all MILP-excluded CERTs launched

Path v7.2 on E0 row 164: TIMEOUT (no exclusion; the spurious "Infeasible" is gone), row 3 still
CERT. Regression on known CERTs (`results/n053/n112_regress_*.jsonl`): acasxu 7, dist_shift 12,
relusplitter 2 and 89, safenlp 280, cgan 13 reproduced; **cersyve 7 is no longer excluded within
100 s** (its earlier exclusion came from SCIP "infeasible" at 60-83 s and may have been spurious
as well). Counting the sources in the completed runs: v14 has 1,025 CERTs, of which 553 retained
and 18 gains were decided by the rigorous LP alone (unaffected) and 442 retained and 12 gains by
MILP exclusion (339 stage exclusions by "Infeasible" status, 138 by dual bound); N109 had 32 E0
CERT gains by MILP exclusion and 2 by LP alone.

`n113_reverify_v14_milp_certs/` (PREREG + FREEZE): every MILP-excluded CERT of v14 (454 rows,
two workers) and of N109 (32 rows) is re-run with path v7.2 at the original budgets. A reproduced
CERT is re-verified; a row no longer excluded is withdrawn from the vector. Until this finishes,
the v14 vector is: LP-only CERTs 571 confirmed, MILP CERTs 454 pending, ADVs 767 audited.

## 2026-10-04 N114 — re-verification complete: the re-verified v14 vector

N113 re-ran every MILP-excluded CERT of N039 v14 (454 rows) and of E0 N109 (32 rows) with path v7.2
(numerically well-posed plans). 448 of 454 formal CERTs reproduced; 6 withdrawn: malbeware 135 and
137 (retained CERTs, now 106 s and 126 s on a 100 s budget), cersyve 3 (UNKNOWN at 95 s: the
widened all-layers plan no longer excludes within the budget) and cersyve 7 (TIMEOUT), relusplitter
119 and 131 (gains, now 184 s and 182 s on 180 s). All six are near-budget rows; whether their
earlier exclusions were spurious or merely faster cannot be told apart, so they are withdrawn.
E0: 29 of 32 reproduced; withdrawn CIFAR 98, 163, 168 (gains) and **164 (the false CERT; now
TIMEOUT, conflict resolved)**.

**Re-verified N039 v14 vector (domain-only path, one frozen code set, audited):** retained
**1,746 / 1,870** baseline solves, **0 conflicts**, gains **38 = 28 CERT + 10 ADV** (relusplitter 25
CERT, cgan 2 CERT + 2 ADV, cora 7 ADV, tll 1 ADV, vit 1 CERT); 571 CERTs by the rigorous LP, 448 by
re-verified MILP exclusion, 767 ADVs S1-valid. Table: `results/reverified_v14_table.md`. Formal
promotion gate: FAIL (124 losses). Formal score unchanged at 1870/2413.

E0 N109 (170 of 400 CIFAR rows, re-verified): 31 CERT gains on baseline-UNKNOWN rows (2 by LP, 29 by
re-verified MILP), anchors 16 kept and 5 lost (rows 2, 83, 114, 155, 164; 164 was the false CERT, now TIMEOUT), 0 conflicts. Incomplete; the
full E0 replay is relaunched with path v7.2 as N115.

## 2026-10-04 N116-N117 — row-coupled softmax (engine n021, adopted); LP-tightened products (n022, negative)

- n021 (`nhz_sound_v21.py`): per softmax row, the differences s_j - s_{i*} to the token with the
  largest centre score are bounded through the element's rows by the rigorous GPU LP and merged into
  the difference box; ratio rows p_j <= e^{U_j} p_{i*}, p_j >= e^{L_j} p_{i*} are added on the output
  value maps. In our construction the attention product references the softmax fresh factors, so
  these rows reach the terminal LP (the baseline's equivalent rows were disconnected). Terminal LP
  bounds (`results/n116_vit_coupled.jsonl`): PGD row 15 0.296 -> **-0.328** (CERT at the LP stage);
  IBP rows 101/102/104 unchanged (0.109/0.094/0.077 vs 0.109/0.093/0.077). End to end with path
  v7.2 (`results/n053/n116_vit_pgd_*.jsonl`): 15 CERT 18.6 s, 62 CERT 7.0 s, **78 CERT 11.8 s (lost
  in v14)**, 25 and 39 TIMEOUT, 2 TIMEOUT. Propagation +6-11 s per ViT row.
- n022 (`nhz_sound_v22.py`): LP-tightened per-coordinate factor ranges in the state-state product
  remainder (rad2 = sum_t rx'_t ry'_t, chosen per coordinate against the DeepZ remainder). IBP rows:
  unchanged (the tightened remainder wins on 241 of 10,098 product coordinates); PGD 15: -0.839; cost
  +30-60 s per row. Not adopted.
Reading: on the IBP model the first attention block precedes every ReLU, so no row exists yet and
the Q K^T scores are pure zonotope ranges; the softmax interval already equals its linear part
(N077). The remaining IBP gap needs an attention element whose output stays affine in Q, K, V
generators with a tight remainder (THEORY 16, direction 2), which this session did not reach.

## 2026-10-04 N118 — stage statistics of v14 and the plan for the next candidate

Multi-layer rows with MILP stages in v14: stage D (last layer gamma) excluded 53 disjunct instances
(acasxu 23, relusplitter 15, vit 10, dist_shift 3, cgan 2) and stage E 133; D spent 4,200-6,800 s
per family without excluding on acasxu, relusplitter, vit and cora. A fixed cascade change is not
clearly better, and selecting stages by LP state is not allowed (hard limit 3); the cascade stays.

Next candidate (not yet replayed; a replay now would recover about one ViT row): path v7.2 +
engine n021 (`nhz_sound_v21.py`). The remaining 124 losses split into (a) 47 ADV rows that need
an incumbent the domain MILP does not find cold (acasxu 26, cora 16, malbeware 5): pending the
user's decision on a warm start from the element's own centre phases or the lattice level-0 LP
optimum (LOG N103, N105); (b) vit 48 CERT (attention relaxation; THEORY 16); (c) linearizenn 10
and the near-budget rows (cersyve 2, malbeware 4, dist_shift 4, relusplitter 4, safenlp 2, tll 2).
E0: N115 (path v7.2) is running alone; its CERT gains will be audited when it completes.

## 2026-10-04 N119 — N115 (path v7.2) vs N109 (path v7.0) on the first 170 E0 CIFAR rows

Identical outcomes on 164 rows. Differences: rows 30, 52, 82, 101 (baseline UNKNOWN) are now CERT
within budget (the numerically well-posed plans solve faster: 74-98 s versus timeouts); row 98 is
now TIMEOUT (CERT at 81 s before); row 164 is TIMEOUT instead of the false CERT. Rows 163 and 168,
which the targeted re-run N113 had not reproduced (100.7 s and 102.4 s), are CERT again at full
budget: budget-edge timing, not exclusion changes.

## 2026-10-04 N120 — N115 CIFAR100 half complete (pre-audit)

200 CIFAR rows with path v7.2 + engine n020.1, 100 s each, alone on the machine: **39 CERT on the
175 baseline-UNKNOWN rows** (domain LP/MILP only), anchors 18 of 25 kept as ADV (domain MILP
incumbents), 7 anchors lost (attack-found witnesses; E0 retention fails as expected), 0
conflicts. The CIFAR witness and CERT-sampling audits run in the background while the
TinyImageNet half proceeds.

## 2026-10-04 N121 — N115 CIFAR100 half audited

Witness audit (`results/audit_n115_cifar_witness.jsonl`, independent reader, exact rationals): all
18 CIFAR ADV witnesses are S1- and S2-valid. CERT sampling audit
(`results/audit_n115_cifar_cert.jsonl`): all 39 CIFAR CERT gains, 2,000 uniform samples each, 0
violations, minimum unsafe slack 0.70, independent box reader agrees. **CIFAR100 E0 half (path
v7.2, domain-only): 39 new CERTs on the 175 UNKNOWN rows, audited; anchors 18/25 kept.**

## 2026-10-04 N122 — PAUSED by the user

The user asked to pause all running experiments and all goals. N115 (E0, path v7.2) was stopped by
PID after 303 of 400 rows (CIFAR100 half complete and audited: 39 CERT gains, anchors 18/25; the
TinyImageNet half is partial and unaudited). Nothing of this project is running. State at the
pause: formal score 1870/2413 unchanged; re-verified N039 v14 vector 1,746/1,870 retained, 0
conflicts, 38 gains (LOG N114); next candidate engine n021 (LOG N116) not yet replayed; the user's
three open decisions are listed in FINAL_REPORT_20261003.md. Resume entry points: path v7.2
(nhz_path_v7_2.py), engine n021 (nhz_sound_v21.py), runners run_n115_e0_path_v72.py and
run_n112_targeted.py / run_n116_targeted.py.
