# MoE Verification Experiment Record

This document is the tracked protocol and provenance record for the controlled
weighted top-2 Route A experiments. Large datasets, checkpoints, logs, and raw
results remain under `/data1/Kane/MOE/ACT/data/moe` and are not committed.

## Safety and Git boundary

- Run only from `/data1/Kane/MOE/ACT` on `feat/moe-route-verification`.
- Use `/data1/Kane/miniconda3/envs/act-py312/bin/python`.
- Keep all datasets, caches, checkpoints, logs, and results under
  `/data1/Kane/MOE`.
- Do not modify `main`, create a pull request, or force-push.
- Experiment 1C refuses to run from another branch, a dirty worktree, or a
  different Python interpreter.

## Frozen Experiment 1 development cohort

The first 100 deterministic clean-correct CIFAR-10 samples (sample ranks 0--99)
form the development cohort. They have already informed algorithm and scheduling
choices and must not be presented as a confirmatory cohort. The main checkpoint is
`data/moe/checkpoints/cifar10_top2_e8_seed0_bal010.pt`.

Frozen artifacts:

| Artifact | SHA-256 |
|---|---|
| checkpoint | `fbaa7c871d28763ac5acb29a9502dc5d146e1d5af0b4a03e9911899251bd43f7` |
| raw result CSV | `1251b54449ad1223e5a0b83e4ba3e3126a49a147a8bfc25d96a050d1cbde367b` |
| runtime config | `6c531a5b214d51a9da478c2e800b4fbdab5434489b00ad38c9d6989d9b0b8f50` |
| tracked source config | `a1faac3672267db2261540c1d4df43472773affda38c84593bd30664a4fbd2ba` |
| sample indices | `73c891e95e33d508ed0e50ccbcf4d48572fc29a2e71492e96f90bcfc23c143cf` |

The frozen implementation HEAD is
`f183ecc71c4cb72d41f80a5d9de1775451dee331`. The independent audit found 400
fixed-radius rows, 55 boundary rows, 100 distinct samples, no duplicate or missing
fixed-radius rows, and no unvalidated `UNSAFE` witness.

Development-only findings are:

- exact-router HZ strictly reduced the ordinary-zonotope candidate set in 67 of
  90 route-unstable rows (74.4%);
- among route-unstable rows, the route-conditioned/monolithic binary-width ratio
  had median 0.465 and 90th percentile 0.577;
- three of 45 route-unstable samples had a fixed-radius unique `SAFE` certificate;
- fixed-radius solver coverage was 31.75%, so the unique-certificate yield is a
  censored lower bound, not an estimated population rate;
- the original fast bound path ignored retained guard constraints, so its 0%
  guarded ReLU reduction is not evidence against guard-aware tightening.

## Experiment 1C closure protocol

Experiment 1C diagnoses solver censoring before any weighted-gate fallback is
implemented. Its tracked configuration is
`act/pipeline/moe/configs/experiment1c_bal010.json`.

The first launch at implementation HEAD `4ad8181ca9a0c79d5a6c5973716d4d24a75ab9ff`
stopped before its first result row because a zero-radius, 3072-dimensional point
box was not retained by the sparse HybridZ propagation path. Its audit files are
preserved in `data/moe/results/experiment1c_bal010`. The bracket now certifies the
zero-radius endpoint from the strict concrete router margin and uses exact sparse
HZ feasibility only at positive radii. The corrected run writes to
`data/moe/results/experiment1c_bal010_r1`; it does not overwrite the failed launch.
That run preserved 15 rows from five samples before a numerically undecided
midpoint stopped the process. These partial artifacts are also retained. The
runner now retries an undecided midpoint with the configured higher budget. If it
remains undecided, it keeps the last formally stable/unstable endpoints, records
`bisection_complete=false`, and continues without claiming the requested bracket
precision. The next clean run writes to `data/moe/results/experiment1c_bal010_r2`.

### Deterministic diagnostic cohort

Select the first 20 **distinct sample ranks** after sorting the frozen fixed-radius
rows by `(sample_rank, epsilon)`, retaining only rows that are route-set unstable
and `UNKNOWN` or `TIMEOUT`. Selecting distinct ranks preserves the clean-sample
cluster as the statistical unit and avoids choosing apparently easy cases.

The frozen selection is:

- sample ranks: `0, 4, 6, 7, 8, 10, 11, 13, 14, 15, 20, 21, 22, 25, 26, 33, 36, 38, 42, 47`;
- CIFAR-10 test indices: `0, 5, 10, 11, 12, 14, 15, 19, 21, 23, 30, 32, 33, 38, 41, 53, 60, 64, 71, 79`.

For each sample, the router-only exact HZ checks whether any expert outside the
clean unordered top-2 set can belong to a legal top-2 set. Inequalities are
tie-inclusive. A bisection produces a certified bracket

`stable lower < minimum route-set-change radius <= unstable upper`.

The three preregistered diagnostic radii are 1.01, 1.05, and 1.10 times the
unstable upper endpoint. The initial upper endpoint comes from the first exact
route-unstable fixed radius in the frozen development CSV.

### Constraint-aware guarded support

For every feasible expert branch:

1. propagate the exact router HZ and attach the route-membership guard to the
   shared input generator frame;
2. propagate that constrained HZ into the expert;
3. use fast generator bounds to identify unstable preactivations;
4. run LP support only for the closest-to-stable configured neurons;
5. run exact MILP support only for the remaining configured critical neurons;
6. allocate a ReLU binary only if the resulting sound support bounds still cross
   zero.

Failed or timed-out support sides fall back to the original unconstrained fast
bound. An incumbent is never used as a bound. The result records selected neurons,
per-side solver status and gap, eliminated binaries, and tightening time.

### UNKNOWN taxonomy

Each branch and each radius uses one of these reasons:

| Reason | Meaning |
|---|---|
| `SAFE_PROVED` | every required expert branch is certified |
| `UNSAFE_FULL_FORWARD` | a reachable witness changes the concrete full-model prediction |
| `UNKNOWN_GATE_SUFFICIENCY` | expert safety is insufficient for the selected-softmax mixture |
| `UNKNOWN_EXPERT_WITNESS_NOT_LIFTED` | an expert witness is not a valid full-model witness |
| `UNKNOWN_SOLVER_LIMIT` | a feasibility or property solve remains undecided |
| `UNKNOWN_NUMERICAL` | a verifier/backend inconsistency prevents a semantic result |
| `TIMEOUT_SUPPORT` | guarded support falls back because its budget expires |
| `TIMEOUT_EXPERT_SOLVE` | the expert property solve reaches its budget |

For weighted top-2, a single expert violation remains `UNKNOWN`. Only a
full-model forward-validated reachable witness is `UNSAFE`.

### Staged schedule and outputs

The schedule is: cheap concrete prediction attack for validated `UNSAFE`, guarded
support, a low-budget expert solve, then escalation only for unresolved branches.
Propagation is reused between solver stages. Nested radii use only sound monotonic
reuse: larger-radius `SAFE` implies smaller-radius `SAFE`, and a smaller-radius
validated `UNSAFE` witness implies larger-radius `UNSAFE`.

Run with:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python \
  -m act.pipeline.moe.experiment1c \
  --config act/pipeline/moe/configs/experiment1c_bal010.json
```

The runner incrementally flushes
`diagnostics.jsonl` and `diagnostics.csv`, and writes `selection.json`,
`config.json`, `summary.json`, and `experiment1c.log` under the configured output
directory. Existing outputs are never overwritten. Each branch records candidate
and feasible route sets, solver stages, a labeled fast-HZ property lower bound,
guarded-support metadata, binary widths before and after guard-aware support,
witness validation, taxonomy reason, and split timings.

Ranks 100--199 remain untouched. They become the confirmatory cohort only after
the taxonomy, support schedule, and any decision to add a restricted weighted
top-2 McCormick fallback are frozen from these 20 diagnostics.

## Experiment 1C diagnostic result

The complete clean run used implementation HEAD
`d275fef5901c17e38e8ad28466be2cb10182de0a` and produced 60 rows for 20 distinct
samples. Every sample has exactly the three preregistered multipliers and every
reported `UNSAFE` has a full-model forward-validated witness. Rank 10 retained a
strict stable/unstable bracket after one numerically undecided midpoint; it is
marked `bisection_complete=false` rather than claiming the requested precision.

At the closest 1.01x radius, the sample-level outcomes are:

| Outcome | Samples |
|---|---:|
| `SAFE_PROVED` | 2 |
| `UNSAFE_FULL_FORWARD` | 4 |
| `UNKNOWN_GATE_SUFFICIENCY` | 12 |
| `UNKNOWN_SOLVER_LIMIT` | 1 |
| `TIMEOUT_EXPERT_SOLVE` | 1 |

Ranks 4 and 21 are the two route-unstable unique `SAFE` samples. The diagnostic
yield is 2/20 = 10.0% (Wilson 95% interval 2.8%--30.1%). This is positive closure
evidence, but it does not meet the preregistered requirement of at least 10 unique
samples and must not be presented as a confirmatory population estimate.

Across all 60 radii, the outcomes are 6 `SAFE`, 14 validated `UNSAFE`, 39
`UNKNOWN`, and one `TIMEOUT`. Of the 40 unresolved rows, 38 (95%) are classified as
gate-sufficiency or expert-witness-not-lifted incompleteness. This exceeds the
one-third trigger for the restricted weighted top-2 fallback; increasing solver
timeouts alone is therefore not the next action.

True constraint-aware support was evaluated on 126 expert branches. It reduced
the actual expert ReLU binary total from 1550 to 1159, eliminating 391 binaries
(25.2%) across 74 branches. The per-branch width-ratio median is 0.867 (IQR
0.602--1.000, p90 1.000), equivalently a median binary reduction of 13.3%. Support
cost 107.8 seconds in aggregate. This supports guard-aware binary elimination as a
secondary structural result, but no end-to-end runtime benefit is claimed without
a matched no-support solve.

The guard accounting uses two different measurement universes and is therefore
reported with the following explicit identity:

```text
actual expert binary elimination       = 1550 - 1159 = 391
direct LP support elimination           = 158
direct MILP support elimination         = 175
propagation-or-structural residual       =  58
                                         ----
total                                   = 391
```

The `fast_unstable=1433` count covers only preactivations in the support
statistics, whereas `1550` is the actual expert ReLU-binary count over all 126
branches before guarded propagation. The residual 58 is defined as the accounting
difference and is not attributed to LP or MILP support. Until a layer-matched
decomposition is available, the paper wording is limited to: constraint-aware
support eliminated binaries on 74 of 126 branches. It must not attribute the full
25.2% reduction directly to support optimization.

The runner summary's `support_after_milp_unstable` field is three too high because
layers fully stabilized by LP retained the pre-LP display value. Actual binary
allocation and certificates are unaffected. The independent audit reconstructs
the correct value as `1433 - 158 - 175 = 1100`; the reporting path is fixed for
future runs.

Decision: keep candidate reduction and route-unstable width separation as core
results; retain guard-aware elimination as a secondary structural claim; implement
the restricted weighted top-2 McCormick fallback before touching ranks 100--199.
Official baseline training and new model sweeps remain paused.

The independent audit implementation was committed at
`a51df8d0067b538dc75b9200702fa50782dc82e9`. Its report has no integrity issues
and is saved with the raw results. Frozen result hashes are:

| Artifact | SHA-256 |
|---|---|
| `config.json` | `1457ef26d8e3192ee71e52cac83c7182eb3786503799baddd67bd6304f064450` |
| `selection.json` | `8c16269d9366b2f234e9b349c7b97c49d23d5b2b0c61c79d174231b2f718390f` |
| `diagnostics.jsonl` | `1daa8ef2ceb06b248fb92b38fbeead6445b2fd97a81ec00d96302be93ec2f4bc` |
| `diagnostics.csv` | `757884616b98add5c477cd55f7502f048d8b923712ea6ede3090fec8e32e150a` |
| `summary.json` | `ee42d732d66af35eb87e2ed0388270cd037f30875e0ebf064bf6bc489ecaa6fa` |
| `experiment1c.log` | `6172e116639f67eaf76bd3e05237f881755a82a620321b2542e639c1a773e9fa` |
| `independent_audit.json` | `0be4d4eb9875bad7753229ef23ae1740069786507050afd034dff845d9b69af0` |

## Experiment 1F0 restricted weighted fallback protocol

Experiment 1C left 38 of 40 unresolved rows in semantic-incompleteness classes:
34 `UNKNOWN_GATE_SUFFICIENCY` and four
`UNKNOWN_EXPERT_WITNESS_NOT_LIFTED`. This triggers the restricted weighted top-2
fallback; longer solver timeouts alone are not the next action.

The F0 implementation and soundness contract are frozen in
`act/pipeline/moe/docs/weighted_top2_fallback.md`. Its tracked configuration is
`act/pipeline/moe/configs/experiment1f0_bal010.json`. The diagnostic is restricted
to those 38 parent rows and verifies the frozen parent JSONL SHA-256
`1daa8ef2ceb06b248fb92b38fbeead6445b2fd97a81ec00d96302be93ec2f4bc`.
Every new row is linked by parent artifact hash, source line number, source-row
hash, and a derived row ID. Raw F0 results use a new directory and cannot modify
`experiment1c_bal010_r2`.

F0 uses a numeric sigmoid range followed by a property-directed McCormick hull.
The encoded optimization problem contains no exponential, division, sigmoid
segment, or sigmoid binary. A negative relaxation optimum remains `UNKNOWN`; only
a saved and independently replayed full-model witness can be `UNSAFE`.

The preregistered F0 threshold is at least 10 resolved parent rows or at least two
new non-repeated unique `SAFE` samples. Below that threshold, F0 remains a formal
ablation and the next code stage is the preregistered F1 two-segment fallback.
Baseline reproduction, new training, and confirmatory ranks 100--199 remain
paused.

The first F0 launch at implementation HEAD `21bb1cf86a19b8355832c7a36358ef7ed9368f1d`
was stopped after eight rows because the result serializer requested obsolete
support metadata names (`complete_exact` and `gaps`) rather than the public
`exact` and `solver_gap` fields. Pair propagation and optimization ran, but all
eight rows were consequently mislabeled `UNKNOWN_WEIGHTED_NUMERICAL`; they are
not scientific F0 results. The partial directory
`data/moe/results/experiment1f0_bal010` is preserved and will not be overwritten.
The corrected clean run writes to `data/moe/results/experiment1f0_bal010_r1`.

## Experiment 1F0 diagnostic result

The corrected run used implementation HEAD
`7f6d36b415a080a5b47b0f4f0c12c3686266f2a2` and completed all 38 frozen parent
rows, representing 14 distinct clean samples. This is a selected
semantic-incompleteness diagnostic cohort, not a natural-sample prevalence
cohort. The results are:

| Result | Rows |
|---|---:|
| `SAFE_WEIGHTED_RANGE` | 26 |
| `UNSAFE_FULL_FORWARD_FALLBACK` | 5 |
| `UNKNOWN_WEIGHTED_RELAXATION` | 4 |
| `UNKNOWN_WEIGHTED_SOLVER_LIMIT` | 3 |

F0 resolved 31/38 rows (81.6%). It added nine non-repeated unique `SAFE` sample
ranks: `0, 7, 11, 13, 15, 20, 22, 36, 47`. These are closure cases selected from
previously unresolved rows; 9/14 must not be reported as a population estimate.
The five unsafe rows occur on ranks 8, 25, and 33. Every unsafe result has a saved
concrete input and was independently replayed against the full selected-softmax
model.

By parent taxonomy, the 34 `UNKNOWN_GATE_SUFFICIENCY` rows became 24 `SAFE`, three
validated `UNSAFE`, and seven still `UNKNOWN`. The four
`UNKNOWN_EXPERT_WITNESS_NOT_LIFTED` rows became two `SAFE` and two validated
`UNSAFE`. No pair-level exception or numerical result remained in the corrected
run.

Total per-row time was 1056.0 seconds: 32.5 seconds candidate enumeration, 124.7
seconds guarded expert propagation/tightening, and 852.0 seconds property solving.
Median row time was 14.0 seconds and the 90th percentile was 60.7 seconds. The
three solver-limit rows all belong to rank 38. F0 therefore has a localized
budget long tail rather than a cohort-wide solver failure.

The independent audit reported zero issues. It checked all 26 safe rows for
complete exact-pair coverage and all nine class-margin properties per pair, and it
replayed all five unsafe witnesses. The preregistered threshold (at least 10
resolved rows or at least two new unique safe samples) is passed on both criteria.
Decision: freeze F0 as the restricted weighted fallback; retain the seven
unresolved rows as its range-only/solver-limit ablation boundary; do not implement
or run F1 unless a later frozen evaluation requires more resolution. Confirmatory
ranks 100--199, baseline reproduction, and training remain paused pending the
next explicit stage decision.

The tracked machine-readable result manifest is
`act/pipeline/moe/configs/experiment1f0_bal010_manifest.json`. Core artifact
hashes are:

| Artifact | SHA-256 |
|---|---|
| `config.json` | `3d44ff3016a6ec8d4a18d17d0807557a4721fd39c233b6aa3110b02bc29a3787` |
| `selection.json` | `48e4dd4713acdcaaae3a8ff63bc690e5f944e2584d8d125eeba2af4d3c8a93aa` |
| `results.jsonl` | `a1da40c7ecbdb3b64ad86d825b41720265f97c7a08713b138e6d09523ca51057` |
| `results.csv` | `cf8ad59948dc176e666532987129d1bed50cd53f3489871247fa3e7f6d8bcb33` |
| `summary.json` | `3585df86bc9e692b15b95966c00f3b02de520d37c5b883cbefb474d79332effa` |
| `experiment1f0.log` | `0d3c085e0b0071e0b4b853ec35ced3b56d776889059a8a36cd1354877d696bfe` |
| `independent_audit.json` | `84cdc1879a38b593192c0b06cf459dfe587968228e7c6febb488a9ceead9b619` |

## Experiment 1 confirmatory preregistration

F0 is frozen after resolving 31/38 (81.6%) semantic-incompleteness rows without
encoding exponentiation, division, or sigmoid segmentation. F1 is not triggered.
The unseen deterministic clean-correct ranks 100--199 are preregistered as the
confirmatory cohort. Official baseline reproduction, new training, and balance
sweeps remain paused until this cohort is independently audited.

The protocol is split into two non-overwriting stages. The fixed-radius census
records 100 samples at `{0.25,0.5,1,2}/255` and performs router candidates,
exact feasible unordered top-2 sets, structural width, and guarded support only;
it does not solve the output property. The route-boundary stage uses one primary
radius per sample, `1.05 * certified route-unstable upper endpoint`, and runs
gate elimination followed by F0 only for frozen semantic-incompleteness reasons.
It never enters F1 automatically. The tracked protocol is
`act/pipeline/moe/docs/experiment1_confirmatory.md`, the config is
`act/pipeline/moe/configs/experiment1_confirmatory_bal010.json`, and the
machine-readable preregistration is
`act/pipeline/moe/configs/experiment1_confirmatory_protocol_manifest.json`.

Future guard accounting now enforces the branch-level invariant

```text
binaries_before - binaries_after
  = lp_support_eliminated
  + milp_support_eliminated
  + structural_or_propagation_eliminated.
```

This closes the development aggregate as `391 = 158 + 175 + 58` without
misattributing the 58 structural/propagation residual to support solves.
`fast_unstable=1433` counts only preactivations entering the direct support
statistics, while `binaries_before=1550` counts actual expert ReLU variables over
all guarded branches, so those totals intentionally have different universes.

The numerical SAFE policy is frozen in code and config: successful status-0
solves only, finite `mip_dual_bound` for MILP, a status-0 optimum for pure LP,
zero requested relative MIP gap, `1e-7` feasibility/integrality tolerances,
absolute-plus-relative `1e-9` outward correction followed by `nextafter`, and a
strict corrected SAFE margin above `1e-7`. A non-optimal primal incumbent can
never certify SAFE. Every concrete `UNSAFE` must still be replayed against the
full selected-softmax model.

Before scientific launch, a single rank-100 engineering probe at `0.25/255`
reproduced the frozen development prefix, completed exact candidate and route-set
analysis, and closed every guard identity. It produced no confirmatory artifact
and is not included in any endpoint.

The first confirmatory launch at implementation HEAD `3faabea9f` completed its
400-row census, then exposed a timeout-enforcement bug during boundary solving.
The runner recorded a 300-second limit but did not terminate the row: rank 155
took 302.3 seconds, and rank 171 returned an `UNSAFE` after 382.5 seconds. The
task was stopped immediately after that complete 72nd boundary row. Directory
`data/moe/results/experiment1_confirmatory_bal010` is preserved and permanently
excluded; neither its census nor any partial boundary verdict is scientific
evidence. Its failure and frozen artifact hashes are recorded in
`experiment1_confirmatory_protocol_manifest.json`.

The corrected `_r1` runner executes every boundary row in a separate spawned
process and terminates it at the registered 300-second wall deadline. A killed
row is `TIMEOUT/INSTANCE_HARD_DEADLINE`; partial artifacts stay quarantined and
no partial verdict is promoted. The scientific config is otherwise unchanged.
The new config and preregistration are
`experiment1_confirmatory_bal010_r1.json` and
`experiment1_confirmatory_protocol_manifest_r1.json`. The full census is rerun
under the corrected implementation so both stages share one frozen HEAD.

## Experiment 1 confirmatory audited result

The corrected `_r1` experiment completed under implementation HEAD
`1a67922c43f4e21f526e3aa12ef7b2f4e3242cba`. Its independent audit reported
zero issues and replayed all 20 unsafe witnesses against the full weighted
selected-softmax model. Ranks 155 and 171 hit the enforced wall deadline at
300.35 and 300.21 seconds; both remained `TIMEOUT`, with partial artifacts
quarantined and no late verdict promoted.

| Confirmatory endpoint | Result | Threshold | Decision |
|---|---:|---:|---|
| Exact < IBP on route-unstable rows | 83/86 = 96.5% | at least 20% | pass |
| Exact < zonotope on route-unstable rows | 75/86 = 87.2% | at least 20% | pass |
| Conditional width median / p90 | 0.430 / 0.530 | <0.7 / <1 | pass |
| Unique route-changing SAFE | 36/100 = 36.0% | at least 10 and 10% | pass |
| F0 semantic resolution | 43/60 = 71.7% | at least 25% | pass |
| End-to-end solved | 56/100 = 56.0% | at least 60% | **fail** |
| Independent audit / unsafe replay | 0 issues / 20 of 20 | 0 / all | pass |

The unique-safe Wilson 95% interval is 27.3%--45.8%. Sample-cluster bootstrap
95% intervals for candidate reduction are 91.3%--100% against IBP and
77.8%--95.6% against ordinary zonotope. The route-unstable width IQR is
0.386--0.473. These confirm candidate reduction, conditional binary-width
separation, and the central Route A unique-certificate claim on the explicitly
constructed route-boundary cohort.

The sole incomplete bisection, rank 195, retained a strict stable/unstable
bracket of width `1.5318627450980338e-05`, used `1.05 * upper`, and finished
`SAFE`; it was not dropped or silently replaced.

F0 added 31 safe certificates and 12 full-forward unsafe witnesses; 17 of its
60 invocations remained unresolved. Its paired runtime overhead was median 28.1
seconds (IQR 7.4--63.5, p90 115.1). Guard support eliminated 1610/10076 expert
binaries (16.0%) across 356/903 branches. The accounting identity is
`1610 = 1183 LP + 380 MILP + 47 structural/propagation`. Support used 706.3
seconds, 0.439 seconds per eliminated binary, and improved matched branch solved
coverage from 83.1% to 89.3%.

The 44 unresolved rows comprise 24 samples with no route boundary found in the
registered search through `4/255`, 14 weighted solver limits, two expert-solve
timeouts, two hard deadlines, one base solver limit, and one weighted range
relaxation unknown. Since solved rate is the only failed GO condition, public
baseline reproduction remains locked. This result does not trigger F1, training,
or a larger cohort automatically.

The final machine-readable result and core artifact hashes are in
`act/pipeline/moe/configs/experiment1_confirmatory_protocol_manifest_r1.json`.

## Experiment 1D applicable-unresolved closure protocol

The confirmatory 56/100 overall solved-rate endpoint remains a preregistered
failure and will never be overwritten. Its interpretation is decomposed into
76/100 samples with a certified route boundary inside the frozen search cap and
56/76 (73.7%) conditional verification coverage. The 24 samples with no boundary
through `4/255` are inapplicable to the route-boundary endpoint, not solver
failures, but they remain in the immutable all-sample denominator.

Experiment 1D freezes all 20 applicable unresolved rows: 14 weighted solver
limits, two expert timeouts, two hard deadlines, one base solver limit, and one
range-relaxation unknown. The tracked selection identifies each row by parent
artifact hash, line, row hash, sample rank, dataset index, and reuse mode. Radius,
candidate and pair semantics, guarded support, F0, and the numerical SAFE policy
are unchanged. F1, a larger route cap, baseline reproduction, and training remain
disabled.

D0 reuses completed expert/property decisions and reruns only unresolved ones.
Since the parent JSONL did not serialize live HZ/MILP Python objects, the required
state is deterministically rematerialized and its candidate, pair, and support
signature is checked against the parent record. The two killed rows have only a
fixed epsilon and Tier-1 progress record; they are explicitly labeled
`fixed_radius_rebuild` and do not repeat boundary search or bisection.

Every row has a 900-second wall deadline and a 300-second state record containing
the active branch and, where the F0 objective exists, incumbent, dual bound, and
gap. The Tier-1 feasibility formulation marks these objective quantities as not
applicable. SciPy/HiGHS does not expose a serializable warm start through this
backend, so a still-unresolved identical encoding is restarted with the remaining
budget after the checkpoint; no mathematical constraint changes.

The baseline unlock is preregistered as all 20 rows run, zero audit issues, all
new unsafe witnesses replayed, at least five newly solved applicable rows,
conditional coverage of at least 61/76 (80%), and no silent numerical fallback.
Closure results remain follow-up evidence and are never backfilled into the
original 56% endpoint. Full details are in
`act/pipeline/moe/docs/experiment1d_closure.md`.

The first D0 launch at HEAD `152fa8c5f` stopped after ranks 110 and 120 were
rejected by an engineering support-identity assertion. It compared the fast
preactivation count with the post-support expert ReLU binary count, which are
different accounting universes. No scientific solver verdict was returned. The
partial `experiment1d_bal010_d0` directory is preserved and excluded; the fixed
runner writes to `experiment1d_bal010_d0_r1` without changing selection, radius,
encoding, or budgets.

The `_r1` launch stopped at rank 129 after a second engineering assertion found
fallback-side count 6 instead of 5. All structural support quantities were
identical: fast and post-support unstable counts, LP/MILP eliminations, and
binary width. Because fallback-side status is time-budget dependent, `_r2`
requires exact structural identity and records that status drift separately.
The partial `_r1` directory is preserved and excluded; mathematical encoding and
all preregistered choices remain unchanged.

## Experiment 1D audited result

The clean `_r2` run completed all 20 frozen applicable-unresolved rows under
implementation HEAD `5f1b15ad2`. Independent audit reported zero issues and
replayed both new unsafe witnesses against the full selected-softmax model.

| Closure result | Rows |
|---|---:|
| `SAFE_WEIGHTED_RANGE` | 10 |
| `UNSAFE_FULL_FORWARD_FALLBACK` | 2 |
| `UNKNOWN_WEIGHTED_SOLVER_LIMIT` | 4 |
| `UNKNOWN_WEIGHTED_RELAXATION` | 2 |
| `UNKNOWN_SOLVER_LIMIT` | 1 |
| `INSTANCE_HARD_DEADLINE` | 1 |

D0 solved 12/20 rows. The parent confirmatory 56/100 overall solved-rate failure
remains immutable. Applicable coverage increases only as follow-up closure from
56/76 to 68/76 (89.5%). The run reused 201 completed property rows, reran 135
unresolved properties, and wrote 20/20 checkpoint records. Total row time was
5151.1 seconds (median 149.9, maximum 900.4 seconds).

The matched guard table over 225 branches is `n00=21, n01=17, n10=3, n11=184`.
The net solved gain is 14 branches and the exact two-sided McNemar p-value is
`0.00258`. Median paired support-minus-no-support solve time is `-0.069` seconds.
Binary elimination is positively associated with support-only transitions, but
this remains a secondary association, not an unconditional runtime claim.

All preregistered D0 unlock conditions pass: 20 rows, zero audit issues, 2/2
unsafe replays, 12 newly solved rows, 68/76 conditional coverage, and no silent
numerical fallback. Official baseline work is therefore unlocked but was not
started in this stage. F1 and training remain paused. Frozen hashes are recorded
in `act/pipeline/moe/configs/experiment1d_bal010_manifest_r2.json`.

## Post-1D positioning audit and ICML 2025 baseline B0

The post-1D code and artifact audit is recorded in
`act/pipeline/moe/docs/fse_positioning.md`. It corrects an external review's
central factual error: the bal010 checkpoint has a nonlinear
`3072 -> 128 -> 8` ReLU router, not the factory's empty-hidden default. The
confirmatory exact-router result therefore remains a correlation-preserving
nonlinear route-feasibility result. The audit also freezes important limitations:
the current model is verification-scale, the reported monolithic result is
structural width rather than a monolithic runtime, route-set enumeration is
combinatorial, and repeated fresh SciPy/HiGHS solves are a major implementation
cost.

The next scientific priority remains the official ICML 2025 RT-ER hard-top-1
ResNet18 baseline. Phase B0 is now specified in
`act/pipeline/moe/docs/baseline_icml2025_protocol.md`, with machine-readable
provenance in
`act/pipeline/moe/configs/baseline_icml2025_provenance.json`. The official remote
HEAD still equals the audited commit
`30ef94d77b5451595b82e739aa8938e1f4c4521f`, and the external clone is clean.

B0 found code/paper discrepancies that must remain visible: the paper uses 130
epochs while the script defaults to 200; the code defines no seed; W&B cannot be
disabled by the advertised flag because of a bitwise-complement bug; checkpoint
selection overwrites the last ten-epoch evaluation rather than selecting the
best model; and the repository contains no implementation of the paper's
analytic certificate. The certificate comparison must therefore be labeled an
author-paper formula reimplementation and must audit the formula's continuous
router-weight assumptions against the released hard-argmax model.

The existing `act-py312` environment lacks FFCV, timm, einops, and W&B and uses
newer torch/torchvision versions than the author pins. Dependency installation is
not authorized. Consequently B0 documentation is complete, but B1 smoke and
training remain blocked pending a separate dependency decision. No baseline,
training, F1, larger cohort, or ACT conversion was started by B0.

## Certificate applicability and property-independent F0 margin

A follow-up audit distinguishes the public RT-ER training/attack/model source
from the unpublished assets: the authors provide neither trained checkpoint
parameters nor an implementation of Theorem 5.4's certified radius. The theorem,
paper artifact, and permitted result labels are recorded in
`act/pipeline/moe/docs/icml2025_certificate_applicability.md`. In particular,
the hard-argmax route and raw-logit expert code require explicit applicability
checks for the theorem's Lipschitz router-weight and bounded expert-output
assumptions. Assumption failure is not called an unsound certificate without the
exact checkpoint, formula procedure, and independently replayed evidence.

ACT now provides `affine_top1_route_boundary`, an exact piecewise-linear
hard-route oracle for affine routers with optional input clipping and explicit
normalization folding. It is unit-tested but has not been run on RT-ER because a
trained official router is unavailable. Random initialization from the public
source is not a scientific substitute.

The F0 router-margin support has also been lifted out of the property loop. It
depends only on the guarded feasible pair, so future runs compute it once and
reuse one immutable `WeightedTop2GateRange` across all property rows. On the
frozen confirmatory execution shape, this would reduce margin support from 918
calls (1836 lower/upper solves) to 111 calls (222 solves), avoiding 1614 repeated
solver calls without changing any McCormick row. A differential regression test
requires cached and legacy encodings to have identical bounds and constraints.
The historical confirmatory and D0 artifacts are not rerun or rewritten.

This code/documentation stage launches no baseline, training, F1, additional
cohort, large-model download, or certificate experiment.

## Batch route oracle, paper questions, and backend interface

The affine hard-top1 oracle now returns a concrete route-boundary witness and
provides `affine_top1_route_boundary_batch`. The general finite-box algorithm
groups inputs by clean expert and competitor and uses breakpoint
`sort+cumsum`, never a scalar Python loop over test points. A declared
`capacity_grid_steps=255` fast path first validates exact uint8-derived
capacities, then performs one GPU-resident weighted histogram per route pair.
On a synthetic 10,000 by 3,072, four-expert workload, the general NumPy path
took 39.51 seconds, general CUDA sorting took 16.45 seconds, and the exact
quantized-grid path took 0.712 seconds on the available RTX PRO 6000 Blackwell.
These are implementation benchmarks, not RT-ER results.

Regression tests cover scalar/batch agreement, independent SciPy LP agreement,
upper-bracket witness replay, clean ties, unreachable competitors, finite-box
clipping, CUDA/NumPy agreement, quantized-grid rejection, and the released
CIFAR uint8 normalization folded into `[0,1]` pixels.

A full paper audit answered the two certificate questions. Theorem 5.4/5.5 has
no numerical certified-radius experiment, and the paper provides no procedure
or values for `L_Ri` and `r_Ri`. The five-leaf preregistration now separates an
uninstantiated formula, a sound but benchmark-vacuous formula, an applicable
non-vacuous formula, a hard-route assumption failure, and a Route A certificate
beyond the exact route boundary. Details are in
`act/pipeline/moe/docs/icml2025_certificate_applicability.md`.

Training-process telemetry is preregistered for seeds `0,1,2` and immutable
epochs `10,20,...,130` in
`act/pipeline/moe/docs/icml2025_route_telemetry.md` and its config JSON. It uses
the label **official-code, paper-config reproduction** and does not impersonate
the unavailable author checkpoint.

The official α,β-CROWN repository was audited read-only at commit
`e5c7e17bf0488843acb77b7519f59876717a49f4`. Its VNNLIB and expression front
ends accept coordinate input bounds rather than arbitrary route halfspaces.
The first sound scalable adapter is therefore CROWN over a guarded coordinate
box hull; an augmented-router-output Clip-and-Verify adapter remains unvalidated.
The backend contract is in `act/pipeline/moe/docs/expert_backend_interface.md`.

No α,β-CROWN or RT-ER dependency was installed, no environment was created, and
no baseline, training, N1, N2, F1, or theorem-radius experiment was started.

## Certification-gap mini-survey preregistration

The systematic mini-survey protocol is frozen in
`act/pipeline/moe/docs/certification_gap_survey.md`, with a machine-readable
mirror in
`act/pipeline/moe/configs/certification_gap_survey_protocol.json`. It defines a
2017-01-01 through 2026-08-29 window, uncapped eligible corpus, frozen Boolean
concept query, one-hop snowballing, explicit inclusion/exclusion codes, paper-
family deduplication, six evidence dimensions, official-artifact rules,
constant/semantics extraction, two-reviewer requirements, author-contact policy,
and artifact-centered wording. The already audited ICML 2025 paper is disclosed
as a motivating/calibration case, and the final analysis must be repeated
without it.

No search or author contact was executed by this preregistration stage. Silence
may only be labeled `NO_RESPONSE`, and an explicit refusal is required for
`DECLINED`. Contact remains blocked without user authorization.

The proposed zero-margin augmented-output reduction was also checked before
implementation. For `g=max_j(r_j-r_i)`, `s=min_k(C_k E_i+d_k)`, the property
`max(g,s)>=0` is unsound for `ANY_LEGAL_TOPK`: a legal tie has `g=0`, so an
unsafe `s<0` incorrectly passes. The backend interface now forbids that
compiler. Exact constrained implication or a disclosed conservative
`max(g-eta,s)>=0` reduction are the admissible future designs.

No environment, dependency, training, survey search, external message,
augmented adapter, highspy box hull, N1, N2, or F1 execution was started.

## Theorem 5.4 constants-provider implementation

The first executable analytic-certificate component is now implemented in
`act.pipeline.moe.certificate_constants`. It distinguishes sound global
induced-norm bounds, sampled-gradient diagnostics, and author-unspecified
constants at the type/status level. Supported sound compositions cover the
operators needed for the audited official CIFAR ResNet18 structure; unknown
graphs fail closed. Softmax probability outputs receive a sound scalar
`1/2` composition and `M_Ri=1`, whereas raw logits do not silently inherit that
bound. Hard argmax routing is explicitly `NOT_APPLICABLE` at reachable ties and
is never assigned a global router constant of zero.

The Equation (8) evaluator refuses missing constants and unnormalized gates,
and it cannot upgrade sampled gradients to a formal label. Eleven focused unit
tests pass. This stage computes no RT-ER certificate because no reproduced
checkpoint exists yet; it supplies the audited provider machinery required by
that future execution.

## Incremental guarded-box hull backend

The first scalable expert-adapter component is implemented as
`guarded_hz_box_hull_highs`. One guarded HZ is lowered to one HiGHS LP, and
coordinate objectives are changed in place. Incomplete objectives retain sound
fast generator bounds; binary HZ variables are relaxed and cannot receive an
exact label. Solver telemetry records model builds, objective changes, status,
iterations, solve time, and accepted basis submissions without claiming
unobservable internal warm-start use.

Six new tests and the 47-test MoE regression set pass. A non-scientific
official-shaped engineering check with 3,072 coordinates and three guard rows
performed 6,144 support solves from one model build in 2.67 seconds. This is an
implementation check, not a CROWN result or an RT-ER result. The original
confirmatory endpoint is unchanged.

The first real frozen-branch run (`_r1`) was stopped after one branch and is
retained as a failed engineering artifact. HiGHS returned `kWarning` while
adding the guarded matrix because its default `small_matrix_value=1e-9` would
drop a legitimate router coefficient of approximately `9.70e-10`; the wrapper
correctly refused to treat that as a complete solve. The backend now requests
the smallest supported threshold (`1e-12`), canonicalizes the CSR matrix, and
still fails closed if HiGHS reports another warning. A regression test covers
the sub-`1e-9` guard coefficient. On the same diagnostic branch, the corrected
backend completed all 6,144 objectives in 5.14 seconds with zero fallback
sides; SciPy had taken 81.37 seconds and the recorded hulls differed by at most
`1.98e-9`. This single branch is a fix-validation observation, not the frozen
paired speed result.

The separate `_r2` run completed all 20 frozen samples and all 43 exact
feasible unordered top-2 route branches. Both backends solved 264,192
coordinate objectives with zero fallback sides, and all 43 paired hulls agreed
within the frozen `1e-8` tolerance (recorded maximum absolute difference 0).
The alternating order produced 17 `highspy->scipy` and 26 `scipy->highspy`
branches. Incremental HiGHS used 43 model builds and 43 cold starts, taking
230.83 wall seconds; the SciPy reference used 264,192 model builds/cold starts
and took 3,491.51 seconds. The paired branch speed ratio has median 15.03x and
the aggregate wall ratio is 15.13x. This is a descriptive engineering result
for guarded coordinate hull construction, not a solver-internal warm-start
claim and not an end-to-end verification speedup.

An independent audit replayed all 43 compressed bound artifacts, verified
their hashes, rematerialized all exact feasible pairs, recomputed every bound
hash and difference, and found zero issues or disagreements. The immutable
publishable summary is
`act/pipeline/moe/results/guarded_box_hull_benchmark_20260829.json`. The
original confirmatory 56/100 endpoint remains unchanged.

## N2 normalized weighted top-k generalization

The F0 decomposition is generalized from selected-softmax top-2 to any
normalized non-negative top-k gate. For a canonical anchor, the encoding uses
exactly `k-1` property-directed products and retains the omitted anchor weight
through the simplex-intersection-box constraints. Selected softmax, normalized
sigmoid, and hard top-1 are supported; unnormalized `switch_prob` is rejected
because it requires an additional scale product.

The implementation preserves shared input generators and route constraints
while assigning disjoint private expert factors. Thirteen N2 tests cover the
gate support matrix, top-k ties, concrete gate boxes, `k-1` decomposition
identity, shared/private factor identity, McCormick consistency, simplex
constraints, top-2 F0 compatibility, cross-domain rejection, extreme sigmoid
scores, relaxation status semantics, and hard-top1 certification. The proof is
recorded in `act/back_end/moe/proofs/normalized_topk_decomposition.md`. This is
a mechanism result; no new model or experiment result is claimed by the code
stage.

The N2 data path is now executable for the frozen alternate gate family:
CIFAR-10, 8 experts, normalized-sigmoid top-3, router width 128, expert widths
256/128, Switch coefficient 0.10, seed 0, 50 epochs, and batch size 256. The
runner selects deterministic clean-correct ranks 0--4, exactly enumerates all
tie-inclusive unordered top-3 sets, retains each set guard over the shared
input frame, tries expert-wise gate elimination first, and invokes the generic
range fallback only when needed. Every fallback property asserts exactly
`k-1=2` products. A negative relaxation candidate remains `UNKNOWN`; only a
full selected-model replay may produce `UNSAFE`.

The frozen seed-0 checkpoint is now complete and hash-locked at
`32ca3b25b11dfc2021929b983a9ebde6626bf4b901891a03bc0281fbce998f73`.
Best validation accuracy is 51.22% at epoch 40 and test accuracy is 50.35%.
Test load entropy is 2.0633, effective expert count is 7.872/8, and the maximum
and minimum expert loads are 16.40% and 8.52%, respectively. Thus the alternate
gate model supplies all eight experts without the collapse that invalidated the
earliest pilot. The immutable training manifest is
`act/pipeline/moe/results/experiment1n2_top3_training_20260829.json`.

New checkpoints record epochs, batch size, learning rate, and weight decay in
addition to the prior training provenance; the N2 runner rejects legacy,
mismatched, or hash-changed payloads instead of inferring those values. No
top-3 verifier coverage, runtime, or certificate number is claimed until the
separate five-sample N2 run and its independent audit complete.

The first N2 launch (`_r1`) failed before sample selection and created no result
directory. `map_location=cuda` controlled state-tensor loading but did not move
the reconstructed module, so a CUDA selection batch reached a CPU router. The
launcher log is preserved and excluded in
`act/pipeline/moe/results/experiment1n2_top3_seed0_r1_failed_20260830.json`.
The runner now explicitly moves the selection module to the frozen selection
device and tests that placement. `_r2` starts all five samples from scratch in
a new directory; verification semantics and budgets are unchanged.

The clean `_r2` run completed all five ranks at implementation HEAD
`f13e749ffc3f60ab7be62950f809512ee55db749`. Across eight exactly feasible,
tie-inclusive unordered top-3 sets, four branch obligations were SAFE, two
were full-model-replayed UNSAFE, and two remained solver-limit UNKNOWN. At the
sample level this is one SAFE, two UNSAFE, and two UNKNOWN. The run evaluated
55 fallback property rows; every row used exactly `k-1=2` McCormick products.
One SAFE sample closed at gate elimination, while three individual route sets
were proved by the generic weighted-range fallback.

The runner audit reports zero issues. A separate external audit rebuilt the
exact router HZ for every sample, reproduced all eight feasible sets, checked
the seven-file runner manifest, and independently replayed both UNSAFE
witnesses through the complete normalized-sigmoid top-3 model. Its final r2
audit reports zero issues. The first external audit is retained with two false
positives because it incorrectly demanded all nine property rows after an
already validated early UNSAFE witness.

This promotes N2 from code-only mechanism evidence to a five-sample engineering
generality result. It demonstrates an alternate normalized gate family and
non-top2 execution, but it is not a prevalence, scalability, accuracy, or
baseline-superiority result. The tracked summary is
`act/pipeline/moe/results/experiment1n2_top3_seed0_r2_20260830.json`.

## Tie-safe eta implication compiler

The augmented-output backend now has a sound hard-top1 compiler for
tie-inclusive routing. It verifies `max(g_i-eta,s_i)>=0`, expressed as a scalar
affine/ReLU DAG, instead of the unsound zero-margin `max(g_i,s_i)>=0`. Seven
tests include the explicit tie counterexample, randomized implication checks,
direct graph-semantic equality, varying branches, and eta-band accounting.

The formal proposition and proof are in
`act/back_end/moe/proofs/tie_safe_eta_implication.md`. The exact additional
obligation domain is `0<g_i<eta`; `g_i=0` remains a required legal tie. The
default `eta` is the frozen `safe_positive_margin=1e-7`, and the audit reports
the mathematical band separately from numerical-boundary tolerance. This code
has now passed a pinned auto_LiRPA 0.7.2 toy conformance on the Blackwell GPU.
Four analytically constant cases cover a safe legal tie, an unsafe legal tie,
the strict eta overcheck band, and a non-member point beyond that band. CROWN
reproduces each exact scalar range and rejects the unsafe tie; the naive zero-
margin compiler would incorrectly return zero. This is only a
`TOY_CONFORMANCE_PASSED` adapter result, not an official-model certificate or
an outward-rounding validation.

Two isolated environments were created under `/data1/Kane/MOE` without
changing `act-py312`. The α,β-CROWN environment imports auto_LiRPA 0.7.2 and
runs CUDA kernels on the installed Blackwell GPU. The exact official RT-ER pin
imports successfully, but its PyTorch 2.4.0+cu121 binary supports through
`sm_90` and fails its first CUDA kernel on the machine's `sm_120` GPU. The
author-pin incompatibility is retained as a result; B1 training was not
started and no newer PyTorch was silently substituted. Exact versions and
repository commits are recorded in the environment manifest.

The checkpoint telemetry protocol now has an executable runner rather than
only an oracle/config. It loads an immutable official checkpoint from the
clean pinned source tree, folds the uint8-domain normalization into the affine
router, checks folded scores and clean routes against the official model,
cross-checks 100 points against the general oracle, and writes all 10,000
per-input brackets. It also records clean correctness, route-load entropy,
directed/unordered boundary pairs, the frozen 200-input witness cohort, and the
strict eta-only overcheck band. Missing PGD metrics remain missing unless an
epoch metrics artifact is explicitly supplied. The runner has not been used
on a trained checkpoint because the author-pin CUDA gate remains blocked.

## N1 retained-path conditioned difference support

N1 now partitions an affine router margin into closed intervals, retains each
interval as two weak constraints on the shared HZ factor frame, and recomputes
property-directed expert-difference support per segment. Adjacent intervals
overlap at every cut, so tie points are covered by both sides. A segment is
dropped only after an infeasibility proof; solver-unknown segments remain in
the sound union, and incomplete support sides fall back to the unconditional
bound with explicit telemetry.

This is not F1: no sigmoid is encoded or segmented. The implementation records
`segmentation_axis=affine_path_margin`, `gate_function_encoded=false`, and
`sigmoid_segments=0`. It also checks the retained constraint prefix in addition
to frame identity, preventing cross-domain reuse when a same-ID HZ has lost a
path condition. Eight focused tests cover tightening, cut/tie coverage,
concrete interval containment, monotonicity, zero-budget fallback, F1
separation, constraint loss, and frame mismatch. The support-monotonicity and
interval-union proof is in
`act/back_end/moe/proofs/conditioned_support_monotonicity.md`. This stage is a
mechanism result. The mechanism is now connected to F0 as one ordinary range-
only McCormick encoding per active closed margin segment. Every segment must be
SAFE for `SAFE_WEIGHTED_SEGMENTED`; a relaxation candidate remains UNKNOWN
pending full-model replay. The implementation reuses an identity-bound
conditioned difference range instead of silently recomputing it on a different
domain. Two additional tests cover end-to-end segmented SAFE aggregation and
the prohibition on direct relaxation UNSAFE. Its engineering performance rerun
is reported below.

## Certification-gap survey partial execution

The frozen search has begun, but several preregistered source-native exports
and one-hop snowballing remain incomplete. The current corpus is therefore
labeled `PARTIAL_SOURCE_EXECUTION_RECONCILED_NO_PREVALENCE_CLAIM`; no ecosystem
proportion is reported. No authors were contacted.

Two reviewers independently screened the same 321-record frozen corpus. Binary
eligibility agreement was 317/321 (98.75%, Cohen's kappa 0.661); three-class
include/exclude/duplicate agreement was 315/321 (98.13%, kappa 0.618). Six
written adjudications resolved four scope decisions and two additional version
duplicates. The reconciled partial corpus contains 318 provisional families,
of which nine received full-text review, one was excluded, and eight were
retained for the future six-dimension artifact audit. These are screening-flow
counts only, not prevalence estimates. Reviewer inputs, hashes, disagreement
rationales, and the executable reconciliation are committed with the partial
result.

Primary-source extraction is now complete for those eight already-adjudicated
families. The partial matrix contains 48/48 frozen dimension cells, each with a
URL and section/page/path locator, and retains the stronger label
`PARTIAL_RETRIEVAL_NO_PREVALENCE`. An independent schema/evidence-completeness
audit reports zero issues. The matrix does not close the preregistered search:
source-native exports and snowballing remain incomplete, SpecSphere's official
anonymous code link returned HTTP 403, and no author contact was authorized.

The remaining source-access and one-hop snowball pass is now recorded as a
second partial artifact, still with no prevalence claim. Eleven source rows
record the exact endpoint outcome: arXiv and the OpenAlex cross-index exports
are available; DBLP cannot express the required abstract/keyword conjunction;
and ACM, IEEE metadata, PMLR, CVF, USENIX, Springer metadata, OpenReview, and
Semantic Scholar remain blocked, incomplete, or lack a suitable complete
native export. In particular, structured zero-citation counts are explicitly
not treated as evidence that a seed has no citations.

One-hop backward/forward retrieval produced 13 unique non-seed candidates and
reencountered four seed families that were deduplicated. Applying the frozen
criteria yielded no new included family: the 13 preliminary exclusions are
partitioned among no input-dependent dynamic routing, attack/empirical-only
robustness, and no formal bounded-input property. These are candidate-level
screening records, not a recall estimate. The artifact
`act/pipeline/moe/results/survey/snowball_partial_20260829.json` stores every
primary URL, discovery edge, dedup key, exclusion code, source endpoint, and
coverage limitation. Its independent structural audit reports zero issues.

## ICML 2025 B1 author-pin compatibility probe

The exact official dependency pins now import from the isolated
`/data1/Kane/MOE/envs/rt-er-repro` environment, and the external official clone
remains clean at `30ef94d7`. The B1 probe fails at the first CUDA tensor kernel:
PyTorch 2.4.0+cu121 supplies architectures through `sm_90`, while the installed
Blackwell GPU is `sm_120`. The author script unconditionally calls CUDA, so no
dataset conversion, training smoke, or epoch was started. This is recorded as
an exact-pin compatibility result. A newer-PyTorch run requires a separately
labeled Blackwell-compatible reproduction and cannot silently replace the
author-pin result.

The separately labeled Blackwell environment now passes the compatibility gate
as `official-code, Blackwell-compatible deps + FFCV`. It uses Python 3.11.14,
PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, and a locally built FFCV 1.0.2;
the exact-pin and CROWN environments remain unchanged. The first CUDA kernel
passes on `sm_120`. FFCV initially failed because no OpenCV pkg-config entry was
available, then built after installing headless libopencv and build support
inside the new environment only.

The official torchvision-to-FFCV import order exposes a JPEG ABI conflict. The
required launch condition is recorded explicitly as
`LD_PRELOAD=/data1/Kane/MOE/envs/rt-er-blackwell/lib/libjpeg.so.8`; it is not a
silent environment substitution. With that condition, 16 deterministic
synthetic CIFAR-shaped samples traversed FFCV and the official
`MOE_Resnet18(num_experts=4)` completed a finite forward/backward pass on the
Blackwell GPU. No optimizer step, epoch, checkpoint, accuracy measurement, or
training result was produced. A slow real CIFAR archive download was stopped at
2,719,744 bytes and left unextracted. The compatibility manifest records the
source hashes, dependency versions, explicit launch condition, smoke scope, and
disk accounting; B1 real-data training remains a separate next stage.

## ICML 2025 B1 Blackwell seed-0 execution code

The real-data B1 path is now executable without modifying the official clone.
The compatibility launcher freezes Python, NumPy, Torch, CUDA, cuDNN, and FFCV
random state and records two non-scientific shims: seed injection and a local
JSONL W&B replacement for the author's broken `--nowandb` branch. The smoke
uses the audited real CIFAR-10 archive, official FFCV transforms, official
four-ResNet18 MoE, PGD-10 training objective with the author's fixed 512 loss
denominator, one Adam step, PGD-50 evaluation, and a strict checkpoint
roundtrip. It is a compatibility gate, not an epoch or accuracy endpoint.

The seed-0 supervisor freezes the 130-epoch schedule and all 13 ten-epoch
checkpoints. At each checkpoint it waits for the matching author metrics,
pauses the complete training process group, creates an immutable epoch-qualified
copy, runs exact 10,000-input route telemetry synchronously, verifies artifact
identity, and resumes training. A telemetry error terminates the run and leaves
an explicit failure artifact. The external clone must be fully clean before,
during, and after execution. This section records an implemented, tested code
stage only; no real-data smoke or training result is claimed until the separate
result stage completes.

Two pre-objective smoke launches are retained as excluded engineering failures.
Smoke r1 used a module entry point that imported ACT's Python-3.12-only package
surface in the Python 3.11 Blackwell environment; the direct standalone entry
point avoids that unrelated import. Smoke r2 completed real CIFAR decoding and
FFCV conversion, then stopped in the smoke-only route counter because FP16 FFCV
input was inspected outside autocast. It did not enter PGD or take an optimizer
step. The route counter now uses the same autocast dtype contract as the author
forward. Neither failed launch contributes a scientific result.

Smoke r3 passed the corrected gate in 13.18 seconds. It used the audited real
CIFAR-10 archive and official FFCV transforms, executed the PGD-10 RT-ER loss
and one Adam update with finite gradients, ran a fixed-batch PGD-50 evaluation
with finite outputs, and strictly reloaded all four experts plus the router.
The restored logits matched exactly (maximum absolute error 0). An independent
hash/schema/state audit reports zero issues, and the official clone remained
fully clean. The compact result is
`act/pipeline/moe/results/environments/rt_er_b1_real_smoke_20260830.json`;
the 800 MB raw smoke artifacts stay outside Git under the isolated run root.
This passes the compatibility gate only and does not claim reproduced accuracy.

The running seed-0 reproduction has completed its first frozen endpoint at
epoch 10. Clean and PGD-50 test accuracy are 30.81% and 25.11%. The exact route
radius median is 0.0025525 (0.651/255) and p90 is 0.0074740 (1.906/255): 66.39%
of test inputs admit a route boundary within 1/255, 91.17% within 2/255, 99.68%
within 4/255, and all 10,000 within 8/255. Loads are 16.01%, 18.95%, 37.91%, and
27.13% (effective experts 3.782/4). All 200 concrete boundary witnesses replay,
the 100-input reference crosscheck differs by at most 1.23e-16, and the frozen
eta overcheck band contains 0/10,000 inputs. This is a scheduled trajectory
endpoint, not the completed single-seed or multi-seed result. Its compact
manifest is
`act/pipeline/moe/results/icml2025_rt_er/seed0_epoch010_route_telemetry_20260830.json`.

The epoch-20 endpoint raises clean accuracy from 30.81% to 33.95% and PGD-50
accuracy from 25.11% to 27.00%, but its route geometry is unchanged from epoch
10. This is not a rounded-summary coincidence: the two router tensors have
identical SHA-256 content at both checkpoints and zero changed tensors, whereas
all 248 expert parameter tensors changed. The Adam checkpoint contains state
for exactly those 248 expert parameters and no state for either router
parameter. Clean expert assignments and boundary competitors are exactly
equal; independently recomputed radii differ by at most `1.04e-17`.

The released code explains the observation: `Router.forward` returns hard
`argmax` indices, `get_second_expert` returns integer `topk` indices used as
masks, and neither path places router scores in a differentiable loss. Thus
this official-code reproduction trains experts while its randomly initialized
router remains fixed. The 130-epoch run continues as frozen, since it remains
the required official-code architecture reproduction and an expert-training
baseline, but its checkpoint trajectory cannot support a claim that this
training procedure pushes route boundaries outward. The compact endpoint is
`act/pipeline/moe/results/icml2025_rt_er/seed0_epoch020_route_telemetry_20260830.json`.

The static route geometry has now been materialized as three audited figures
and a 13-row table. Strictly below `2/255`, `4/255`, and `8/255`, the exact
route-boundary counts are respectively 9,117, 9,968, and 10,000 of 10,000.
Thus the route-invariance precondition fails somewhere inside the `8/255` ball
for every test input, although this says nothing by itself about output safety.
The expert loads are 16.01%, 18.95%, 37.91%, and 27.13% (effective expert count
3.782). The most common unordered nearest-boundary pairs are `0-2` (27.22%)
and `1-3` (21.81%). The compact audit manifest is
`act/pipeline/moe/results/icml2025_rt_er/seed0_static_route_geometry_20260830.json`;
the hashed SVG/CSV artifacts remain under
`data/moe/results/icml2025_rt_er/seed0_static_route_figures_20260830_r1`.

A pinned source audit also inspected all four official training entry points,
not only CIFAR-10 RT-ER. They share the same integer `argmax`/`topk` router and
none places router scores in a differentiable loss. The paper treats the router
as part of the optimized robust MoE and motivates Assumption 5.3 through
training-induced router-score separation, but does not specify a differentiable
hard-routing estimator. We classify the missing estimator as paper
underspecification and the absent stated optimization effect in the released
scripts as an artifact divergence. Dynamic optimizer/checkpoint confirmation
remains scoped to the current CIFAR-10 RT-ER seed-0 run. Static routing itself
is not treated as invalid. The audited evidence is
`act/pipeline/moe/results/icml2025_rt_er/released_training_router_gradient_audit_20260830.json`;
responsible disclosure has not been sent and requires explicit authorization.

To distinguish the seed-0 route census from a single random draw, a separate
20-seed initialization census is frozen. Each seed reconstructs the complete
official four-ResNet18 model before reading the router, preserving the official
random-number consumption order; a standalone Linear draw is not admitted.
Seed 0 must match the immutable epoch-10 router bit for bit and reproduce all
10,000 immutable route-boundary rows. The census reports strict and outward-safe
counts at `{0.5,1,2,4,8}/255`, per-seed loads and boundary pairs, and the full
minimum--maximum range across all preregistered seeds. It is an initialization
distribution result, not a trained-model certificate.

## ICML 2025 B3 executable comparison stage

The final-checkpoint B3 execution layer is implemented but not run. It freezes
the first 20 deterministic clean-correct test indices whose exact affine route
upper bracket remains within the 8/255 cap after multiplication by 1.05. For
each input it builds every tie-inclusive hard-top1 guard, decides candidate
feasibility, constructs a guarded coordinate hull with the incremental LP
backend, and serializes immutable boxes for a separate CROWN-environment worker.
The route-invariance baseline and Route A therefore use the same expert backend;
their applicability difference comes only from the former's route-stability
precondition.

The same runner instantiates Equation (8) under three provenance-preserving
families: sound global constants for a continuous-softmax/probability reading,
the released hard-argmax/raw-logit reading, and author-unspecified constants.
It assigns the preregistered five-leaf outcome per input and never silently
equates the continuous surrogate with the released hard-dispatch program. The
CROWN worker retains the existing numerical discipline: positive raw
auto_LiRPA margins are `CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE`, negative
bounds are `UNKNOWN`, and `UNSAFE` requires a concrete full-model replay. Thus
the code can collect conformance data now but cannot manufacture the missing
outward-rounded formal endpoint. The monolithic CPU comparison remains a
separate post-B1 experiment and is disabled in this runner.

Following the observed epoch-20 static-router result, B3 also contains an
audit-only provenance gate. It links optimizer slots to the final model's
named parameters and compares the final router bit-for-bit with the immutable
epoch-10 checkpoint. The frozen execution requires zero router optimizer state
and exact equality to that reference. This addition changes no cohort,
epsilon, solver, property, or numerical policy; it prevents the final B3 table
from silently describing the released checkpoint's fixed router as learned.

## Dynamic-MoE verifier-front-end probe

The parser-probe execution layer exports the
frozen bal010 weighted top-2 model and the immutable official RT-ER epoch-10
checkpoint with their real dynamic dispatch (`TopK` or `ArgMax` followed by
`Gather`) to ONNX opset 17. Four deterministic route-diverse CIFAR-10 inputs
must agree first with ONNX Runtime; the RT-ER tensorized export wrapper must
also agree with the author's grouped hard-dispatch forward before export.

The resulting graph is then passed through both the exact
`complete_verifier.load_model_onnx` loader and its ONNX conversion and bounded
graph components in the frozen alpha-beta-CROWN checkout. Outcomes
distinguish export rejection, reference-ONNX rejection, verifier-front-end
rejection, silent semantic mismatch, and semantic-preserving acceptance. This
experiment measures only whether an existing verifier front end can consume
the unmodified dynamic program. It does not measure certification coverage or
runtime, and acceptance is not a robustness certificate.

The audited r2 run completed with zero issues. Both models exported to valid
ONNX and matched PyTorch on four route-diverse inputs: maximum absolute errors
were `2.38e-7` for bal010 and `4.77e-7` for RT-ER. The RT-ER tensorized wrapper
also matched the author's grouped forward (`1.34e-6` maximum output error,
exact routes, and zero router-score error). Nevertheless, both the isolated
conversion component and the exact `complete_verifier.load_model_onnx` entry
point rejected both programs with `NotImplementedError: Conversion not
implemented for op_type=GatherElements.` The VNNLIB and auto_LiRPA stages were
therefore not reached. This establishes a program-consumption gap for these
two frozen dynamic-dispatch encodings; it does not claim that alpha-beta-CROWN
cannot verify the individual experts after Route A specializes dispatch.

The first component-only run and a single-model full-loader implementation
smoke remain preserved in the raw result tree but are explicitly excluded.
The compact audited result is
`act/pipeline/moe/results/crown/dynamic_moe_parser_probe_20260830.json`.

The complementary specialization probe is frozen separately. It fixes each of
the four official hard routes, exports the resulting raw-pixel ResNet18 branch,
requires zero `TopK`, `ArgMax`, `Gather`, or `GatherElements` operators, and
passes the graph through the same ONNX conversion, official model loader, and
auto_LiRPA `BoundedModule` stages. Its only intended conclusion is a
program-consumption closure: dynamic MoE rejected, Route A specialized expert
accepted. Acceptance is not a robustness certificate or a coverage/runtime
claim. CROWN child processes are restricted to one CPU thread while B1 runs.

The first specialization execution is retained as an excluded fail-closed
engineering result. Its initial syntactic rule treated every ONNX `Gather` as
expert dispatch, but the ResNet18 export uses one `Shape -> Gather` with a
constant index solely to recover a tensor dimension. The r2 audit therefore
forbids data-dependent `TopK`, `ArgMax`, `Gather`, and `GatherElements`, while
explicitly classifying only constant-index `Shape -> Gather` as non-dispatch
bookkeeping. Unit tests require a feature-tensor `Gather` to remain forbidden.

The audited r2 execution closes the program-consumption loop. The complete
dynamic RT-ER graph remains rejected at `GatherElements`; after fixing each
route, all four specialized ResNet18 graphs contain zero data-dependent
dispatch nodes, match PyTorch under ONNX Runtime, and are accepted by both the
same official CROWN loader and auto_LiRPA `BoundedModule`. Each graph retains
one constant-index `Shape -> Gather` dimension lookup, which the independent
audit confirms is not data dispatch. All 4/4 branches pass and the audit reports
zero issues. This is front-end composability evidence only, not a robustness
certificate. The compact result is
`act/pipeline/moe/results/crown/rt_er_specialization_probe_20260830.json`.

## N1 engineering rerun code freeze

The N1 performance runner freezes the original 20-row applicable-unresolved
Experiment 1D selection, parent row hashes, radii, route sets, guarded-support
policy, numerical SAFE policy, and 900-second per-row deadline. It compares the
already completed unsegmented D0 result against one natural affine-margin cut
at zero. Every row is run; no threshold-triggered early stopping or selective
rerun is allowed. Existing solved parent property rows are reused exactly, and
only unresolved property rows receive N1.

This is explicitly an `engineering_performance_rerun_not_confirmatory_overwrite`.
The original confirmatory 56/100 endpoint is immutable. The output reports
paired status transitions, solver metadata, support telemetry, segment counts,
runtime differences, and full-forward witness validation. F1 remains disabled:
the config fixes `sigmoid_segments=0` and `gate_function_encoded=false`.

A non-result runner integration smoke on frozen rank 110 traversed candidate
reconstruction, guarded shared-input pair propagation, margin segmentation,
and two property solves before the deliberately shorter 180-second outer smoke
limit stopped the third property. Its progress artifact reports an explicit
weighted solver limit. The incomplete smoke directory is retained and excluded
from all summaries; only the 900-second child-process run below can produce the
paired engineering result.

The complete 20-row run finished without semantic conflicts. The unsegmented
D0 baseline solved 12/20 rows (60%); N1 solved 14/20 (70%). The paired table is
`n11=12, n10=0, n01=2, n00=6`, so the net gain is two rows and the exact
two-sided paired binomial/McNemar p-value is 0.5. N1 added one SAFE result
(rank 125) and one full-forward-validated UNSAFE result (rank 120). All three
UNSAFE rows in the completed artifact replay successfully through the frozen
weighted top-2 model. The original confirmatory endpoint remains 56/100 and is
not overwritten by this engineering comparison.

Retained-margin support was evaluated for 157 properties and 168 active
segments. Eleven properties had at least one strict expert-difference support
tightening. The median paired runtime difference was +15.91 seconds and the
median ratio was 1.212, so this run supports a small coverage gain but not a
runtime-speedup claim. The initial independent audit file is retained as a
failed audit artifact: it emitted false positives because it did not inherit a
pair-level `reused_parent` marker and compared two parent rows that had timed
out before recording F0 pairs against an empty set. Audit schema v2 fixes those
checks and independently rematerializes exact feasible top-2 sets for every
non-timeout row. It reports zero issues, 20 unique ranks, no semantic-source
drift, and 3/3 UNSAFE witness replays. The immutable publishable summary is
`act/pipeline/moe/results/experiment1n1_engineering_20260829.json`.

## Explicit route-invariance baseline execution

P0a executes the missing end-to-end route-invariance baseline on the frozen
confirmatory ranks 100--199. It does not infer property safety from router
status alone. Exact, tie-inclusive feasibility must establish one unique
unordered top-2 set before the baseline may invoke the property verifier. A
second legal set returns `UNKNOWN/ROUTE_INVARIANCE_PRECONDITION_FAILED` without
an expert solve.

The comparison isolates that precondition: after it passes, the baseline and
Route A use the same guarded HZ propagation, support policy, expert-wise gate
elimination, F0 fallback, numerical SAFE policy, and full-model witness replay.
Thus the baseline is not weakened by a different expert backend. For the 76
samples with a registered route-boundary endpoint, the runner consumes the
hash-locked confirmatory endpoint and exact route sets. For the 24 samples whose
exact search found no route change through `4/255`, the endpoint is the
registered `4/255` cap and the downstream solve is executed afresh. The two
original hard-deadline rows obtain only their exact pair/radius identity from
the hash-locked Experiment 1D artifact; their original confirmatory TIMEOUT
verdict remains the primary Route A entry, while closure is a separate field.

Runtime is labeled artifact-backed endpoint accounting. Route-unstable baseline
rows stop after the frozen exact candidate stage; invariant rows share one fresh
downstream execution with Route A. This comparison may establish coverage cost
and Route A-only certificates, but it is not a same-process solver speedup
benchmark. The original confirmatory `56/100` endpoint remains immutable.

P0a completed all 100 ranks with an independent zero-issue audit. The exact
route-invariance precondition held for 24 samples and failed for 76. The
explicit baseline solved 12/100 (2 SAFE, 10 full-model UNSAFE), while staged
Route A solved 68/100 on the same per-sample endpoints (38 SAFE, 30 full-model
UNSAFE). The paired coverage gain is 56 samples. All 36 route-changing SAFE
certificates are Route A-only, so the previously reported unique-certificate
count is now backed by an executed end-to-end baseline rather than only a
router-precondition label.

The baseline accounts for 1,956.5 seconds and Route A for 8,836.5 seconds
(4.516x). This is the expected coverage-cost tradeoff: route invariance is
cheaper precisely because it abandons all 76 route-unstable endpoints. It is
not a Route A runtime-speedup result. The separate Experiment 1D closure would
raise Route A solved coverage to 80/100, but it remains a follow-up field and
does not overwrite either this P0a primary comparison or the immutable original
confirmatory `56/100`. The independent audit replays all 30 primary UNSAFE
witnesses through the complete selected-softmax model and reports zero issues.

## Incremental expert HZ backend code freeze

The expert support/property path now has an opt-in `highspy` session that
lowers one sparse-HZ branch once, changes column objectives incrementally, and
reuses a scratch-row pool for property queries. It preserves the existing
support/minimum/property return types and three-valued verdict policy. Only an
exact output HZ that shares the input frame may return a falsified witness; a
relaxed witness remains `UNKNOWN`. Any HiGHS warning or partial update
invalidates the session, after which support falls back to sound fast bounds
and properties fail closed to `UNKNOWN`.

The implementation sets `small_matrix_value=1e-12` and rejects smaller nonzero
base or scratch-row coefficients before HiGHS can silently drop them. Telemetry
separates model builds, objective and row updates, solve counts, accepted basis
submissions, iterations, nodes, and time. An accepted basis submission is not
reported as proof of solver-internal warm-start use. Synthetic LP/MILP,
support, minimum, and property tests compare against the existing SciPy path.

The opt-in backend is now connected to all three repeated-solve sites used by
the frozen closure: guarded preactivation support, expert property solving, and
the final F0 augmented property HZ. Defaults remain `scipy`; a run must request
`highspy_incremental` separately for support, expert properties, and F0. The
same branch model is retained across low-budget and escalation attempts through
an explicit cumulative budget extension. Different F0 properties still produce
different McCormick-augmented HZs, so they deliberately receive independent
sessions; telemetry records `cross_augmented_hz_reuse=false`.

`experiment1_highspy_engineering.py` freezes all 20 applicable-unresolved D0
rows, their original radii and parent hashes, and the 900-second deadline in a
new engineering-only result directory. It aggregates model builds, objective
and scratch-row updates, solve calls, basis submissions, iterations, nodes,
budget extensions, and paired status/runtime transitions. The immutable
confirmatory endpoints remain `56/100` overall and `56/76` applicable. This is
still a code/mechanism result until the 20-row engineering rerun and independent
audit complete.

The first full launch (`_r1`) was stopped after rank 110 failed the D0
support-identity guard. This was an execution-scope mismatch: D0 correctly
requires rematerialized SciPy support to reproduce its parent structure, while
the engineering treatment intentionally substitutes incremental HiGHS for the
time-limited support solver and may therefore soundly eliminate a different
binary subset. The fix does not relax D0. Only the explicit engineering config
may record structural support-signature drift; candidate sets, exact route sets,
radii, parent hashes, support budgets, and numerical SAFE rules remain frozen.
The `_r1` row is excluded and retained in
`act/pipeline/moe/results/experiment1_highspy_engineering_r1_failed_20260829.json`;
the clean `_r2` run starts all 20 rows again in a new directory.

The `_r2` launch then exposed a distinct control-status bug on the first row:
HiGHS returns `kWarning` from `run()` when an explicitly configured time limit
is reached, while the model status is `kTimeLimit`. Treating that expected pair
as a partial-update warning invalidated the session before escalation, making
the intended budget extension impossible. `_r2` was stopped and excluded. The
backend now accepts only this exact run/model-status pair and counts it as
`run_time_limit_warnings_accepted`; all update, matrix, basis, and other run
warnings still invalidate the session. The record is
`act/pipeline/moe/results/experiment1_highspy_engineering_r2_failed_20260829.json`.
The `_r3` run again starts all 20 rows from scratch.

The `_r3` launch completed four rows and finished the rank-126 computation, but
the `/data1` filesystem reached zero available bytes during the atomic row JSON
flush. The child raised `ENOSPC`, and the parent correctly failed closed. This
is an external storage failure, not a solver verdict. The partial rows are
excluded and retained in
`act/pipeline/moe/results/experiment1_highspy_engineering_r3_failed_20260830.json`.
Only reproducible package/download caches under `/data1/Kane/MOE/cache` were
cleared; all environments were then import-checked, and checkpoints, raw
results, and user files were left intact. The `_r4` launch starts all 20 frozen
rows from scratch in a new directory with identical scientific and engineering
configuration.

The clean `_r4` launch completed all 20 rows at implementation HEAD
`2e344b43bb8042b764712eddbad68da0e440811d`. It returned 11 SAFE, two
full-model-replayed UNSAFE, five UNKNOWN, and two TIMEOUT. Relative to the
frozen D0 result, the paired transitions were `SAFE->SAFE=10`,
`UNSAFE->UNSAFE=2`, `UNKNOWN->SAFE=1`, `UNKNOWN->UNKNOWN=5`,
`UNKNOWN->TIMEOUT=1`, and `TIMEOUT->TIMEOUT=1`. Thus solved rows increased only
from 12/20 to 13/20, with no solved regression. This is a separately labeled
engineering result and does not rewrite either the confirmatory 56/100 or D0
68/76 endpoint.

End-to-end runtime did not improve. D0 took 5,151.10 seconds and incremental
HiGHS took 5,674.07 seconds, a ratio of 1.102. The paired median difference was
+9.88 seconds; 6/20 rows were faster and 14/20 were slower. The 15.03x guarded
coordinate-hull construction result therefore does not transfer to the full
expert/F0 pipeline. Telemetry contains 292 sessions and model builds, 3,402
solves, 12,348 row updates, and 47 budget extensions. All 75 accepted
`run()==kWarning` events correspond exactly to 75 `kTimeLimit` model statuses;
there were zero build failures and no warm-start claim.

The original Experiment 1D audit reports 132 issues because it only recognizes
SciPy's status-0/bound-kind fields. That audit remains preserved. A dedicated
incremental-backend audit independently checks HiGHS `kOptimal`, finite dual
bounds, outward-corrected positive minima, complete pair/property coverage,
session warning accounting, and concrete witness replay. Its final r3 audit
has zero issues and replays 2/2 UNSAFE witnesses. Two earlier audit reports are
also retained: r1 had 297 false positives because it failed to propagate
pair-level `reused_parent`, and r2 passed soundness but mislabeled an inherited
D0 threshold as an incremental-vs-D0 preregistration. The immutable tracked
summary is
`act/pipeline/moe/results/experiment1_highspy_incremental_engineering_r4_20260830.json`.

The engineering decision is to retain incremental HiGHS as an opt-in backend
and a negative end-to-end performance result. It is not made the default, and
no end-to-end speedup claim is permitted.

## P0b CROWN adapter consistency code freeze

The frozen 43-branch bal010 cohort compares four representations of the same
expert-wise convex obligation: exact HZ with the unordered pair guard retained,
the guarded coordinate hull passed to CROWN, the original perturbation box
passed to CROWN, and a tie-safe eta implication passed to CROWN. Both member
experts must prove every clean-prediction margin row before a pair is counted as
certified. The eta graph uses
`g_S = max(outside scores) - min(inside scores)` and
`max(g_S - eta, minimum property margin)`, with weak tie-inclusive route
semantics and `eta=1e-7`.

Negative HZ or CROWN bounds never become an `UNSAFE` verdict in this cohort.
CROWN uses a positive-margin filter and is explicitly labeled as not
outward-rounded, so P0b is adapter-consistency evidence rather than a main
formal-certificate table. Every exact-HZ refutation is checked against any
CROWN positive-margin result, and the flushed JSONL is independently re-read
against the frozen source-line and guarded-hull hashes.

The CROWN environment uses Python 3.11. Its `typing.override` compatibility
shim must execute before importing the ACT package, so the runner and tests are
launched as repository file paths with `PYTHONPATH=/data1/Kane/MOE/ACT`, not via
`python -m act.pipeline...`. A real bal010 smoke exposed a crash in dynamic
score slicing; fixed one-hot `nn.Linear` projectors now lower successfully in
auto_LiRPA.

The first full `_r1` execution is retained as an excluded engineering artifact.
It exposed a second, non-semantic censoring path: the runner repeated an exact
route-set query with a 10-second limit even though every source pair had already
been frozen, hash-linked, and independently audited. A repeated query timeout
therefore rejected a valid frozen branch without finding a discrepancy. The
`_r2` code validates the source pair identities against the frozen parent exact
pair list, rebuilds the shared model/router HZ frame, and applies each pair guard
directly. It never treats a repeat feasibility query as an execution gate. The
failed `_r1` directory remains immutable and excluded; `_r2` writes a new result
directory. This section records the corrected code freeze until all 43 branches
and the independent audit complete.

The completed failure record is
`act/pipeline/moe/results/crown/crown_adapter_consistency_bal010_43_r1_failed_20260829.json`.
The launcher exited normally after flushing all 43 rows, but four rows contain
the repeated-query error and the independent audit reports 12 structural issues
(three expected missing-result fields per error row). There were zero soundness
cross-check issues and no `UNSAFE` statuses. The 39 otherwise valid rows are
still excluded from scientific aggregation so `_r2` remains a clean full-cohort
replication, not a four-row patch.

The completed `_r2` artifact will also be read by a separate audit module rather
than relying only on the runner's in-process summary. That pass freezes the raw
JSONL hash, revalidates the 43 source identities and tie/UNSAFE rules, and
reports expert-level status counts plus paired guarded-box versus original-box
CROWN lower-bound differences. Bound ordering is observed rather than assumed;
the audit does not convert a CROWN negative relaxation into `UNSAFE` or claim a
runtime speedup.

The clean `_r2` run completed all 43 branches and 86 expert obligations with
zero branch errors, zero soundness cross-check issues, and two independent
zero-issue audits. Guarded-box and original-box CROWN each certified the same
four pairs and the same 29 expert obligations. Across 774 property rows their
lower-bound difference had median `-2.38e-7` and range
`[-4.77e-6, 2.62e-6]`; because these CROWN values are not outward-rounded,
the micro-scale signed differences are numerical observations rather than a
tightening claim. The coordinate hull therefore preserved no pair-certificate
advantage over the original box in this cohort.

The tie-safe eta reduction certified 0/43 pairs. Retained-guard exact HZ also
certified 0/43 pairs at the frozen budget: 71/86 expert obligations were
incomplete, 12 completed with a non-positive exact lower bound, and three were
certified. Thus P0b does not establish a complete backend ordering. Its useful
negative result is that coordinate boxing erased the guard's certificate value,
while the sound eta compilation was too relaxed under CROWN. The immutable
summary is
`act/pipeline/moe/results/crown/crown_adapter_consistency_bal010_43_r2_20260829.json`.
