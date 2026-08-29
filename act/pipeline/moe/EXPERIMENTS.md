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
