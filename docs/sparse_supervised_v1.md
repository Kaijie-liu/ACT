# Large sparse basis: separate unified-budget supervision

This stage binds the frozen `sparse_basis/` component to **new**
`sparse_supervised/` entry points. Starting branch was
`feat/moe-route-verification`, HEAD `f476de5042d103f80be92ea6bf41a6b506f9d5ca`,
clean and synchronized. No old component, limit, result or acceptance threshold
is replaced. The real diagnostic is frozen separately, **not launched**.

## Execution contract

The supplied-LP clock starts before input reading, hashing and decoding. Each
phase runs in an owned subprocess with the same original monotonic start.

| Phase | Absolute offset from original start |
| --- | ---: |
| Load and validate unchanged LP/source/property |218 seconds|
| Native capture, including imports/model submission/readback |218 seconds|
| Map captured basis to original coordinates |218 seconds|
| Sparse exact candidate construction and serialization |218 seconds|
| Copy standalone checker and candidate bundle |298 seconds|
| `python -I -S` full original-LP check |298 seconds|
| Terminal review, inventory and publication |300 seconds|

These are shared deadlines, not fresh per-phase grants. The native call remains
one call, at most10 seconds, with the unchanged highspy1.14.0 options. Sparse
size/fill/heap/bit/operation limits are copied into the plan and independently
checked against the frozen component. No alternate basis, retry or repaired
historical point is attempted. Controlled component exhaustion returns `LIMIT`,
not LP infeasibility. Unexpected errors stop the ordered roster; later jobs
remain explicit `NOT_RUN_AFTER_ERROR` denominators. Timeouts/unresolved results
do not delete jobs. Output directories and evidence files cannot be overwritten.

The owned watchdog terminates only its process tree. Cleanup, hashing,
serialization and publication remain charged; late publication revokes a
positive acceptance. This is not a hard-real-time/reboot-recovery guarantee.
Uninterruptible cleanup can overrun; that overrun must stay visible as TIMEOUT.

## Partial evidence and cost

`native/raw_native.json` is saved immediately after native return. A subsequent
readback error or cutoff may leave this file without `capture.json`. The new
cost reader checks its LP/statement identities, original input and submission,
options and duration against the recorded capture window. A valid raw return
therefore retains **one recorded native call and its duration**, even though no
mapping or proof may proceed. If the raw return is missing, malformed or has an
invalid binding, native count/time remain null, never inferred as zero or one.
This is provenance/cost checking, not certification of the native point.

Complete and raw capture records must agree. Construction schema, component
policy, hint, statement and original deadline are bound. The independent checker
receives the unchanged original LP; every E/A/box constraint is still checked.
A basic E residual never waives an equality. Completing the checker can validly
produce `NOT_EXACTLY_FEASIBLE`, upper=null; `CHECKED_LP_DIAGNOSTIC` is **not** a
synonym for a successful exact-feasibility certificate.

Cost identity:

```
whole supplied-LP publication clock
  = disjoint recorded phase windows + residual overhead
```

Native and exact-construction component durations are nested within these
windows, not added again. Missing phase exits are right-censored; an observed
interruption window is reported separately from a missing completed duration.
Resource waiting, launch preflight and fresh post-terminal audits are separate
batch overhead. Historical network propagation, range proofs and F0 construction
are **not rerun or claimed free end-to-end work**: this experiment begins with a
supplied archived LP. No MoE speedup claim follows from these analytic times.

## Controls and review

Numbered control receipts preserve each attempt. Attempt001 passes120/120;
[attempt002](sparse_supervised_controls_attempt002.json) passes **122/122**
(27 new supervisor controls and95 existing regressions). The second version
adds the review/freeze gate and plan/roster controls.
[Fresh review](sparse_supervised_v1_review.json) checks854 retained artifact
files,15 terminal/cost records and4 moved isolated packages: **PASS,0 issues**.
The complete large analytic path took2.6143s including publication in this
control run; this is not a real-network or comparative speed measurement.
Controls cover:

- complete capture→map→construct→pack→isolated-check execution;
- native, exact-construction and checker cutoffs; outer cutoff and late publish;
- readback cutoff/exception after a valid raw native return;
- missing/mutated artifacts, raw/complete disagreement, original LP/statement,
  source/property, policy, schema, deadline and mapping identity mutations;
- explicit LIMIT, no overwrite, error-stop roster and missing-cost accounting;
- redundant E rows retained, and float-collapsed unequal rational RHS rejected;
- a structured4,096-variable/8,192-E-row analytic native system through the
  full supervisor, with checked U=−4096/3, plus component regressions.

The large control is singleton-rich synthetic geometry, **not a real-network
performance forecast**. The fresh review rebuilds terminal/cost summaries,
checks saved artifact hashes, moves complete packages and rechecks them using
`-I -S` without model, dataset, solver or original-directory imports. It does
not run another native solve or reconstruct another basis candidate.

## Separately frozen real diagnostic

`docs/sparse_supervised_real_v1_freeze.json` binds four **unchanged** source LPs:

| Input/property | Pair | Variables | E+A rows | Stored nnz |
| --- | --- | ---: | ---: | ---: |
|220/p0|{1,2}|7397|4331|246558|
|222/p1|{0,1}|7682|4616|236502|
|230/p2|{0,3}|9482|6416|348915|
|232/p0|{0,1}|9095|6029|329664|

These are the same first nonpositive obligations selected in the sealed old
diagnostic, not newly selected favorable rows. Original bytes, statements and
ordering are reconstructed from that archive. Static dimensions fit the **new**
component; real native basis validity, mapping, rank, fill, bit growth and runtime
remain unmeasured. Old small-interface incompatibility findings remain correct.

New output: `data/moe/results/sparse_supervised_real_20260919_v1`.
The [freeze](sparse_supervised_real_v1_freeze.json) is **FROZEN_NOT_EXECUTED**.
Its [selection review](sparse_supervised_real_v1_selection_review.json) separately
reconstructs the four original identities and static compatibility. The output
directory has not been created; real native calls and reconstructions are zero.

No checkpoint, new property, extra sample or changed LP is introduced. The old
four inexact-primal results remain sealed. Prior checked lower bounds are frozen
context only; this new candidate bundle proposes no dual and makes no exact-gap
or optimality claim. A checked feasible U≤0 limits the given LP relaxation,
not the original network. U>0 alone does not prove LP safety. Inexact or missing
points leave the cause unresolved. Network→HZ, guard/route exclusion and F0
lowering remain trusted upstream; this is not a complete MoE request proof.

Freeze requires passing current controls and a fresh control review. A separate
process reconstructs the source selection and compatibility before launch is
permitted. Launch additionally requires a clean, pushed feature branch, resource
gate and exclusive new directory. Freeze does not launch:

```
/data1/Kane/miniconda3/envs/act-py312/bin/python -m sparse_supervised.study freeze --controls <final-receipt>
/data1/Kane/miniconda3/envs/act-py312/bin/python -m sparse_supervised.study reconstruct
```

Only a subsequent execution decision should invoke `study launch`. No real
native call or exact reconstruction occurred during this implementation/freeze.
