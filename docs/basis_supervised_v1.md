# Single-budget native-basis evidence supervision (analytic scope)

## Result and scope

New namespace `basis_supervised/` connects the existing restricted native basis
adapter to original-coordinate rational reconstruction, portable packaging, and
the unchanged standalone original-LP checker. No earlier module, frozen method,
four real diagnostic results, holdout or numerical acceptance gate was changed.

[Attempt001](basis_supervised_controls_attempt001.json) passed71/71 controls.
The strengthened [attempt002](basis_supervised_controls_attempt002.json) passed
**73/73**, comprising19 new supervision controls plus54 prior regressions.
Both receipts and local test artifacts are retained. The second attempt adds
full-outer fault injection and nested-component cost validation, not a changed
mathematical acceptance threshold.

A [fresh-process review](basis_supervised_v1_review.json) reconstructed10
terminal/cost records, compared the five main observations to the receipt, and
rechecked the relocated analytic package under `python -I -S`: PASS,0 issues.
No solver was called by that review. It is a structural audit plus one exact
analytic LP recheck, not an independent proof of any real network.

The new supervision controls perform **4 completed native analytic captures**:
normal, unsupported mapping, before synthetic reconstruction cutoff, and before
synthetic mapping exception. Prior native-adapter regressions additionally have
6 captures; other earlier LP component tests remain regression tests. Synthetic
native stalls replace `Highs.run` rather than execute a long optimization. Their
recorded native call count is **unknown/null** when capture is incomplete, even
though the fault harness knows no actual solve took place. There are **0 real
network LP calls/reconstructions**. No real experiment is frozen or launched.

## One original clock

`supervise(spec, new_directory, started=original_monotonic_start)` consumes a
caller-owned clock; it cannot allocate a fresh300 seconds to each phase. All
worker processes inherit this clock. `started` must be finite and not in the
future. The spec binds an existing serialized `{lp, statement}` by file SHA256,
the exact statement and its canonical hash. Parsing/validation happens inside
the load phase. Input tensors, network lowering and earlier LP creation are not
repeated or claimed as part of this supplied-LP diagnostic.

| Phase / boundary | Absolute deadline relative to original start |
| --- | ---: |
| Load/validate/import; native capture; mapping; exact construction |218s|
| Portable package; isolated original-LP check |298s|
| Outer terminal publication/admission |300s|

The native adapter still receives **at most10 seconds**, one call, simplex,
presolveOFF, scaling0, threads1, parallelOFF, with existing highspy1.14.0.
Nothing installs or upgrades dependencies. Reserving80 seconds for packaging
and checking does not promise those stages will finish. It also does not waive
their time cost. The driver and each phase use the existing owned-process-tree
wait/cleanup implementation; unrelated processes are never signaled. Worker
imports, serialization, cross-process startup and source hashing are charged.

Watchdog cleanup may finish after the deadline. The observed overrun remains in
cost records; it cannot produce an accepted late check. Driver work ends at298,
and publication observed at/after300, or an explicit publication-timeout marker,
revokes acceptance. The parent also inventories surviving partial files after
owned cleanup. This is a process-level watchdog, not a claim of hard-real-time
termination or power-loss/reboot recovery if the supervisor itself is killed.

## Immutable stages and terminal rules

The stages are `load → capture → map → construct → package → check`.
Plan, phase-entry, phase-exit, candidate, outer and publication records use
no-overwrite creation; reusing an output directory raises an error. There is
no retry/resume or automatic change of basis. Original native input/submission
records precede the solve; raw capture precedes all mapping and checking.

Successful mapping is only `MAPPED_HINT_ONLY`. `UNSUPPORTED_MAPPING` stops
before construction with raw capture intact. Rank/sparsity/arithmetic limits
remain `UNRESOLVED_SINGULAR_BASIS` or `LIMIT`, not LP infeasibility. Construction
only emits `CANDIDATE_ONLY`; its `feasibility_certified` flag remains false.
The original LP and statement must remain byte/identity consistent through
packing, and the copied checker must match the bound source hash. The sole
checking subprocess uses `-I -S`, no solver/model imports, all original LP
constraints, exact rational objective and zero feasibility tolerance.

`CHECKED_LP_DIAGNOSTIC` means the independent check completed. Its separate
classification and `primal_status` decide whether a feasible upper bound was
actually checked. It is not synonymous with optimality, a positive lower bound,
complete MoE SAFE, or network UNSAFE. Missing/invalid primal evidence cannot be
promoted through native success status. In particular, a feasible nonpositive
point of an outer LP is not a concrete full-model counterexample.

The original real diagnostics and their failed exact-feasibility outcomes
remain sealed. Network→HZ, guards, route exclusions and F0 lowering remain
trusted upstream; this release does not check or replace them.

## Cost and evidence contract

`audit(directory)` rechecks phase order, original deadlines, immutable file
inventories, capture/mapping/LP/statement bindings, nested timing constraints,
and terminal/checker flags. It does not re-solve the LP. The independent checker
is what checks rational feasibility, not the JSON audit. Source hashes bind
new orchestration, existing components and selected runtime helpers; this is
not a machine proof of the interpreter or native binary implementation.

`costs(directory)` reconstructs:

```
whole supplied-LP publication clock
  = sum(disjoint recorded phase windows) + residual clock
```

Native solve time is nested in `capture`, rational construction time in
`construct`; neither is added again. Parent admission/audit, startup gaps,
inventory hashing and publication fall in residual time. Missing component
records yield **null**, never a guessed0. Interrupted phases without a complete
exit record retain a censored observed window separately; that window is not
double-added to the full clock. Malformed partial JSON is tolerated only for
ERROR/TIMEOUT cost forensics, never to accept evidence. All partial bytes remain
hash-bound in the terminal inventory. Explicit failures after a completed
capture can retain its real nested cost.

Post-terminal independent reviews and resource waits are outside this request
clock and must be separately recorded by a future experimental launcher. There
is no full-MoE runtime or speedup claim from this analytic release. Backdated
clocks in cutoff tests simulate already consumed budgets and are not empirical
algorithm timing measurements.

`batch.loop`/`summarize` provide ordered terminal accounting without selecting a
real cohort: ERROR stops subsequent work, preserving `NOT_RUN_AFTER_ERROR`
rows; timeout/unsupported/checked-unresolved outcomes continue. Missing roots
cannot stand in for completed requests. Pre-launch ERROR or unrun rows have
unknown cost, not zero. Denominator and counted/uncosted costs remain explicit.

## Controls and reproducibility

Run controls with the existing interpreter:

```
/data1/Kane/miniconda3/envs/act-py312/bin/python -m basis_supervised.controls
```

Each invocation uses a new numbered receipt and retained local artifact root.
It verifies earlier source freezes both before and after the suite. The main
analytic LP is x+y=1,3x<=1,x,y∈[0,1], objective−x−1. The complete supervised
path reconstructs(1/3,2/3); its isolated checker returns U=−4/3. A moved package
containing just `verify.py` and `bundle.json` returns the same conclusion without
checkpoint, training data, historic run directories or solver imports.

Controls additionally cover actual unsupported redundant-E mapping; inherited
proposal cutoff; expired/future clocks; owned driver cutoff; injected stalls at
the native-call and exact-elimination boundaries; partial checker output;
post-capture OSError; whole-supervisor fault terminals and costs; hash deletion,
missing check output, re-signed wrong mappings/LPs/checker flags/clocks; late
publication; component time overclaims; partial JSON censoring; no-overwrite;
and mixed terminal rosters. `fault_worker.py` is an explicit analytic harness:
the production driver never dispatches to it and exposes no fault switch.

For fresh read-only terminal/cost review, the receipt's
`control_artifact_root` contains `good`, `unsupported`, `error`, `expired`,
`reserve`, `outer_cutoff`, `full_native_stall`, `full_construct_stall`,
`full_map_exception`, and `batch_timeout`. Run `audit` and `costs` on those ten
roots; compare the five recorded main observations with receipt `observations`.
Mutation fixtures are separate directories and are intentionally invalid.
The relocated `moved/verify.py` can be rerun with the bound bundle and statement
hashes in `good/packing.json`. No source solver execution is necessary.

## Next decision, not an authorized real execution

This closes the requested **small supported-interface supervision controls**.
It does not lift64-variable/64-equation, sparse fill-in, rational-bit or operation
caps; basic/free equality-row mappings remain unsupported. Consequently, this
is not yet a drop-in reconstruction mode for the large frozen network LPs.
Any real diagnostic needs a separate compatibility/selection/freeze decision
within the unchanged caps, or separately designed sparse/mapping support and
controls. Do not silently relax caps, repair/relabel old points, alter ranges,
increase time, or rerun the sealed four diagnostics.
