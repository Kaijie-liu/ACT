# Primitive arithmetic: real diagnostic preparation V1

## Status: frozen, reviewed, NOT executed

The instruction after integration was to complete preparation for a real rerun.
No new research guidance is needed for this bounded step: the four original LPs,
fixed arithmetic contract and independent original-LP checker already define it.
This stage adds a **separate batch protocol**, not another arithmetic change.
There are zero real LP solves/reconstructions in this preparation. The new real
output directory does not exist at selection review. Do not interpret this
document as a result or an automatic launch instruction.

Evidence:

- `primitive_diagnostic_controls_attempt002.json`: 15/15 tests PASS, including
  11 new batch/identity/cost/archive controls and four archive regressions.
- `primitive_diagnostic_v1_controls_review.json`: fresh-process PASS, zero
  issues; 2,086 artifacts, three complete batch ledgers / 12 denominator rows,
  four relocated `python -I -S` checks with identical mathematical outcomes.
- `primitive_diagnostic_v1_freeze.json`: source/runtime/input-bound protocol.
- `primitive_diagnostic_v1_selection_review.json`: independent ordered selection
  and compatibility reconstruction PASS, zero native solves/reconstructions.
- Prior integration remains the unchanged 101-test receipt and five moved-check
  review in `primitive_supervised_v1.md`; this stage does not overwrite them.

Attempt001 is retained as FAIL. An analytic test incorrectly expected two
nonpositive objectives after setting offsets to 0,1,2,3. Its exact values are
`-1/3,2/3,5/3,8/3`, hence one, not two. The correction changes only that test
expectation and additionally checks each rational objective explicitly. It does
not modify inputs, feasibility acceptance, a threshold or real-run outcomes.

## Fixed question and obligations

The question is whether primitive-row integer reconstruction completes and
produces an independently checked feasible point for any of the **same four**
saved original LPs that stopped under native-fidelity V2. It is not a new cohort,
a complete MoE request, a proof of network unsafety, or a timing competition.

| Frozen job | Variables | Equality rows | Inequality rows | Stored matrix entries |
| --- | ---: | ---: | ---: | ---: |
| input220_p0 | 7,397 | 1,441 | 2,890 | 246,558 |
| input222_p1 | 7,682 | 1,536 | 3,080 | 236,502 |
| input230_p2 | 9,482 | 2,136 | 4,280 | 348,915 |
| input232_p0 | 9,095 | 2,007 | 4,022 | 329,664 |

The complete ordered job dictionaries match V2 and original selection, including
export bytes, LP identity, statement, request, pair, property and original lower
bound context. Source decoding and native import admission pass read-only
preflight. This does **not** predict new fill, LCM growth, feasibility or time.

Keep 300 seconds per supplied-LP request, 218-second construction cutoff,
298-second check/work cutoff, one native call <=10 seconds and one basis attempt.
Keep the 4096-bit cap and all structural/operation caps, original rows and
coefficients, native fidelity options and exact original-LP feasibility tests.
No new dual is proposed; prior lower bounds remain explicitly labelled context.
No resume, retry, alternate basis, fallback algorithm, extra sample or extra time.

New directory:

```
data/moe/results/primitive_diagnostic_real_20260920_v1
```

The native proposer runs once again in the **new** execution; no old hint is
silently reused for free. The archive compares old/new basis coordinate structure
and assembled-system hashes. If either is different or unobserved, it forbids
pure arithmetic attribution. Even when matched, timings are non-interleaved,
descriptive observations, not a causal speed comparison.

## Execution and audit contract

`primitive_diagnostic/` binds and reuses the unchanged `primitive_supervised/`
implementation. It does not add files inside the frozen component namespace.

Launch requires the ACT interpreter, clean feature branch, remotely synchronized
HEAD, passing controls and selection review, exact frozen sources/runtime, and
an absent new output directory. The existing ACT batch writer lock is held
through execution and final audit; a second writer fails instead of taking over.
Request subprocess ownership and all cutoff behavior remain unchanged.

Resource checks retain minimum16 GiB available RAM, minimum5 GiB free disk,
maximum0.5 load/core, one worker/thread and no GPU. Resource polling has a bounded
24-hour wait and is outside each request clock, explicitly charged separately.
The preparation observation was about90.5 GiB available RAM,784.3 GiB free disk
and0.228 load/core; it is **not** a reservation or permission to interrupt jobs.
The gate is checked afresh before each future request.

Each attempted job records resource events, invocation start, outer terminal,
post-terminal audit time and total attempt time. ERROR stops subsequent jobs,
which receive explicit NOT_RUN_AFTER_ERROR rows with null evidence/costs. LIMIT,
TIMEOUT and unsupported/singular-basis states remain unresolved and permit the
next registered job. No denominator shrinks. Publication/final-audit failure is
retained and never labelled PASS. A machine kill that prevents even publication
leaves an incomplete batch; the auditor rejects it, and there is no auto-resume.

The terminal reader rejects missing/extra/reordered rows, status/outer mismatch,
different request or original clock, attempted work after ERROR, modified
summary, negative/nonfinite/impossible costs, and unrecorded successful checks.
Partial arithmetic and malformed tail records remain diagnostic-only under the
already tested component rules. Archival parsing marks missing/unparseable
records explicitly; it never fills an absent objective with a native value.

Cost levels are nested:

```
supplied-LP request = stages + residual, including its proof/terminal work
attempt = resource wait + request + post-terminal audit + wrapper overhead
batch = attempts + batch overhead
preflight and final-summary publication/audit are separately reported
```

Do not add these levels again. Constructor phases and serialization are nested
inside the request ledger. Missing costs are null, not zero. Historical upstream
network propagation, gate/range proofs and F0 construction are not rerun and
are excluded: this cannot become a full-network acceleration claim.

## Controls and their limits

Four analytic native LPs run the actual new batch and unchanged supervisor;
their exact primal points and objectives are independently checked. Other
controls exercise resource failure, timeout followed by exception, error-stop
roster, no retry, missing/extra rows, bad clocks/costs, changed policy/runtime,
dirty/unpushed/wrong-branch launch, archive partials and basis mismatch.

The timeout roster control uses an accelerated original-start fixture and
explicitly adjusts its synthetic wrapper clock; it is not a real301-second
measurement. Actual owned-process deadline/exception/partial-write behavior was
tested in the sealed 101-test integration stage, not weakened by this fixture.
The two stages jointly support preparation, not real-LP efficacy.

The fresh control review only recomputes ledgers and reruns standalone original-
LP checks. It does not call a native solver or reconstruct another basis.
The separate selection review took about5.69 seconds and performs no elimination.

## Interpretation frozen before execution

| Future outcome | Permitted conclusion |
| --- | --- |
| Exact feasible point with checked U <= 0 | This LP relaxation cannot prove strict positivity; not network UNSAFE |
| Exact feasible point with U > 0 | Feasible upper bound only; not a positive lower-bound certificate |
| Original-LP check rejects reconstructed point | That proposed point is not exactly feasible; not proof of LP infeasibility |
| Arithmetic LIMIT | Record first phase/operation/bit maximum, preserve partials; final solution size may still be unknown |
| TIMEOUT or incomplete proof | Execution did not complete within the fixed budget; no retrospective acceptance |
| ERROR | Stop batch, preserve all raw/partial records and unstarted denominator rows |

Archive all four terminal rows, full costs, first-limit metadata, original
bit-policy identity, basis/system comparison and proof outcome. Native Optimal
and native negative objectives stay untrusted. A completed-check label alone
does not imply feasibility, and an archive PASS is not a new mathematical proof.

## Next action, deliberately not run by preparation

After this preparation commit is clean and pushed, the explicitly invoked
once-only command is:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -m primitive_diagnostic.run launch --execute-frozen
```

Then run a fresh, read-only archive process:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -m primitive_diagnostic.archive --output docs/primitive_diagnostic_v1_execution_results.json
```

Both controls and the freeze are already complete; do not run `freeze` again.
Do not change strategy after seeing any of these four outcomes. New research
guidance becomes useful after this bounded execution identifies the remaining
blocker, or before proposing new samples, larger caps, alternative bases or a
different proof contract. None of those expansions is part of this preparation.
