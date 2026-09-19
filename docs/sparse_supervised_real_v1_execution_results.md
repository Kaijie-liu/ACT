# Frozen sparse-basis real diagnostic: stopped on native model-import warning

Execution authorized2026-09-20 on clean, pushed feature HEAD
`23dc688f92448df222d463ff20eb49d2b9600fff`. The four-job batch was launched
**once** using the unchanged frozen runner and resource gate. No sample, LP,
property, native option, deadline or acceptance threshold was changed.

## Outcome: execution failure, faithfully retained

| Frozen job | Terminal | What actually ran |
| --- | --- | --- |
|input220_p0, pair{1,2}|ERROR|Load completed; native model submission failed|
|input222_p1, pair{0,1}|NOT_RUN_AFTER_ERROR|Not started under the registered stop rule|
|input230_p2, pair{0,3}|NOT_RUN_AFTER_ERROR|Not started under the registered stop rule|
|input232_p0, pair{0,1}|NOT_RUN_AFTER_ERROR|Not started under the registered stop rule|

Denominator remains **four**. This is not four completed numerical diagnostics.
The execution status is `AUDITED_WITH_ERRORS`; there are zero complete exact
checks and zero newly checked feasible upper bounds. No LP or network safety/
unsafety conclusion follows. Real-basis validity, mapping, exact reconstruction,
fill growth and runtime remain unmeasured, not failed mathematical properties.

The saved traceback identifies the boundary precisely:

```
sparse_basis/native.py:116   ok(h.passModel(model))
ValueError: native API status: HighsStatus.kWarning
```

The registered adapter accepts only `kOk`. This failure occurs before native
model readback, `submission.json`, the call to `h.run()`, and raw return capture.
`prepared.json` and `native/input.json` (including the intended submitted float
matrix) remain intact. There is no `raw_native.json`, basis map, constructed
candidate, portable bundle or checker output. The absence of optimization follows
from the saved traceback and frozen control flow, **not from treating a missing
native-cost field as zero**. Native API construction/submission did occur.

The saved native logging configuration has `output_flag=False`, so the detailed
native warning is not in the log. We have **not** established whether tiny-entry
filtering, scaling or any other import rule caused the warning. It is not licensed
to ignore `kWarning`, declare the submitted model equivalent, or call this a
solver time-limit failure. No new import/optimization call was used to investigate
the warning in this stage.

## Complete available cost, with missingness preserved

| Recorded component | Seconds |
| --- | ---: |
|Input220 load window|0.414452|
|Input220 capture/import window, including failed native submission|1.516251|
|Residual request overhead|0.462805|
|Input220 full publication clock|**2.393508**|
|Batch launch preflight, outside request clock|2.622400|
|Resource check/wait, outside request clock|0.000091|
|Immediate post-terminal audit|0.312447|
|Automatic final batch audit|0.791455|

Full request cost equals the two disjoint phase windows plus residual overhead.
Native optimization duration/count are null in the frozen cost schema, as no
raw-return record exists. Unstarted requests have no cost record; they are not
three zero-cost completed diagnostics. Aggregate sums over available records are
labeled with their coverage/missing counts. There was no timeout, retry, extra
budget, extra property, native-warning bypass, or automatic recovery run.

Costs start at the archived supplied LP; historical network propagation, range
certification and F0 construction are excluded. No end-to-end MoE speed claim
or successful large sparse real-LP feasibility construction is supported.

## Archive and independent review

- [Frozen protocol](sparse_supervised_real_v1_freeze.json) is unchanged. Its
  `FROZEN_NOT_EXECUTED` describes the original registration, not current progress.
- Raw run remains at `data/moe/results/sparse_supervised_real_20260919_v1`;
  the date in that path was fixed before execution and has not been renamed.
- [Compact machine archive](sparse_supervised_real_v1_execution_results.json)
  binds the launch HEAD, all four terminals, complete available cost, traceback,
  source identities and raw artifact hashes. It excludes raw matrices/checkpoints.
- `sparse_diagnostic_archive/` only reads saved records. Four analytic archive
  controls pass, including missing-denominator, continued-after-error, missing-
  cost and false-check acceptance rejection. A fresh process reconstructs the
  frozen summary and verifies the resulting archive hashes, without optimization,
  new basis reconstruction or another bound-check attempt.

Archive `PASS` means **records and accounting reconstruct consistently**. It
does not turn `AUDITED_WITH_ERRORS` into a successful numerical experiment or
an independent network proof. Old four inexact-point outcomes remain unchanged.

## Next boundary

This run is closed and must not be resumed or overwritten. The next research
step is a separately scoped native-import compatibility investigation: identify
why this exact submitted model triggered `kWarning`, and determine whether the
native readback preserves every intended coefficient and constraint. Any proposed
handling needs analytic controls and a new execution identity before a real rerun.
Do not simply accept warnings, lower a coefficient threshold, expand the native
time limit, or manually execute the three skipped jobs in this sealed directory.
