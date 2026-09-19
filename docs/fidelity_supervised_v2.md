# Native import fidelity and sparse exact construction V2

This is a separately versioned follow-up under the 2026-09-20 instruction to
continue the research within a 20-hour work window. The window does **not**
increase any diagnostic's budget. V1 and its failed execution remain sealed.

## Change and trust boundary

`fidelity_supervised/` retains the original sparse constructor, original LP,
single basis attempt, and standalone exact original-LP feasibility checker.
The native proposer is separately versioned. It fixes HiGHS 1.14.0
`small_matrix_value=1e-12`, explicitly binds the existing large/infinity
thresholds, enables retained native logging, and reads all options back.

Before import, the intended binary64 matrix is rejected if a stored nonzero
rounds to zero or has magnitude at/below the fixed small threshold. Values at
the native large/infinity thresholds are also rejected. No coefficient is
deleted or scaled. After import, the complete matrix, objective, row bounds,
column bounds and sense must equal the intended binary64 submission. Import
status and full readback are saved; only `kOk` plus equality permits the one
native optimization call. Warnings are not accepted. Post-solve readback is
checked as before. Threshold changes and old capture schemas are rejected.

This is fidelity to the **intended binary64 proposal model**, not exact
equivalence of that model to the original rational LP. Rational-to-binary64
rounding still exists. Native basis and values are untrusted hints; exact
reconstruction and the unchanged isolated checker must establish feasibility
against every original rational constraint and bound. No native tolerance,
objective, `Optimal` status, or successful structural audit can replace that
check. There is no dual proposal, optimality claim or network SAFE/UNSAFE claim.

## Execution and complete accounting

One supplied-LP request clock remains 300 seconds: load/import/native capture,
map and exact construction by 218 seconds; packaging and checking by 298;
terminal publication by 300. The native call remains at most 10 seconds, one
attempt, one worker and one thread. The sparse arithmetic/fill/bit caps are
unchanged. All input loading, preflight, logging, readback, serialization,
subprocess startup, supervision, cleanup and in-request review are charged.
Nested component times are not added twice. Missing evidence/time remains null.
Resource waiting and post-terminal/final audits are separately reported.
Historical network/HZ/F0 construction is not rerun or billed as part of these
supplied-LP diagnostics; this is not a full-network speed comparison.

## Controls and freeze order

The controls include threshold boundary/underflow, unchanged tiny entries,
warning injection, changed readback, option/schema mutation, exact original-LP
checking, deadline and exception paths, partial native returns, and prior
regressions. Control attempt001 is retained: it exposed a stale V1 schema
predicate in the new mapping interface, not a numerical failure. The fix is
confined to the new namespace; subsequent receipts record their own source hashes.
Attempt002 additionally exposed a fault-injection import still pointing at V1;
that analytic harness was corrected to target V2. Its unchanged legacy regression
also encountered a 20-ms startup-window race (no child was started before the
deadline). The failed receipt is retained; no legacy source or deadline changed.

After a passing full control receipt and a fresh-process review (including
relocated `python -I -S` checks), `fidelity_supervised.study freeze` binds the
same four LP files, statements and order as V1. Static compatibility includes
the new import admission policy; actual real readback, mapping and arithmetic
behavior remain unmeasured until execution. `reconstruct` independently repeats
the selection/identity check, not optimization.

The freeze is committed and pushed before `launch`. A fresh directory is used:
`data/moe/results/fidelity_supervised_real_20260920_v2`. No resume, retry,
alternative basis, extra request, budget increase or new range is permitted.
An ERROR stops the batch and leaves the remaining registered jobs explicitly
NOT_RUN_AFTER_ERROR. Limits and timeouts remain unresolved results, not errors
to erase. All four denominators and all failed attempts must be archived.

Control attempt003: **158/158 pass**. Fresh-process review reconstructs 973
retained artifacts and 16 terminal/cost records, and repeats five relocated
isolated checks, PASS with zero issues. The separate archive reader has eight
passing controls (four new import checks plus four historical denominator/cost
controls). None of these controls optimizes a real-network LP.

## Interpretation

A checked exact feasible point supplies an upper bound U for this original LP.
U <= 0 can establish that this LP cannot prove strict positivity; it is not an
unsafe witness for the full network. U > 0 does not prove a positive LP minimum.
If reconstruction or checking does not complete, the LP relaxation versus
candidate-quality explanation remains unresolved. Successful import only
closes the former import-fidelity blocker; it does not establish basis efficacy.
