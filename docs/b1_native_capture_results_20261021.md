# B1 native capture: synthetic integration PASS

Execution commit `b2c7cd8866b8cc6e917b7e1702ab76ed81fbff4c`,
[frozen protocol](b1_native_capture_protocol_20261021.md),
[manifest](b1_native_capture_freeze_20261021.json),
[lossless saved records](b1_native_capture_records_20261021.json).
One attempt under `DEV_MARGIN_20261021_V1`; no repaired/repeated control.
The saved-only audit exited0. **This is diagnostic integration, not a new
algorithm, full-request result, exact feasible certificate or network proof.**

## Results and scope

All8 registered native slots were entered:6 ordinary off/on calls and2
post-native capture-deadline injections. All29 additional controls passed.
The two injections propagated `CaptureDeadline` and produced no complete
capture; no numerical UNKNOWN swallowed them. Zero-variable and supplied
native-return fixtures are separate controls, not additional solver calls.

| Fixed scalar fixture | Off/on outcome | Interpretation |
| --- | --- | --- |
| Positive pair | Identical `SAFE_WEIGHTED_RANGE`, accepted minimum1.7499999974999998 | Numerical-policy result for the supplied scalar HZ only |
| Two-branch monolithic | Identical `UNKNOWN_MONOLITHIC_RELAXATION`, minimum−0.5000000015000001 | Nonpositive relaxation result, not full-model UNSAFE |
| Infeasible equality source | Identical UNKNOWN, solver status2, no point | Missing candidate preserved, not upgraded to a proof |

Every non-elapsed decision field was compared, including active pair, objective,
bound kind, raw bound, solver gap and candidate input. The positive pair's raw
solver bound0.25 and its stored objective offset1.5 illustrate why retaining
only a scalar bound or recomputing a matrix would lose diagnostic information.
This is not evidence that a new bound was found.

Ten original records and their byte-identical relocated copies passed same-
deadline structural reception. Relocated execution used `python -I -S -B` and
did not import ACT, NumPy, SciPy or torch. No checkpoint, dataset, historical
directory or solver was needed by that receiver. Expected scopes came from the
fixed fixture roster rather than only the worker's self-reported scope.

The receiver rejected rehashed missing/wrong structure and bindings. Nonfinite
raw point bits remained `RAW_UNCHECKED`, not feasible evidence. Source matrices
and candidates changed after capture could not alter the immutable record.
Constant, lowering-failure and supplied backend/solver-failure records retained
their distinct call-status semantics. Tests cover the declared integration
boundary; they do not prove all possible captures or every upstream transform.

## Complete batch cost and ledger

| Recorded scope | Seconds |
| --- | ---: |
| Driver entry through terminal-export preparation |3.600631|
| Driver entry through completed terminal export |3.602320|
| Owned native session execution / cleanup |3.207722 /0.040685|
| Worker case body after numerical imports |0.149309|
| Online structural reception |0.005935|
| Relocation copying and identity checks |0.004118|
| Relocated receiver entry through export preparation |0.068505|
| Owned portable session execution / cleanup |0.167264 /0.032815|

These scopes are nested, not additive. The native session includes imports,
construction, solves, capture, control checks and export. Each call's inclusive
time and its contained capture time are retained in the event ledger. The first
off query pays native initialization; comparing its microtime with the warm on
query would be invalid speed evidence. The driver's pre-interpreter startup,
final stdout/exit tail and post-run offline audit are not mislabeled online
full-cold-start cost. All online reception occurred before the original300s
deadline; offline audit did not rescue a timeout.

Work coefficients accumulated4555 →5697 →6839 from worker to online to portable
reception. Arithmetic count0 correctly reflects structural checks, **not** an
exact lower-bound proof. Main capture-ledger bytes56,202 and capacity-control
ledger bytes885 are separately labeled. The full run, including relocated code
and data, is254,029 bytes; the lossless archive is206,579 bytes, well below the
shared128MiB ceiling. Per-process8GiB AS remained enforced. Native direct-child
`wait4.ru_maxrss` was729,116,672 bytes, explicitly not group simultaneous peak.

Both owned sessions exited0 and reported `LEADER_REAPED_NO_LIVE_SESSION` before
their reserved cleanup deadlines. FD count remained4→4. The existing cleanup
contract reports identity-based no-live-session evidence; it does not claim
escaped-session coverage or complete reaping of all historical zombies.

## Scientific consequence and next boundary

B1 can now retain the actual native F0 matrix, lower-level source HZ, raw point
and branch/objective maps without changing the default decision. This closes
the **native-return capture** gap; it does not yet distinguish the real tasks'
search limits from relaxation weakness.

Next, connect the actual guarded entry/router/joint expert source and complete
pair/property roster to this optional interface, including selection of the
first returned unresolved obligation. Do not rebuild a different source using
the stopped fixed-tilt strategy. Then finish the dataset-aware candidate and
bounded diagnostic-query freeze before any real request. Exact feasible-point
or bound checks, if available, must bind these stored coefficients; otherwise
the attribution stays unseparated.

Real-stage usage remains0/0/0. A2 fixed-tilt STOP, B2 failed dependency-audit
attempt, historical source gaps and sealed experiments remain unchanged.
G1–G6 stay OPEN. Separate read-only AI review is not independent human review.
That review independently compared original/relocated/archived record hashes,
the8-slot/29-control/10-record denominators, equal deadline, continued Work,
contained capture costs and the cleanup scope; it found no blocking discrepancy.
