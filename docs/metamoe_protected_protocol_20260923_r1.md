# Protected expert property budget, R1

## Scope frozen before real execution

The preceding observation established **zero native output-property queries**:
base feasibility overran its30s soft limit and starved the19 output rows. This
version changes execution scheduling only, not HZ propagation, guard/candidate
analysis, support, model/inputs, constraints, integer factors, property tolerance,
or acceptance mathematics. Singleton guard elimination is explicitly excluded.

ONE original MNIST0 ACT request, same physical normalizedfloat64 box at2/255,
same checkpoint/19 properties/seed100/2threads/2GiB CSR/8GiB sampled group RSS,
same300s outer request limit. Original expert allocation remains <=30s. No author
rerun, fresh cohort, parameter search, historical evidence reuse or automatic
retry. R4 and observation R1 are preserved. This is a functionality/development
control, not a new paired speed or coverage experiment.

## Protected schedule and acceptance

Expert clock starts before lowering/export. Base deadline is10% of that SAME
allocation (3s if30s total), fixed before data. Remaining ordered property rows
share remaining time: current row gets remaining/(rows still to visit). A
contracted witness check, when required by original tolerance logic, shares its
expanded row deadline. Unused time is available to later rows. No subset chosen
after observing margins. Unknown base no longer prevents diagnostic property
attempts but NEVER licenses CERTIFIED. An infeasible base retains old UNKNOWN.

Native SciPy runs in a private persistent child, same installed binary/options;
the parent stops it at local deadline, ignoring partial/late output. There is no
retry of that query; a later distinct row may start a new child. Import, transfer,
serialization, matrix hashing, loading, native work, original point validation,
cleanup and evidence publication are charged to the expert/request clocks.
The child remains in the outer-owned process group. Native time_limit alone is
NOT trusted as hard wall-clock enforcement. Polling/OS kill/cleanup can have
small overshoot, which is counted, not a free extension or claimed instantaneous
resource guarantee. Final expert publication after deadline demotes conclusions.

The parent retains original `_valid_milp_point` bounds/integrality/row checks.
Valid incumbent can establish feasibility at native status1, as before;
native status2 excludes the region under the existing numerical policy. No
non-optimal objective is accepted as a bound. A conclusive output requires:

- CERTIFIED: base feasible AND every required expanded violation query infeasible;
- FALSIFIED candidate: exact/shared-frame recovery plus original strict violation
  threshold; original full dynamic-model replay is still required for UNSAFE;
- everything else UNKNOWN with full per-row and per-query records.

V1 protected API supports only one-lane, nonempty LINEAR_LE. Unsupported requests
fail closed; default HZSolver on disk is unchanged. Temporary class substitution
is explicitly opt-in, main-thread/process-local and restored on exit. New files
are added to the inherited identity manifest; no frozen source is rebound.

## Evidence and checks

Each evaluation saves its plan, same constraint matrix + output linear map,
property matrix/thresholds, query scopes/extra rows and unique-token/hash-bound
native results. Every property is listed, including NOT_STARTED. The auditor
reconstructs property rows from the stored base/output map and C matrix, checks
row order, binding, deadline/outer precedence, and repeats existing float point
validation without solving. This is NOT a source-complete or exact proof.
All partial/fault artifacts remain. Raw matrices/checkpoints are not committed.
`native_started` counts the durable pre-call entry marker. An interrupted entry
is not a completed solver query; report completed native returns and accepted
infeasibility results separately. A kill between entry publication and the call
cannot establish the exact amount of internal native work.

Controls before freeze: real tiny safe/unsafe differential, base hang followed
by real property calls, unknown-base refusal, missing obligations, contract
drift, invalid binding, child exception, partial/late publication, no-spawn after
deadline, restoring opt-in scope and complete budget/cleanup accounting. Original
outer controls are rerun. Keep first failed control outputs in work notes; no
dependency install or changes to original acceptance gates.

Pre-freeze controls (2026-09-23): 54 distinct tests PASS in `act-py312`:
21 protected solver/deadline controls,11 saved-evidence/outer/audit controls,
10 existing CSR outer controls,3 original numerical-policy controls,9 existing
class-separated top-1 controls. The new21+11 also PASS in the pinned intake
environment. No real MNIST query has run at this preparation stage. Reproduce
with `python -m unittest discover -s tests -p test_protected_hz_solver.py -q`
and the analogous `test_metamoe_protected_smoke.py`,
`test_metamoe_csr_execution.py`, plus
`python -m unittest act.back_end.solver.test_solver_hz_policy -q`.
The class-separated suite requires `initialize_device('cpu','float64')`
before discovery. The initial plain-discovery invocation failed7/9 because
the global ACT device was not initialized; rerunning the correct launcher
passed9/9. No production fix/dependency change was made for that invocation.
Gurobi availability warnings are retained; the frozen native backend is SciPy.

Commit tested code, `metamoe_protected_smoke_r1.py --freeze`, commit/push config,
inspect current jobs/RAM, then `--run` ONCE. New output root:
`/data1/Kane/MOE/baseline_runs/metamoe_protected_20260923_r1`.
Use `audit_metamoe_protected_smoke_r1.py` in a separate process. Parent hashes of
completed/partial evidence are postflight, explicitly timed and included in the
batch, not hidden inside the300s request clock. The independent audit is separate.

## Predeclared decision

Primary engineering endpoint: do native output-property queries now run without
base monopolization, within unchanged total limits? Report attempted, excluded,
unknown and unstarted properties separately. Full SAFE is not required to call
budget isolation effective, and such effectiveness is not certificate success.
If properties remain limited, record that. If all are excluded but base remains
unproved, retain UNKNOWN. No limit increase, relaxation tuning, reopening R1 or
formal cohort follows automatically from this control.
