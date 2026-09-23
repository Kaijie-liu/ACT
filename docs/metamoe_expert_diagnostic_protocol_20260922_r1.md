# MetaMoE expert UNKNOWN: one observation-only diagnostic

User asks to distinguish solver limits from representation precision, not to
raise the budget until a positive result appears. Historical R4 is sealed.

Source inspection finds three losses: native SciPy status/message/exception
are reduced to feasibility UNKNOWN; `base_unknown` returns before statistics;
the class-separated frontend drops expert VerifyResult metadata. The current
expert path is integer feasibility (not LP margin minimization). Router
coverage and nonzero selected-score checks have already completed in R4.
300 seconds is the outer budget; each expert's base and property feasibility
queries share at most 30 seconds. Propagation and lowering have separate cost.

## Frozen scope

ONE original `mnist_0` ACT request. Same checkpoint, materialized normalized
float64 input box, epsilon 2/255, 19 properties, tie policy, seed/threads,
2 GiB CSR representation admission, 8 GiB sampled own-group RSS, 300 seconds.
No author rerun, cohort, sample selection, retry, extra query, objective change,
support tightening, solver option change or numerical acceptance change.
All old source/environment/data hashes are inherited with NO source rebind.
New directory: `/data1/Kane/MOE/baseline_runs/metamoe_expert_trace_20260922_r1`.

Runtime wrappers call the original functions exactly once with unchanged
arguments and return the original object. They record router/candidate phases,
expert propagation layers, lowering, evaluate_spec, each feasibility/native
MILP/point-validation call and score support. Native status/message/options,
incumbent presence and existing validation result are retained. Base/property
row, exact-witness eligibility and source callsite are recorded when present.
No raw tensor/matrix serialization or additional feasibility validation occurs.
Durable, synchronous hash-chained spans survive outer termination. Logging
failure is fatal rather than silently dropping data. Overhead is charged; the
observation can perturb deadline-sensitive outcomes. This is not a matched
speed comparison and cannot reconstruct the exact old native trajectory.

## Controls / sequence

Controls cover unchanged object/arguments, exceptions, early-base metadata,
status-1 valid incumbent (no forced timeout), invalid incumbent, trace identity,
tampering, partial tail/open span, outer timeout precedence, frozen settings
and single-call/no-overwrite behavior. Existing outer process deadline/error/
partial-evidence controls are rerun. No large native query in controls.

Commit tested implementation; run `metamoe_expert_diagnostic.py --freeze`,
commit/push manifest; inspect jobs/resources then use `--run` once. A separate
process runs `audit_metamoe_expert_diagnostic.py --config ... --output ...`.
Archive compact trace analysis and raw hashes; keep all raw evidence and R4.
No automatic follow-up. Truncated/error evidence remains a diagnostic outcome,
not a passing smoke gate. Audit does not independently reprove SAFE or witness.

## Interpretation fixed before execution

- Base limit before any property query: observed inability to establish base
  feasibility under the allocation, not evidence of insufficient output bounds.
- Native limit / exhausted local deadline: execution limitation; a valid
  incumbent may still establish feasibility under the original float policy.
- Native error or rejected incumbent: record separately, not as model unsafety.
- Feasible violation region without a full-model witness: representation or
  tolerance ambiguity remains; not automatic UNSAFE or proved LP obstruction.
- Completed infeasible violation region: property excluded under existing HZ
  policy, not an independently checked floating-point network proof.
- A nonpositive margin is not recorded by this zero-objective feasibility path.
  Do not invent an LP dual gap or unique precision explanation.

Decide next action from this one trace. No larger time limit, backend switch,
formal cohort or precision experiment is authorized by this diagnostic.

## Implementation preparation (2026-09-23)

14 synthetic controls pass in both `act-py312` and the pinned ACT author-intake
environment; 10 existing outer deadline/error/partial controls pass in
`act-py312`. No native model or real LP/MILP query was run by these tests.
Initial `pytest` invocation could not run (not installed); no dependency change,
tests use unittest. The first control run had one erroneous assertion about a
function-local support import; it was corrected and both complete suites rerun.
Original ACT, model, environment and R4 execution files remain unchanged.
This protocol was drafted September 22 and prepared September 23; the R1
identifier keeps its original date, while execution timestamps are recorded.
