# Native return/receipt reserve — single-factor development protocol R1

This is an execution study, not a relaxation or source-proof repair. The prior
20-request LAS comparison remains frozen (ACT 4 policy acceptances, author 9
numerical filters and one outer timeout). Its output-query records motivate an
investigation of short query deadlines and repeated worker startup. They do not
establish insufficient relaxation precision.

## Fixed mechanism and obligations

Both new ACT arms retain checked expert-base assignment, checked routing
assignment and score-nonzero precheck. Only the **native soft solve limit for
expanded/contracted expert output queries** changes: `full_native=1.0` versus
`receipt_reserve=0.8` of the then remaining query time. This is an a priori
mechanism choice, not an optimized proportion. Base/fallback and router queries
remain at 1.0. Native presolve, zero MIP gap, tolerance, original point checking,
infeasibility acceptance and all output obligations are unchanged.

The treatment leaves room within the existing deadline for native return,
serialization, parent validation and receipt. It may instead reduce solve
capability, and a native call may overrun its soft cap. The unchanged parent
deadline kills late workers; neither arm gets grace time. Status 1 without a
checked incumbent is UNKNOWN, never an infeasibility proof. A completed unknown
can keep the persistent worker; a deadline/error retains the original restart
behavior. Both arms use the same new budget instrumentation and pay its cost.

The whole-request cap is 300 seconds; each expert retains the existing at-most
30 seconds and ordered equal-share property deadlines. No second pass, row
reordering, additional query, modified relaxation or gate, or larger budget.

## Scope, freeze and execution

After controls pass, separately freeze `metamoe_receipt_reserve_r1.json`.
Select the first two requests **in each dataset's original registered order**
from `metamoe_las_followup_r1.json`: CIFAR10 indices 1,2 and MNIST indices 1,3.
These are observed development inputs, not a holdout or success-selected set.
Same checkpoint, materialized tensor, normalized 2/255 domain and 19 global
class properties. Four inputs, two new ACT executions each, eight calls total;
alternate order by input. There is no author arm or comparison to historical
elapsed time. The parent protocol binds the original raw-top1, class-separated,
any-legal-tie semantics. This is separate from weighted-top2 proof work.

The new directory is
`/data1/Kane/MOE/baseline_runs/metamoe_receipt_reserve_20260923_r1`.
No resume, overwrite, automatic follow-up or real execution during preparation.
The one changed private native worker is explicitly rebound; old configs and
results are not edited. Execution requires the committed source/control hashes,
a clean checkout and the frozen commit as ancestor. Errors stop remaining calls
with explicit NOT_STARTED terminals; timeouts remain in the denominator.

## Controls and accounting

Before freezing: tiny real-native same-matrix differential; output-only fraction
scope; context restoration; fixed numerical gates/all properties; status-1
receipt handling; deadline/exception/partial and late-evidence rejection; wrong
identity; outer-timeout precedence; full terminal/receipt/cost audit and mutation
tests. No checkpoint or dataset solving is used for this gate.

The unchanged outer supervisor charges imports, validation, loading,
conversion, all queries, checks and worker publication to the 300 seconds.
Cleanup remains charged even if it overruns the limit. Parent stream hashes,
terminal inventories and independent audit are separately disclosed overhead,
never free solver time; the batch clock includes orchestration/postflight.
Missing native runtime is censored/unknown, not zero. Preserve each worker start,
query, proposed/effective cap, matrix hash, terminal and partial file.

The final saved-only auditor verifies bindings, the original obligation policy,
applied fractions, native caps, outer precedence and cost. Independently replay
any full-model UNSAFE witness before the final audit may pass. Report all four
input-level paired statuses (positive, replayed unsafe, unknown, outer timeout;
errors separately), gained/lost positives and solved, and costs for **all** calls.
Native runtime, worker starts and receipt overhead are explanatory, not a new
success threshold. A budgeted UNKNOWN is not a solved-fast case.

Positive results remain `HZ_POLICY_ACCEPTED`, not source-complete strict
certificates. Source/guard soundness and deployed floating-point semantics are
not repaired by this execution change. No effect is claimed before execution.
