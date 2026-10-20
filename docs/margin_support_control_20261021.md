# Finite margin support implementation and controls

The new optional prototype tests the
[finite tilt rule](margin_conditioned_tilt_policy_20261021.md) on given affine
sources, not trained networks. The user replied “继续” after the bounded
implementation/control next step. This stage uses the standing authorization
for default-off implementation and one frozen standard-library batch. It does
not retry M1b/M3 or change their failed online results.

## Scientific question and finite implementation

Does the exact source-derived tilt produce the predicted positive scalar bound
on the fixed nonconstant shared/private example, while retaining the diamond
counterexample as nonpositive? This tests a specific rule and its limitations,
not external leadership, runtime improvement or G1 completion.

The producer is `act/back_end/moe/margin_conditioned_support.py`. No production
entry imports it or enables it. Controls load the file directly, avoiding ACT's
numerical package initialization. It accepts immutable source bytes and finite
candidate multipliers; no solver generates those candidates in this stage.

The source contains original affine outputs for both experts and router scores,
the complete property vector and constant, ordered factor identities and ownership,
finite boxes, all retained inequalities/equalities, pair orientation and a declared
continuous semantics. Projection to b,d,m is exact; the source hash is externally
bound. The router cannot use expert-private factors and neither expert can use
the other's private factors. Network lowering and actual route reachability
are outside this given-source contract.

Two anchors come from a rational sigmoid enclosure; each uses zero tilt, and
the on arm may add one box-weighted-median tilt. Source P and all its constraints
are unchanged. There are at most four objective slots, not four necessary solves.
The checker recomputes the original objective and the dual residual lower bound:

    y <= 0, rho = objective - A^T*y - E^T*z
    L = offset + rhs^T*y + equality_rhs^T*z
        + sum_i min(rho_i*lo_i, rho_i*hi_i).

The producer and checker have separate residual and cell-aggregation arithmetic.
They share the exact source parser, projection, finite policy and analytic sigmoid
enclosure kernel; this is not two independent implementations of those premises.
No float exponential is accepted. Sigmoid monotonicity, sigma(0)=1/2 and
sigma'<=1/4 supply the rational cell enclosure.

All cells covering the declared margin box are reconstructed; a missing or altered
cell, support, policy, identity or range is rejected. Fixed selected-softmax gates
are supported here when the checked margin is identically zero, giving weight1/2.
The returned grade is `CHECKED_LOWER_BOUND_ON_GIVEN_CONTINUOUS_SOURCE`.
Positive is not a production SAFE transition. Nonpositive is a normal completed
bound, not UNSAFE. No production optimal-status or numerical margin gate changes.

## Frozen roster and acceptance

One new exclusive directory:
`/data1/Kane/MOE/results/margin_support_20261021_r1`.
The companion freeze binds code, original source hashes, policies and this order:

1. Nonconstant shared/private example: zero versus median tilt, 128 closed cells.
2. Diamond: retain -1/16 with no rescue or refinement.
3. Pair reversal with expert/router and ownership reversal.
4. Two output classes and an exactly cancelling nonzero property constant.
5. Actual fixed half gate: one support and one degenerate margin cell.
6. Median tie and equivalent-source representation limitation.
7. Nonzero inequality and equality multipliers; one retained diamond-slice support.
8. Source/request/pair/constant/factor/router/private/constraint binding faults.
9. Missing/incorrect support, dual sign, wrong source and valid-but-wrong policy.
10. Missing/duplicate cells, false sigmoid premise and forged cell/aggregate bounds.
11. Immutable captured bytes versus mutated live input.
12. Factor/cell/stored-bit/intermediate-bit capacities and nonfinite rational text.
13. Private injected deadline and Work stops during checking, without resetting
    the real ledger or changing a global module/clock.

The fixed exact support expectations on the main example are
zero: `-89/256,-25/256`; on: `-89/256,-25/256,-25/256,39/256`.
The on aggregate must be at least 3/512 and the off aggregate negative.
The diamond remains -1/16; the single constrained-slice support is 1/16.
The receiver rechecks eight saved complete source certificates and that single
support. No model/optimizer calls or arbitrary case search occur.

These comparisons do not execute native N1/M3, stronger subdivision or GenBaB.
The stronger-baseline limitations in the mathematical record remain binding.
The sources are literal continuous examples, not route-changing model witnesses.

## Budget and reuse of existing supervision

One attempt, one admission, no resume or repair-and-rerun. Freeze/commit/push before
execution. Existing `owned_session.execute` is reused without modification.
Its original RSS acceptance remains fail-closed, with terminal_confirmation off.
An unexpected error saves the available prefix and consumes the attempt.

The controller's first timestamp starts the cumulative 300-second deadline.
Worker phase240/run250/cleanup270 reserves are inside it. Mathematical Work,
candidate reception and independent online recheck share the original deadline
and cumulative ledger: 8M visits/arithmetic, never reset per cell or reception.
The prototype narrows capacity to16 factors,32 constraints,8 classes,128 cells
and1MiB members; these are not increases to any existing allowance.
Stored/intermediate rational limits remain4096/8192 bits; a large computed output
that cannot satisfy the stored contract is refused, not rounded or relabeled.

Each managed process has 8GiB hard AS; session plus controller sampled RSS remains
2GiB, not an instantaneous hard aggregate guarantee. Generated artifacts, including
atomic publication reserves and32KiB external logs, stay within128MiB. One fresh
admission requires4GiB available memory and128MiB free disk within60 seconds.
Creation/loading, capture, exact construction, checks, serialization, reception,
cleanup and terminal export are charged. Work denotes this prototype's explicit
coefficient/row/arithmetic accounting, not every interpreter or kernel instruction;
those remain in wall time.

The interpreter's time before the first script timestamp and final stdout/exit tail
are not included in the observed export duration; `full_cold_start_seconds=null`.
The hard alarm remains armed through exit. This boundary is inherited from the
accepted standard-library control protocol, not adopted as a model timing policy.
No measured request speedup will be inferred.

## Continue or stop

Only a complete timely batch and online reception, unchanged caps, clean owned
termination and balanced FDs count as acceptance. Offline recheck cannot repair
missing or late acceptance. Preserve negative examples and any failure.

If accepted, the next research decision concerns a separately scoped HybridZ
support-candidate integration with strong native comparisons, not immediate
real-model/GPU execution or an observation repair chain. This prototype alone
does not close source conversion, G1, G3, G4 or any other project goal.
