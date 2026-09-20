# Candidate execution repair V2: standard-library import ordering

V1 executed at26894fa35 and ended ERROR before validation/optimization: the
fresh-process audit hook rejected decimal imported internally by fractions.
The analytic tests had already imported fractions and therefore missed this
startup path. V1 raw files and freeze remain unchanged,0native calls.
Recorded independent terminal audit PASS0; elapsed0.04210s. Missing proposal
and check costs are null, not zero-duration completed stages.

This is a separate, bounded engineering repair under the continuing task:
preload standard-library Fraction before installing the dependency audit hook,
as in the existing checked_gate.review. The checker does NOT use Decimal for
its bound arithmetic. A fresh python-S import regression now exercises exactly
this path. No dependency/environment changes.

All scientific choices in candidate_v1 remain fixed: same stored LP, one
candidate,90s native/300s total,80/120/80 phases, same tolerance/gate/supports,
same complete-request checker and conditional trusted base. V2 differs only
in import bootstrap, regression, execution/output identity and corresponding
audit protocol reference. No solver option or precision change.

Freeze V2 and commit/push before its single launch into the new directory.
Do not retry within V1 or overwrite its files. A nonpositive or failed V2
result is retained; no automatic further scientific search is scheduled.
