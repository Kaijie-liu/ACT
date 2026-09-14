# Direct rational request proof R3 — completed

Execution `f30f8ccf4`, raw `data/moe/results/request_lp_rational_20260914_r3`.
Independent read-only review:
`../results/request_lp_rational_review_20260914_r3.json`, PASS, zero issues.
28 focused tests passed before execution. No dependency changes.

All 18 required output obligations check: 15 inherited scoped facts and three
new direct rational McCormick LP bounds. New residual lower bounds are about
2.60543882, 1.55273374 and 4.10959609. Overall minimum remains the reused
bound 0.18304675. Worker envelope 35.68 seconds; not a performance comparison.
Same frozen input/model, pair coverage, property and checked ranges as R2.
No tightening, retry, new sigmoid estimate or production acceptance change.

The request now excludes
`F0_outer_HZ_construction_and_floating_coefficients` from its trusted base.
The independent checker validates the exact sparse construction from the
original shared expert HZ and then checks the proposed dual arithmetically.
Both original expert functions retain the same factor vector. Old floating
F0 proofs remain archived but do not discharge R3's residual obligations.

The remaining assumptions are network/input-to-HZ, membership/pair guard
lowering, and router infeasibility exclusions. The verdict remains
`CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_LOWERING`, not an independent proof
of the entire network or its deployed floating-point implementation. R1's
UNKNOWN and R2's positive conditional result remain unchanged.
