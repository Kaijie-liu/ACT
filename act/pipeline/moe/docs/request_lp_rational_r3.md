# R3: direct rational construction, separate from external comparison

Frozen before queries: same seed0/index3000 request, parent order-R2 artifact
hashes bound by `request_lp_order_review_20260914_r2.json`. No new model,
input, radius, sigmoid approximation, range refinement, or production gate.
Three residual LP proposals, at most ten solver seconds each; 600-second
outer watchdog includes parent rechecking. No retry/overwrite on failure.

Start from the original serialized shared expert HZ used by both checked
disagreement supports, NOT a floating F0 output. Project `u=q E_b+c` and
`d=q(E_a-E_b)` with rational arithmetic on exact binary coefficients, retaining
the identical factor vector. Append lambda and w directly and four rational
McCormick inequalities; bound w by rational corner products. Binary factors
remain explicitly relaxed to [-1,1]. No dense constraint materialization.

The independent checker imports neither the builder nor floating F0 nor a
solver. It reconstructs the objective, factor constraints, projections,
product planes and bounds, then checks a proposed dual certificate with
exact residual arithmetic. Request aggregation checks gate and difference
proofs, request/pair/property/source identities, full route partition,
and all reused/residual properties. Mutation and degenerate-case controls
precede real queries; a positive result is not required for acceptance.

Success means removing only
`F0_outer_HZ_construction_and_floating_coefficients` from the trusted base.
Network/input-to-HZ, membership/pair guard lowering, and route infeasibility
exclusions remain trusted. Nonpositive bounds/failed proposals are retained.
Old R1 UNKNOWN and R2 conditional positive remain unchanged. This is not a
high-accuracy experiment, performance result, complete MILP checker or native
floating-point proof. External-path comparison is a separate protocol.

Run: `python -m act.pipeline.moe.request_lp_rational --run <new-root>`.
Audit: `python -m act.pipeline.moe.review_request_lp_rational --root <new-root>
--output <new-compact-json>`.
