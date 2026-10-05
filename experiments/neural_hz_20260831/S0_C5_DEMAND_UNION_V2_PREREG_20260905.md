# C5-v2: union exact coefficient demand across spatial occurrences

C5-v1's actual prefix shadow computes 65,853,568 instead of 385,835,008 channel
products, but its projection branch uses 8,909,056/21,536,768 and misses the
per-branch quarter-work gate. V1 is not promoted; this observation motivates a
new algebraic sharing implementation, not a relaxed threshold or selected arm.

For a channel-stationary middle diagonal and channel-stationary inner row mask,
the coefficient indexed by `(output channel, outer tap, inner tap, input
channel)` is independent of the spatial occurrence requesting it. Enumerate
every requested output row and every exact live source coordinate as in V1,
union these coefficient demands, contract EACH demanded coefficient once in
ascending middle-channel order, then emit every original spatial contribution
in exactly V1's order. Duplicate ADD terms/sources/predicates are not merged.

No coefficient is estimated, approximated, rounded into a preferred value or
obtained from a solver. V2 must be bitwise equal to V1's source-restricted
matrix on all tests, including non-dyadic payloads, because each reused scalar
has identical operands and order. The prior difference from the independently
associated spatial-CSR oracle remains a separate, unresolved rounding boundary.
Nonstationary scale/mask rejects V2; no hidden alternate representation path.

Before any real V2 result, freeze the same all-two-branch ADD16/Conv17 component
shadow, source snapshot, selected rows, original budgets and per-branch quarter
gate. All 200M/256M product and 64M emission limits stay unchanged. Charge the
union masks, coefficient payloads, emission CSR, two spatial enumeration passes,
additions, sigma work and Python/RSS overhead; discard compile cache after
emission. Do not count fewer multiplications as free traversal or whole-HZ
physical savings. Source HZ/predicate roots and frame remain intact.

Tests cover exact V1 equality, real-valued source binding, repeated spatial
requests, zero support, borders, nonstationarity, masks and caps. Actual-source
shadow must repeat ALL terms, not only the failed projection branch. It cannot
rerun a verifier, skip baseline rows, change a score or authorize ReLU36/default
integration by itself. Whole-state, runtime-lineage, numerical exactness,
four-concurrent, family and complete 2413/E0 gates remain outstanding.
