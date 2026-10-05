# Next question: exact larger circuits with denominator-bearing definitions

This is a prospective research handoff, NOT a preregistered numerical run or a
claim of a new candidate/gain. C117 whole-program reuse and C118 exact-column
reuse have complete negative results. Full Neural-HZ goal and gates remain.

## Correct the dominant-block intuition

Nodes 16 and 19 were already included in all 163 eligible tiles of C87/C88.
C88's blanket F(2,3) circuit increased 5,357,312 to 6,190,476 nnz. Twenty local
nnz winners required 36,888 factors before storage screening. All 163 tiles
passed native coefficient checks. C89 rejected sixteen of the twenty winners
on physical storage; the four remaining node-19 tiles all fitted cumulative
reserves. Node 16 had no storage-positive tile to admit.

The implemented local bill for N new factors, O outputs and S saved nnz is

    byte_delta  = 88N + 16O + 64 - 12S
    entry_delta = 13N + 3O + 8 - 2S.

Closest node-16 byte candidate (8,10): N=1675,S=5073,O=512 gives +94,780 B.
Closest node-19 byte loser (12,4): N=1857,S=13693,O=512 gives +7,356 B.
These are declared bills, not complete LIVE gains. Raising a cap does not
repair either positive byte delta. C109 also closed additional two-parent
partial sharing while retaining V factors under ordinary injective geometry.

Evidence: C87_MASKED_TILE_COST_AUDIT_20260913.md,
C88_INLINE_TILE_AUDIT_20260913.md, C89_QUOTIENT_BUDGET_AUDIT_20260913.md,
c89_quotient_budget_v1.py, and C109_EQUIVALENCE_AND_COST_20260913.md.

## A materially different mathematical question

The primary Lavin/Gray paper gives F(4,3) transforms in equation 15 and their
two-dimensional nesting. A 4x4 output block uses 36 transformed products,
versus 64 from four 2x2 F2 blocks. This operation-count fact is NOT a predicate,
factor, memory or verification-speed claim: larger transforms add work and
may densify sparse source masks. [Primary paper, §4.3](https://arxiv.org/pdf/1509.09308).

The G transform has non-dyadic denominators. Emitting rounded 1/3 is forbidden.
Instead investigate clearing each known transform denominator onto a NEW
auxiliary defining pivot:

    D_t * 2^e * m_t = sum_c N_(t,c) * 2^e_c * v_c,
    semantic M_t = 2^e * m_t.

Here D_t is an exact positive integer, the parent transformation/equivalence
must be proved, and each z-box must be independently redundant. Integers such
as 3 and 9 are exactly native-representable even though their reciprocals are
not. The proposed representation keeps original inputs, binaries/predicates
and exact inverse; it is not a convex replacement or solver rescue.

## Existing compatibility is not an admission certificate

No prior F4 experiment/closure was found in the bounded local history search.
C96._emit_row currently installs a power-of-two pivot and conflates that pivot
with semantic normalization. A NEW versioned constructor must distinguish:

1. semantic factor unit 2^e;
2. actual defining pivot D_t*2^e;
3. a positive power-of-two whole-row gauge.

C90's independent polynomial oracle, C91 extension/physical records, and C107
publication store/accept positive actual native pivots and divide via exact
Fraction; they do not universally forbid odd integer pivots. Their other
limits, including the 512-bit rational bound, remain unchanged. Original MAIN
scalar machinery remains power-of-two-specific and must not receive these new
definitions without a separately proved construction boundary.

Do NOT clear an odd denominator on an original output equation: the original
equation oracle removes only a dyadic row gauge. Keep denominators in new
definitions and route semantic M correctly. Do not inherit M-inlining blindly:
eliminating an odd-pivot M can reintroduce non-native rational coefficients.
Retaining those factors changes the actual cost and may defeat the proposal.

## Required next experiment, before any target/source run

Pre-register a bounded exact basis-identity proof for the published F4 formula
and a denominator-cleared native constructor on ordinary dense and masked
nonconvex source fixtures. Preserve the untouched original output equations.
Use exact algebra for all basis terms, full coefficient/window/gauge checks,
redundant boxes, all-factor inverse and complete physical bills. Compare the
entire construction against direct representation and inherited F2, never
multiplication count alone. If ordinary full-source storage/work fail, stop
before target replay; do not lower fees or resurrect closed partial-sharing.

Only after this premise is proved should the complete same-structure original
mask/weight population be inspected under a new frozen budget. Actual source
and native ownership/authentication, BOTH 1 GiB/64M, 256M/200M source bounds,
shared 16384/131072/16M reserves, unchanged BASE-inclusive 45 s verification,
concrete witnesses and full 2413 replay are still mandatory for any promotion.
