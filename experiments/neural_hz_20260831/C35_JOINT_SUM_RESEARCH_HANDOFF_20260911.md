# Remaining SAME-structure question: exact joint sums, not rounded products

C35 is COMPLETE and the ordered/disjoint non-unit extension is CLOSED:31519
actual pairs,31396 inexact products,123 window-first failures,0 survivors. No
C34/C35 retry, new journal/terminal, timeout/window/cap change or score gain.
Formal1870,13 families,E061, original goal and all prohibitions remain intact.

Read the C35 audit and C14 complete early-rejection audit before further work.
C14 checked scalar products BEFORE consumer collisions, so its negative
product guard is not a proof that an exact JOINT coefficient could never work.
C35 intentionally requires consumer-first/ordered-disjoint support. Its70784
other EQ consumers remain excluded by that shape guard, not proved admissible
or proved overlapping. This is a measurable remaining SAME single-consumer,
binary-free defining-EQ population, not permission to open another structure.

## Exact mathematical distinction

For D:p*z+a*u=d and C:q*z+t*u+v=c (or <=c), r=-q/p gives

    (r*a+t)*u + v = c+r*d  (or <=).

The mathematical operation is a single exact dyadic sum, not a rounded product
followed by a tolerance check. A binary64-inexact product does NOT logically
imply an inexact final sum when an actual overlapping coefficient is present.
The current ordered rule has t=0, where that rescue is impossible.

A small, NORMAL-window algebra sanity check was evaluated with Fraction (no
network, solver or real-target coefficient read):

    p = 2
    a = 1 + 2^-27
    r = 1 + 2^-26
    q = -2*r
    t = 1 - 2^-53
    d = c = 0

The product r*a is NOT exactly representable in binary64. The exact combined
coefficient r*a+t = 2 + 2^-26 + 2^-27 IS exactly representable, hex
0x1.0000003000000p+1. Every nonzero coefficient operand/result stays in the
UNCHANGED [2^-20,2^40] window; |a|<=p proves the removed factor box redundant.
For u=-1,-1/3,0,1/3,1, exact reconstructed z=-a*u/p preserves both the box and
the EQ/INEQ residual. This is an algebraic sanity check, NOT evidence that a
significant real cohort has this cancellation. Do not optimize a synthetic
numeric coincidence or infer capability from these five assignments.

## Bounded next action, if pursued

Before reading new actual coefficients, preregister a distinct complete
READ-ONLY joint-sum census on the same bound C34 state. A single uniform rule
can merge actual ordered supports using exact dyadic/integer arithmetic;
convert to binary64 ONLY after proving the ENTIRE resulting coefficient/RHS
is exactly representable. No FMA rounding, tolerance, interval relaxation or
changed numerical window. Unit/disjoint cases are algebraic special cases,
not a solver-status/menu fallback.

First prove and test the complete necessary conditions and work accounting:
actual degree two, current UID/row mapping, real shared-column identities,
output liveness, binary-free defining row, exact redundant box, strictly lower
complete nnz, exact RHS and every merged coefficient. Count non-head rows
with NO actual overlaps separately. Complete the population or reject it;
never use C35 table ids as a runtime/diagnostic selection whitelist. No new
target or same-version census retry is implied here.

C35's actual current UID/incidence branch already costs150324194. A new exact
merge algorithm must pay for all support searches and joint arithmetic inside
the SAME whole256M/branch200M, entries64M, measured1GiB, AS16GiB and240s.
Replacing a necessary guard with a different mathematical proof must be
declared before target results; do not merely lower old tariffs or hide scans.

Even positive arithmetic would not authorize generation or a terminal. The
one-bit sign lineage cannot handle general r. Moreover a collided coefficient
r*a+t does not by itself recover a without retaining/reconstructing t; exact
cancellations can remove parent incidence and require NEW ownership transfer.
Price all such metadata and reverse reconstruction, retain original input/
binary/global identities, and bind entirely NEW source/predicate proofs.
Only a fresh paid first-write implementation with strict whole-LIVE reduction
could advance. No root-domain totality certificate exists; base MILP remains.

No C36 source, target or preregistration exists at this checkpoint. This is
a concrete untested hypothesis, not a promised improvement. If complete
normal-window joint sums also have no survivors or no affordable physical
implementation, record that boundary and stop that version without changing
the user's restrictions or silently moving to CIFAR/new families.
