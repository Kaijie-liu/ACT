# C9 V1: exact radix-factor predicates and one-emission affine definitions

Previous goal turn was progress: tested C8 implementation plus complete target
rejection and source/predicate diagnosis, sealed by the eleventh manifest.
C8 is immutable and closed. Its diagnostic result is
fa3dd96dfbf1547797839af50be53a1e56024ac944f0eb89fce490742d8e242b.

## Uniform exact row representation (registered before implementation)

A logical row contains original continuous/binary coefficients a_i, integer
binary exponents p_i, and RHS b. It denotes sum a_i*2^p_i*x_i = b (or <=b).
The continuous/binary variable domains and identities do not change. A new
affine definition is the same row type, including its fresh designated output
coordinate with coefficient 2^E. All inherited equalities/inequalities use
this encoder too; no small old predicate is exempt or silently discarded.

Use the existing fixed coefficient window [2^-20,2^40]. From ORIGINAL nonzero
coefficient exponents compute the feasible integer interval of common row
shifts k satisfying BOTH ends of the window, and choose its closest-to-zero
member (already well-scaled rows keep k=0). If that interval is nonempty, emit each coefficient
with ONE combined ldexp(a_i,p_i+k), and emit RHS ldexp(b,k). Each operation
must be finite and exactly reversible. This fuses normalization/row balancing;
it does not widen the window or change native solver thresholds.

If a row's dynamic range does not fit, partition its coefficients by consecutive
24-bit exponent bands (starting at its minimum exponent). For each nonempty
band introduce eta = band_sum/2^E_band, with E_band a conservative exact integer
sum-envelope bound and NO minimum-unit floor for that partial sum. Eta's box
[-1,1] is therefore redundant. Coefficients within a band span <24 binary
exponents; at most 64M summands give a finite conservative local normalization.
Combine band values in ascending band order with exact binary-scale sum nodes.
Before combining scales more than 24 exponents apart, insert uniquely defined
scale-relay coordinates with unit changes of at most 24 exponents each.
Relay boxes are redundant because their normalized values only shrink.
The final root equality/inequality is the original RHS divided by the positive
root unit. Preserve a false constant predicate such as 0=1 exactly.

Every new coordinate is continuously and uniquely defined; all old binary
factors remain binary. Original feasible assignments extend uniquely through
the radix definitions, and projecting onto the original coordinates recovers
the original predicate and value. Re-encoding an existing defining row may
place its designated output among radix leaves; that is NOT deleting a phase
or introducing a convex relaxation. Proof is by eliminating only the new
radix factors, then using the original affine DAG's triangular definitions.
This is algebraic coefficient factoring, NOT input/phase splitting or BaB.

Keep explicit numeric root-row/row-scale/auxiliary-definition-row maps. Seal
and charge all these arrays and retained originals. Independent elimination
must recover every ORIGINAL coefficient via powers-of-two path weights; do
not use approximate sparse subtraction as the equivalence oracle. Require
positive dyadic internal links, redundant boxes, both inequality directions,
all prefixes and original source/operator multiplicities. Focused Fraction
tests independently verify all equations, feasibility equivalence and output.

## Fixed target and construction accounting

The full target remains the SAME sealed ADD75 checkpoint/native Dense77 tail,
all 14 terms, 9 sources, 200 outputs. Keep the C8 exact integer sum envelope
and floor 0 for MAIN affine coordinates; radix partial coordinates have their
proved local units. Never rerun C8 or select a different iid on failure.

Shared DAG base encoding is charged 12 abstract arithmetic operations per
retained coefficient/center/new row instead of C8's 16: one combined reversible
emission removes the first ldexp/inverse/comparison and redundant magnitude
pass. This is not a changed work ceiling or CPU instruction/timing claim.
Charge old predicate entries/rows at the same 12, including ALL equalities and
inequalities. Add unchanged support visits. Reserve and charge up to 16M
additional work for band masks/sorting, partial/relay bookkeeping and their
coefficient emissions. Count comparisons conservatively by n*ceil(log2(n))
for sorting and charge every scanned band mask, not only selected entries.
Hard auxiliary reserve 16,384; additional numeric-entry reserve 131,072. These
are new candidate internal caps, not relaxations of the existing global caps.
Check complete base+reserved work before construction, and every reserve
consumption before allocation/emission. Unused reserved buffers remain in the
physical ledger if retained. No uncharged second full-matrix conversion pass.

Unchanged global ceilings: whole work 256M, branch 200M, stored HZ entries64M,
measured construction1GiB, CPU threads1, worker16GiB, wall240s, tests60s.
Preflight must include complete old/new predicates and reserved auxiliary
storage/work. The same two-leaf native expanded reference lower bound and
strict COMPLETE numeric bytes AND entries decrease remain mandatory.

First implement/prove the general row primitive on focused nonconvex HZ
fixtures, binary/continuous mixed rows, inequalities, false constants,
non-dyadic stored coefficients, large exponent gaps/relays, exact bit recovery,
mutations and cap/default-off rejection. Test single-emission equivalence to
the two-step C8 rule where C8 is defined. Then qualify all inherited predicates
of the SAME actual checkpoint with the fixed caps, complete coefficient proof,
native ingestion and honest storage accounting. This prerequisite alone is
not the full affine-suffix candidate, a speed gain or a promotion.

Exclusive initial outputs results/c9_predicate_rows_20260905_v1/; automatic
source freeze, tests/logs/result-or-failure/exit retention. A complete integrated
suffix needs an additive run card and fresh exclusive directory AFTER the row
primitive qualifies; no source version may be changed after its target run.

No production/historical edits, no solver-option changes, no optimization or
verdict in the initial prerequisite. Formal1870/2413, E0 61/400, nonconvex and
witness invariants, live78/terminal/shadow/full-replay/four-concurrent gates
are unchanged. No isolated prerequisite can mark the overall goal complete.
