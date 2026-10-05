# C10 read-only defining-predicate census

Preregistered before inspecting C10 target census output. C9 terminal V1 is
closed UNKNOWN/base_unknown with gain0; its live/terminal-entry representation
proofs remain immutable. This transaction neither reruns a solver nor changes
an HZ, threshold, parameter, baseline, source checkpoint or old result.

Read exactly the sealed fresh live ReLU78 checkpoint and final HZ from C9.
Verify their file hashes, final/post-HZ content hashes, and unchanged embedded
pre-ReLU predicate prefix before analysis. Use C9's explicit main-slot and
logical-row maps, not a guessed largest-column pivot for unregistered rows.
The C9 proof of unique definitions/redundant main boxes is prerequisite evidence,
not a fresh projection proof. Original-prefix and binary factors are protected.

Measure all continuous columns: final value liveness; equality/inequality
degrees; MAIN/radix/original/ReLU role; physical defining-row support and whether
the MAIN pivot is directly present. Report exact support histograms and counts
for dead-value factors, single-consumer definitions and homogeneous two-column
aliases. For a direct dead-value definition of width r and total predicate
degree d, report the single-elimination no-collision upper bound
delta_nnz <= -r + (d-1)*(r-2). This is a structural upper bound, not proof that
floating-point multiplication/addition or the removed factor's box is safe.

For homogeneous two-continuous-column aliases with dyadic pivot, compute the
exact represented ratio using a reversible power-of-two scale. Count ratios
whose magnitude<=1 (local box redundancy), power-of-two ratios, and per-alias
whether EVERY remaining coefficient product is exactly representable as a
finite float64 and remains in the unchanged C9 window[2^-20,2^40]. Use an
independently Fraction-tested odd-significand product criterion with a final
finite/reversible range check; scan coefficient occurrences in fixed65536-entry
chunks. No coefficient is rounded, truncated or rewritten in the HZ.

IMPORTANT: Product exactness alone does not prove sums at colliding parent
columns or simultaneous alias-chain elimination. The census must label those
as unproved, and must not count its hypothetical deltas as realized compression.
No object/HZ construction, live publication, native ingestion or solver call
is authorized by this census. No target cohort advancement follows mere counts.

Limits: exact/read-only CPU1, worker240s, tests60s, address space16GiB;
whole logical census work<=256M, input stored coefficient entries<=64M,
measured census construction/temporary cap1GiB unchanged. Count global scans,
per-main classification and per-coefficient exact-product checks conservatively.
No cap increase after observation. Exclusive directory
results/c10_predicate_census_20260908_v1/; per-column numeric table, aggregate
report, source/artifact hashes, full logs and exit status automatically retained.
Run all384 C9 prerequisites plus focused census mathematics tests first.

If eligible structure is material, next preregister an exact fill-safe
continuous-only transformation with all-box/sum/reconstruction and whole-state
gates; otherwise close this direction at its measured bound. Formal1870/2413
and independent E061/400 unchanged. Historical HyZor storage remains read-only.
