# S0-C5: exact value-support sliced affine contraction, V1

2026-09-05, `redu-hz`, base `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
The corrected interval-only diagnostic finds 41 requested rows in 13 channels
at Tiny143 ReLU36 and a zero ReLU28. This motivates a source-support rule, but
does not supply its mathematical authority. C4 remains closed.

## Exact HZ identity and uniform rule

For an EXACT source HZ `h=c+Gc*xi+Gb*z`, define L as every row with a nonzero
center OR at least one explicitly nonzero continuous/binary coefficient.
Then `h = E_L E_L^T h` pointwise for every SAME latent assignment satisfying
its original equality/inequality predicates. A row is dropped from value work
only by inspecting those exact stored coefficients, never by a near-zero test,
an interval-only claim, an iid or a solver result. No latent column, binary
phase, predicate or source frame is deleted. A zero-valued source may still
constrain all other branches: its predicate owner remains present.

For complete two-Conv affine paths and requested rows R, construct only
`R B D A E_L E_L^T` and apply it to that source HZ. For each spatial tap pair,
intersect its valid input coordinates with L BEFORE the middle-channel dot
product. This avoids both full-channel descriptor compilation and products
whose source value is identically zero. All valid spatial padding/stride/
dilation events remain explicit; no effective-kernel border approximation.
V1 ordinary Conv groups=1 only; other geometries reject rather than changing
their existing representation. Source/outer row masks are explicitly respected.
The result is an HZ-restricted operator, NOT an equivalent arbitrary-vector
Conv matrix; its source is part of its semantic identity.

Bias travels through the original complete affine DAG once. A branch's Conv
bias is not removed when its value-source term is zero. All repeated ADD path
occurrences, different sources and shared frame/predicate ownership survive.
No interval, zonotope or constrained-zonotope state replaces HZ.

## First execution: acquire actual source evidence

One instrumented, corrected-loader `sparse_phase_implicit_relu_census` prefix
run of Tiny143, stop after ADD32, before the existing expensive ReLU36 tail.
16 GiB, 64M sparse entries, OMP/BLAS/MKL=1, CPU, supervisor wall 240 seconds.
No terminal solver; solver timeout argument remains 45 seconds but is unused.
Capture actual HZ and affine-expression caches plus prefix bounds and graph
certificate in an exclusive trusted local snapshot. This is evidence acquisition,
not candidate integration. A timeout or missing HZ source is retained honestly;
interval masks cannot be substituted as HZ sparsity authority.

## Mathematical and resource tests

Implement a bounded pure numerical contraction and HZ application prototype.
Test dyadic exact coefficients against independently expanded Conv composition,
constant-only rows, binary-only rows, shared predicates, whole-zero sources,
nonzero bias, borders, stride/dilation, and invalid/non-exact frame rejection.
Exhaustive small witness checks must use the SAME continuous/binary assignment.
Non-dyadic reassociation is not a bitwise/universal rounding proof; no production
exactness credit follows from tolerance-only comparisons.

Every compile counts actual retained and skipped channel products, spatial
visits, emitted nnz and CSR numeric bytes. Frozen caps remain 200M products per
branch, 256M per whole request, 64M emitted nnz, 1 GiB controlled transient;
the existing one-quarter work and whole-state strict physical-reduction gates
remain required for advancement. A smaller local operator alone is not a
whole-HZ reduction. Store mask bytes and all retained source/predicate roots.
Python/allocator work is separately measured, never called free.

No selected retry, phase/bounds repair, budget change, source-value substitution,
branch subset, attack, BaB, split, backward or dual rescue is permitted. The
math prototype does not alter the prefix probe. Before actual ReLU36 integration,
bind it to the observed complete runtime source/operator lineage, same-frame
shadows, actual forward support and exactness/resource gates. If any fails,
close V1; no later layer/instance is authorized by partial success. BN's full
retention gate and the full 2413 replay remain open. All defaults/scores stay
unchanged. Outputs are new files only in this isolated experiment directory.
