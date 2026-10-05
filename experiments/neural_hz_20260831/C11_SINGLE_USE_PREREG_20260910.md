# C11 remaining single-use affine definitions — read-only preregistration

2026-09-10, redu-hz. Same S0 continuous-definition simplification, not a new
family or solver-status-dependent path. Close/preserve C10's outer240s TIMEOUT
at gain0. Do not rerun its solver or change any budget/window/default.

Analyze the saved exact C10 final HZ (file SHA
192f3a5a95637933bbbed8b57b10fd20f7d237b4c6f2066573e5dfa14e6b1971), using the
tagged MAIN/radix maps from the qualified live checkpoint (SHA
1c242085191040cdea7db9975aba5a8c2134e4b2776213cfee2c99a7e3ec1c9d).
Preserve the original input prefix, all binary factors, radix/later ReLU slots,
all source checkpoints and the production path. No HZ rewrite is authorized.

For each remaining output-dead MAIN variable j with a direct positive-dyadic
pivot a in its registered definition, require exactly ONE other predicate
occurrence. Allow wider affine definitions and nonzero RHS:
  a*x_j + sum(b_i*x_i) + sum(d_k*z_k) = h.
In its sole consumer u*x_j + other = t (or <=t), substitution replaces the
column by -(u/a)*b_i and -(u/a)*d_k and RHS by t-(u/a)*h. The inequality direction
does not change. A dyadic upward L1 envelope of b,d,h <=a proves the removed
[-1,1] box redundant, so exact substitution is a two-way set identity.
No binary is pivoted away. Per-definition no-fill delta is at most-2 coefficient
nonzeros, improved by overlaps/cancellations; counts are NOT jointly additive.

Read-only tests establish each candidate's box bound, exact float64 products,
unchanged [2^-20,2^40] coefficient window, exact RHS update and exact overlapping
column sums. Independent Fraction fixtures check both equality/inequality
consumers, offsets/binaries, collisions/cancellation, rejected arithmetic,
retained alias tags and protected frames. Record separately power-two consumer
multipliers, coefficient widths and products/collisions. No benchmark input
sampling, terminal solve, presolve, numerical rescue or phase fixing.

Frozen census work256M, entries64M,1GiB measured construction, worker240s,
tests60s,address space16GiB,CPU1/GPU0. Work charges8*input coefficient nnz plus
32*logical MAIN columns,64 per replacement/RHS term and32 per consumer-row
coefficient inspected for an individually evaluated definition. Charge all
candidate arithmetic before inspecting outcomes. If the full arithmetic census
does not fit, record the structural preflight and reject; do not select a
convenient prefix or relax the gate. No simultaneous-elimination proof is claimed.

After census, time ONE ordinary _lower_hz_milp call with exactly the existing
prune/coalesce options and projection/phase-fix off. This builds the model only;
no scipy/native optimizer or presolve is called. Preserve model/input content,
record elapsed/measured growth/dimensions/nnz and source maps. This isolates
lowering cost without repeating the failed capability solve. It cannot prove
where the earlier killed call was executing; report that distinction.
The lowering diagnostic is independent: a census arithmetic-budget rejection
is saved as a rejection before lowering the unchanged input. It does not permit
any prefix arithmetic, HZ rewrite or extra census attempt. A completed lowering
diagnostic cannot turn a rejected census into a pass. A hard outer timeout stops
both. Both stages retain the same independent1GiB measured construction gate.

Exclusive results/c11_single_use_census_20260910_v1/. Freeze inherited/new
sources, both checkpoints and provenance; save tests, events, per-factor table,
intermediate census result, lowering diagnostics, outcome/failure and exit
hashes automatically. Formal1870/2413, independent E061/400, gain0.
Any subsequent generalized fused substitution still needs preregistered
simultaneous/reconstruction/whole-work/live-state proofs BEFORE a new terminal
attempt; this census is not permission to append another over-budget postpass.
