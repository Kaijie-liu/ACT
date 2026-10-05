# C5 ordered union: complete HZ bitwise result, native boundary still gated

2026-09-05, branch `redu-hz`, base
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
The preceding status-only turn made no progress. This continuation implements
and tests an ordered source-bound coefficient compiler and produces two new
complete-field HZ comparisons. No production source/default, old archive,
baseline authority or score ledger changes. Formal **1870/2413**, E0 **61/400**,
TLL capability **27/32** remain unchanged; formal gain is zero.

## Structural identity and floating order

For a bound source HZ x = c + Gc xi + Gb z satisfying its original predicates,
a coordinate whose c/Gc/Gb row is exactly zero is zero for every admissible
assignment. Omitting its value contribution leaves the SAME factors,
predicates, frame and witness assignment valid; the source is retained even
if all its output values vanish. No binary or continuous column is deleted.

For two ordinary convolutions with channel-stationary intervening/output
scales, each requested output/input spatial pair defines an ordered list of
valid outer/inner taps. Coefficients are reusable across spatial positions
only when output channel, input channel and that ordered list agree. Kernels,
geometry, stationary inner mask and scales are fixed within the compilation.
The source-support union changes which coefficients are requested, not their
values or summation order.

V3 accumulates middle channel first, then outer spatial taps, with each term
evaluated as ((output_scale * B) * middle_scale) * A. It resets exact
cancellation to positive zero like the original dictionary oracle and uses no
reduction or fused multiply-add. Independent final-column accumulators permit
exact source-zero pruning without changing retained accumulation order.
The oracle uses existing scalar _left_compose_rows, with no coefficient-demand
reuse; toy tests also compare against its unfiltered full source-column form.

Complete materialization coalesces equal source identities BEFORE applying
HZ maps, joins distinct sources in original order, and adds full native bias
once, including bias outside the requested rows. Tests cover all fields,
multiple batches, non-dyadic payloads, borders/stride/dilation, masks, duplicate
sources, zero/cancellation, mutation and resource rejection. The new suite has
21 passing tests; a combined replay of the preceding 31 plus new 21 tests
passes **52/52**. This is a focused suite, not a full repository or 2413 replay.

## First full-expression shadow

Both actual ADD16 terms, through Conv17/SCALE18/BIAS19, pass byte-for-byte
comparison of ALL retained operator coefficients and ALL joined HZ fields.
The result has 25088 output rows, 11156 continuous factors, 874 binary factors,
874 equalities and 1748 inequalities, retaining frame 1 and exact=True.

| Source | Unrestricted products | Ordered V3 products | Quarter-work |
|---|---:|---:|---|
| ReLU9 | 364298240 | 29376000 | pass |
| ReLU5 | 21536768 | 2767360 | pass |
| Total | 385835008 | 32143360 | both pass |

Ordered arithmetic costs slightly more than V2's 30486400 products, but still
removes about 91.7% of unrestricted channel products (about 12x). It resolves
the prior arithmetic-association difference with the implicit implementation,
without changing tolerances. It does not assert equality to the differently
associated spatial-CSR oracle or a universal NN outward-rounding certificate.

Complete candidate materialization took 1.459 seconds; the source-restricted
scalar test oracle took 16.474 seconds. These are instrumented one-off
diagnostics, not a controlled speed gate or full-prefix speedup. Peak RSS
including oracle/observations was 863592 KiB. The supervisor completed in
28.827 seconds, exit 0, no source/provenance drift.

Evidence: `evidence/c5_ordered_full_hz_shadow_20260905_v3.json`, SHA-256
`b79c6330990c03ff9fe296beb20a383bf34816d002b8da9674315fb9abdde9e2`.

## Native mask correction and newly established blocker

The first shadow used the native bias helpers with an UNMASKED outer Conv.
Production's deferred path masks stable-negative outer rows. Thus the first
result proves its complete paired expression, but cannot be promoted as the
entire native runtime expression: unselected bias rows can differ.

A separate preregistered diagnostic invokes the existing deferred preparation
on the sealed ADD16 state and captures its actual masked expression at entry
to the phase-selective routine. It runs no ReLU, publication, solver or prefix
retry. Every field of this native masked HZ ALSO matches the scalar oracle
byte-for-byte. Candidate materialization is 1.625 seconds; oracle 19.216 seconds.
Its selected frontier remains 310 unstable plus eight positive = 318 rows.

The unchanged phase-selective savings condition is now directly evaluated:
the eight positive probe rows have **7711** generator nnz, below **25875**
required entries. Production therefore rejects this selective probe and next
requests ALL **1097** non-stable-negative rows (310 U + 787 P). Passing a
318-row probe must not be misreported as sufficient to complete Conv17.
No condition or threshold was changed, and the 1097-row transfer was not run
in this continuation.

The explicit-root representation ledger attempted to include network input
and specification objects in addition to all saved HZ/expression/bounds roots.
It correctly rejects `network.0.labeled_input: LabeledInputTensor`, for which
this adapter has no registered traversal. No partial storage sum or physical
gate is claimed. Expanded comparator arrays were constructed and validated
against every implicit operator row before that rejection; peak RSS including
them was 1369524 KiB, not a candidate-only memory measurement.
The ledger diagnostic V1 is closed with this recorded failure. No alias rule
or unknown-object rejection was weakened.

Evidence: `evidence/c5_native_boundary_ledger_20260905_v1.json`, SHA-256
`1002f92628376039b539130ec5d53011ba1a7bbbec9082664e28e26269a44b09`.
Supervisor exit 0 indicates a completed diagnostic, NOT a passed ledger.
Its exit record reports 31.997 seconds and no source/provenance drift.

## Exact next work and limits

1. Register the full 1097-row native materialization boundary, using the same
   ordered compiler and unchanged caps. Count ALL branches, both prior probe
   work and native follow-up work where they coexist in a runtime transaction.
   An isolated successful follow-up must not hide the preceding probe cost.
2. In a NEW ledger adapter version, explicitly traverse the known
   LabeledInputTensor tensor/label and registered InputSpec/OutputSpec fields,
   rejecting unknown fields/classes. Test these roots and aliases before the
   real ledger. Preserve the V1 failure and all original owner-aware rules.
3. Complete same-boundary physical and construction-workspace accounting and
   runtime publication/rollback gates before attempting integrated progression
   toward Tiny ReLU36/63/71, CIFAR targets, E0 and the 13-family full replay.

C5-v3 is a component-level positive result, not an enabled runtime candidate.
The two earlier 240-second prefix timeouts remain failures. No completed
ReLU20, ReLU36, full prefix, formal CERT/ADV or four-concurrent qualification
is inferred from these shadows. Goal remains active. All jobs are terminal.
