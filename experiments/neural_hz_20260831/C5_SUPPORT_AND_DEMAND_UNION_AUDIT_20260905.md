# Actual HZ support and coefficient-demand union: positive component checkpoint

2026-09-05, `redu-hz`, base `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
The preceding goal turn made progress (complete thread diagnostic and C4
closure). This continuation preserved both earlier seals and obtained new
actual-source mathematical/component evidence. Formal 1870/2413 and E0 61/400
remain unchanged. No production source/default has changed in this continuation.

## Corrected support is not the old 2180-row census

The source/corrected IntervalTF diagnostic on the pinned Tiny143 graph passes
all fixed-center inclusion checks. Corrected ReLU36 has 25,047 N, 7 P and 34 U:
41 selected rows in 13/128 channels. ReLU28 is zero under this cheap pipeline
interval analysis. Channel-sliced C4 contraction would prospectively cost
52,715,520 instead of 519,045,120 products, but interval evidence alone cannot
authorize HZ sparsity or a corrected runtime bound. The independent interval
source run is 11,156 U, not Trial9's HZ-tightened 2172 U; these scopes are not
conflated. No independent outward-rounded interval proof is claimed.

Evidence `evidence/corrected_phase_support_census_20260905_v1.json`, SHA-256:
`425e3997e6c68b388145b4b7e4e86b211d115880fb7c17871704f0594f0b30ff`.

## Actual HZ prefix acquisition, with both failures preserved

The V1 240-second acquisition timed out before ADD32 and produced no snapshot
or result JSON. The distinct V2 changes only observation granularity, sealing
each completed ReLU/ADD and recording every layer start/end. It also times out
at 240 seconds, but proves that Conv17 is the first unfinished transfer: ADD16
and its snapshot completed at 2.279 seconds. Both exits show no source drift.
Neither run reaches ADD32, ReLU36 or a terminal property. Fragments are not
unioned into a completed prefix. No solver call or verdict retry occurs.

The immutable ADD16 snapshot is 75,790,481 bytes, SHA-256
`3cb6d229300aaa8755cfcf6794f5671f9b0486fc50d0375e4d106a2b1757af8d`.
The `.partial` names in the completed early snapshots are retained hard-link
aliases of the sealed files, not extra copies and not inferred completion;
the per-layer JSON hash seal is the authority. Unsealed partials must not be
loaded as completed evidence.

The real expression has TWO terms, both in frame 1:

| Source | Live value rows | Continuous | Binary | Equalities | Inequalities |
|---|---:|---:|---:|---:|---:|
| ReLU9 | 3906 / 25088 | 11156 | 874 | 874 | 1748 |
| ReLU5 | 18815 / 46656 | 10432 | 512 | 512 | 1024 |

These live masks inspect exact c/Gc/Gb coefficients; they are not sourced from
interval assumptions. Every source and all its predicates remain retained,
including when a whole value contribution vanishes. No binary or latent
column is projected out.

## Complete mandatory-prefix component shadow

The actual ADD16 expression, full graph paths, inner Conv/SCALE payloads,
Conv17 and SCALE18/BIAS19 were matched to the trusted runtime snapshot. Both
source terms are measured, not only a favorable branch. The same forward
calculation selects 310 U plus eight P = 318 rows in 63 channels before
ReLU20. This is a same-structure mandatory-prefix shadow, not a new target
substituted for ReLU36.

All selected operator columns and resulting HZ center/continuous/binary maps
are compared with an independent spatial-CSR/SciPy product oracle. All factor
widths, frames, equality/inequality matrices and RHS vectors are preserved.
The fixed concrete graph center differs by at most 2.1094237467877974e-15.

| Source branch | Unrestricted spatial products | C5-v1 | C5-v2 demand union | V2 quarter-product gate |
|---|---:|---:|---:|---|
| ReLU9 | 364,298,240 | 56,944,512 | 27,719,040 | pass |
| ReLU5 | 21,536,768 | 8,909,056 | 2,767,360 | pass |
| **Total** | **385,835,008** | **65,853,568** | **30,486,400** | both pass |

V1's projection branch misses its individual quarter-work condition. V2
removes repeated coefficient dot products across spatial occurrences; it does
not change that gate, omit a branch or tune by identity. Both V2 matrices are
BITWISE IDENTICAL to V1 for every actual emitted coefficient. The new
coefficient-demand union is a genuine computation change, not a solver rescue.

V2 additionally charges 685,824 sigma products, 31,000,881 additions, and
49,728 spatial visits (two enumeration passes). The compile caches contain
5,542,272 and 315,072 numeric bytes and are discarded after emission. Emitted
CSR numeric bytes are 1,627,264 and 836,500. Those are local buffers, not the
whole HZ/live-cache ledger; source owners, predicates, biases and other graph
consumers remain live. Python dictionaries/allocator workspace are not hidden
inside these numeric-byte counts; peak process RSS in the evidence includes
the independent oracle and is not a candidate-only measurement.

Measured V2 contraction times are 1.164 s and 0.120 s (1.284 s total), compared
with V1's first shadow at 3.731 s and 1.004 s. These separate one-off component
measurements are not a controlled speed gate, full-prefix timing, or a 12.7x
verifier speedup. The roughly 12.7x figure refers ONLY to counted channel
products relative to unrestricted spatial composition.

## Exactness limits and open promotion requirements

The real matrices are not bitwise identical to the independently associated
spatial-CSR oracle: maximum restricted-operator discrepancy is about 1.17e-15;
continuous-map discrepancy about 1.48e-17; binary-map discrepancy is zero in
these two shadows. Exact dyadic theorem tests and V2-to-V1 bitwise reuse tests
do not turn those real floating comparisons into a universal rounding proof.
The evidence explicitly leaves numerical exactness authority unproved.

No full HZ materialization transaction, publication/rollback, exact runtime
whole-state comparison or four-concurrent qualification has run. No early
snapshot becomes a completed prefix. Full corrected BN retention, ReLU36,
E0 and the formal 2413 replay remain outstanding. The components are isolated,
default-off, and carry gain 0. C5-v2 is a positive component candidate, not a
promoted result and not completion of the user's goal.

The next step is a complete same-frame materialization shadow from this actual
snapshot using the same rule and explicit whole-state/numerical accounting,
then the still-gated runtime prefix/ReLU36 attempt. It must not substitute a
local CSR saving or one favorable timing for those requirements.

Evidence:

- V1 `evidence/c5_real_prefix_component_shadow_20260905_v1.json`, SHA-256
  `fcf1c7d3d3cf5e1f6904382d48587d8dc54b450a0a160045d9e077838100b316`.
- V2 `evidence/c5_real_prefix_component_shadow_20260905_v2.json`, SHA-256
  `8cf371d1eef090cc05188d70ca2cfc785f0d5998b214686d9067a9e0047daeec`.
- Focused new tests: 7 phase-census + 14 C5-v1 + 10 C5-v2 = 31 tests.
