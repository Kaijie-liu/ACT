# Exact-domain and live-object assessment — static decision

2026-10-14; source HEAD `4ad9c1d34ad579ffdfa2fff434a583a666f0c0a5`.
Continues the [direct unique-root design](hz_unique_root_protocol_20261014.md)
and [source/lifetime contract](hz_source_lifetime_protocol_20261014.md).
The [machine-readable assessment](hz_exact_domain_20261014.json) binds inspected
local source files. No production implementation, quota increase, checkpoint,
dataset, numerical library, GPU, solver, or timing experiment ran.

**Decision: the current Python/native implementation does not yet have the
requested complete cost upper. Stop inferring one by adding partial ledgers.**
The new useful result is a restricted dyadic arithmetic contract; the concrete
lifetime findings below show what a separately scoped bounded implementation
would have to change. All mathematical checks, their acceptance points, the
300-second budget, old limits and frozen results remain unchanged. G1–G6 remain
OPEN. This is not a new algorithm performance or certificate result.

## 1. Exact values have two domains, not one generic rational pool

For the **native binary64 candidate profile only**, complete source checking
establishes the following dataflow. A dyadic is an integer divided by a power
of two; addition, subtraction, multiplication and absolute value preserve it.

| Part | Domain and reason |
|---|---|
| Initial radius/clip and input inclusion | General rational; original literal semantics remain |
| Rounded input center/scale; affine, ReLU, guard | Dyadic; stored binary64 plus checked dyadic combinations |
| Objective coefficients `alpha` | Dyadic; classification coefficients and gate endpoints 0, 1/2, 1 |
| Native duals; pullback; dual RHS; residual; weighted norm | Dyadic, by induction through the exact updates |
| Objective constant and final scalar bound | May be general rational because of the margin |
| Strict final comparison | Also involves the non-dyadic threshold 1/10,000,000 |

`objective_roots()` accumulates margin in the scalar `d`, not in `alpha`.
Reference parsing alone is insufficient: local coefficients, roots and scales
are binary-checked, while the complete source reconstruction binds references
to source-weight sums or ReLU/guard signs. This proves dyadic closure, not that
every intermediate is exactly representable as binary64. Public `Session`
still accepts generic rational duals such as 1/3; those are **not** covered.
The new format must bind its profile, not infer it from an optimistic filename.

For canonical inputs `p/2^e` with `bits(abs(p)) ≤ B` and `0 ≤ e ≤ B−1`:

| Expression | Numerator bit upper | Denominator exponent upper |
|---|---:|---:|
| `a ± b` | 2B | B−1 |
| `a*b` | 2B | 2B−2 |
| `a + b*c` | 3B | 2B−2 |
| Sum of four absolute differences | 2B+2 | B−1 |

Proof: addition aligns both numerators to the larger exponent, at most B−1;
an aligned numerator has at most 2B−1 bits. Multiplication adds numerator widths
and exponents. In the fused expression the common exponent is at most 2B−2;
aligning the two terms and adding is bounded by 3B bits. For four differences,
align the eight original operands once to their maximum exponent; each
difference has at most 2B bits, and summing four adds at most two bits.
Canonical reduction cannot increase either bound.

At B=8192 the four-difference numerator envelope is **16,386 bits**, not the
generic rational envelope 65,540. This is a mathematical common-denominator
bound, **not** a bound on every temporary of CPython `Fraction`. It applies only
when the original operands have actually passed B-bit checks. Unchecked
`delta=abs(error-q)`, source ingress and other composite operands need their
own bounds. Do not insert an early B-bit rejection inside a fused expression:
large intermediate terms can cancel to a result allowed by the original gate.
Future compact dyadic storage must check the canonical numerator and `e+1` at
the existing acceptance points; it must retain the general-rational boundary.

## 2. Logical bit length is not allocated capacity

The inspected runtime is CPython 3.12.12 (conda-forge). Upstream
[`long_normalize`](https://github.com/python/cpython/blob/v3.12.12/Objects/longobject.c#L103)
updates logical digit count without shrinking the block. The
[subtraction implementation](https://github.com/python/cpython/blob/v3.12.12/Objects/longobject.c#L3234)
can allocate from operand sizes before borrow cancellation.
For example, take K a multiple of 30, `x=2^60+1`, K>90:
`2^K − (2^K−x) = x`. The first operand has an extra base-2^30 digit; subtraction
allocates its size, but the noncompact result needs only three digits.
This is source-level reasoning, not a measured retained-memory experiment.
The superficially similar `(2^K+x)−2^K` is **not** a valid general example:
equal-size subtraction first skips equal leading digits.

Thus accepted bit lengths alone do not bound backing capacity, hash-cache
payload, allocator rounding or RSS. Recursive integer multiplication/division
workspace also needs a call/lifetime contract. The installed `_pylong.py` is
hash-bound for review, but this assessment does **not** claim its large-division
branch is reached by this native dyadic profile. Nor does it prove the whole
binary executable matches the upstream tag. No global cache, check, or runtime
has been replaced. This is a reason to specify bounded ownership, not a reason
to add an empirical multiplier to the old slot count.

## 3. Current objects do not match the proposed typed arena

The term `2w(N+F)` in the earlier slot formula has a concrete correspondence:
prepared objective values `wF`, original dual alpha `wN`, pullback alpha `wN`,
and returned dual/residual values `wF`. It is a conservative slot count;
shallow copies can share immutable values initially. It is **not** an allocation
count. The remaining 237,328 proposed slots have no complete object-by-object
implementation proof and must not be treated as one.

Additional current-path state includes non-wire Fraction coefficient banks,
view visibility sets and membership lists, container freezing, source-row
generator buffers, retained raw candidate bytes and receipt dictionaries.
The wire path removes some eager coefficient storage, but reconstructs legacy
row representations and does not implement the new direct protocol.

The caller also preserves the previous `records` and `raw` while the next
`propose()` RHS runs; serialization can overlap native loop tensors. Native
operator, exact session and forest coexist. `finish()` does not release all
their owners, and an exception can retain frames. Finally the public object
APIs serialize arbitrary Python objects **before** checking encoded length.
The byte cap therefore cannot serve as their preallocation bound. A future
bounded durable-byte entry can avoid this issue; it cannot inherit that property
merely by calling the old object API.

## 4. Native inspection removes one uncertainty, not all workspace

Pinned Torch source is commit `5811a8d7da873dd699ff6687092c225caffcf1bb`.
For the inspected CPU/double/dim-0/int64 path, tile width ≤8 excludes the
expanded-index sorted/CSR scatter optimization, which requires width ≥16.
The ordinary double scatter accumulator aliases `self` rather than allocating
a promoted whole-tensor buffer. This is a conditional source-path conclusion,
not a new observed dispatch or a proof of full runtime memory.
([dispatch](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/TensorAdvancedIndexing.cpp#L1927),
[accumulator](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/cpu/ScatterGatherKernel.cpp#L143))

The caller's index-select and product still allocate. Named data envelopes:

| Data component | Width 1 | Width 8 |
|---|---:|---:|
| Initial alpha, groups, dual banks and objective | 242,384 B | 1,636,672 B |
| Retained prior loop locals | 589,929 B | 4,719,432 B |
| One level's select plus product | 4,436,480 B | 35,491,840 B |

The constructor `isfinite` also needs more than one mask: a conservative named
temporary envelope is `11L=74,059,920 B`, from double absolute values and boolean
results in the [pinned implementation](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/TensorCompare.cpp#L432).
These components are not a sum purporting to cover the full peak. Tensor
metadata, Python constructor lists, allocator high-water, library/thread state
and the exact consumer are additional. Frees need not return memory to the OS.

## 5. Bounded next decision, not another open-ended runtime survey

The direct protocol already separates unique roots from checked memberships
and removes old JSON logical-byte replay **in its design**. It cannot yet borrow
an implementation memory bound from these existing Python/native paths.

The smallest useful next implementation scope would be a versioned, optional
bounded representation/ownership layer: typed unique-root/source tables;
dyadic numerator/exponent storage plus an explicitly bounded general-rational
boundary; fixed candidate buffers; bounded byte ingress/encoding; and an owner
graph that prevents old/new tile overlap from escaping the stated envelope.
Keep the existing Python/Torch path as a reference. If candidate arithmetic or
reduction order changes, disclose and test it; never claim bit-identical
optimization trajectories merely because acceptance remains exact.

**That implementation is not authorized or performed here.** Before its use,
every allocation site, arithmetic scratch term, source snapshot, native/OS
component and phase transition needs a correspondence or explicit resource
contract. A hard budget gives a permitted failure outcome, not guaranteed
completion. Candidate, fallback, checks, serialization, termination and cleanup
remain charged to the same request; partial/late evidence cannot establish a
complete positive result. Do not select a larger quota or run real/GPU cases
while total memory and full-cost admission remain null.

Reproduce the static formulas and controls with the existing `act-py312` Python,
`-B -I -S`, inserting this repository into `sys.path`, then run
`scripts.test_exact_domain`. The tests exercise small exact-value identities,
scope exclusions, source drift and unchanged gates; they are neither a new
checker implementation nor an end-to-end capacity control. The stored JSON
must reproduce exactly from `scripts.assess_exact_domain.analyze()`.

Validation: **9 new static tests + 61 prior regressions = 70 passing tests**.
Three separate read-only reviews checked the arithmetic-domain/formula scope,
native/lifetime accounting and CPython borrow example; none found a required
correction. Only the
primary agent wrote files. These are AI/code reviews, not the independent human
technical review required for the paper. No performance result is inferred
from the test runner's duration.
