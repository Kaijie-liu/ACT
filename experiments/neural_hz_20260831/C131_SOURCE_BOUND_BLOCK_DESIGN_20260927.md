# C131 — source-bound, demand-born Neural-HZ block design

Status: **mathematical/source-design preflight, not a qualified implementation**.
Branch `redu-hz`, starting commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`; production provenance
`15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75`.
This new document extends, and does not edit or supersede the disposition of,
the sealed C130 handoff. Formal **1870/2413** and separate E0
**CIFAR100 25 / TinyImageNet 36 = 61/400** are unchanged.

Work includes source inspection, authenticated saved-JSON scalar arithmetic,
mathematical design and one new unqualified kernel-source draft. Validation
of that draft is AST/static review only. No experiment module imports, test
collection/execution, numerical artifact loads, model/HZ restoration, solver,
default change, production edit, commit or push has occurred in this step.

## 1. The representation change

The same S0 residual-CNN convolution is represented by one exact channel-linear
equation: selected channels use shared F4 V/M definitions, and every residual
channel remains direct in the **same original output equation**. Selection is
the existing C122 structural rule, all and only channels with `12*q+88*a < 0`.
There is no second solver path, per-instance switch, alternative phase choice,
search over channel subsets, or fallback after a selected construction fails.
Evaluate all 52 fresh structural routes before choosing at most one block by
deterministic complete structural/resource bounds. Saved node/tile labels are
diagnostic labels only and never dispatch keys. A failed chosen block closes
the attempt; it does not cause an uncharged retry of another block.

The proposed new source-born implementation has three coupled parts:

1. Decode/check every original coefficient, but construct transformed kernels
   only for channels that have consumers. No unused transformed tensor is born.
2. Express exact row coefficients as a normalized 53-bit significand and a
   binary exponent; compile support once, and compare every final native bit.
3. Defer the new V/M slot binding until original radix allocation finishes.
   Preserve all original MAIN slots and semantic powers; append new factors
   after the actual radix region, without ghost slots or identifier collisions.

Continuous and binary factors, EQ/INEQ, original shared frame/latent identities,
source UIDs, all unrelated rows, and the composed original inverse are retained.
The added auxiliary equations have unique positive pivots and redundant boxes.
This is an exact extended formulation of the same genuinely nonconvex HZ, not
a zonotope/CZ/interval replacement or an approximate kernel transform.

## 2. Raw-only, demand-born kernel factory

The new closed-call factory accepts the actual original kernel and the freshly
proved complete selected/residual partition. It must observe and independently
certify **all 9KC original coefficients**, including zero coefficients. Only
**all 36KS selected transformed coefficients** are constructed; every one is
independently checked, including zeros. The independent nine-basis tensor
identity and the original-source anchors/triangular reconstruction remain.
Omitted unselected transformed cells have no consumers, which is proved from
the complete partition rather than asserted in a producer report.

For every nonzero original coefficient the raw form is `m * 2**e`, with
`2**23 <= abs(m) < 2**24` and `-149 <= e <= 104`. Independent exact restoration
to the original binary32 value or exact binary64 lift proves the words; zeros
use zero words, while the source snapshot preserves signed-zero bytes.
Finite/domain checks precede conversion and scaling. All KC alignment spans,
ignoring zero words and using span zero for an all-zero kernel, remain at most
33. Thus aligned magnitudes are below `2**57`; producer and
independent-checker partials remain below `49 * 2**57 < 2**63`.

Unlike the previous proposal, this factory has no canonical odd-mantissa
consumer. Therefore it does not compute, retain, or re-check a second canonical
pair. This is elimination of an unused representation, not a reduced fee for
the old canonical program. The conservative source checker remains 64 units
per source coefficient; the rejected 32/48-unit checker estimate is not used.

The eight owned arrays are:

| Array | Shape | Type |
| --- | --- | --- |
| Original source snapshot | K,C,3,3 | original float32/float64 |
| Raw mantissa | K,C,3,3 | int64 |
| Raw exponent | K,C,3,3 | int32 |
| Selected aligned source | K,S,3,3 | int64 |
| Selected transformed numerator | K,S,6,6 | int64 |
| Selected common exponent | K,S | int32 |
| Complete selection mask | C | bool |
| Selected channel IDs | S | int32 |

Every array is uniquely owned and their buffers do not overlap. They contain
`E = 27KC + 46KS + C + S` entries and, for a binary64 source,
`180KC + 364KS + C + 4S` bytes. The proposed complete factory work is

`KERNEL = 66560 + 32C + 16S + 1480KC + 3912KS`.

The KC term includes source custody, raw producer, independent source check,
complete span and census. The KS term includes both alignments, both complete
transform programs, all comparison/envelope/census passes and retained stores.
The fixed term covers full basis/coverage proof and mapping. These are design
reserves requiring a matching straight-line implementation, not measured work.

All data stay private to a single construct-and-prove lifetime. There is no
public prepared receipt, hash-only proof, saved kernel cache, or caller-supplied
"already checked" flag. Recheck the actual source bytes at completion.

## 3. Source-bound 53-bit coefficient theorem

For a nonzero exact coefficient write

`a = sign * M * 2**(F - 52)`, where `2**52 <= M < 2**53`.

An original raw word gives `M = abs(m) << 29`, `F = e + 23`. A selected
transformed integer first needs an exact integer-to-binary64-to-integer
roundtrip, checked once before extracting its normalized fields. Every cell
gets an exactness flag; consuming an inexact cell rejects the candidate.
Unused cells remain fully integer-transform-proved, but need not be native
binary64 coefficients. No inexact consumed coefficient is accepted.
Crucially, certify the **actual stored** M/F/sign/zero/fit fields against the
freshly independently proved integer numerator and common exponent before any
shared consumer sees them. The signature compiler reserves **48Q**, split into
24Q derivation/stores and 24Q complete stored-field read-back certification.
The previous provisional 32Q is insufficient for this explicit contract.
Only this certificate permits omission of a duplicate signature compiler in
the independent row checker; it never omits actual final coefficient checks.

The actual C8 producer returns `max(0, ...)`, and C97 stores that result in
the original node's `exponents`. Fresh binding to those semantic exponents,
and a complete nonnegative scan, are mandatory. Native row gauges and
`eq_scales` are **not** interchangeable with semantic powers.

Original raw exponents are at least -149. A nonzero transformed coefficient
is a nonzero integer times a common `2**e` with `e >= -149`, so its floor
exponent `F >= -149`. With actual parent semantic power `p >= 0`,
`q = F+p >= -149`. Consequently the normalized representation's unreduced
denominator is at most `2**201` (202 bits). Its numerator has at most
`max(53, q+1)` bits. Checking `q <= 511` therefore proves the inherited
512-bit numerator/denominator domain without a trailing-zero computation.

This theorem is restricted to the freshly bound C97 source-origin class.
It does not weaken or remove C126's general negative-map APIs or tests.
Odd defining denominators and exact L1 reduction are still checked separately.

For each occurrence, after proving the native window, the expected full IEEE
word is synthesized from sign, fraction `M-2**52`, and exponent field
`q + gauge + 1023`. Every actual final 64-bit coefficient must equal it.
For all coefficients set `L=max(-20-q)` and
`U=min(40-q-int(M != 2**52))`; use `gauge=min(max(0,L),U)` and reject `L>U`.
The strict upper-end correction matters: floor-exponent span alone does not
prove the `2**40` endpoint for non-power-of-two coefficients.
All row columns, ordering, support/cardinality, pivots, roles, semantic powers,
denominators, gauges, RHS and binary support remain checked.

## 4. Exact normalization and original C8 semantics

For eligible **nonpivot V/M normalization terms**, the floor-exponent span is
at most 60. Aligning normalized
53-bit magnitudes to the minimum floor exponent gives terms of at most 113
bits. At most 128 terms give a sum of at most 120 bits. Four base-`2**32`
limbs suffice; each pre-carry uint64 digit sum is at most
`128*(2**32-1)`. Incoming carry is at most 127, so a carry addition stays below
`128*2**32`. Check span, shifts and fan-in **before** arithmetic.
This is exact integer L1, not a floating sum or bound approximation.
Mixed output rows can have more than 128 terms and are not covered by this
auxiliary L1 lemma; their powers use the complete distinct original C8 path.

For M, the defining denominator is the unchanged product of F4 denominators;
its odd part is at most 9. Remove the sum's own power-of-two factors and keep
exact small-divisor gcd/reduction, both reduced
512-bit limits, and integer ceil-log comparison. The resulting V/M powers are
nonnegative, and their actual equations prove their redundant `[-1,1]` boxes.

Original output powers must remain the distinct C8 pre-coalescence envelope:

```
f_i = floorlog2(abs(w_i)) + 1 + parent_power_i
h = max(f_i) - 26
T = sum(2**max(f_i - h, 0))
p = max(0, h + bit_length(T - 1))
```

Neither exact-L1 ceil nor the native gauge may replace that rule. Fresh source
support, dense-original-kernel eligibility, injective surviving coordinates,
and monotone actual slots are proved before using a sort-free row plan.
Unsupported aliases are ineligible for this new source-bound primitive; they
are not silently treated as distinct or removed from the old general API.

## 5. Deferred birth and inverse preservation

An unchanged C97 encoder/tracker cannot simply accept additional V/M rows.
It fixes `nc` to the original MAIN frame, allocates radix at `nc+len(def_rows)`,
and assumes a particular raw-row/owner partition. A new sibling lifetime and
schema must explicitly implement this partition:

- Preserve all original MAIN slots, UIDs and original C8 powers.
- Prove each replaced output originally has at least two distinct nonzero
  parents, no original binary/RHS term, and native encoding without radix.
- Prove every parent used by a new row lies outside the old raw-factor cohort.
  Never classify V/M as old raw factors or feed them to old pivot-last logic.
- Omit the whole original direct encoding path, and retain private local V/M
  descriptors and original output handles. No temporary negative alias tag is
  exposed through `eq_roots`.
- After original radix allocation finishes, bind V/M slots at
  `logical_nc + actual_radix`, and UIDs after the actual original radix UIDs.
  Resolve the output handles, then run the preserved forest recurrence over
  the explicitly enlarged frame.
- Partition all actual producer counters into emitted original and deferred
  original rows. A deferred row earns at most `8D` credit only if the entire
  old direct path is absent. New C8/support work is paid, not treated as free.

Independent final owner deltas still use all actual old/new incidences and
the full coordinate width. Inverse records bind the actual positive pivots,
new slots/UIDs, gauges and boxes. Extension follows actual auxiliary equations;
recovery drops auxiliaries and composes the complete original local inverse.

## 6. Existing stages, not a narrower verification boundary

C97 already separates fresh source construction from C31 complete offline
original-source proof, physical publication and fresh restore. That boundary
is retained. C129/C130's ordinary-fixture all-in-one 24E snapshot protocol is
not imposed as a new real-source requirement. Conversely, no work vanishes:
all work is paid at the stage that actually performs it, and every stage keeps
its full original source, inverse, physical and evidence obligations.

Fresh construction takes only the original expression, keep mask and frame
widths. It must not consume an archived reference HZ or an old proof receipt.
Offline proof retains **both full C9 and full C31 inputs**, plus the candidate.
It proves all original rows/maps/sharing/binaries and all final new rows.
Do not allocate an additional full C130 control HZ: that would exceed the
64M-entry union and is not required by the existing original-source proof.

The kernel evidence and new temporary plans remain live through the complete
proof. Save all eight arrays and the full source/mapping manifest in a new
exclusive diagnostic archive, close/sync it, and bind its hash into the full
proof. Only then may construction-only arrays be retired. Verify all such
owners are dead and no candidate `.base`, closure or report retains them.
Keep the original C9/C31 inputs, candidate expression/kernels/native HZ, all
inverse maps and auxiliary/output/block records. No archive is deleted.

This is the existing C97 graph-retirement/C91 standalone-state boundary, not
an exclusion of live buffers from the physical ledger. Peak/union measurement
includes each buffer for its whole actual lifetime. Fresh restore, source
hashes, complete numerical evidence and full proof bytes remain compulsory.
M/F/sign/fit buffers, incidence plans and limb workspaces are paid, private,
derived scratch rather than additional independent source coefficients.
They stay live through their full checks and are then retired with alias
checks. As in the original C127/C97 lifecycle, no requirement is invented to
archive every arithmetic temporary; all specified factory arrays and all
actual native/source/inverse/proof data are still saved. If the implementation
retains any scratch in the returned state, it joins the physical/archive scope.

## 7. Admission still outstanding

The accompanying joint-preflight record must account for the complete source
program, every max-parent branch, all 52 fresh routes and masks, original and
new coefficient proofs, deferred owner/inverse work, complete evidence,
metadata, strict physical reduction, storage liveness and fresh restoration.
Arithmetic headroom alone is not numerical qualification.

All original caps remain: whole 256M / branch 200M, 64M entries, both 1 GiB
RSS-growth and trace-peak plus tracer metadata, 16 GiB address space, CPU1/GPU0,
worker 240s, and complete inherited-plus-new test collection/execution <=60s.
Shared reserves remain 16384 auxiliaries, 131072 positive entries and 16M
**all-new-row emission**, including original radix usage. Native magnitudes
stay in `[2**-20, 2**40]`; reduced rational numerator and denominator <=512 bits.

The new [raw-only kernel draft](c131_demand_kernel_v1.py) implements only the
private kernel primitive, not the integrated demand/block-native source path.
No real-target execution or integrated source replacement is admitted before
the complete joint bound and explicit lifetime/schema obligations close.
New qualification must retain the same ordinary source controls and
all original nonzero inverse-point populations, not reduce them to fit.
No already failed version is retried unchanged, and no frozen archive changes.

The full research goal remains active. A source primitive or bound is not a
new CERT/ADV: every old solve and every one of the 13 family counts must survive
the full 2413 replay before formal promotion. E0's complete 400-row retention
and capability four-concurrent >=1.0 gate remain separate and unchanged.
