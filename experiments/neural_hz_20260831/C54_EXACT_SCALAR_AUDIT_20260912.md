# C54 exact general-scalar HZ: mathematical/storage prototype, not solver admission

Branch redu-hz, starting commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac.
Production candidate SHA256 15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75 is unchanged.
Formal baseline remains 1870/2413 and separate E0 remains CIFAR25/Tiny36.
All runtime outputs completed and saved before the subsequent read-only instruction audit;
this document completes their archive on resuming the research goal. No instruction-audit
proposal has been applied.

## Representation and exactness

Source-first substitution removes an output-dead continuous definition z=r*x
only when its defining pivot is positive dyadic, its constant is zero, it has no
binary term after exact shared-root coalescing, and 0<|r|<=1. A retained root
is in [-1,1], so the removed coordinate's box follows from this relation.
Every original coordinate has an exact inverse; original EQ/INEQ semantics,
global identities and binary factors are preserved. Binary factors are not pivots.

Finite binary64 source values are exact dyadics. Products and sums use signed
integer mantissas/exponents, not rounded floating products. Canonical scalar IDs
index owned uint64 limbs, exponents, signs and offsets. The fixed limits remain
512 mantissa bits and the nonzero magnitude interval [2^-20,2^40]. This format
is not directly admissible to an existing binary64/native consumer.

v1 uses split continuous/binary CSR matrices and two inverse arrays.
v2 packs EQ/INEQ/output into one CSR with explicit continuous/binary column and
row partitions, and packs the inverse root/scalar IDs into one uint64 record.
The unchanged v1 builder is measured before v2 packing; all23 unshared split
numeric arrays are weak-retired per cohort. Global/UID/scalar arrays remain
explicitly shared. This is not yet a fused real-network source writer.

## Fixed complete-source comparisons

All cohorts use width128 ordinary general-scalar chains, shared Add, and Conv/ReLU.
Complete original source is retained on BOTH sides. The reference is unchanged
C52 binary64 reference generation for the same program, not an inflated exact
reference and not current C31 or the complete real phase-selective verifier.

| Cohort | Continuous factors | Predicate nnz | Numeric bytes | Numeric entries | Numeric + reported shallow bytes |
| --- | --- | --- | --- | --- | --- |
| chain | 131 -> 3 | 280 -> 9 | 12132 -> 8279 | 1240 -> 1079 | 126096 -> 124423 |
| shared Add | 195 -> 3 | 520 -> 9 | 20900 -> 13338 | 2152 -> 1702 | 188016 -> 181805 |
| Conv/ReLU | 133 -> 5 | 274 -> 17 | 12052 -> 8327 | 1230 -> 1096 | 128148 -> 126735 |

v2 passes all four fixed physical comparisons on all three cohorts.
Numeric-byte savings are approximately31%,36%,31%; combined reported-accounting
savings are only1.33%,3.30%,1.10%. Neither is whole-process RSS, runtime or benchmark
gain. The shallow-object ledger is reported accounting, not exact allocator
occupancy; its object/array-header conventions can overlap separately charged
numeric ownership. Measured transient bounds are reported independently.

v1 remains rejected: chain combined126096->127968B; Conv numeric entries1230->1242
and combined128148->130304B. Shared Add passed. The diagnostic repeated exactly
the two registered failures and never reclassified them as success.

## Tests, proofs, measured transactions

v1 frozen focused suite:23 tests,21 pass and2 registered physical failures in1.04s.
Worker exit2 means complete negative diagnostic, not accepted representation.
v2 frozen focused suite:26 pass in1.16s; worker and supervisor exit0.

Independent Fraction induction checks every original source row, inverse,
predicate, output and UID. Maximum scalar mantissa widths are417/107/417;
each cohort contains a surviving matrix coefficient not exactly binary64.
The48 DISTINCT feasible toy-network vectors (6/6/36) passed complete inverse
and exact-function checks, then the same48 were replayed after serialization.
They are not96 distinct vectors and none is a benchmark counterexample.
This is an executable exact audit, not a proof-assistant certificate.

v1 measured window1.3513998687267303s; worker3.2980410680174828s;
supervisor5.987286081537604s; work1365227. HWM-minus-entry-RSS6819840B;
traced peak3762285B plus metadata899232B. Both1GiB guards passed.

v2 measured window2.872807084582746s; worker4.825493813492358s;
supervisor7.640510271303356s; work2485065. HWM-minus-entry-RSS19750912B;
traced peak9561610B plus metadata919328B. Both1GiB guards passed.
Measured work includes full source/reference/exact construction, packing and
retirement, proofs, points, ownership, protocol5 export, unchanged C41 authenticated
decode, restored proofs/points/physical checks, and fresh+restored root accounting.
All CPU1/GPU0,AS16GiB,64M-entry,256M-whole/200M-branch,60s-test/240s-worker limits
are unchanged. Per-proof work readings may be cumulative on a shared work pool.

Fresh v2 archive retains57772 numeric bytes/6342 entries/1924 numeric roots and
409173 reported shallow bytes. Fresh plus restored retains115544 numeric bytes,
12684 entries and791926 reported shallow bytes. Decoder1984 numeric reducers
uses writable-bytearray storage, not a claim of whole-transport zero-copy or
restored original view-alias identity.

## Immutable runtime records

- v1: results/c54_exact_scalar_negative_20260912_v1/
  result SHA256 07dd6126ebdbad28d1b43ea8460f61b811d4dc4a971c2e8a3b6b2b76989ae080;
  complete protocol5 archive133290B, SHA25658f8d736b1f0685c60fca11b7f94b3bfbc864398e21886f4bb994d8a6ded8f8d.
- v2: results/c54_packed_exact_scalar_20260912_v2/
  result SHA256 dd7a50c5bf2394860d3ad7450b72bf8e0665e36913c484e79e6b44917c135f75;
  complete protocol5 archive131585B, SHA2567c9ed1990e5e6a65d3d6511931738fc6df162be5f78d85c93942e3b4ea90fe73.

All logged artifact hashes and781/786 source paths rechecked with no drift.
All62 earlier checkpoint manifests,2207 unique files/28308295247B verified unchanged.
No C54 worker remains to poll or restart. Native lowering, real target, same-structure
shadows, family/full replay and timing gates remain UNPROVED, formal gain0.
