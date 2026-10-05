# C131 — joint source-generation / complete offline preflight

Status: **prospective source-design bounds; no numerical qualification**.
Read with [the mathematical and lifetime design](C131_SOURCE_BOUND_BLOCK_DESIGN_20260927.md).
All values below are source-derived formulas evaluated on authenticated saved
scalar census records, not new coefficient observations or measured work.
No instance/tile labels enter the proposed rule. The C130 checkpoint and all
earlier sources, results, failed dispositions and score ledgers are unchanged.

## 1. What changed from the C130 handoff

The joint plan now removes genuinely unused canonical kernel words, shares
freshly certified stored 53-bit signatures within one closed transaction,
preserves original deterministic C8/gauge witnesses, and uses a single-row
working frontier. It does not reduce coefficient coverage, reprice the old
implementation, or require an additional full old HZ to be materialized.

Generation and complete offline proof/publication remain the existing C97/C31
stages. Costs are assigned to the stage doing the work; neither sum is an
all-process CPU-work or whole-LIVE certificate. All original inputs, complete
source comparisons, owner/inverse proofs and fresh restore remain required.

## 2. New bounded row programs

Definitions: C input channels, K filters, S selected channels, O demanded
original outputs, A new V/M factors, R=A+O new rows, N complete new row nnz,
D original direct nnz including output pivots, and T V/M nonpivot incidences.

```
U = 36C + 16K
Q = 36KS
P = 144C + 484S + Q + 324K
F = 66560 + 32C + 16S + 1480KC + 3912KS
G = 65536 + 16U + 16P + 48Q + 16D + 128O + 40N + 240R + 12T
J_math  = 65536 + 16P + 16D + 128O + 32T + 192R
J_final = 16U + 32N + 64R
I = 65536 + 4(D+N) + 96A + 192O
```

F is the complete raw-only kernel factory. G's charges cover:

| Term | Paid scope |
| --- | --- |
| 65536 | Headers, fixed identities and bounded scalar metadata |
| 16U | Actual source maps, powers, ranges, injectivity/order and map custody |
| 16P | Complete support/row planning, prefix/population work and owned plans |
| 48Q | Signature derivation/stores AND certification of actual stored fields |
| 16D+128O | Original support/C8 producer, original direct-native/no-radix eligibility |
| 16N+128R | Native arithmetic, full IEEE synthesis, gauge/domain decisions |
| 16N+80R | Complete new-row emission, not a net emission difference |
| 8N+32R | Late logical-to-physical binding and final owned CSR copy |
| 12T | Bounded exact producer L1 |

The earlier G32Q proposal did not explicitly certify the stored M/F/sign
fields; it is withdrawn. G48Q contains 24Q derivation/stores plus 24Q stored
read-back certification against independently proved integer numerators.
Only then is a duplicate J32Q signature compiler unnecessary.

J_math independently rebuilds the support and selected/residual recipes,
original C8 powers and **original deterministic direct-row gauge**, full
direct native/no-radix eligibility, four-limb auxiliary L1, odd denominators,
512-bit bounds, semantic powers and redundant boxes. It does not reuse a
producer row-planning or normalization helper. It may share only the fresh,
private, already certified immutable source/kernel signatures.

J_final reads the final actual CSR, comparing every coefficient's complete
64-bit word and actual column, full row population/order/roles, positive
pivots, original UID/MAIN mapping, gauges, powers, denominator, RHS and Ab.
The 64R part is row metadata; 32N is complete per-coefficient comparison and
associated native checks. Complete inverse-record tuple checking is separate.

I covers actual old/new owner updates, full inverse/UID/block records, the
enlarged forest frame/range checks, deferred handles and partitioned producer
counters. It does not include the separately paid 8N+32R final binding twice.
More precisely, use 4(D+N) incidences, 64A+32O+64B inverse/route/block records,
7A forest enlargement, 3A owner-range checking, and 8W for the one affected
node's ordered skip/mask, with the structural guard W<=16O. The remaining
budget is at least 22A+32O+65536-64B for bounded deferred handles/counters.
There is no new whole-MAIN scan, full graph copy, extra D scan or second N
packet. Dispatch once per node; G owns the only late binding/CSR copy.
The actual saved operator width is 25088 (shape [128,14,14]), not 32768; the
latter is only its safe padded grid upper. With O=2048 the guard holds.
This schedule must be realized literally by the new source sibling; an
implementation exceeding a category rejects, not borrows capacity.

## 3. Generation accounting on the saved structural census

The C130 full-source scalar base, full coupled reserve 46,305,837 and all 52
route calls give 166,771,129 whole / 119,850,773 branch before any birth credit.
Fresh mask packing costs 5,603,328 whole; complete max-parent placement gives
2,195,456 branch, including the 65,536 fixed header. Every original parent path
must be recomputed at runtime; the saved maximizing path is not authority.

Only complete omission of the old direct path earns `8D`, and the new C8
producer above is explicitly paid. There is no 12D emission credit, no 22O
refund and no deletion of the original coupled reserve.

| Saved label | S | N | R | D | T | A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 17 | 283948 | 7266 | 339584 | 83274 | 5218 |
| 5 | 26 | 285608 | 7470 | 382720 | 110842 | 5422 |
| 6 | 24 | 232768 | 7303 | 321536 | 88633 | 5255 |
| 9 | 28 | 319390 | 7585 | 382208 | 124925 | 5537 |
| 10 | 23 | 229772 | 7254 | 314240 | 81974 | 5206 |

All have C=K=128, O=2048. Generation includes F+G+J_math+I:

| Saved label | Whole | Maximum branch | J_final, paid offline |
| --- | ---: | ---: | ---: |
| 1 | 244179889 | 193851661 | 9657856 |
| 5 | 254745361 | 204417133 | 9724032 |
| 6 | 247872117 | 197543889 | 8022464 |
| 9 | 258667565 | 208339337 | 10812416 |
| 10 | 246332257 | 196004029 | 7923456 |

The complete global-selection / routing-evidence allowance is **1999600**:

| Category | Bound |
| --- | ---: |
| All numerical evidence encoding, including complete archive hash | 1110912 |
| C62 numeric headers and owner layout | 214656 |
| Complete metadata | 96000 |
| Full before/after numerical fingerprints | 140784 |
| Full compact report and ledger | 262144 |
| All 52 candidates and complete all-parent DAG selector | 175104 |

Its fixed population is 169 owned arrays / 69368 entries / 454896 payload
bytes: all 156 C122 arrays, 12 full packed input/output/position arrays, and
one int64[52,16] resource table. With fixed names <=64 ASCII bytes and 128-byte
NPY headers, full NPZ size is <=514406 bytes. Header H<=12000 includes all
52 complete records, every array and all 36 shared graph-cost/parent records.
Compact complete report and ledger caps are 196608 and 65536 bytes.
No redundant freeze/source dictionary is inserted into this routing stage.

The 16 resource columns are S,A,R,N,D,T,O,F,G,J_math,J_final,I,whole,branch,
offline and conditional packet-byte saving; full original C122 reports retain
the remaining emission/entry/auxiliary facts. Selection visits every candidate
and recomputes all source paths under one deterministic resource rule; it is
not a search over candidate subsets. All 52 routes, not just five positives,
are freshly computed and saved. Nonchosen routes retain their **conditional**
dense/native flags: this does not pretend that every original kernel was
newly numerically certified by the packed masks. The selected kernel/source
alone receives the fresh complete F check, and failure closes the attempt.

Adding this allowance to both generation bases leaves:

| Saved label | Complete generation whole | Complete generation branch |
| --- | ---: | ---: |
| 6 | 249871717 | 199543489 |
| 10 | 248331857 | 198003629 |

Only one full kernel preparation is assumed. Post-selection failure does not
get another free preparation; multi-tile reuse is not assumed. Old and new
auxiliary/positive-entry/all-emission usage share the original reserves. Any
unknown original usage must be bounded or checked before consuming the
reserved remainder; historical 28-radix observations are not runtime authority.

## 4. Why full original C9 comparison is not omitted

The original C65 source proof prepays `2*original_predicate_nnz + 8*rows` for
full original row data/index equality and row headers. On replaced original
rows, that same 2D equality compares actual C9 with the independently rebuilt
**original** recipe, not C9 directly with a mathematically different mixed row.
J_final separately compares all actual mixed rows.

The fresh J_math witnesses include original C8 unit, original deterministic
gauge and original no-radix status for every replaced output. Offline must
match actual C9 maps, gauge, pivot, RHS, Ab and row count to those witnesses.
Only after that match can exact literal equality transfer the already proved
native endpoint/window property. The original gauge is not the mixed gauge.

A proposed extra composite 10D covers source-word and parent/power gathers,
expected original-bit construction and row-sized frontier stores. Together
with the retained old 2D this is **12D total**, not a newly asserted 8D pass.
For the actual source-bound ImplicitConv2DOp, the privately certified source
snapshot is float64, as required by its original `_numeric_array` constructor.
For each nonpivot coefficient its signed int64 expected bit word is

```
(original_source_bits XOR INT64_MIN) + ((parent_power + original_gauge) << 52)
```

The original normal-binary32 floor lies in [-126,127] and the proved native
floor in [-20,40], so the shift is in [-147,166]. The shifted integer and final
sign-preserving sum stay inside signed int64; this is not unchecked wraparound.
The positive output pivot is handled separately and does not flip sign.
Full actual column/bit equality remains. Standalone float32 factory input must
never be reinterpreted as int64; it is not the source-bound operator contract.

## 5. Complete offline source/proof/publication bound

The original audit upper is rederived from source dimensions, full original
source products/sorts/gauges, conservative owner-delta population and decoder
bounds: **103157044**, not the historical measured 99835284. It retains every
original row, source graph array, UID, alias, inequality, binary and inverse.
Input fingerprints cost 88431792 for all 44214872 original C9/C31 entries.
Candidate fingerprint/export is bounded by strict candidate entries below the
old C31 20974618-entry state. All original inputs stay held.

The complete fixed original-audit/input/publication portion is 233588402.
With `E=27KC+46KS+C+S`, the new complete offline formula is:

```
W = 233588402
  + 8A
  + J_final
  + 8(D+N) + 16(254898+A)      # actual owner incidences AND full width
  + 10D                       # extra original recipe; old2D stays paid
  + 64A + 32O + 64             # complete inverse/output/block records
  + 2E + 17408                # full kernel evidence fingerprint/publication
  + 926464                    # complete schema/header/metadata/report bound
```

The evidence term applies only to all eight freshly proved, private, immutable,
owned arrays: one full E fingerprint and one existing-composite E protocol-5
exclusive publication. There is no extra 8E snapshot or unpriced mutable
receipt. Full original-source before/after fingerprints remain. Header,
layout, metadata, serialization, exclusive close/sync/hash binding are paid.
Fresh restored read-only buffers still incur 9E copy/bit-check plus E restored
fingerprint in the existing separate restore stage; 2E is not a roundtrip bill.

| Saved label | Complete offline W | Remaining under 256M |
| --- | ---: | ---: |
| 1 | 258262596 | -2262596 |
| 5 | 259242454 | -3242454 |
| 6 | 255978602 | 21398 |
| 9 | 260625554 | -4625554 |
| 10 | 255708208 | 291792 |

Only labels 6 and 10 fit this conditional joint source/offline design. They
are not runtime admissions. There is especially little offline slack; every
listed schema and lifecycle premise must be enforced before accepting work.
Their corresponding complete actual-proof branch bounds, excluding publication,
are 123435708 and 123177092, below the unchanged 200M branch limit.

### Header derivation, not a historical measured allowance

The compact schema keeps complete original C65 fields/report/proof, original
source dimensions, `auxiliary_records[A,8]`, `output_routes[O,3]`, and complete
block records. No coefficient-level Python object tree is introduced.

- C9 numeric-header H<=3072: complete 21-field frame, <=128 prefix HZ entries,
  36 source nodes, full <=1024-header report, <=128-scalar identity record and
  three-key original provenance. All original fields remain included.
- C31 numeric-header H<=1024: complete original report and 16 fields/proof.
- Candidate H<=6144: complete original report, complete 36-node C130 proof,
  deferred counter records, mixed summary/descriptors and full proof wrappers.
- Five complete C62 numeric walks: total H<=27696 and registered roots <=768,
  costing 546560. Unsupported schema/shape closes the candidate; no root skips.
- Two full non-opaque metadata walks: H_old<=6144 and H_new<=11264, costing
  139264. This includes expression/operator fields, CSR bodies and ndarray bases.
- Complete eight-array diagnostic layout: 11264.
- Complete compact proof/report serialization: 196608, allowing both report
  passes, full selected/source proof and manifests. The complete 36-node
  certificate upper is 59037 compact JSON bytes; original report <=16384 bytes.
- Fixed schema/closure/retirement handling: 32768.

Sum: **926464**. These are explicit prospective schema bounds and prepaid
guards, not the historical approximately 154K measurement. The implementation
must enumerate/validate every allowed field and pay the check before walking.

## 6. Single-row workspace, with all original graphs retained

Use `B=9C+37` (1189 here), covering even the complete original direct C8 row.
The producer-only plan is retired before allocating the independent plan.
Complete J_math finishes before packet allocation. Emission and final actual
comparison each use a single bounded row frontier, preserving every row and
coefficient; this is not a smaller verification population.

The additional local entry upper, including all eight factory arrays, is

```
E + 4Q + 6P + 2U + (20R+1) + (2N+10R+C+2) + (14B+64).
```

Terms are respectively factory, signatures, independent templates, maps,
mathematical/original-gauge witnesses, full owned packet and bounded scratch.
No full-N term-descriptor or comparison workspace is required.

| Saved label | Local peak entries | Local peak bytes |
| --- | ---: | ---: |
| 1 | 2550790 | 12140104 |
| 5 | 3054087 | 14015908 |
| 6 | 2833651 | 12950916 |
| 9 | 3234847 | 14844300 |
| 10 | 2771316 | 12704752 |

All original C9/C31 states, source graphs, actual candidate CSR, owner/inverse
roots and publication buffers remain in the complete union. This local bound
does not by itself prove 64M entries or either 1 GiB process measure. It avoids
the earlier unnecessary whole-N scratch envelope, not the actual final CSR.
The full ledger and both unchanged RSS/trace gates remain mandatory.

A more detailed, still unimplemented offline schedule now has a finite bound:

| Offline phase | Label 6 entries | Label 10 entries |
| --- | ---: | ---: |
| J_final with every private row-proof array retained | 61128403 | 61066068 |
| C65 source-map construction | 63334016 | 63328127 |
| Complete C65 original row proof | 63016460 | 63010571 |
| Independent complete owner incidence | 62007642 | 61972079 |

This requires a NEW lifetime-only source-audit implementation: after J_final
has consumed every actual row/recipe, retire only its private packet,
signatures and incidence plans; retain all eight factory arrays and all
original direct unit/gauge/no-radix witnesses. Original C9/C31 roots and both
source graphs remain throughout. Retire header/row-local arrays only at their
proved last use, and perform owner incidence after the source audit returns.
No original input or proof population is removed. Source_maps contributes at
most 4449288 temporary entries; the full row-loop envelope is
`5H+M+6EQ+24B=4131732`, with H=254898, M=243162, EQ=244340, B=47835.

The new audit also needs a same-width unsigned view for **full bit equality**:
compare each complete uint64/uint32/uint8 word rather than allocating one
boolean per byte. Signed-zero and all bit patterns are still checked; old
comparison fees are retained. Do not edit the frozen old helper or call old
C91 audit wholesale, which would allocate a large contiguous comparison and
require an unnecessary extra old HZ. Its full owner/inverse theorem is reused
only inside the new complete source-plus-mixed-row proof partition.

The historical 56056008-entry union is post-construction, not a bound on every
generation temporary. The old encoder retains owned rows while assembling
final CSR. Keep the original C31/C97 64M complete retained-root checks and both
1 GiB transient gates; do not claim a stronger all-transient-entry theorem or
a measured RSS/trace pass from these tables. Factory scratch can finish before
encoder creation, since routing depends on structure, not parent semantic
powers. Exact archive/restore liveness and observed process peaks still need
their original full qualification. CSR index/pointer bytes are never omitted;
their inherited entry convention is data-size-only, not an invented new metric.

## 7. Implementation and execution status

The conditional mathematical design is sufficiently concrete to draft a
default-off new raw-only factory. Such a draft is **unqualified** until its
actual passes, schemas and liveness match these bounds. It does not authorize
real-target execution, a replacement source generator, or a score claim.

Still close before numerical qualification: exact integrated sibling/source/inverse schema;
whole held-state liveness including restoration; same four ordinary controls,
all original nonzero point populations and all inherited 3660 tests plus new
cases, with fresh full worker/category bounds and <=60s collection+execution.
No easier fixture, omitted old solve or altered cap is permitted.

The historical 200M ordinary-source reservation is not an intrinsic obstacle.
For the unchanged C16/K32/H6 fixtures, geometry gives a<=576 active source rows
and E_conv<=73728 incidences. The unchanged source fee formulas are
`whole_base=36a+8E_conv+93752`,
`control_branch_base=48a+12E_conv+104008`, and
`complete_coupled=67a+209268`. Powers and coefficients prove zero radix usage;
the only raw aliases are the same 512 depth-one .75 diagonal rows, with the
same protected .5 consumers. Thus the full source skeleton is <=1264252 on
either work axis, and a **new** preregistration can reserve 2M per unchanged
fresh skeleton (16M for all eight), without repricing or editing C130.
The three new-algorithm fixture bounds are 13557824, 12661952 and 11083960.
Even retaining full numerical snapshots/restore and all-36 Fraction references
gives an approximately 209M feasibility subtotal; actual additional owner,
JSON/schema and integration inventory must still close before a freeze.
The last 3660-test gate took 58.337s; a static cost bound cannot guarantee the
unchanged 60-second wall gate, and no inherited test may be omitted.

Formal1870/2413, all 13 families/every old solve, invalidADV=0, separate E0
61/400, capability four-concurrent >=1.0, all verification prohibitions and
the full replay promotion ladder are unchanged. The full goal remains active.
