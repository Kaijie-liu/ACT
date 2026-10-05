# C128 preflight — exact conditional whole and longest-branch source bounds

Disposition: prospective accounting only. A new dense-support/packed-owner
engine is under implementation, not qualified by this document. The saved
complete graph permits an exact nodewise longest-branch recomputation; its
branch saving is28926816, NOT the whole-graph saving70554624. A combined real
source/birth path is still not admitted, even if the proposed graph primitive
later qualifies. No numerical experiment is run for this preflight.

All observations below use code reads, authenticated saved scalar JSON, integer
arithmetic and hashes. No model, NPZ numerical payload or archived HZ was loaded,
and no source restoration, solver call, test or numerical proof was executed.

## Authenticated inputs and scope

Paths are relative to `experiments/neural_hz_20260831`.

| Saved input | SHA256 |
| --- | --- |
| `results/c117_affine_block_census_20260922_v1/complete_census.json` | `b67c962b181259fe9f82e368138d01610116abb17c684aec544fa231cec42075` |
| `results/c97_once_power_20260913_v1/preflight/result.json` | `68039db23422bf8f6878e18a40a5f79f5a35b50694f550c94e54ad01f189713a` |
| `results/c97_once_power_20260913_v1/actual/result.json` | `74fa0e9b8ba9895a95638b4737ac9ce65adc130b96ef234f884606b4589f41bc` |
| `results/c127_systematic_kernel_20260927_v2/result.json` | `519ca415a32b406be523779519bee8d21b0288bfd911130604e9437b2bbb8bea` |

C117's census hash appears directly in C127 v2's frozen dependency manifest.
The two C97 hashes match their inherited historical checkpoint seal, whose
SHA256 is `5e1ce6d6c315ccf93cec98f75d98e3c301209d89c53916b748dc9eea27cdc91e`;
that seal is itself frozen in the qualified dependency chain. C127 v2's result
matches its terminal exit artifact hash. These are historical inputs, not a
new source-authentication or runtime-custody certificate.

The census contains all36 node records, every topological parent list, root35,
complete operator geometry and original count/support work. Every node's kind,
width, auxiliary count, continuous/binary/center edges and support work agrees
with the saved C97 actual source report. Thus parent or path data are NOT
missing for this scalar recurrence. Current fresh semantic powers, emitted
rows, alias decisions and runtime ownership remain separate obligations.

## Final proposed fresh-call tariff

The [C127 handoff](C127_DENSE_SUPPORT_HANDOFF_20260927.md) proposed
`Q=1024+4W+8(NI+NO)+16E`. C128's design explicitly adds the axis-planning work:

```
Q' = 1024 + 4W + 16(KH+KW) + 8(NI+NO) + 16E
E  = B*G * sum_kh(valid output heights for kh)
             * sum_kw(valid output widths for kw)
```

W counts ALL kernel entries, NI/NO count complete flattened input/output widths,
and E counts every batch/group/geometrically valid stencil incidence. The
16(KH+KW) term requires scalar interval/clipping planning, not repeated full
arange buffers whose work could be nonzero when E=0. The execution must prepay
its header/axis work before that planning, then prepay density/vector/stencil
work before scans or allocation. Density inspection is fresh, never borrowed
from a mutable flag or a historical hash.

Forward channel reduction, spatial gather/add and broadcast/full row mask;
reverse full packed-label row mask, channel reduction, transpose scatter and
broadcast; all temporaries and complete returned vectors must fit this new
program's tariffs and resource proof. Sparse kernels retain the exact old V2
loop price plus fresh inspection; CSR behavior is unchanged. Inherited zero
input/label shortcuts and exact cache-key/source mutation checks remain. The
dispatch is structural, not selected separately for saved node IDs.

For the seven nonempty dense nodes, both fresh forward and packed-reverse calls
are required. The unchanged graph wrapper is2NO+4NI. The final needed-count
pass uses the original exact forward cache, not a new prepared-kernel cache.

| Node | NI | NO | W | E | Q' | Old support | New node upper | Saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 25088 | 25088 | 147456 | 1600 | 1017952 | 14334336 | 2186432 | 12147904 |
| 16 | 25088 | 25088 | 147456 | 1600 | 1017952 | 27181184 | 2186432 | 24994752 |
| 19 | 25088 | 25088 | 147456 | 1600 | 1017952 | 30144128 | 2186432 | 27957696 |
| 22 | 46656 | 25088 | 8192 | 196 | 610912 | 2471040 | 1458624 | 1012416 |
| 25 | 25088 | 6272 | 147456 | 400 | 848224 | 545152 | 1809344 | -1264192 |
| 30 | 6272 | 6272 | 147456 | 361 | 697072 | 6168704 | 1431776 | 4736928 |
| 32 | 25088 | 6272 | 16384 | 49 | 318256 | 1718528 | 749408 | 969120 |

Node25's regression is included; there is no per-node cost menu. Empty-support
Conv nodes1/4/7 retain6272 each and node13 retains25088. All other support work
is unchanged. The resulting graph-support upper is:

```
86963144 - 82563072 + 12008448 = 16408520
whole support saving = 70554624
```

This is1088 more work than the original handoff's Q-only projection. The
primitive implementation and complete qualification must establish the tariff;
this table is not a measured new graph execution.

## Exact conservative longest-branch recurrence

C97 keeps its ORIGINAL conservative branch encoding price, even though its
whole price removes proven duplicate coefficient operations. For each node i:

```
encoding_i = 12*(continuous_edges_i + binary_edges_i
                + center_edges_i + auxiliaries_i)
P_i = encoding_i + support_i + max(P_parent for parent in parents_i)
max(empty parents) = 0
branch_base = max_i(P_i) + 12*old_predicate_entries
```

The saved graph reconstructs the original maximum116456236 exactly. The saved
C97 branch base124863436 fixes the unchanged predicate term8407200, corresponding
to700600 old predicate entries. Replacing ONLY the seven node support debits
above and rerunning all36 recurrences gives the following complete path table.
Values exclude the shared old predicate term, which is added only once.

| Node | Parents | Original P | Revised P |
| --- | --- | ---: | ---: |
| 0 | none | 6272 | 6272 |
| 1 | 0 | 12544 | 12544 |
| 2 | 1 | 18816 | 18816 |
| 3 | none | 6272 | 6272 |
| 4 | 3 | 12544 | 12544 |
| 5 | 4 | 18816 | 18816 |
| 6 | none | 6272 | 6272 |
| 7 | 6 | 12544 | 12544 |
| 8 | 7 | 18816 | 18816 |
| 9 | none | 456848 | 456848 |
| 10 | 9 | 15290384 | 3142480 |
| 11 | 10 | 15816720 | 3668816 |
| 12 | none | 25088 | 25088 |
| 13 | 12 | 50176 | 50176 |
| 14 | 13 | 75264 | 75264 |
| 15 | none | 13726208 | 13726208 |
| 16 | 15 | 53889664 | 28894912 |
| 17 | 16 | 54686336 | 29691584 |
| 18 | none | 6037820 | 6037820 |
| 19 | 18 | 84315580 | 56357884 |
| 20 | 19 | 85112252 | 57154556 |
| 21 | none | 1731924 | 1731924 |
| 22 | 21 | 12148692 | 11136276 |
| 23 | 22 | 12945364 | 11932948 |
| 24 | 11,14,17,20,23 | 86943676 | 58985980 |
| 25 | 24 | 89096972 | 62403468 |
| 26 | 25 | 89150124 | 62456620 |
| 27 | 26 | 89190980 | 62497476 |
| 28 | none | 1240772 | 1240772 |
| 29 | 27,28 | 89265564 | 62572060 |
| 30 | 29 | 98106908 | 66676476 |
| 31 | 30 | 98307612 | 66877180 |
| 32 | 24 | 98371260 | 69444444 |
| 33 | 32 | 98571964 | 69645148 |
| 34 | 2,5,8,31,33 | 98866748 | 69939932 |
| 35 | 34 | 116456236 | 87529420 |

The maximizing path remains18→19→20→24→32→33→34→35. Its changed nodes19/32
save27957696+969120=28926816. The competing node31 path includes node25's cost
regression and is fully recomputed, not discarded. Therefore:

```
new branch base = 87529420 + 8407200 = 95936620
new whole base = 176407180 - 70554624 = 105852556
```

Adding the UNCHANGED complete conservative C97 coupled upper46305837 yields
prospective source-only bounds152158393 whole /142242457 branch. Their capacity
remainders are103841607 /57757543. This is a conditional same-source tariff
projection, NOT current source-generation qualification; no existing power,
predicate, quotient, radix, inverse or ownership debit was discounted.

## Why the combined birth path is still not admitted

The ordinary-qualified C127 systematic kernel fee at K128/C128 is55148544;
unchanged C120 constructor preparation costs45350912. Together they cost
100499456 BEFORE any new non-kernel constructor or native proof operation.

For the most generous saved single node19 tile, D=382720. Grant the entire8D
whole /12D branch original direct-encoding removal, even though a future birth
program must prove which operations truly disappear. Recomputing the credited
branch still leaves the same maximizing path. Then:

| Conditional subtotal | Whole | Branch |
| --- | ---: | ---: |
| Revised base minus8D/12D plus both full kernel preparations; ALL coupled work omitted | 203290252 | 191843436 |
| Same subtotal with unchanged coupled46305837 retained | 249596089 | 238149273 |

The retained-coupled branch already exceeds200M by38149273 before every new
non-kernel row/support/power/native/source-proof/physical/ledger/evidence cost.
The whole subtotal leaves only6403911 before those costs. Consequently merely
combining the new graph engine with both current preparations does not yet
authorize a real source-birth experiment. The graph change removes real work,
but any further removal from coupled ownership/quotient/assembly groups needs
a genuinely different program and a complete renewed proof, not a fee waiver.

C123's later40–47M binder was not in the C97 source base and cannot be deducted
again. Historical ledgers or measurements do not pay current runtime roots.
Original semantic powers, MAIN/radix/V/M slots, UID ranges, exact owner retagging,
alias choice, inverse reconstruction, source custody and all positive-entry/
emission reserves still require their original or newly proved implementations.
Unknown integration terms remain UNKNOWN, never zero.

## Ordinary fixture qualification and prospective archive budget

The planned isolated worker keeps ALL four original C16/K32/h6 source fixtures.
It builds both unchanged and new-graph complete C97 sources, compares every
actual source array plus graph/UID/owner/semantic scalar fields, checks complete
ownership/inverse points and saves all source numerical evidence. At this
geometry one fresh dense Q' call is30560; two calls plus the graph wrapper cost
64448. Zero/no-hit decisions must follow actual support/needed masks, not the
fixture name. These kernel fees do not stand in for complete source work.

The initial source200M/setup2M/comparison8M/proof4M/evidence16M/ledger24M
proposal was insufficient: the complete source proof needs4802560 and numeric
archive encoding alone now needs16681824. Before any freeze/run the replacement
partition is source200M=(32M+32M+32M+4M)*2, setup2M, comparison10M, proof6M,
evidence20M and ledger16M. Sum254M and global256M are unchanged. Nothing is
funded by smaller actual source counters or an after-result category transfer.

The now-written `c128_complete_source_v1.packet` contains48 arrays per source
version:9 field arrays,3 HZ vectors,18 HZ CSR payload arrays,16 complete node
support/needed/slot/power arrays and2 physical UID arrays. Original source
payloads contribute29 more arrays per fixture and full independent old/new
owner oracles contribute2, giving127 archived arrays per fixture. Four separate
numeric stage archives per fixture preserve this complete inventory. The scalar
inventory follows directly from the fixed source expression and authenticated
C127 complete-source counts:

| Fixture | Surviving source coordinates s | Direct Conv nnz D | Entries per source version | Original payload entries | Both owner-oracle entries | Full NPZ entries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 576 | 74240 | 178267 | 12246 | 4224 | 373004 |
| Historical masked | 432 | 55808 | 139532 | 11670 | 3936 | 294670 |
| Heterogeneous | 436 | 55936 | 139840 | 11686 | 3944 | 295310 |
| No-hit | 16 | 5120 | 32748 | 10006 | 3104 | 78606 |

Derivation: MAIN=s+1536, logical/physical continuous count=s+2112, EQ count
q=s+1025, Ac nnz=D+2s+1025, Ab nnz=s+1, Gc nnz512, Gb nnz0, Auc/Aub nnz1
each, one INEQ and no radix auxiliaries. The complete four-node widths sum2112.
Dense UID slabs count1; other fixtures count2. Hence a source packet contains
`22298+13s+2D+slabs` entries, source payloads contain`9942+4s`, and both owner
oracles contain`2(s+1536)`. This includes EVERY CSR indices/indptr element,
which C62's numeric-entry convention excludes while counting their bytes.

The per-version packet byte counts1057568/824296/826168/181096 also reconcile
with the saved C127 original-source identity after adding its original source
payload, explicit maps and outer keep. Masked/heterogeneous retain3456/3360
additional bytes of unused CSR backing capacity after zero elimination; C62
must still count that reachable capacity. NPZ payment uses every logical array
element actually serialized, not a falsely converted C62 entry count.

The final preregistration design saves FOUR stage NPZ files per fixture:
29 original input arrays before source generation,48 complete old-source arrays
before the new build,48 complete new-source arrays before comparison, then2
complete independent owner-oracle arrays. These16 files still contain1041590
total numeric entries; no semantic array is dropped or redundantly reserialized.
Their exact encoding charge is`16*1041590+16*1024=16681824`.

Each of the16 stages also prepays a65536-byte JSON artifact/manifest allowance,
adding1048576. Four full point JSON allowances262144 and four full source JSON
allowances131072 add1572864. Complete evidence reservation is therefore
`16681824+1048576+1572864=19303264<20M`, leaving696736 in that category. Relative
to one NPZ per fixture, the extra12 archive headers cost12288 and the16 new
stage JSON allowances cost1048576; neither is silently paid by another category.

There are24 evidence JSON receipts plus the separately prepaid complete ledger
receipt,25 in total; terminal result is additionally checked against its own
allowance. Success/failure/ledger JSON allowances524288/65536/131072 stay in the
ledger category and must fit actual exact bytes+1024.

Saving a completed stage does NOT qualify it: stage artifacts must explicitly
remain unproved/unadmitted until complete later checks succeed. Original inputs
and the full old/new source populations are published before comparison can
fail; any later rejection preserves already completed stage files instead of
discarding the control/candidate evidence. This does not promise that an
in-progress builder which raises before returning can expose a complete result,
nor convert incomplete or failed evidence into successful qualification.

Complete scalar/array comparison pays`4*16384+8*980774=7911728<10M`; the fixed
header part must precede packet/header/manifest materialization. Complete owner
and inverse proof pays for BOTH versions:

```
2 * sum_cases(1024 + 8*(Ac.nnz+Auc.nnz)
                  + 32*(n_eq+n_ineq) + 64*n_cont) = 4802560 < 6M
```

Current three original expression bindings per fixture plus fixture creation
reserves total1083872<2M. Source payload geometry/shapes/sharing and origin
binding must also be retained portably; any added work must be explicitly paid.
Complete held-root metadata/fingerprints/reporting still needs its actual
bounded ledger measurement; no saved historical measurement substitutes for it.
These figures are static preflight predictions, not new numeric run results.

No cheaper geometry, omitted old source control, shortened proof or borrowed
category capacity is permitted. All later inventory/schema additions must be
bounded before freezing and any failure must close the registered version.

All3584 qualified inherited tests and complete60s collection/execution gate
remain. C127 v2 used56.39011875540018s, an observed margin of only3.61s, not a
guarantee for the next run. New independent explicit-incidence tests should be
bounded ordinary cases while still covering full integer counts/packed labels,
groups/batches, row masks, stride/padding/dilation, sparse fallback, mutation,
zero paths, overflow/prepayment and complete graph identity. They cannot omit
required coverage merely to fit a time cap.

All64M-entry/BOTH1GiB, signed-word and UID-domain guards remain. The new engine
must prove nonnegative reduction/fan-in/fan-out bounds before int64 arithmetic,
preserve owned read-only result vectors and inherited complete cache limits,
and account for transient group/stencil/mask buffers. This preflight grants no
prepared cache, native dispatch, LIVE, solver, timing, capability or score gain.
Formal1870/2413 and independent E0 CIFAR25/Tiny36=61/400 remain unchanged; all13
families/every old solve stay protected, with production/default/archives intact.
