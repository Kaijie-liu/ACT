# Full native materialization and retained-owner ledger checkpoint

2026-09-05, `redu-hz`, base
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`. The preceding goal turn made
progress. All four previous checkpoint manifests verified before and after
this continuation. No production/default source, historical archive, authority
or score ledger changed. Formal **1870/2413**, E0 **61/400**, TLL capability
**27/32** remain unchanged. Goal active; no completion or blocked claim.

## The actual native workload is now tested, not only its probe

The sealed ADD16 source and the native masked Conv17/SCALE18/BIAS19 preparation
are unchanged. The native 318-row probe rejects selective materialization:
7711 positive generator nnz is below its unchanged 25875 threshold. The
existing native continuation therefore requests all 1097 non-stable-negative
rows (310 U + 787 P). Both requests now execute with the same frozen C5-v3
compiler. The budget wrapper deducts work across stages BEFORE compiling the
next branch; no branch or stage resets the frozen caps.

| Stage | Source9 channel products | Source5 channel products | Total |
|---|---:|---:|---:|
| 318-row probe | 29376000 | 2767360 | 32143360 |
| 1097-row native follow-up | 45209216 | 3286528 | 48495744 |
| Complete sequence | 74585216 | 6053888 | **80639104** |

Both branch-sequence totals are below 200M, their sum below 256M, and every
stage/branch retains its quarter-work condition. The corresponding complete
unrestricted spatial channel-product count is 1702961152. The roughly 21.1x
ratio is a COUNTED WORK ratio, not a verifier speedup. Scaling products,
additions, enumeration work, cache bytes and emitted CSR sizes remain separate
fields in every stage's raw evidence.

ALL retained operator coefficients and ALL HZ fields match the unchanged
source-restricted scalar row oracle byte-for-byte for BOTH stages. The toy
suite also covers the unfiltered source-column oracle. No tolerance, clipping,
predicate deletion, phase/latent projection or witness repair is used. The
source hashes stay unchanged and the resulting HZ retains 25088 output rows,
11156 continuous factors, 874 binary factors, 874 equalities, 1748 inequalities
and frame 1. Bitwise implementation agreement is not a universal NN rounding
certificate or a terminal verdict.

First full-sequence evidence:
`evidence/c5_full_native_transaction_20260905_v1.json`, SHA-256
`d25080c9318c3ffadc7c0735ebb29f3939f21769503be53e392e8e75d0a6ba0d`.
The candidate stages took 1.455 and 3.251 seconds; scalar test oracles took
22.640 and 58.073 seconds. The supervisor finished at 100.785 seconds, exit 0,
no source/provenance drift. These are diagnostic, not controlled speed gates.

## Storage measurement: two real ownership issues are explicitly handled

Read-only inspection established that the saved graph has a LabeledInputTensor
with exactly tensor/label fields. It also established that protocol-5 loaded
NumPy arrays retain memoryview(bytearray) owners. The new schema adapter
traverses registered LabeledInputTensor/InputSpec/OutputSpec fields, preserves
aliases and rejects unknown classes, extra/missing fields and cycles. The new
buffer visitor accepts only the checked contiguous bytearray-backed form and
charges the full backing allocation, never an array copy.

That adapter successfully measured entry, then the first full diagnostic
correctly rejected completed probe Gc.data: SciPy retained a larger native
allocation behind a shorter CSR data view. The first diagnostic's failure is
preserved as `csr_data_not_full_owner:active['probe'].value.Gc.data`.

A separate V3 visitor handles this real allocation pattern without changing
the metric or copying/compacting source/result data: physical bytes charge the
FULL owner; stored CSR entries count the union of active data spans. Exact
aliases deduplicate, disjoint spans add; partial overlaps, gapped views,
dtype conflicts and incompatible dense/index/data aliases still reject.
The original and V2 visitors are unchanged and still reject the unsupported
short-data-owner case in their tests.

During initial V3 test development, 7 tests failed because the SciPy test
constructor copied the intended short views despite copy=False. The fixture
was corrected to install and assert the public data-view identity; no expected
metric or rejection threshold changed. See the separate development note.
The final source-frozen suite passes **73 tests**, including all 26 original
ledger tests, 21 ordered compiler tests, schema/buffer/sequence tests and nine
partial-owner tests. This is not a full repository or 2413-case replay.

## All four registered retained-state boundaries pass

The accounting-only repetition keeps the same native requests, compiler,
budget and BOTH complete scalar-oracle comparisons. It only changes the
measurement adapter. All numerical checks pass again. No solver or verdict
retry occurs.

The representation comparator remains the pre-registered
**phase_selective_expanded_v1**. ALL three uniquely reachable Conv operators
are replaced on its side by exact expanded CSR, with every scalar row checked:
Conv10 has 26214400 entries, Conv13 1605632 and masked Conv17 1149440.
The total 28969472 is below the 64M preflight cap. Source, predicate, bias,
operator sharing and cache roots are preserved. There are 163 network numeric
parameter roots. No consumer GC is applied on either side.

| Registered boundary | Candidate bytes | Expanded comparator bytes | Candidate entries | Comparator entries |
|---|---:|---:|---:|---:|
| Entry | 64805800 | 410290612 | 7470548 | 36111828 |
| Completed probe | 71403704 | 416888516 | 7938545 | 36579825 |
| Completed follow-up | 86220836 | 431705648 | 8997603 | 37638883 |
| Both results retained (conservative) | 92818740 | 438303552 | 9465600 | 38106880 |

Every boundary is strictly smaller in bytes AND entries. The constant byte
difference is 345484812. The last boundary deliberately retains both results
as an over-retention stress case; the native caller need not retain a rejected
probe. These numbers describe explicitly registered LOADED-SNAPSHOT owners,
not the original worker heap, Python metadata or construction peak.

Importantly, this is the COMBINED implicit representation against its frozen
expanded comparator. It does not prove C5-v3 alone shrinks persistent HZ state
relative to Trial9's already-implicit implementation. The new compiler's
independent contribution is ordered source-support/coefficient work reuse.

Final evidence: `evidence/c5_full_native_owner_ledger_20260905_v2.json`, SHA-256
`add3dd4f63ac5241d97245cbbed695d9cf4be4233f874282f001b10f85a4fc6a`.
Its candidate stages took 1.437 and 1.990 seconds; oracles took 16.567 and
41.960 seconds. Fixed-order one-off measurements do not establish speed gates.
Peak RSS including test/expanded oracles was 1406136 KiB (not candidate-only).
The supervisor finished at 71.701 seconds, exit 0, no source/provenance drift.

## Next gate and non-claims

The next qualification is a LIVE transaction at the same native boundary:
register original source/predicate/frame/slot metadata, before/after Facts,
Layer caches and every relevant transfer-function cache/strong root; account
construction transients and test publication/rollback without dropping roots
or substituting serialized ownership. The completed numerical/retained-state
evidence justifies implementing that adapter, not silently skipping it.

Only after those gates should the isolated rule be connected to an integrated
prefix attempt toward ReLU20 and the fixed ReLU36/63/71 progression, then CIFAR
targets, same-structure shadows, corrected BN retention, E0 and full 2413.
This continuation executed NO ReLU transform, cache publication or terminal
solver and completed NO new network layer/prefix/property. Four-concurrent
qualification and the 13-family retention replay remain outstanding.

All experimental processes are terminal and their exclusive logs/exits are
saved. Old prefix timeouts, old ledger failures and old scores remain intact.
