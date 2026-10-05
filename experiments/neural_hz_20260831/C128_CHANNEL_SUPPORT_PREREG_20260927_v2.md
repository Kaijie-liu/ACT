# C128 v2 — validate the original derived budget diagnostic

Branch `redu-hz`, commit `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`;
production provenance
`15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75`.
Exclusive sole run: `results/c128_channel_support_20260927_v2`.

## Precisely changed comparison program

[C128 v1 preregistration](C128_CHANNEL_SUPPORT_PREREG_20260927.md) is inherited
in full except the explicit version, test inventory, frozen file list and
derived-capacity comparison changes below. Its nine frozen files and failed
run are immutable. [Failure audit](C128_V1_FAILURE_AUDIT_20260927.md) records
3632 passing tests but failed full qualification; v1 is not retroactively passed.
Both completed dense NPZs have the identical full-file hash, but v1 did not
finish its complete array comparison, owner/inverse proof or four-case ledger.

The original C10 WorkPool defines
`capacity = min(max_work - whole_base, max_branch - branch_base)`.
Both limits equal each registered source reservation in these four fixtures.
C64 publishes this diagnostic as `report.alias_quotient.coupled_extra_capacity`.
Less support work legitimately increases it. The old comparison incorrectly
required equality even after allowing the underlying work totals to differ.

NEW `c128_complete_source_v2.py` validates nonnegative integer bases, capacity
and used work; both bases must fit the unchanged source reservation, capacity
must equal that exact original minimum, and used work must fit capacity. Only
then does a comparison copy replace this ONE field with a constant marker.
Original metadata and all saved evidence remain untouched. Every other alias
field, including actual coupled work, still compares equal. Same-named fields
elsewhere are not recursively omitted. No HZ array, incidence, predicate, frame,
binary, UID, sharing, owner or inverse criterion changes.

NEW supervisor independently validates both saved formulas for all eight
builds, including total work = whole base + used and branch work = branch base
+ used. This is an exact test of a derived accounting value, not a cap waiver,
smaller proof population, changed work fee or authorization to omit evidence.

## Sole execution and unchanged full scope

Freeze the entire qualified C127 chain, all frozen C128 v1 sources and all16
failed-v1 artifacts, plus these six new files before collection or execution:

- `c128_complete_source_v2.py`
- `c128_source_worker_v2.py`
- `test_c128_budget_metadata_v2.py`
- `run_c128_source_supervisor_v2.py`
- `C128_CHANNEL_SUPPORT_PREREG_20260927_v2.md`
- `C128_V1_FAILURE_AUDIT_20260927.md`

The factorized engine, graph adapter, birth adapter and original48 tests are
reused unchanged and authenticated through the failed-v1 freeze. Add exactly
four dictionary-only tests: original exact capacity and immutable input,
incorrect capacity/excess actual work rejection, whole/branch/tied bottlenecks,
and complete preservation of all other metadata. Full inventory is
3584 qualified C127 +48 unchanged C128 +4 new =3636 tests/157 files; exact
collection/execution match, zero skips/errors/failures, combined wall<=60s.

Each old/new source is independently freshly built on each SAME dense, masked,
heterogeneous and no-hit C16/K32/h6 fixture. All48 arrays per full source packet,
all29 original source/operator arrays, both complete sparse owner oracles and
every original/recovered rational point are saved. No numeric archive restore.
Full original source/frame/binary/EQ/INEQ/graph/predicate/UID/inverse equality
and aggregate support saving remain mandatory. All completed stages are saved
before subsequent comparisons, exactly as in v1. No source fixture shrinks.

## Unchanged payment and resource gates

Category caps remain source200M/setup2M/comparison10M/proof6M/evidence20M/
ledger16M =254M. Whole256M, branch200M; source reservations per old AND new
build remain32M/32M/32M/4M. The extra scalar formula validation is covered by
the unchanged16384-per-case complete metadata/header reservation. All density,
stencil, original-source, coupled, owner/inverse, ledger and serialization fees
remain identical. No recovered capacity funds another category or fixture.

Predicted full comparison7911728, owner/inverse4802560, setup1083872 and
complete evidence19303264 are unchanged. Sixteen complete NPZs contain1041590
numeric entries;25 full JSON receipts include complete stage/case/point/ledger
evidence. Each actual compact byte string +1024 must fit its unchanged
allowance. Success524288/failure65536/ledger131072 prepaid upfront, stage65536,
case131072 and full points262144. Complete held ledger still must independently
fit16M. All originals, both versions, proofs and evidence remain held and counted.

Same64M numeric entries, BOTH1GiB RSS growth and traced peak + tracer metadata,
AS16GiB/CPU1/GPU0, worker240s, native[2^-20,2^40],512-bit exact bounds and shared
16384 auxiliaries/131072 positive entries/16M ALL new row emission limits.
Fatal-only fault handler. Hash authentication is not falsely claimed covered
by the logical generation counter. No wall-speed or complete allocator claim.

## Disposition

This version gets one frozen complete run. A failure remains failed with every
completed stage retained; no unchanged retry or retroactive reserve increase.
C127 v2 remains ordinary-qualified, C127 v1/C124/C128 v1 remain failed.
[Real-source preflight](C128_SOURCE_BUDGET_PREFLIGHT_20260927.md) still rejects
the optimistic combined branch238149273 BEFORE missing costs. This source
component cannot admit a real target or a combined F4 circuit on its own.

Formal1870/2413=1063CERT+807validatedADV and all13 families/every old solve stay
protected; separate E0 CIFAR25/Tiny36=61/400 unchanged. Full45s BASE,
same-structure shadows, timing gate and2413/400 replays still precede promotion.
No convex replacement, binary pivot, instance menu, attack/PGD/BaB/split/
backward/dual rescue, solver/default/production/commit/push or historical
archive mutation. Broader research goal ACTIVE; formal gain remains0.
