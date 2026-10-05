# C102: exact ordinary integer text/page locality pays for the existing HZ path

Registered2026-09-13 on redu-hz/f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac,
production candidate15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75.
C101 is PROGRESS as negative actual-runtime evidence, not a solved gain. It
completed three full measured walks totaling73.556s and timed out240s without
a solver start. It is closed; do not rerun it unchanged or weaken its gates.

## Exact bounded replacement

The C78 native integer loop emits the same C5 path/type/value bytes, but creates
a Python decimal string and pointer-page key per integer occurrence. C102 uses
PyLong_AsLongLongAndOverflow plus ordinary signed decimal formatting in a64B
stack array for machine-word values, and keeps the original interpreter path
for already-supported non-word integers. A one-entry local pointer-page cache
avoids reconstructing a dictionary key when the exact high pointer bits agree.
The still-held private append-only page dictionary owns all cached pages; no
numeric or mutable value-based identity merging, omitted tokens or ownership
changes. All heap workspaces remain Python objects visible to the same tracer.
Additional fixed stack state is one64B digit array, a pointer and an address
word plus ordinary scalar control fields; no hidden native heap allocation.

The complete collector is copied unchanged except the native module import and
explicit EXTRA work: one unit per possible prefix integer for the added page
control, plus8 units per prefix for local control initialization. Every old C78
8-per-integer,32768 buffer and1024-per-page charge stays. Word formatting replaces
the old string formatting/conversion rather than discounting it. Six added
terminal-boundary markers cost6*64 extra units in that diagnostic pool. The single
shared256M runtime diagnostic pool remains; no source generation price is cut.
All full numeric hashes remain under the separately reported C32 authentication
boundary; repeated byte traffic is NOT represented by unique resident entries
or the token pool, and all CPU work is NOT claimed to fit256M generation.

## Qualification and one actual run

Retain every2556 existing test. Add the original21 complete C5-oracle tests on
the changed collector and6 new-circuit/runtime-boundary tests on its new route:
ALL120files/2583tests collect+execute<=60s. Freeze C/Python/binary/compiler/
header/source/configuration hashes before the full run. Reuse the C78 compiler
workflow; no compiler, serialization or test-framework feature campaign.

Then one<=60s payment diagnostic uses6744304 integer visits/1314088 distinct
ordinary objects, held identically across old-C78 and new-C102 traced walks.
Compare every field against original C5; both RSS-growth and traced-peak plus
metadata1GiB must pass. One shared256M token pool covers both timed arms.
Require old_time/new_time>=1.5 before any new full network run. This synthetic
go/no-go gate is neither original-runtime performance qualification, F-speed
promotion nor neural-verification capability. Counts select the diagnostic
shape only and are not visible to the algorithm. No archived numeric HZ is used.

On success only, execute one fresh original Tiny143 network. Same C100 text-only
proof reuse, fresh C98 construction and C100v2/C99 native splice, all original
input/affine/property/source/native guards, complete four live-root walks,
full629346312B/52428800-entry comparator and actual native coefficient/bound/
integrality ingestion. Add before/after events around final root measurement,
reference construction and native ingestion; no omitted check or new solver.

Unchanged CPU1/GPU0,AS16GiB,256Mwhole/200Mbranch generation,64M numeric entries,
both1GiB transient, shared16384aux/131072entries/16Mextra work. Native21882407
payload separately paid. Ordinary MILP45s includes base feasibility; total
original worker240s, with no timeout increase, preload, warm/reset, or fallback.
All199 unit/100965 local/7357 circuit inverse relations and1350 binaries remain.
Save any actual point before complete inverse and concrete original-model/
property replay; invalid ADV zero. Full terminal must RETURN for stage success;
UNKNOWN is evidence only, and any missing gate closes this version.

Exclusive results/c102_word_runtime_20260913_v1 retains logs/results/exit/hashes
automatically. No old source/archive, /data1/Kane/HyZor, production/default or
charter edits; no commit/push. Formal1870/all13/2413 endpoint, E0 CIFAR25/Tiny36
and400 retention replay, later structural/generalization/PLDI objectives remain
ACTIVE. No attack/PGD/BaB/split/backward/dual or LP-status/marginal/ray rescue,
binary pivot, convex replacement or identity menu. If the measured hot-loop
change fails, stop this payment route and return to actual affine/predicate
generation; do not grow an unrelated diagnostic framework.
