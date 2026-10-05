# C5 fresh ReLU63 prefix: actual state reached, terminal still unproved

This additive trial-log entry follows `C5_RELU36_BOUNDARY_AUDIT_20260905.md`.
It does not rewrite the frozen trial log, previous failed runs or any score.
The goal continuation made progress: complete ReLU36 numerical/physical
boundary qualification, a tested conservative whole-state certificate, and
a fresh unchanged-runtime prefix that really reaches ReLU63. No blocker or
completion status is warranted; the overall 2413/2413 goal remains active.

## Frozen run and actual outcome

Run: `results/c5_integrated_relu63_20260905_v1/`.
Result SHA-256:
`62f33f7aafd320e9804fbc5f22a78fc8b2465552ade32161b7e71de5436a051d`.
73 tests passed, including original exact source-support/witness tests. Worker
and tests exit zero; source/provenance drift false. CPU numerical threads 1,
16 GiB process cap, 240-second wall cap; no changed compiler/work/nnz limits.
Propagation to the deliberate layer-63 stop: 11.749476 seconds; supervisor:
18.159186 seconds. These are prefix timings with observation, not a speed gate.

The independently replayable read-only snapshot audit is
`evidence/c5_relu63_snapshot_audit_20260905_v1.json`, SHA-256
`083d36d41da59e409f3373a2b4cbd5724e5457864981aaac973d0e67587eb760`.
It verifies candidate source hashes before loading local pickles, every
snapshot seal, actual HZ/expression presence, finite numeric data and frame 1.

ReLU63's actual HZ has 11,708 continuous factors, 1,150 binary factors, 1,150
equalities and 2,300 inequalities, exact flag true. Its output Gc/Gb are zero
because all 6,272 rows at this ReLU are stable negative; the nonconvex latent
factors and all predicates are still retained. The target pickle SHA-256 is
`f0328e4f94981011ca1c171adef6209a9ef362ef644ea6a52b15ae66047ed0a5`.
ReLU55 likewise retains this full predicate/factor state with zero output map.

The result is UNKNOWN / missing terminal HZ because terminal layer 79 was
intentionally not executed. ReLU63 is not a new CERT/ADV, and cannot count as
a CIFAR/Tiny or formal solved-case gain.

## Newly exercised branch that needs qualification

At Conv41 / ReLU44, five C5 source terms require a single 165-row native probe
(157 unstable plus the first eight of 124 positive rows). Cumulative channel
products are 26,198,528; all five quarter-work tests pass. Unlike the previous
ReLU20/36 boundaries, this probe is ADMITTED: positive generator nnz 28,354
exceeds the native mask/bias threshold 6,396.

The unchanged native phase-selective code leaves the 124 stable-positive
rows as affine paths and constructs the exact unstable core. Actual ReLU44
state is therefore an affine expression, not a missing state: five masked
source paths plus one same-frame identity core term. The six sources are
finite and exact with frame 1; the core has nc=11,708, nb=1,150, neq=1,150,
nineq=2,300. Its Gc has 102,657 nnz. The isolated `sparse_cached=false` result
field must be read alongside this actual `expr_cache[44]` state.

This local admission reports a 21,958-entry saving, but that is NOT a complete
live physical proof. The ReLU36 rejection/follow-up proof cannot substitute
for this new admitted branch, and its positive-path/core publication has not
yet received an independent real-boundary equivalence audit. The subsequent
empty native frontiers at ReLU55/63 carry 11/12 source terms; no C5 channel
contraction is selected for either. These mixed-length retained paths also
need to be kept in the next full ownership/source audit.

## Exact next work, not a smaller goal

Before advancing to ReLU71, pre-register the Conv41/ReLU44 admitted-phase
boundary: compare the single C5 probe and all five branches against the
original ordered oracle, verify the native positive affine paths plus binary
unstable core and its frame/slot/cache publication, and prove the complete
live metric at the actual post-publication boundary. Retain the zero-output
predicate-bearing states and every shared ancestor; no local-core shortcut.
The previously tested reference-subset certificate may bound the same full
expanded comparator, but it must include the NEW full candidate root set.
Do not force a full follow-up that the native algorithm never requests.

Then continue the fixed ReLU71/terminal, CIFAR166/153, structural guards,
same-family and full-2413 campaign. Formal 1870/2413 and separate E0 61/400
remain unchanged, and candidate defaults remain disabled. The old production
worktree edits are preserved, not modified by this continuation. All three
new supervised runs and the read-only auditor have terminated. Their raw
positive/negative records and linked staging snapshots are retained.
