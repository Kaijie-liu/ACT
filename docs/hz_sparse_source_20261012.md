# Sparse enclosure connected to checked MoE templates and block support

The [frozen integration](hz_sparse_source_design_20261012.md) passes all twelve
control groups and a separate saved-evidence audit. The new opt-in path replaces
local row projection with checked sparse enclosure while retaining exact source
checks, once-per-expert propagation and direct common/private block support.
All original domains, objectives and final endpoint bounds match the old path.
There are no new real-model certificates, physical GPU results or performance
claims. G1–G6 remain OPEN.

## What changed, and what did not

[Source construction](../scoped_source/hz_sparse_source.py) uses a new
`CHECKED_SPARSE_HZ_BLOCK_SOURCE_V1` package and request context.
The [independent checker](../scoped_source/check_hz_sparse_source.py) first
checks each exact input, affine, ReLU or conditional-guard reference, then its
sparse enclosure and actual SparseHZono coefficient snapshot. It retains the
ReLU row permutation, shared factor IDs, expert-private activation/error factors
and original binary mapping. Unexpected new factors in a guard are refused.

Each expert is still propagated once from the common input/router domain.
Pair-specific guards do not become global template facts. Endpoint support
still operates directly on common/private blocks, without joint HZ construction,
global sparse remapping or full LP export. All legal pairs and properties remain
obligations. The CPU block candidate algorithm, 128 updates and acceptance
threshold are unchanged; both arms generate fresh candidates.

The new row kernel parses each supplied source/target once per enclosure check.
This does **not** mean the complete network checker parses each state only once:
source equations, ownership, snapshots and block evidence perform their own
checks. `sparse_core_work` counts only the row kernel; the independent archive
audit reconstructs those visits from stored sparse matrices.

Old source limits (128 factors/outputs and 256 total constraints), block limits
and new sixteen-row-per-kind limits intersect. Nothing here raises production,
portable, property or operator admission. The new primitive's 4096-factor limit
is not a network admission limit. This is an experimental source connection,
not a change to production defaults.

## Frozen results

[R2 compact independent audit](hz_sparse_source_20261012_r2.json):

| Fixed declaration | Positive duties, old / new | Endpoints per arm | New checked sparse lifts |
|---|---:|---:|---:|
| weighted_sign | 3/3 / 3/3 | 6 | 14 |
| tied_partial_reuse | 18/18 / 18/18 | 18 | 20 |
| unsafe_tied | 0/6 / 0/6 | 6 | 14 |
| unresolved_sign | 6/6 / 6/6 | 12 | 14 |

The eight complete calls retain 42 endpoint queries per arm. Every source target,
factor inventory, block domain/objective and final endpoint bound agrees, not
only the positive counts. The unsafe control remains nonpositive; that is not
an independently validated full-model UNSAFE witness.

Six additional operator calls cover the original three affine/ReLU/guard
rounding bridges in both arms. Exact targets agree. Affine and ReLU bridges
need nonzero added compensation; the guard bridge changes its RHS outward.
The normal four declarations still use zero nonzero lift compensation, so they
alone would not test these cases.

The fixed dense nine-factor row is refused by the old local kernel and accepted
by the new enclosure, without a support query. The 129-factor source and
17-output row remain refused. Twenty-one concrete negative objects are
independently reconstructed and rejected with their prescribed exception type
and text. They include source equations, row permutation, guard, ownership,
maps/frame, inward compensation, omission, property/gate and old-proof reuse.

Both partial packages check 15 endpoints and retain 3 missing endpoints as
`UNKNOWN_MISSING_EVIDENCE`. Five fault receipts bind expired producer/checker,
delayed live snapshot, source mutation before proposals and package mutation
after endpoint checking. Source mutation produces no extra candidate calls;
late package mutation never becomes accepted evidence.

The checker calls no source/enclosure/candidate producer or native optimizer.
Its ACT package imports still load numerical libraries; this stage does not
claim a portable standard-library-only checker. Its remaining trusted boundary
includes correspondence of the declaration to the intended program and the
exact checker implementation. It does not prove native floating-point execution.

## Attempts, costs and regressions

Execution binds freeze HEAD `6e46b210e41dae8491e8fffe28b595f088e74331` and
design hash `82c3f17ee8a2ecf67c948cbf00e3fe0a086d729b00453452104147a94526549b`.
Both R1 and R2 are preserved. R1 passed its mathematical controls/audit but its
initial receipt omitted heavy ACT imports from the reported interval. R2 moves
those imports inside the timer, includes concrete negative construction and
partial copying in their auxiliary budgets, and preserves bridge failure costs.
All eight normal packages, six bridge objects and the dense target are unchanged
across attempts. These are accounting repairs, not additional scientific cases.

| R2 observed accounting interval | Seconds |
|---|---:|
| ACT/control module imports | 1.4419 |
| Complete control suite | 4.6407 |
| Observation archive publication | 0.0353 |
| Total including implementation snapshot setup | 6.1249 |
| Separate checking-only audit, including heavy imports | 2.5065 |

Each full call has one cooperative 300-second deadline including source creation,
construction, proposals, serialization and checking. Auxiliary calls have 30
seconds; the suite envelope is 1800 seconds, with at most three repair attempts.
No hard-watchdog claim is made. The reported suite total excludes interpreter/
standard-library bootstrap and the final summary receipt write itself. Auxiliary
intervals and test windows are retained, ordered and checked for overlap.

These tiny one-run intervals are accounting observations, not a speed benchmark.
The Gurobi license warning is an import diagnostic; no native optimizer was run.
There was no model/data load, CUDA call, training or sealed-input retry.

An additional 109 tests pass: sparse/binary64 primitives, old source connection,
expert templates, block support, capacity, project layout and sealed-proof/cost
ledgers. Navigation checks retain all 619 historical source bindings and 4,632
archived handoff lines. Read-only AI reviews are not independent human review.

Raw append-only artifacts remain under
`/data1/Kane/MOE/baseline_runs/hz_sparse_source_20261012_r1` and
`/data1/Kane/MOE/baseline_runs/hz_sparse_source_20261012_r2`.
R2 summary SHA-256:
`b4aab09bb18ceba0a4c24f1e3771380f0409d0fc3b9f682a9d54e15680f66eea`.
Recheck using the recorded implementation and existing ACT environment:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /data1/Kane/miniconda3/envs/act-py312/bin/python -B -m scripts.run_hz_sparse_source audit /data1/Kane/MOE/baseline_runs/hz_sparse_source_20261012_r2
```

The two raw archives occupy 6,316,032 allocated bytes. No scientific files or
caches were deleted. A separate whole-workspace reading is 225,908,912,128 bytes;
its growth is not all attributed to these calls.

## Continue / stop decision

Stop this small-source integration study: its equality and rejection questions
are answered. Do not collect more tiny timings or alter the controls to obtain
new positive duties. Continue with one bounded end-to-end capacity/compatibility
decision for **this combined source/block schema**. Check complete source rows,
factors, properties, endpoint queries, snapshot repetition, operators and
evidence volume against the registered real recipes before proposing any new
capacity implementation or real execution. Existing metadata already shows that
the input, property and operator gates do not disappear with sparse rows.

Do not automatically add another supervisor, increase caps, retry the physical
GPU refusal or reopen sealed inputs. A block-compatible GPU candidate kernel
and actual-device admission remain separate required work; flat-matrix CPU/
simulated controls do not satisfy them. The aim is still a compatible algorithm,
source, device and proof chain with real full-request benefit—not a growing
count of independent control suites.
