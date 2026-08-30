# HybridZ CIFAR100 / TinyImageNet pause handoff

Date: 2026-08-09 (Australia/Sydney)
Branch: `hybridz-fse-20260727`
Git base: `391785fe7fa7c02b927ac2a9240a09d293f1363c`
State: active exact-only Conv/ADD exploration; no real or large worker is running.

This is the restart point for the current exploration.  The detailed history
remains in:

- `docs/HYBRIDZ_LARGECLS_EXPLORATION_20260727.md`;
- `docs/HYBRIDZ_CIFAR100_TINYIMAGENET_FAILURE_BUCKETS_20260729.md`.

The working tree contains many untracked experiment files, so the Git base is
not a recoverable snapshot of the research state.  Content SHA-256 values
below are the freeze identifiers.

## Executive conclusion

There is real progress, but not a benchmark-wide breakthrough.

1. CIFAR100 has two authoritative real outcomes in the fixed Gate-6 attempt:
   one live `CERTIFIED` result and one strictly replayed `FALSIFIED` result.
2. TinyImageNet has one real, zero-tolerance strictly replayed
   `FALSIFIED` counterexample.  Its real `CERTIFIED` result is an older
   SAFE-side diagnostic with top-level `proof_authority=false`; it is not an
   authoritative SAFE artifact and cannot be upgraded retrospectively.
3. DAG last-use release changed CIFAR100-large/TinyImageNet from repeated OOM
   to four completed fixed-sentinel runs.  This is a substantial memory and
   runnability gain.
4. The real CIFAR100-medium PCOH K2 build-only experiment produced a sound
   materialized structural upper improvement of about 1.427%, but it did not
   reach the preregistered strong threshold and has no verdict authority.
5. Gate-6 did not pass: it stopped at 2/6 on an orchestration-policy defect.
   Gate-14, Gate-40, and the full benchmark remain unopened.

Therefore it is accurate to say that individual real verification and
counterexample chains succeeded, and that two enabling mechanisms made
substantial progress.  It is not accurate to claim a CIFAR100/TinyImageNet
aggregate improvement, a TinyImageNet SAFE proof, a PCOH verdict gain, or a
path to 400/400.

## Corrected research target: compact exact-ReLU HybridZ

The active target was explicitly corrected after the controlled RC-MPH and
PC-CRF experiments.  Production work must use the **forward HybridZ compact
exact-ReLU formulation**.  In particular:

- every selected unstable ReLU keeps its exact phase/binary semantics;
- no ReLU triangle relaxation may replace that exact representation;
- no branch-and-bound route is part of this research target;
- no backward or dual-bound propagation is part of this research target;
- the immediate performance question is where CIFAR100/TinyImageNet spend
  time and memory in forward construction, constraint assembly, native-model
  loading, and solution;
- any compaction must preserve the exact feasible set while reducing rows,
  nonzeros, repeated phase/link structure, or repeated materialization.

RC-MPH and PC-CRF remain useful, fully recorded controlled experiments, but
they are now **side branches only**: RC-MPH is a dual hypograph primitive and
PC-CRF repairs independently relaxed ReLU triangles.  Neither may be wired
into the corrected production path.  The next work package is a forensic
decomposition of CIFAR/Tiny constraint count, nnz, wall time, and RSS, followed
by a forward-only exact-equivalence compaction toy.

### Bottleneck-prioritized exact work program

The first CIFAR100-medium construction census changes the implementation
order.  It attributes about 85.56% of forward build wall time to convolution,
8.13% to ADD, and 4.96% to ReLU.  In the corresponding materialized operator
frame, ADD accounts for about 65.27% of constraint nonzeros.  These figures are
a profiling result, not a proof result, and come from the existing C89 medium
artifact rather than a completed generic compact-exact sparse-HZ benchmark.

Work is therefore ordered as follows while preserving the exact-only rules
above:

1. **Conv first:** replace per-output-pixel Python construction with an exact
   reusable sparse topology template plus vectorized/preallocated CSR value
   filling.  Candidate and baseline matrices must be bit-exact after CSR
   canonicalization, including groups, padding, stride, dilation, zero
   weights, and bias.  No numerical tolerance may decide equivalence.
2. **ADD second:** represent shared constraint ancestry as immutable DAG blocks
   or stable row handles, union shared ancestors once, and materialize the flat
   CSR only at the solver boundary.  This may remove duplicate physical rows
   and repeated copies, but it may not discard a merely implied or
   approximately equal row.
3. **ReLU third:** retain the compressed exact binary graph.  Reuse a phase
   handle only when two complete preactivation predicates and their bounds are
   exactly identical.  A later extension may admit an exactly proved positive
   rational scale.  Similar bounds, sign correlation, negative scale, offsets,
   and one-way implication are insufficient.

Static ONNX shape inference gives the topology cache a real, though not yet
timed, reuse opportunity.  CIFAR100-medium has 19 Conv nodes but only seven
distinct `(input shape, output shape, kernel, stride, padding, dilation,
groups)` geometries; TinyImageNet-medium is also 19 to seven, and
CIFAR100-large is 20 to eleven.  Thus twelve, twelve, and nine Conv nodes,
respectively, can reuse a spatial index template.  Weight values remain
layer-specific and are never cached as semantic aliases.

Each candidate starts disconnected from production.  It must first pass exact
CSR/point/Jacobian/Fraction/MILP/raw-property equivalence and hostile mutation
tests, then a same-topology synthetic timing and RSS gate.  The first isolated
performance gate is at least 1.50x.  The integrated four-thread five-pair gate
requires median speedup at least 2.00x and a paired-bootstrap 95% lower bound
at least 1.80x, with no verification-result conflict.

The current cleanest structural snapshot is the CIFAR100-medium iid2 C89
Operator-HZ candidate.  Its source has 52,359 continuous columns, 4 binary
columns, 98,378 upper rows, and 9,267,556 constraint nonzeros.  The matched
exact-zero frame differs only by four binaries, four upper rows, and twelve
nonzeros.  The four selected exact cells therefore add one binary, one net row,
and three net nonzeros each relative to the old two-row relaxed cells.  This is
evidence about the existing Operator-HZ formulation; it is not a benchmark of
the generic `2C+1B+1E+2U` compact-exact sparse-HZ implementation and must not be
reported as one.

The same snapshot decomposes the exact-zero frame as follows:

| source | continuous columns | upper rows | constraint nnz |
|---|---:|---:|---:|
| input box | 3,072 | 0 | 0 |
| ten ReLU layers | 8,227 | 16,454 | 3,218,776 |
| eight materialized ADD layers | 40,960 | 81,920 | 6,048,768 |
| output roundoff | 100 | 0 | 0 |
| total | 52,359 | 98,374 | 9,267,544 |

Thus ADD materialization owns 83.27% of rows and 65.27% of constraint nnz in
this frame.  A consumer-aware `ADD -> ReLU` deferral is consequently a primary
exact candidate: when the ADD has a supported direct ReLU consumer, encode the
ReLU graph against the summed affine expression and omit the intermediate ADD
box/equality band.  Unsupported fanout or affine chains must retain the current
materialization.  This is first tested with every unstable ReLU exact; the old
triangle-only add-fusion tests are not evidence for the corrected target.

A second, orthogonal ADD candidate operates only at the native solver boundary.
The existing equality band stores `A x <= u` and `-A x <= v` as two upper
rows.  HiGHS natively accepts one ranged row, `-v <= A x <= u`.  A strict
signed-row-pair folder can therefore retain the proof-source HZ unchanged while
halving those rows and storing the coefficient payload once.  Folding is
allowed only after canonical CSR indices match and every stored coefficient is
the exact sign-negation of its mate; hashes are indexing aids, never equality
proofs.  Candidates are solved on the ranged model and independently replayed
against the original two-row source.  On the C89 tag counts this would reduce
the 81,920 ADD band rows to 40,960 and the 12,748 active-ReLU band rows to
6,374 before any solver-specific presolve.  These are structural projections,
not measured runtime gains yet.

The C89 candidate loaded a 98,378 by 52,363 model with 9,167,448 native-load
nonzeros.  It finished only one of six pair queries and produced no edge,
clique, cut, proof, or verdict.  TinyImageNet still has no comparable
exact-compact shape/build/native/RSS receipt.  Accordingly, these measurements
prioritize engineering but do not constitute a CIFAR/Tiny breakthrough.

### Exact-only implementation checkpoints

The first reusable measurement primitive is now frozen and independently
reviewed.  It exercises only the compressed exact-ReLU forward primitive and
records per-layer `C/B/E/U`, constraint/value nonzeros, payload bytes, and wall
time.  Its immutable checkpoint is:

| File | SHA-256 | Result |
|---|---|---|
| `forward_exact_relu_census.py` | `698f23e591de9ceb31426b6a5214483e4b56c68fc8515e847ddafd4f6b36e177` | 17/17; independent NO-BLOCKER |
| `test_forward_exact_relu_census.py` | `e19461e3051ba429ba11c9fab6a1a2cab00d9c340142ccdbe3d7f10e0410d192` | same |

For every selected unstable cell it replays the intended structural increment
`+2C,+1B,+1E,+2U` and `p+7` constraint nonzeros, where `p` is the complete
stored preactivation support.  Its payload walker counts aliased ndarray/CSR
storage by byte-interval union and fails closed on unsupported containers,
subclasses, negative strides, invalid dtypes, and non-finite values.  It is a
synthetic census only: every proof/verdict/production flag is false and no real
or large instance was run.

**Numeric exactness blocker discovered after the structural census.**  The
existing generic `sparse_hz_apply_relu_exact` cannot currently be treated as
an exact-real graph encoder for arbitrary binary64 inputs.  For the stored
affine expression `x=0.1+0.7*xi`, bounds `[-0.6,0.8]`, its linking equality
stores `fl(0.1-0.8/2)=-0.30000000000000004`.  The resulting one-ULP constant
error permits a true `x=2^-56>0` to select the inactive phase and output zero.
At the minimum subnormal, `alpha/2` and `beta/2` also underflow and delete an
endpoint.  This affects both materialized and deferred generic sparse-HZ
routes; it is not evidence against ADD deferral itself.  Structural census
results remain valid, but no generic compact-ReLU exactness or promotion claim
is allowed until an arithmetic-safe encoding passes independent Fraction
phase replay.  Widening the equality is not an acceptable fix because that
would be a relaxation rather than the requested exact graph.

A disconnected arithmetic-safe replacement candidate is now frozen and has
passed independent review for its declared no-authority scope:

| File | SHA-256 | Local result |
|---|---|---|
| `forward_exact_relu_numeric_candidate.py` | `4edc2c04e1cadd5736c3681eaf59498e5b3a6cb1baea691b743cb645a3dce933` | 17/17; independent NO-BLOCKER |
| `test_forward_exact_relu_numeric_candidate.py` | `e0fc2347b1aaad476823179e7fefb956ea170326e7cfabd937bf62c67940c2b4` | same |

For ordinary rows it preserves the compact graph but represents the exact
stored-real difference `c-beta/2` as `hi+lo`.  If any row has a nonzero `lo`,
one layer-shared continuous factor is fixed by the exact equality `kappa=1`,
and only those link rows receive the residual coefficient.  If either half is
not exactly representable, the row switches to the division-free graph
`x=alpha*q+beta*s, y=beta*s`, with phase-dependent nonnegative `q,s`; it does
not round away the endpoint or introduce an equality band.  Bounds are
recomputed from immutable raw binary64 storage with exact `Fraction` sums and
outward conversion.  The `0.1+0.7*xi` and minimum-subnormal counterexamples
are covered locally, together with phase/point/Jacobian and independent MILP
checks.

This repair changes the old unconditional resource law.  For `k` unstable
rows, `h` division-free rows, `r` compact rows with a nonzero low residual,
and `I=1[r>0]`, the increment is
`C=2k+I, B=k, E=k+I, U=2k+2h`; constraint nonzeros are
`sum_compact(p_i+7) + sum_half_free(p_i+8) + r + I`.  The old census
`2C/1B/1E/2U, p+7` remains a structural measurement of the legacy primitive,
not the new generic exact resource promise.  The replacement is still
disconnected and has no proof, production, or authenticity authority; its
production snapshot/performance boundary must still be designed before any
integration.  The independent review additionally replayed 600 arbitrary
finite binary64 rows, 1,000 general unstable rows, extreme normal/subnormal
cases, old continuous/binary/equality/upper constraints, malformed CSR
storage, and allocator-edge failures.  Production still needs an authenticated
owner because public `hz` attributes can be rebound, and solver handling of
subnormal coefficients is a separate numerical boundary; neither limitation
is hidden by the candidate receipt.

The disconnected exact-ReLU DAG interning toy is **not promotable** despite a
roughly `1.96x` row-replay microbenchmark and a `12/32 -> 6/16` row/nonzero
reduction.  Independent review found two generic-API counterexamples: an
allocator can reuse the ID of a live factor that is not visible in the current
predicate/arena rows, and row views from two arenas with the same numeric row
ID can be unioned without preserving both contents.  Production use therefore
requires a global typed factor allocator plus arena-bound row provenance.  The
fixed single-arena toy remains a diagnostic; it is not evidence of a CIFAR or
TinyImageNet hit, and direct duplicate ReLU inputs were absent in the three
screened ONNX graphs.

The ADD signed-row/ranged-row primitive has passed its disconnected hostile
gate after closing CSR dtype/canonical-cache ingress holes:

| File | SHA-256 | Result |
|---|---|---|
| `exact_ranged_row_compaction.py` | `e7ea8cfa6aec7d065b55d20a589c83fc3c25d4028d2929b98eba3874af211d78` | independent NO-BLOCKER |
| `test_exact_ranged_row_compaction.py` | `087efc9bc3bee6b3bc82b6f4dee5638324dbfa6a89e039b9186200ba90a60575` | ranged+ADD 27/27 |

The audit replayed 500 random frames, 10,000 exact `Fraction` points, and 100
HiGHS source-versus-range status comparisons, including subnormal and maximum
finite coefficients, signed zero, empty rows, and contradictory bands.  The
three compact exact-ReLU rows in the Operator ADD toy remained bit-identical.
This establishes the folding algebra and strict replay boundary only.  A
separate native-load sentinel was therefore used to measure folding cost,
row-load cost, model closure, and source replay before any solver integration.

That native-load sentinel is now frozen as a **negative end-to-end result with
a positive loader-only sub-result**:

| File | SHA-256 | Result |
|---|---|---|
| `exact_ranged_row_native_load_sentinel.py` | `b166b81e6d39f8f53f446d06e18f177082ebfd33c353939266156e434f1d5f8d` | independent NO-BLOCKER |
| `test_exact_ranged_row_native_load_sentinel.py` | `8fb8da3245e928af906ea6d9fcd4eb995b2ddacdc0de4a8cfb1825a4e532b7f7` | 21 tests / 28 subtests |

At the fixed scale-40 synthetic C89 ratio, native `addRows` for 1,024 ranged
rows was about `2.18x--3.27x` faster than loading the 2,048 signed source rows.
However, strict post-hoc pair discovery, snapshotting, and folding dominated
the work: `fold + addRows` was only `0.093x--0.103x` of the baseline, i.e. about
9.7--10.7 times slower.  Consequently post-hoc folding is closed for
promotion.  The viable next design is builder-owned pair metadata and direct
ranged-row emission at the native boundary while the original source HZ
remains separately replayable.  Loader-only speed must not be reported as an
end-to-end speedup.

The direct-emission design has now been statically scoped.  Pair authority must
be created inside the signed-band builder for exactly two families,
`add_materialize` and stable `relu_active`; tags and caller-supplied schedules
remain diagnostics only.  Final assembly translates private block UIDs into
nonoverlapping row spans, and the native loader sends the forward row once with
`lower=-ub_reverse, upper=ub_forward`.  Unstable/exact-binary ReLU rows,
micro-RLT tails, binary-bearing rows, malformed pairs, and derived/cloned HZs
all use the complete baseline source.  This removes solver-time pair discovery
and full-NNZ folding, but it requires genuinely immutable or versioned
`Auc/Aub/ub` storage bound to a one-use owner lease; NumPy `write=False`, tags,
or a portable digest alone cannot close concurrent mutation.

This boundary is intentionally narrow: the complete two-row source HZ remains
available for independent replay, so source rows, source nnz, and ADD
materialization memory do **not** decrease.  Only native HiGHS rows/nnz,
`addRows`, and presolve/model memory can improve.  A future source-level
constraint DAG is a separate work item; no ranged-loader receipt may claim to
have solved the 65.27% source-nnz bottleneck.

The source-level design has now been narrowed further.  Merely replacing the
current `_ConstraintBlock` list with a DAG cannot remove a logical constraint:
the Operator builder already appends into one global history, so that change
would only reduce flattening, copying, and possibly peak RSS.  Actual source
compression requires an authoritative `RANGE` block created inside the same
transaction that creates an equality band.  It may store one canonical row
only when the independently constructed reverse row has identical CSR
structure and every stored coefficient is its bitwise sign-negation.  The
block retains distinct forward/reverse virtual facet IDs and tags, and exact
replay must reconstruct both historical facets byte-for-byte.  Any mismatch
falls back atomically to the two original `LE` rows; tags, hashes, and numeric
tolerances cannot authorize pairing.

Under that exact representation, the C89 ADD payload projects from 81,920
stored rows / 6,048,768 stored nonzeros to 40,960 ranged rows / 3,024,384
stored nonzeros.  The complete K4 frame projects from 98,378 stored rows /
9,267,556 nonzeros to 57,418 / 6,243,172, while still representing all 98,378
one-sided mathematical facets.  These are static projections, not measured
speedups.  The first implementation is therefore disconnected and must prove
immutable arena ownership, cross-arena fail-closed handles, byte-exact virtual
facet replay, `Fraction` membership, and a bounded C89-ratio performance/RSS
gate before any Operator or solver consumer is changed.

The disconnected RANGE/DAG candidate has now passed its structural and
isolated timing review:

| File | SHA-256 | Result |
|---|---|---|
| `constraint_block_dag_candidate.py` | `4dcace661ea6886c755ff7848cb6de5f1f440742fdcc0f1d6a69dd713ad03f44` | independent NO-BLOCKER in disconnected streaming scope |
| `test_constraint_block_dag_candidate.py` | `fcc3db852852a25d5a357bbac8593f64af472a1125486356fdcdf4eea17da1d0` | candidate portion of the combined 35/35 gate |

The earlier independent scale-40 candidate benchmark measured about `1.616x`, a
retained-payload ratio of `0.5000032`, and a non-pair fallback ratio of about
`0.982x`; source rows and stored nonzeros were exactly halved while every
virtual facet remained replayable.  The review also closed cross-owner factor
ID reuse, validate-to-serialize mutation, nested subclass substitution,
caller-view iteration races, cross-arena handles, and concurrent mutable arena
use.  Mutable owners are thread-confined; only a sealed program may be read
cross-thread.  This receipt deliberately fixes `rss_measured`, production
baseline, full-promotion, and all authority flags to false.  An isolated
subprocess RSS gate and a graph-owned Operator integration design are still
required; the 1.616x candidate result is not a CIFAR or production speedup.

That fresh-child memory sentinel is now independently frozen:

| File | SHA-256 | Result |
|---|---|---|
| `constraint_block_dag_memory_sentinel.py` | `95cf82461a851682f480a5398f6e2f54cae9f110532347c9ce9fd63ed24024c9` | independent NO-BLOCKER streaming measurement protocol |
| `test_constraint_block_dag_memory_sentinel.py` | `c988068b354695a6f476fde72e07dd821272739ce7f8cf46d11ac91e2a753469` | combined candidate+sentinel 35/35 |

The final protocol used eighteen fresh children with fixed 20-second,
512-MiB RSS, and 32-MiB payload limits.  It retains the old fully expanded
replay as a permanently closed diagnostic, but adds a bounded 1--256-row
immutable iterator that reconstructs facets in the exact legacy order without
building a global list or CSR.  Independent medium results were approximately
`0.494` for source build/seal RSS, `0.469` for full streaming replay RSS, and
`0.500002` for retained numeric payload, with about 8.99 seconds total wall
time.  The expanded path remains explicitly closed even when an individual
run jitters below 0.80.  Streaming exact replay therefore passes the complete
disconnected synthetic memory gate.  This is the first full source+replay
offline success in the current bottleneck program, but production promotion,
real-model authority, and Operator/solver integration remain false.

The first production representation layer (Phase A) is now frozen, but still
has no consumer:

| File | SHA-256 | Result |
|---|---|---|
| `constraint_program.py` | `aaf7f36a39ef348733adf6b88621e21d63c0b2f03f99eb696847622e209ad7c2` | independent bounded-only NO-BLOCKER |
| `test_constraint_program.py` | `f9c060b9c4a1fb84d50f14dcf9dcc19aba51eb1c844994d430fd9ce1eb85a132` | production core 60/60; combined 81 tests + 56 subtests |

This module is independent of the disconnected candidate and does not import
the Operator, solver, verifier, dual, or proof paths.  Factor IDs come only
from a captured external allocator; the program stores ordered `LE` and exact
guarded `RANGE` occurrences and streams immutable native or legacy-facet
batches.  Owner, arena, view, prepared append, append result, and bare batch
objects are deliberately non-authoritative.  Only a sealed program can carry
representation/replay authority.

The red-team work replaced mutable multi-field publication with immutable
owner/arena/iterator roots and copy-on-write registry swaps.  The final audit
covered callback reentrancy, ABA and graph rebinding, stale frames and ID
reuse, prepare/commit/abort interruption, staged sealing, program-GC revival,
and native/legacy iterator load/advance/close/GC boundaries.  Independent
counts were trace 40/40, GC double-interruption 4/4, rebinding 26/26, and batch
authority 2/2.  Each bounded injected failure converged to a complete old or
complete new state with at most one live authoritative program.  The scope is
explicitly bounded: cleanup/repair retries at most four times and does not
claim safety against an unbounded stream of asynchronous faults.

Phase A is therefore a production-quality representation primitive, not a
production optimization result.  No Operator or HiGHS consumer is connected,
no real/large run was made, and neither CIFAR100 nor TinyImageNet verification
coverage changed.  Phase B must first use an internal default-off Operator
sink and byte-exact legacy replay; Phase C must then load native `RANGE`/`LE`
batches without full expansion before any end-to-end speedup claim is valid.

The Conv direct-CSR/fused-generator primitive has now closed its first-review
CSR index-domain and extreme-geometry failures and passed independent review:

| File | SHA-256 | Result |
|---|---|---|
| `exact_sparse_conv_csr_candidate.py` | `cc4db6d83beba319fec39ae2d847bc6702a4d35acb4c2c2a7044d286e56f1403` | 19/19; independent NO-BLOCKER |
| `test_exact_sparse_conv_csr_candidate.py` | `b9c4540c67f91b568d9a08377f1a980f4945be95a9da567f908817ce9f701efa` | same |

Against the already-vectorized Operator builder, the independent single-thread
same-topology medians were about `2.05x` for direct canonical CSR and `2.43x`
for the eligible row-local generator transform.  The standalone relabel path
was only `0.40x` and is closed.  These timings cover only exact matrix or
generator construction; center, bias, error propagation, the full network,
and the solver are excluded.  The much larger legacy per-pixel comparison is
not a C89 baseline and must not be used as the headline.  Static C89 topology
suggests 17 of 19 Conv inputs are row-local/empty (the two Conv-to-Conv entries
are not), but this is a hit-rate hypothesis rather than measured end-to-end
speedup.  Production integration remains under toy/wide review, and no
CIFAR100/TinyImageNet run has been authorized from it.

Two small complete-Operator synthetic measurements show why the direct CSR
builder is only a prerequisite.  For `8x16x16 -> 16x16x16` and
`16x24x24 -> 32x24x24`, including input construction, the Conv affine
transform, and final sparse-HZ assembly, paired median speedups were only about
`1.11x` and `1.16x`; an independent rerun measured about `1.17x` and `1.25x`,
respectively.  All output cores were bitwise identical, and every measurement
missed the `1.50x` promotion gate.  The next Conv work item is therefore the
audited fused topology-to-row-local-generator path, which avoids the generic
SpGEMM for eligible inputs; direct-only dispatch is not a performance
breakthrough.

The first production-dispatch review also found three integration blockers,
all outside the direct builder's coefficient algebra: the historical C89 gate
still requires the old `vectorized_exact_csr_v1` metadata value; the imported
module still declares itself disconnected and non-authoritative; and the
K2/K3 implementation-integrity file list does not include that newly executed
dependency.  A production promotion must therefore use an explicitly
production-authoritative Conv core, migrate the gate/schema and negative
tests, and bind the core file into every implementation digest.  Until those
changes and the full-stage speed gate pass, the production integration remains
unfrozen and no real run is allowed.  As an immediate safety action the
direct constructor is now behind the internal, default-false
`_EXPERIMENTAL_CACHED_DIRECT_CONV` switch; the default proof-bearing Operator
path again uses the established vectorized/explicit-fallback constructor, and
the candidate dependency is imported lazily only by explicit toy tests.  The
candidate, Operator, and historical probe suite passed 106/106 after this
withdrawal.  Independent review found no blocker in the withdrawal boundary;
the current safety checkpoint is `operator_hz.py`
`9d2fed8c8a67eb53b931e34a98a181891ee32cbd85e16422bb175850626fbe14`
and `test_operator_conv_csr_builder.py`
`0779208f4bb5a6f9ced2069ea6e948167cfca1c9784331469f18d871367597a3`.
This is a safe research checkpoint, not a Conv promotion.

A follow-up single-pass Conv affine-core prototype tested whether the isolated
builder win could survive the complete `c/G/error` transform.  The profiled
audit target was `exact_sparse_conv_affine_core.py`
`eafd7a4ffdf2acd3afddb53c09323775bba777c1a51588215dcf8cab88dc27d2`
and `test_exact_sparse_conv_affine_core.py`
`0b18dbf3c1af156dd77b43ab4dbefa0e0dac2ccc2baf6ddfb7e7f2582dc8b351`.
The local 41-test gate found bitwise agreement with the existing Operator
oracle across randomized admissible geometries, and it materialized neither a
full sparse convolution operator nor a SpGEMM.  Nevertheless, its complete
paired speed ratio was only about `0.332x` (roughly 2.5--3 times slower), so it
is an explicit **NO-PROMOTION** result.  Independent review also found that
the public receipt/source/core boundary used self-describing hashes and could
be coherently re-signed by a caller; in particular, a forged row-local mapping
or forged final core could be accepted.  Until that boundary is downgraded to
non-authoritative accounting and semantic replay is made total, this module is
only an isolated profiling/correctness reference.  It must not be imported by
the proof-bearing Operator path.

The post-audit safety downgrade is now explicit in schema v2: the receipt's
`linear_primitive_authoritative`, property-proof, and verdict flags are all
fixed `False`, and attempts to construct an authoritative receipt are
rejected.  The current non-authoritative checkpoint is
`exact_sparse_conv_affine_core.py`
`ad65848cca2cf15a060fa570473b4130eff5459edd93c9460304f607b5dfaafb`
and `test_exact_sparse_conv_affine_core.py`
`2fe3dc6d4d1c56750a29b9599d3acdaf31344ab1ee4bd78ce19cd39016b414d1`;
its focused tests pass 12/12 and the narrow safety downgrade received an
independent NO-BLOCKER review.  This downgrade intentionally does not claim
to repair the self-signed source/core/result boundaries; it prevents consumers
from treating the negative experiment as an admitted linear authority.

A read-only stage profile confirms that ordinary Python/NumPy hardening cannot
rescue this design.  On the fixed `1x8x12x12 -> 16x12x12`, `3x3` row-local
synthetic, the warm direct path took about 5.37 ms with a 16.28 MB traced peak,
versus about 2.92 ms / 5.32 MB for the established `W + _affine` path.  Passing
the 1.50x gate would require at most 1.95 ms, a further 63.7% reduction.  Even
the deliberately optimistic sequence of one-use capability, delayed digest,
and ownership-transfer assumptions remained about 3.88 ms.  The coefficient
traversal itself was about 2.23 ms before output emission, while a permuted
generator map added roughly 7.64 ms and is therefore categorically ineligible.
The current core is closed rather than micro-optimized.

The only remaining Conv hypothesis is a separate, model-owned sealed topology
schedule consumed by a native streamed fused kernel, restricted initially to
monotone row-local generators.  Its stop rule is deliberately earlier than a
production integration: raw kernel time must be at most 1.25 ms, complete warm
time at most 1.95 ms, peak memory at most 5.32 MB, and every `c/G/error` bit and
authority replay must match.  Cold construction already misses that target,
so no cold-path or network-level Conv speedup is currently claimed.

The first native-schedule feasibility probe met the small-fixture compute
targets but failed the network memory boundary.  Its disconnected checkpoint
is `exact_sparse_conv_native_schedule_candidate.py`
`593ecaa0b4690d29ac44d7c1a538633a75be072c5e9c6cf37d965241e9d7c23a`
and `test_exact_sparse_conv_native_schedule_candidate.py`
`0e03ce31de7dc8576775039641b4b8cdfc946aaeca34c22a04d2a3d6ea1f2dd0`;
the related Conv regression passed 38/38.  On the fixed synthetic, the native
single loop took about 0.183 ms raw and 0.345 ms including the warm wrapper,
with a 3.178 MB traced peak, while reproducing canonical `c/G/mass/error`
bits.  However, it caches the expanded convolution operator.  At the known
C89 total of 74,100,416 operator nonzeros, the exact buffer lower bound is
890,483,788 bytes (849.232 MiB) before Python overhead, output HZ storage, and
allocator fragmentation--more than 5.3 times the 160 MiB source allowance.
It also relies on undeclared CFFI plus a host runtime compiler.  The result is
therefore **NO-PROMOTION**: it establishes only that a native fused loop has
enough arithmetic headroom.  Any successor must stream directly from compact
weights/topology and must not retain expanded `W`.

That compact successor now exists as a frozen, disconnected audit candidate:
`exact_sparse_conv_native_schedule_candidate.py`
`c0fd674f64eb7e6b64ef057f5e93465185e58736bc5dc9e8661547137c86c2a5`
and `test_exact_sparse_conv_native_schedule_candidate.py`
`a8a1d34bf61e907a887fbef57c4fc4fbd55a94e5ad203600e962d75a535430e3`.
Unlike the historical v1 above, v2 stores only compact weights, per-channel
bias, and 17 integer geometry values, then directly streams canonical output
`c/G/mass/error` without building an expanded convolution matrix or schedule.
On its fixed synthetic, raw/warm/model-cold times were approximately
0.335/0.557/1.493 ms; the matched model-cold baseline was 4.529 ms (`3.03x`),
and the traced small-fixture peak was 3.638 MB.  Twenty-four additional grouped,
strided, padded, and dilated geometries plus subnormal/cancellation cases were
bitwise green.  Static C89 initializer shapes project only 18,516,504 bytes
(17.659 MiB) of persistent numeric schedule storage, below the 160 MiB cap.
These are candidate measurements under independent review, not a production
speedup.  CFFI is not an ACT dependency, runtime compilation is excluded from
the model-cold timing, and all authority/promotion flags remain false.

The first independent review kept the arithmetic evidence but blocked any
promotion.  The C ABI used signed `int` geometry arithmetic without a complete
proof that products such as `oh*stride + kr*dilation` cannot overflow; the hot
API also accepted coherently self-signed schedule/source dataclasses and could
therefore understate source mass.  The 17.659 MiB figure covers only persistent
compact schedule buffers.  It excludes source/output snapshots and transient
freezing: for an 8,192-output, 7,929,856-nnz C89-sized layer, the current output
capacity plus immutable-copy lower bound is already about 182.063 MiB before
IDs, source, allocator, or Python overhead.  Finally, "model-cold" excluded
dynamic source preparation.  The candidate must close those integer,
factory-provenance, receipt-type, and measurement boundaries before its
independent status can change; no Operator integration is authorized.

Those boundaries were repaired in disconnected v3.  The final captured-view
checkpoint is
`exact_sparse_conv_native_schedule_candidate.py`
`1d2889afb14abba9c8c8fe2dbd1f88b05994121acfd39ca8a898ce4a25b153f3`
and `test_exact_sparse_conv_native_schedule_candidate.py`
`dbccba04b1f5361731367c6e4e58b56508bd4213b796bccba87e9467ad3ef706`.
The C ABI now uses checked 64-bit geometry/size arithmetic, hot calls require
factory-produced process-local identities, registry records strongly retain
the admitted arrays, and every post-admission step consumes only immutable
captured references.  Deterministic cross-thread rebind and ABA tests cover
geometry, weights, bias, center, mass, IDs, columns, and digests.  Receipts use
strict types, and the cold comparison includes both schedule and source
preparation.  Its 43 tests pass; raw/full/peak remained within their narrow
gates at about 0.718 ms / 1.011 ms / 3.639 MiB.  The truthful cold ratio,
however, was only `1.454x` (`2.459 ms` versus `3.576 ms`), with earlier runs
also below the threshold.  The predetermined 1.50x stop rule closes v3 as
**NO-PROMOTION**, irrespective of occasional noisy passes.  The 182.063 MiB
large-layer freeze lower bound and unmeasured C89 apply peak remain explicit;
the final captured-view review is NO-BLOCKER only in the disconnected,
non-authoritative, NO-PROMOTION scope.  Its independent cold run happened to
measure `1.555x`, but the earlier repeated sub-threshold results, permanent
`promotable=false`, and closed network-memory gate are not overridden by one
noisy pass.

## Evidence levels

- A `numeric diagnostic` includes build-only geometry, an independently
  checked upper bound, a solver candidate, or a resource measurement.  It is
  not a verdict.
- A real `FALSIFIED` result is accepted only when CPU ONNX Runtime and the raw
  VNNLIB predicate replay the exported input at tolerance `0.0`.
- A real `CERTIFIED` result is authoritative only when the live, process-local
  SAFE capability validates the exact source, graph, config, dtype, solver,
  proof-domain partition, and terminal result.
- A serialized process-local seal is not a portable signature and cannot
  re-authorize an old JSON file.
- Historical S/U labels, solver statuses, incumbents, `UNKNOWN`, timeout, OOM,
  and build-only receipts do not create a verdict.

## Real outcomes

| Dataset / instance | Real result | Evidence level | What may be claimed |
|---|---|---|---|
| CIFAR100 medium iid29 | `CERTIFIED`, 12.413352 s | top-level `proof_authority=true`, live SAFE capability, 99/99 root rows, no GT | Authoritative for this exact live run |
| CIFAR100 medium iid2 | `FALSIFIED`, 19.605805 s | strict ORT/raw-VNNLIB replay, tolerance 0, margin 0.0665418293, top-level authority true | Authoritative counterexample; Gate-6 still failed |
| CIFAR100 medium iid64 | `FALSIFIED`, 18.282674 s | strict replay valid, margin 0.0045942216; legacy top-level authority false | Counterexample itself is valid; not a Gate pass |
| CIFAR100 large iid118 | `FALSIFIED`, 42.470014 s | strict replay valid, margin 0.5550898093; legacy top-level authority false | Counterexample itself is valid; not a Gate pass |
| CIFAR100 large iid113 | `CERTIFIED`, 40.405860 s | legacy SAFE diagnostic, top-level authority false | Diagnostic only |
| TinyImageNet medium iid6 | `FALSIFIED`, 61.085820 s | strict replay valid, tolerance 0, margin 0.8353867604; legacy top-level authority false | Counterexample itself is valid; not a Gate pass |
| TinyImageNet medium iid17 | `CERTIFIED`, 67.480188 s | legacy SAFE diagnostic, top-level authority false | Diagnostic only; no Tiny live SAFE authority |

All six files record `ground_truth_loaded=false`.

### Gate-6 interpretation

The fixed C75 Gate-6 stopped after the first two CIFAR100-medium cases.
The iid29 SAFE artifact is clean.  The iid2 FALSIFIED conclusion is also
sound, but the old probe incorrectly ran the SAFE validator on that
FALSIFIED result and emitted 12 inapplicable promotion errors.  The fixed
no-promotion-error rule therefore stopped the gate at 2/6.  The status-routing
code was later repaired, but the benchmark was not rerun and the old artifact
was not rewritten.  The remaining four cases are historical real diagnostics,
not members of a successful C75 Gate-6 run.

### TinyImageNet interpretation

TinyImageNet is not solved.  iid6 supplies a genuine strict counterexample.
iid17 completed and returned `CERTIFIED`, but its old process is gone and it
never carried a live SAFE capability, so it cannot be promoted to proof
authority after the fact.  Tiny real runs also remain expensive: iid6 used
about 28.138/33.416 GB CUDA allocated/reserved and iid17 about
28.138/33.427 GB.

## Capability breakthroughs

### 1. Strict counterexample authority

The exported-input path now binds model, VNNLIB, input, logits, and output
hashes, checks the input domain, executes CPU ONNX Runtime, evaluates the raw
VNNLIB formula, and requires a zero-tolerance violation.  It validated real
counterexamples on CIFAR100 medium iid2, CIFAR100 medium iid64, CIFAR100
large iid118, and TinyImageNet medium iid6.

### 2. First live SAFE authority

CIFAR100 medium iid29 is the first fixed-gate run with a live
`CERTIFIED` capability and top-level `proof_authority=true`.  It certifies
only that exact process and source snapshot.  It does not make other old SAFE
JSON files authoritative.

### 3. Large-model memory liveness

C50 introduced DAG last-use release of internal forward-bound states without
changing public bounds or bound arithmetic.  Under the same external GPU
pressure that had caused repeated OOM, it allowed the two CIFAR100-large and
two TinyImageNet fixed sentinels to complete.  This is a genuine systems
breakthrough, though not a full benchmark result.

### 4. PCOH real structural tightening

The strongest new geometric evidence is the real CIFAR100-medium iid2 PCOH
K2 build-only artifact:

- source shape: `(O=100, C=52,657, B=4, E=0, R=98,974)`;
- fresh shape: `(100, 52,661, 4, 3, 98,975)`;
- constraint nonzeros: `10,498,232`;
- global checked cube upper: `110.86509745475846`;
- materialized structural upper: `109.28266254174339`;
- absolute drop: `1.5824349130150637`;
- relative drop: about `1.427%`;
- rounding tax: about `3.29e-14`;
- total wall time: `20.972593` s;
- peak process RSS: `2,429,984,768` bytes;
- no ground truth, full-parent LP, solver handoff, SAFE/ADV verdict, or full
  fresh diagnostic LP was used.

The continuation threshold passed.  The strong threshold was
`2.217301949095169`, leaving a gap of `0.6348670360801054`, so the result is
not a strong promotion.  It is a sound, candidate-only structural upper
diagnostic, not proof that the exact parent/fresh LP optima differ and not a
verdict.

The first real K3 attempt ended in terminal-integrity stop-loss before a
usable K3 numeric result.  The artifact records
`k3_transaction_called=false`, zero pair/conditional/local LP calls, and no
stable-bit schedule, fresh dimensions, or tightness gate.  Its old receipt
lost the original helper stage;
forensics strongly suggested a cross-run focus digest that included elapsed
telemetry, but that exact cause cannot be recovered from the overwritten
artifact.  A stable semantic focus anchor was implemented later.  No real K3
rerun was performed.

### 5. K4 / adaptive result

The real C89 adaptive K4 attempt is closed in its unchanged form.  It
completed only 1/6 pair queries, certified zero conflicts, did not
materialize, and peaked at `2,866,225,152` RSS bytes, above the 2.5 GiB gate.
It is not a successful K4 validation.

## What has not been achieved

- no Gate-6 6/6 pass;
- no Gate-14, Gate-40, or full 400-instance run;
- no benchmark-wide improvement or official aggregate score;
- no authoritative TinyImageNet SAFE artifact;
- no real PCOH K3 success and no PCOH/K4 verifier verdict;
- no proof that the real fresh LP optimum is strictly below the parent LP
  optimum;
- no full-profile K3 memory calibration under a bounded cgroup;
- no production integration of the latest candidate-only PCOH path.

Blind reruns, larger timeouts, alternate iids, global 10.5M-nnz LP sweeps,
and reopening the unchanged C89 K4 path remain closed.

## Paused code checkpoint

The K3 resource-stop wrapper and memory sentinel are the last independently
audited frozen toy components:

| File | SHA-256 | Test state |
|---|---|---|
| `operator_phase_conditioned_k3_build_only.py` | `8101c50772c75cd64d436252856c54e0dba03214999ba3a7ca7d371cd4d984c0` | wrapper suite 19/19 |
| `test_operator_phase_conditioned_k3_build_only.py` | `22010fdc6fec2547cfed71eddb07081176f4d83f8451be7c453540ae60a1f559` | same |
| `operator_phase_conditioned_k3_memory_sentinel.py` | `552a0ed940e5b8c8cfda11fb57836ba7f2acb3270305795e318047cee80f3cd3` | sentinel suite 15/15 |
| `test_operator_phase_conditioned_k3_memory_sentinel.py` | `d0955b13763a0a940d3b0a367fa8a3d4f4d88783b7fccad001058c05252c6938` | same |

The two suites ran together as 34/34 on 2026-08-09.  The sentinel's large
profile was not executed; only its inert/fail-closed toy contract ran.

The probe integration is a **WIP checkpoint**, not a frozen production
component:

| File | SHA-256 | Current local test state |
|---|---|---|
| `hybridz_phase_clique_build_probe.py` | `25652675f454758055160246fb1f9353652fd245cab2a0259e707335362b9919` | `py_compile` and 76/76 toy tests pass |
| `test_hybridz_phase_clique_build_probe.py` | `1488aef9a711ababb484fda28b50203b9475b424bfa20eba871021953ef3320d` | same |

The final local edit only made stop-loss `total_seconds` no smaller than any
already recorded stage duration.  It repaired three local regressions.  The
new WIP SHA has not received the independent hostile re-audit required for a
real run.

Known lifecycle work remains before the probe may be called frozen:

1. if strict adoption validation fails, `_adopt_pcoh_k3_trusted_transaction`
   still releases the anchor before placing the original transaction in the
   public body, so the original resource-stop receipt can be lost;
2. a later terminal veto can clear the trusted outcome SHA and the integrity
   receipt's actual counters even when the original resource-stop transaction
   itself was strictly valid;
3. a `BaseException` before the K3 finalizer can release the registry but fail
   to return a sealed public receipt.

The partial exact-type/top-to-detached/live-outcome registration hardening in
the WIP source must also be independently replayed before use.  No real run is
authorized from this checkpoint.

All six files in this section are currently untracked, so these are content
hashes, not committed Git blob identifiers.

## Resource gates that remain in force

- PCOH/K3 outer wall: 60 s;
- source build: at most 27 s;
- process RSS hard gate: 2.5 GiB;
- pre-S and pre-fresh forecast: 384 MiB;
- host/cgroup reserve: 896 MiB;
- K3: exactly 8 patterns and 12 signed-pair queries;
- at most 20 local LP calls and 34 conditional checker calls;
- fresh issue is forbidden after a resource-stop;
- all real outputs must use a new path and must not overwrite an artifact.

The real K2 tightness run left only about 242.6 MiB below the 2.5 GiB cap;
the failed K3 artifact left about 237.2 MiB.  Existing evidence is not enough
to lower the 384/896 MiB gates.  The Python `Fraction` envelope and native
HiGHS allocations are not covered by a complete static allocator bound.

## Artifact content hashes

```text
7aaefa3d32556089d05ccaa14420960f0bb20eb99801897ac420e590e1130062  pcoh_k2_build_only_cifar100_medium_iid2_first.json
01625add9f435eefef20e3eaa6dcaf72f2ce0f50137f19a611c576c1829846b0  pcoh_k2_materialized_tightness_cifar100_medium_iid2_first.json
a13efbcf302077ef3fd6150cf881eda3c549d9f037ca29f599a25045d95f766b  pcoh_k3_pairfirst_tightness_cifar100_medium_iid2_first.json
ef0996404383b56be13ead98fd6784ba1cd15609f01bb57c2c935c4e608ca3c2  gate6_c75a_live_authority_cifar100_medium_iid29_20260729.json
32c8ee09613d32df1c14a5a13edeb87fc90f4bef99ad401ac68582239c1de8df  gate6_c75b_live_authority_cifar100_medium_iid2_20260729.json
85a1921db72de0c8d38c186ea528479837e0e117c46be1a3ffb043a07430f3ef  gate6_c75b_live_authority_cifar100_medium_iid2_20260729.strict_replay.json
19d5eced9597c56e0e67ce9acfb99b1067e125b7ead3abae71a554e0395f0653  gate40_c48g_property_forest_sat_cifar100_medium_iid64_20260729.json
ce0d973f7a32f756f2de76412dcdef76202a6b792e0165ae422a243fe50efefd  gate40_c48g_property_forest_sat_cifar100_medium_iid64_20260729.strict_replay.json
f0d14323c9bcc5ddc77de07b88370be51c4c7f0a4c8b493a61d8b80b7d0d274e  gate1_c50a_lastuse_c48_sat_cifar100_large_iid118_20260729.json
3b9f93de7256e53a37db42a2ffd78061c445a500577cfe78dac7837b807821ab  gate1_c50b_lastuse_c48_sat_tinyimagenet_medium_iid6_20260729.json
91523c0f632365a1500338a1428a9eec728c7499c04c291898da4518b8c49496  gate1_c50b_lastuse_c48_sat_tinyimagenet_medium_iid6_20260729.strict_replay.json
c0f52f27b410c54b6c8e4f98c91f77145fd8c88a8b17b0a7615cdaf2ce7c60ba  gate1_c50c_lastuse_c48_unsatdiag_cifar100_large_iid113_20260729.json
20bfcc801353753ece68b81bf0cbf43846f8029cec2ab3925efe625c5dd65136  gate1_c50d_lastuse_c48_unsatdiag_tinyimagenet_medium_iid17_20260729.json
0a6959437fcff165fc1e0c72391eaaccf6e384d42879b76878f61a07cb2c0fee  gate1_c89_allocatortrim_nativeobjective_explicitclose_rbs_adaptive_k4_cifar100_medium_iid2_20260808.json
```

Recheck them without executing a verifier:

```bash
sha256sum \
  artifacts/hybridz_largecls_gates/gate6_c75a_live_authority_cifar100_medium_iid29_20260729.json \
  artifacts/hybridz_largecls_gates/gate6_c75b_live_authority_cifar100_medium_iid2_20260729.json \
  artifacts/hybridz_largecls_gates/gate6_c75b_live_authority_cifar100_medium_iid2_20260729.strict_replay.json \
  artifacts/hybridz_largecls_gates/gate1_c50b_lastuse_c48_sat_tinyimagenet_medium_iid6_20260729.json \
  artifacts/hybridz_largecls_gates/gate1_c50b_lastuse_c48_sat_tinyimagenet_medium_iid6_20260729.strict_replay.json \
  artifacts/hybridz_largecls_gates/pcoh_k2_materialized_tightness_cifar100_medium_iid2_first.json \
  artifacts/hybridz_largecls_gates/pcoh_k3_pairfirst_tightness_cifar100_medium_iid2_first.json
```

## Safe local checks

These commands are toy/regression only:

```bash
python -m py_compile \
  act/pipeline/verification/hybridz_phase_clique_build_probe.py \
  act/pipeline/verification/test_hybridz_phase_clique_build_probe.py

python -m unittest \
  act.pipeline.verification.test_hybridz_phase_clique_build_probe

python -m unittest \
  act.back_end.hybridz_tf.test_operator_phase_conditioned_k3_build_only \
  act.back_end.hybridz_tf.test_operator_phase_conditioned_k3_memory_sentinel
```

Latest local results are 76/76 and 34/34.  The missing Gurobi licence warning
does not affect these focused tests.

## Only valid restart sequence

1. Treat the current probe as WIP.  Do not run a real benchmark.
2. Close the three lifecycle gaps above and independently replay the full
   exact-type, coherent-rehash, clone-registration, TOCTOU, deadline, and
   `BaseException` hostile matrix.
3. Re-run the focused 76 + 34 tests and the adjacent K2/K3/pair/scheduled/fresh
   regression set.
4. Before any real K3 run, execute the fixed large-topology K3 memory sentinel
   in a bounded cgroup for three cold processes, with explicit opt-in.  Record
   baseline, post-pair, pre-S, every pattern, and terminal aggregate memory.
   This is empirical calibration, not a mathematical allocator bound.
5. Keep the 2.5 GiB, 384 MiB, and 896 MiB gates unless that preregistered
   calibration justifies a stricter empirical gate.
6. Only then decide whether one new CIFAR100-medium iid2 K3 build-only run is
   allowed.  Do not rerun K2, switch iid, add time, run TinyImageNet, run K4,
   or open Gate-6/14/40 as a substitute.

## Pause declaration

Exploration is paused at this checkpoint.  No real/large process is running,
no new benchmark artifact was produced during this handoff, and existing
artifacts were not modified.  Resume from the code-audit and bounded-memory
steps above, not from another benchmark launch.

## 2026-08-09 resumed checkpoint: lifecycle closed, algorithm line reopened

The three K3 probe lifecycle gaps listed above were closed without expanding
the benchmark scope.  The current frozen toy-only checkpoint is:

| File | SHA-256 | Evidence |
|---|---|---|
| `hybridz_phase_clique_build_probe.py` | `5e81f670f9575ad394da39146981d3659300d484ce7b36b252f4c8c4ea4d934a` | `py_compile`, probe 77/77 |
| `test_hybridz_phase_clique_build_probe.py` | `aea35727146f35100dc5561aba28ee3697352263421bd0535e1ac6e23aed4ec2` | same |

An independent narrow audit replayed the four formerly open semantics and
reported **NO-BLOCKER**: adoption validation failure/`KeyboardInterrupt`, a
resource-stop terminal RSS veto, a preceding localized-finalizer
`KeyboardInterrupt`, and a second exception in terminal checksum handling.
In all four cases the original K3 transaction content remained intact, its
actual counters were retained where applicable, no failed transaction was
promoted, and the process-local registry ended empty.  No real or large
profile was run.

This closes the K3 lifecycle work package; it does **not** authorize a K3 real
run.  The current session's cgroup-v2 ancestry has controller files on the
delegated leaf/user subtree but lacks `memory.current`, `memory.max`, and
`memory.peak` at `/sys/fs/cgroup`.  The strict sentinel therefore correctly
reports an incomplete controller boundary.  A bounded `systemd-run --user`
probe can create a 2.5-GiB leaf, but cannot repair the missing ancestor
contract.  The large memory calibration and all real probes remain disabled;
the 2.5-GiB/384-MiB/896-MiB gates are unchanged.

### Pre-registered algorithm candidate: RC-MPH

The resumed main line is algorithmic rather than further infrastructure
hardening.  The first controlled candidate is **Rival-Separable Correlated
Multi-Plane Hypograph Dual** (`RC-MPH`).  It never mixes different rivals.
For each rival independently, it jointly optimizes a simplex over two to four
independently valid suffix upper planes and the dual of one immutable,
residual-correlated prefix frame:

```text
U_r = max_{z in P} min_k f_{r,k}(z),     final upper = max_r U_r.
```

The `min` is a hypograph construction.  Multi-rival execution may share
prefix replay, factorization, and caches only; it has no cross-rival geometric
authority.  This distinguishes the candidate from the C72-invalid joint rival
selector, final-neuron PairHull, single frozen-alpha replay, and the previous
free-cube pairwise mixture.  Mathematically it is a correlated-prefix convex
plane mixture; its value must come from optimizing plane weights and the
prefix constraint dual together.

The decisive dyadic toy is fixed before implementation:

```text
x in [-1,1]
p = x
q = x                    # immutable causal equality p = q
s = p + q
h = ReLU(s)
m1 = p/1 - h - 1/4
m2 = p/2 - h - 1/8

rival 1 planes: p - 1/4,             -q - 1/4
rival 2 planes: p/2 - 1/8,  -p/2 - q - 1/8
```

All four planes must have a strictly positive independent support
(`3/4, 3/4, 3/8, 11/8`).  With the original `p=q` prefix, the two bundle
supports must be exactly `-1/4` and `-1/8`, matching an independent Fraction
graph oracle and exact ReLU MILP.  The expected optimal simplex weights are
`(1/2,1/2)` and `(3/4,1/4)`.  Removing either required plane, deleting the
causal equality, or substituting a same-shape wrong stable ID must eliminate
the SAFE result or fail closed.  A permanently unsafe extra rival must keep
the final OR property UNKNOWN; cross-rival plane mixing is forbidden.

The first implementation stage is restricted to two new, disconnected files:

```text
act/back_end/hybridz_tf/property_correlated_plane_bundle.py
act/back_end/hybridz_tf/test_property_correlated_plane_bundle.py
```

It must provide point consistency, interval-width and affine-Jacobian checks,
Fraction and independent LP/MILP agreement, original-frame outward replay,
rival-separable scalar/batch equivalence, stable-ID and receipt tamper
rejection, and seeded dyadic soundness cases.  Candidate solver status has no
authority; the module remains `proof_authority=false` and
`verdict_authority=false` and is not connected to `operator_hz`, `solver_hz`,
the verifier, BaB, or configuration files.

The track closes immediately if the fixed toy does not cross zero, if the
gain survives removal of `p=q`, if it requires cross-rival mixing, more than
four planes per rival, a dense prefix copy, or unbounded resources.  Passing
the toy authorizes only a controlled primitive and blast-radius regression,
not a real instance.  A later production-shape candidate must first satisfy a
fixed synthetic cost gate and then the existing Gate-6 stop loss; Gate-14/40/
400 remain prohibited until Gate-6 shows an additional correct solve or a
clear registered margin/width gain at acceptable cost.

### Pre-registered complementary candidate: PC-CRF

`RC-MPH` cannot repair every upstream triangle fake point: it only combines
legal suffix planes against the prefix it receives.  The complementary
candidate is a **Property-Conditioned Cross-Layer Residual Facet** (`PC-CRF`).
For two ancestor/descendant unstable ReLUs connected through one authenticated
residual join, write their triangle residuals as

```text
rho_i = y_i - alpha_i z_i
rho_j = y_j - alpha_j z_j.
```

A property-selected direction `d_i rho_i + d_j rho_j` is maximized over all
four joint phase cases while retaining the original recursive pre-ADD shadow,
stable IDs, and local affine equalities.  The resulting finite support is a
valid reusable prefix facet.  Candidate LPs may propose it, but an independent
Fraction/original-frame replay must check every phase case; solver status has
no authority.

The fixed discriminator is:

```text
x in [-1,1]
y = ReLU(x)
z = y - x - 1/2          # genuine residual join
v = ReLU(z)
q = -2x + 3y - 3v
```

Both individual ReLU interval bounds are exact and both ReLUs remain
unstable.  The ordinary composition of their triangle relaxations must have
upper `3/2`, attained at the fake point `(x,y,z,v)=(0,1/2,0,0)`.  The exact
Fraction graph and independent exact-ReLU MILP must both have upper `1`.  With
the fixed secant residuals `rho_i=y-x/2` and `rho_j=v-z/2`, four-phase
projection must derive and independently validate

```text
rho_i - rho_j <= 1/4.
```

Adding only this facet must reduce the LP upper from `3/2` to exactly `1`.
A downstream RC-MPH control using the two legal suffix planes
`min(-2x+3y, x+3/2)` must remain at `3/2`; this proves that the two candidates
address different losses.  Removing the residual join, changing either
stable ID/row tag, reversing a coefficient sign, or omitting a phase must
fail closed or lose the improvement.

The first implementation is again restricted to two disconnected files:

```text
act/back_end/hybridz_tf/property_cross_layer_residual_facet.py
act/back_end/hybridz_tf/test_property_cross_layer_residual_facet.py
```

It must include exact point and Jacobian consistency, layer-width accounting,
the baseline/facet/RC-MPH differential above, Fraction phase enumeration,
independent LP/MILP agreement, wrong-copy/tag and mutation rejection, seeded
dyadic DAG soundness, budget monotonicity, and a hard cap of four selected
pairs.  The track closes if the facet is not strict on this discriminator, is
not derivable from all phase cases, duplicates an already materialized linear
row, requires ground truth, or cannot be represented as a sparse reusable
prefix row.  Even a passing toy remains candidate-only and disconnected from
all verdict paths.

### Controlled results after pre-registration

`PC-CRF` passed its complete controlled gate and an independent hostile
review.  The frozen candidate is:

| File | SHA-256 | Result |
|---|---|---|
| `property_cross_layer_residual_facet.py` | `b413e9e5d4c1d7e8c4585b25f847fc9e71782121974bf8504ffd5cbb90e8fda7` | focused 20/20; independent NO-BLOCKER |
| `test_property_cross_layer_residual_facet.py` | `1deb4ea223b7ccf9c273924a17e63ce351b4aba6073797e16f24431861ba03e3` | same |

The fixed results were exactly the preregistered values:

```text
ordinary two-triangle LP       3/2
downstream RC-MPH control      3/2
derived residual facet         rho_y - rho_v <= 1/4
LP with the facet              1
exact Fraction graph           1
independent exact-ReLU MILP     1
raw q <= 5/4 margin: baseline  +1/4, facet/graph -1/4
```

The independent extended gate covered all `theta=n/256`, `1<=n<=255`, and
130,815 exact graph points.  All phase endpoints and affine phase segments
were contained by the derived hull; every exact LP elimination remained a
`Fraction`.  Hostile tests closed bad-ID/tag coherent self-signing, deep
certificate bool/float equivalence, invalid-facet under-approximation,
validate/consume ABA, reversed/duplicate/negative pair proposals, and
derive-time caller mutation.  The final validator works from an entry deep
snapshot and the only numeric consumer re-derives a private trusted
certificate.

This is a genuine controlled geometry improvement, but not yet a benchmark
breakthrough.  It equals the ideal complete K2 phase hull on this toy; its
hypothesized advantage is compression and all-rival reuse of one sparse
prefix facet.  No Operator-HZ row has yet been materialized, no synthetic
production-shape timing exists, and no CIFAR100/TinyImageNet or Gate-6 run is
authorized from this result.

`RC-MPH` has now also passed its complete controlled gate and an independent
hostile review.  The frozen candidate is:

| File | SHA-256 | Result |
|---|---|---|
| `property_correlated_plane_bundle.py` | `881f32777706310debcb94e375e8313eb6c29d37fbda16488dede9849d846a7b` | focused 19/19; independent NO-BLOCKER |
| `test_property_correlated_plane_bundle.py` | `9b16c282b8f6af594a0f89042194e55f0395eb63c6a53e06ff2b50008e5ea868` | same |

The final review replayed the stable-ID float-truncation, hostile schema
subclass, coherent `bool == int` receipt, frame/CSR/plane ABA, negative-dual,
outward-overflow, and no-dense-conversion attacks.  An independent 64-case
dyadic `Fraction` primal oracle found no under-bound; all outward binary64
values enclosed the exact replay.  The decisive two-rival toy retains exact
bundle uppers `-1/4` and `-1/8`, while all four independent single-plane
supports remain positive.

This freezes RC-MPH only as a disconnected numeric primitive.  Plane-validity
and prefix-provenance authority remain false, and it has no production
consumer, solver handoff, verdict authority, or CIFAR100/TinyImageNet result.

## 2026-08-10 pause checkpoint: Phase A extension audit pending

Work is paused at the user's request.  All implementation and audit agents
were stopped before this checkpoint was written.  No real, large, CIFAR100,
or TinyImageNet execution occurred during this phase, and no verification
coverage changed.

The independently reviewed Phase A base remained:

| File | SHA-256 | Evidence |
|---|---|---|
| `constraint_program.py` | `aaf7f36a39ef348733adf6b88621e21d63c0b2f03f99eb696847622e209ad7c2` | bounded-only NO-BLOCKER before the API extension |
| `test_constraint_program.py` | `f9c060b9c4a1fb84d50f14dcf9dcc19aba51eb1c844994d430fd9ce1eb85a132` | 60/60; combined 81 tests + 56 subtests |

Read-only Phase B preparation then found two genuine API gaps.  The core could
not store an existing complete Operator LE tag without appending another
`:<layer_id>`, and it had no explicit whole-arena terminal discard for an
Operator build that fails after one or more appends.  A consumer-side tag
sidecar and automatic legacy fallback were both rejected because they would
weaken provenance and failure atomicity.

The current working-tree candidate adds only:

- `prepare_le_exact_tag` / `append_le_exact_tag`, which bind an exact builtin,
  nonempty complete tag directly into occurrence authority, native replay,
  legacy replay, and the program digest while leaving the old tag API intact;
- `ConstraintArena.discard()` and the idempotent `close()` alias, which move
  the owner and arena together into an unsealed terminal state, consume all
  pending capabilities, create no program, never roll back external factor
  IDs or sequence numbers, and reject every later owner/arena mutation or
  seal.  Discarding an already sealed arena is rejected and cannot revoke its
  program.

The exact paused candidate is:

| File | SHA-256 | Local evidence at pause |
|---|---|---|
| `constraint_program.py` | `7dcafa2d571afc6184f5c2f1c7a75c7a3db3710d7ccc605b8063f7039636fc35` | `py_compile`; 69/69 focused; self-injected 104/104 old-or-complete-new publication outcomes |
| `test_constraint_program.py` | `3b4c0845ea735b1e9b22a220299d2437ddfd24bfad5945ee0638f89c8a6a3361` | combined core + disconnected oracle + allocator: 90 tests + 64 subtests |

This candidate is **not independently frozen**.  The independent reviewer had
matched both hashes, passed `py_compile` and 69/69, completed the static
exact-tag and arena-first/owner-final discard publication review, and had not
reported a counterexample.  It was still running the Unicode/NUL/tag-rebind,
terminal-state, poisoned-owner, callback/reentrancy, GC/ABA, per-line trace,
double-interruption, and old iterator-authority matrices when the pause was
requested.  The review was interrupted deliberately, so absence of a reported
counterexample is not a NO-BLOCKER verdict.

Phase B made no code changes.  `operator_hz.py` still has no constraint-program
consumer, and no focused sink test was created.  Phase C native HiGHS loading,
Phase D verifier/config/K2/K3 provenance, synthetic production comparison, and
all real gates remain unstarted.

The only valid restart sequence from this pause is:

1. Recompute the two current candidate hashes above and confirm no file moved.
2. Resume the interrupted independent exact-tag/discard review on those exact
   hashes, without editing, and obtain an explicit NO-BLOCKER or fix/re-freeze.
3. Only after NO-BLOCKER, resume the default-false Phase B Operator sink.  Its
   first result must byte-match the legacy `SparseHZono` core, bounds, tags,
   factor IDs, and metadata while recording virtual and physical row counts
   separately.  Any sink failure terminally discards the arena and fails the
   build; it must never retry through the legacy path.
4. Independently review Phase B before beginning native solver consumption.
5. Keep real/large and the 6 -> 14 -> 40 -> 400 sequence prohibited until all
   offline integration, provenance, RSS, and performance gates are green.

At this pause there is one strong disconnected CIFAR-shaped ADD result but no
dataset breakthrough: source and streaming RSS/payload were approximately
halved in the synthetic sentinel, while Conv full-stage candidates remained
below their promotion gate.  TinyImageNet still lacks a comparable exact
Operator-HZ receipt.  Therefore neither dataset has a new verified solve to
report.

## 2026-08-10 resumed checkpoint: Phase A extension frozen

The paused hashes were reproduced exactly before work resumed.  The interrupted
independent review then completed 430 hostile checks and rejected the paused
`7dcafa...` candidate for two reasons:

1. Both old and exact-tag APIs accepted a lone-surrogate builtin string,
   committed it, and leaked `UnicodeEncodeError` only when `seal()` attempted
   the canonical UTF-8 digest.
2. A swallowed nested mutation at the final owner swap set the sticky
   reentrancy flag but still allowed all nine outer operations to publish a
   complete new state; `seal` could consequently register a program instead
   of failing closed.

Both defects were fixed without changing the public scope.  Text validation
now rejects non-UTF-8-encodable strings with `ConstraintProgramError` before
owner/arena activation, staging, ID allocation, or sequence burn.  Owner
finalization now detects sticky reentrancy, restores the captured epoch,
returns the owner to idle, preserves poisoning after external effects, and
raises `ExternalAllocatorContractError`; seal provenance is removed before a
program can become authoritative.

The replacement Phase A extension is independently frozen:

| File | SHA-256 | Evidence |
|---|---|---|
| `constraint_program.py` | `fc150f0e281037fa5baffd28d59bcf5ad4c691fcd693f0a8853ee04cb5e2939d` | focused 72/72; combined 93/93; two independent NO-BLOCKER reviews |
| `test_constraint_program.py` | `b26aeb3e2ad09c909e0d778ee272e7165deddce9e94036aefbeafd80444fdd9a` | Unicode and owner-final-swap regression gates |

The Unicode review rejected 28 surrogate injections and 20 type/empty-string
attacks before staging, while preserving BMP, astral, U+10FFFF, NUL, and
multi-colon tags in multi-batch native/legacy replay and canonical digests.
The owner review covered all nine operations, create/existing and old/exact/
band branches, external-touch poisoning, rollback double interruption, and
old program/iterator authority.  Every case failed closed with an idle owner;
failed seal created no program authority.  Both reviews retained the existing
bounded-only caveat and ran no real/large workload.

Phase B may now depend only on these replacement hashes.  Its lifecycle is
two-stage: any failure before successful seal terminally discards the arena;
after seal, replay or final assembly failure preserves the complete sealed
program but fails the whole build, returns no partial HZ, and never retries the
legacy path.  Revoking or discarding a successfully sealed authority is not an
allowed recovery operation.

## 2026-08-10 stage pause: recoverable construction in progress

Work is again paused at the user's request.  No new algorithm or integration
work may start in the background.  One already-running bounded worker is
allowed only to finish the recoverable-construction tests/audit described
below, write its result to
`artifacts/hybridz_largecls_gates/constraint_program_two_stage_checkpoint_20260810.json`,
and stop.  That JSON, rather than the transient hashes in this paragraph, is
the first artifact to inspect on resume.

Phase B reached a meaningful default-off integration candidate before its
freeze review:

| File | SHA-256 | Evidence before red-team |
|---|---|---|
| `operator_hz.py` | `71c3a87893cb5141b211c7ae36375682b41094f3ca42c1cc822062578ea69439` | exact sink candidate; no solver consumer |
| `test_operator_constraint_program_sink.py` | `3cf3f22c49b7cfd83f5d31efa73d9997ac2203d402f9bfb3e80eecc32f2ffe6b` | focused 12/12 plus 4 subtests |

The default path neither imports nor constructs the constraint core.  The
enabled path is restricted to `exact_budget=-1` and rejects preactivation LP,
property-tail/pruning, and micro-RLT consumers before creating an owner.  It
uses exact-tag `LE` blocks for active/unstable ReLU and affine-chain rows, and
only `ADD_MATERIALIZE` may create a guarded `RANGE`.  A paired residual toy and
all-point input were byte-identical to legacy for all dense/CSR arrays, bounds,
stable IDs, tags, release telemetry, and metadata after excluding the dynamic
clock.  Its one-row ADD was stored as one native/source row and two virtual
facets with zero fallback.  RBS adaptive K4 passed 2/2; existing Operator
add-fusion, residual-normal-form, preactivation-hardening, and micro-RLT tests
passed 43 tests plus 16 subtests.

The Phase B freeze review nevertheless found four lifecycle blockers, so the
candidate is not promoted:

1. `ExternalFactorAllocatorAdapter.bind()` may register a binding and then be
   interrupted before the returned handle is assigned to the sink.
2. `ConstraintProgramOwner(adapter)` has the analogous post-registration,
   pre-assignment window, leaving an open owner without a recoverable handle.
3. A cleanup exception can replace the original build exception.
4. A consumer-body exception during legacy replay does not deterministically
   close the cursor; its registry capture can remain live at offset 256.

The last two are local Operator fixes.  The first two require a public Phase A
construction protocol; private-registry recovery and `object.__new__` hacks
were rejected.  The in-progress core therefore adds exact, non-authoritative
`reserve()` handles followed by one-shot `initialize(...)->None` for both the
adapter and owner.  Adapter publication uses a durable commit intent so its
adapter record, allocator binding, namespace lease, and reservation removal
can expose only terminal `POISONED/OLD` or exact `READY/NEW`.  Owner snapshot
callbacks run once; a complete main entry always wins over repair, while any
pre-publication callback failure terminally poisons the reservation.  Existing
`bind()` and `ConstraintProgramOwner(adapter)` remain compatibility wrappers;
Phase B will use only the recoverable API.

At the moment this pause was recorded, those two files were still moving and
had the non-authoritative transient hashes:

```text
a4f49734256a9782ed5e2236faff015575025ab3cb1f04e62333c8fd9aa5d400  constraint_program.py
00be6ffde1e24522f88e5c5b7e32cf3f22615c2f3f4cb2a6a2df6e40a9bbff38  test_constraint_program.py
```

They had passed `py_compile`, old construction/GC/namespace smoke tests, and an
8-part new fault matrix; a longer focused run encountered the already recorded
environmental futex wait and was being rerun in bounded single-thread groups.
These hashes are not frozen and must not be used as a dependency.  The
background checkpoint must record final hashes, exact test counts, independent
audit verdict or remaining blocker, absence of orphan test processes, and the
first resume action.

Dataset status remains unchanged.  CIFAR100 has a strong disconnected ADD
representation result (approximately half source rows, nonzeros, retained
payload, and streaming RSS) and now a byte-identical default-off Operator toy
integration, but no production promotion or additional verified real case.
Conv full-stage candidates remain below their promotion threshold.
TinyImageNet still has no comparable exact Operator-HZ receipt or authorized
real run.  No CIFAR100/TinyImageNet real or large workload was run in this
stage, so neither dataset has a newly verified solve.

Resume in this order only:

1. Read and hash-check the automatic two-stage checkpoint JSON; finish or fix
   its independent adapter/owner audit if not `frozen_no_blocker`.
2. Update Phase B to use only public `reserve/initialize`, preserve the primary
   build exception across bounded cleanup, and close every replay cursor in a
   `finally`/context boundary; re-freeze and repeat both Phase B red teams.
3. Only after Phase B NO-BLOCKER, run the full synthetic production-equivalence
   and RSS gates, then begin native HiGHS streaming.  Real gates remain closed.

## 2026-08-10 resumed result: recoverable Phase A and default-off Phase B frozen

The automatic checkpoint was read before any edit.  Its final Phase A files
matched exactly, no orphan test process remained, and the previously missing
adapter review was rerun independently.  The replacement recoverable core is
now frozen NO-BLOCKER:

| File | SHA-256 | Evidence |
|---|---|---|
| `constraint_program.py` | `d301ebf546cc01bfdd133a00572ca87dc7632331b81188d3812635b99f7b5ab0` | 80 tests + 103 subtests; independent 27-point executable line-event fault matrix |
| `test_constraint_program.py` | `c41968e7800ed8e819a7440c32814c959fc41c096435ffc9d9833b65da37d205` | adapter and owner two-stage construction, competition, GC/ABA, reentrancy, partial-read and primary-exception gates |

The public construction sequence is now the only integration contract:
`reserve()` first returns a non-authoritative handle, and the caller stores
that handle before invoking one-shot `initialize()`.  The adapter publication
converges to terminal poisoned OLD or complete READY NEW; a complete owner can
always be recovered through its retained handle and terminally discarded
before seal.  The independent review retained the existing bounded-only
fault-model caveat: repeated destruction of every repair attempt is required
to remain fail-closed, but liveness is not promised.

Phase B was then updated only in `operator_hz.py` and its focused test.  The
four previous lifecycle blockers are closed:

1. the sink uses only public adapter and owner `reserve/initialize`, storing
   each handle before initialization;
2. owner initialization or arena-creation public-return interruption is
   recovered through that same owner and terminally discarded;
3. a cleanup `BaseException` is attached as a note and cannot replace the
   primary build exception object;
4. legacy replay owns an explicit cursor and performs bounded close; a cursor
   close failure cannot replace a consumer-body primary exception.

The Phase B default-off synthetic candidate is independently frozen:

| File | SHA-256 | Evidence |
|---|---|---|
| `operator_hz.py` | `4dff6de88b64dd583be8ff2eebc145da720ebdb31f92a3de8065edc4d1ec2d7b` | focused 16/16; combined Operator/RBS 61 tests + 20 subtests |
| `test_operator_constraint_program_sink.py` | `92ea98ac0da1b7d2821946995b8b90596a7b041351c9c3bf9a091fe6ba434f5c` | two-stage return faults, cleanup double fault, offset-256 consumer/close double fault, parity and no-fallback gates |

Both independent reviewers returned NO-BLOCKER for the declared default-off,
synthetic-only scope.  The disabled path still does not import the constraint
core.  Enabled preflight requires `exact_budget=-1` and rejects preactivation
LP, property-tail/pruning and micro-RLT before adapter, owner or factor-ID
creation.  A runtime forbidden-hook gate with three unstable ReLUs produced
three exact binary factors and only exact ReLU rows while triangle and dual
hooks were disabled.  No BaB or backward call was introduced.

ADD materialization remains the only allowed RANGE family, and any per-row
fallback fails the whole build.  An independent width-129 test stored 129
native/source RANGE rows for 258 virtual facets, replayed batches `(0,256)` and
`(256,2)`, and reproduced every legacy dense array, CSR buffer, bound, stable
ID, tag and metadata byte-for-byte with `fallback_pairs=0`.  The all-point
input case also preserved the complete reserved input-ID provenance while the
live factor frame and constraint program remained empty.

The machine-readable records are:

- `artifacts/hybridz_largecls_gates/constraint_program_two_stage_checkpoint_20260810.json`
- `artifacts/hybridz_largecls_gates/operator_constraint_program_phaseb_checkpoint_20260810.json`

Phase B is still not a solver or proof promotion.  The first Phase C audit
found the next concrete boundary: `hz_objbound_decide` receives only the
expanded `SparseHZono`, whereas the authenticated `ConstraintProgram` lives on
`OperatorHZBuild`.  Native HiGHS streaming therefore needs an explicit
one-shot handoff which binds program identity, exact continuous/binary factor
IDs, objective ordering and the final HZ.  It must not rediscover authority
from mutable HZ attributes or silently fall back to the expanded row path.
The first streaming mode must also reject existing whole-matrix consumers
(prefix/dual, equation substitution, singleton/FBBT and connected presolve)
until each is migrated explicitly.

Dataset status is consequently still honest but unchanged at the solve level:
CIFAR100 now has an independently frozen, byte-identical default-off Operator
representation of the approximately half-sized ADD source program, but no
native solver consumption and no additional verified real case.  TinyImageNet
still lacks a comparable exact Operator-HZ receipt.  No real or large workload
was run in this stage.

## 2026-08-10 pause: Phase C disconnected loader is NO-PROMOTION and under red-team

The first native HiGHS RANGE/LE consumer was implemented strictly as a
disconnected two-file candidate.  It streams native program batches, maps
`xi_b = 2 z - 1` using exact dyadic row sums and directed lower/upper rounding,
rejects HiGHS tiny/large/infinite thresholds, and independently replays an
optimal incumbent in the original coordinates.  It imports no production
consumer and grants no producer, proof, verdict, or solver-status authority.

The paused candidate hashes are:

| File | SHA-256 | State |
|---|---|---|
| `constraint_program_highs.py` | `c63061249f86c63894ed2076988c0afea13f4d0145314ad3747c0016ef9aa2db` | frozen for audit, not promotable |
| `test_constraint_program_highs.py` | `6808a8d6cd8ff3cec7376d2a320d1d605b859314045532a4997403ebdfe2be11` | 24/24 focused tests |

The mathematics and normal-path tests are useful, but the complete route is a
clear performance failure.  An independent 96-row AB/BA run measured 17.314 ms
for the candidate and 2.008 ms for the legacy baseline, or 0.11596x.  A
strictly balanced eight-repeat check measured 0.11444x wall and 0.11440x CPU;
timed stages covered at least 99.72% of the candidate wall time.  HWM was not
measured, the full promotion gate is incomplete, and promotion is fixed false.
This negative result closes only the disconnected implementation: it still
builds and seals the program, reconstructs a complete legacy HZ solely to make
the disconnected binding, transforms/loads native rows, solves, and then
replays the original program.

The soundness freeze review has already found promotion-level lifecycle and
receipt defects, so the candidate hash above is withdrawn pending its final
report.  Confirmed defects include hostile cleanup-exception stringification
replacing a primary BaseException, a binding-only receipt falsely reporting
loaded binary integrality, and three terminal-registry/ABA gaps which can leave
or revive a failed handoff after graph restoration.  The frozen review is now
complete with verdict `BLOCKER`; it changed no candidate file and ran no
real/large workload.  Its machine-readable report is
`artifacts/hybridz_largecls_gates/phasec_loader_soundness_redteam_20260810.json`
with SHA-256
`6542bc84de6d015da255920ebd1d9b4239440d97c63fad205eb49907c80dd212`.

There is also a decisive Q=1 stop-loss.  Candidate program build and seal alone
takes about 3.596 ms, while the entire legacy baseline is about 2.015 ms; a
1.5x result would require at most 1.343 ms total.  Therefore the next design
cannot keep the existing core and merely attach a solver sidecar.  It must be
either a solver-ready primary representation, or a real multi-query session in
which both candidate and baseline receive the same reuse opportunity.  The
main fast path must expose native stored blocks directly (especially whole
batches with `A_bin.nnz == 0`) rather than reconstructing 256-row replay
batches, converting int64 indices, and running Python `Fraction` per row.  A
producer-owned input-factor capability plus independent rigorous concrete
forward/property validation is preferred over a second full auxiliary-row
replay.

The complete pause record is
`artifacts/hybridz_largecls_gates/phasec_loader_pause_checkpoint_20260810.json`.
The bounded design worker also completed and stopped after writing
`docs/HYBRIDZ_PHASEC_SOLVER_READY_STREAM_DESIGN_20260810.md` with SHA-256
`34b95421a5b933e18ae5d3ebbba2472b5fd9a8c907c870bc04614bcb260b3a74`.
No CIFAR100 or TinyImageNet real/large job is running, no background task is
left, and no background algorithm change is authorized.

Dataset impact remains limited but concrete.  CIFAR100 now has a frozen
default-off exact Operator representation and synthetic evidence that native
ADD RANGE storage roughly halves source rows, nonzeros, retained payload, and
streaming RSS.  It still has no new verified real case: the first solver route
is blocked and about 8.6x slower.  TinyImageNet still has no comparable exact
Operator-HZ receipt and no authorized real run.  New verified cases for both
datasets in this stage are therefore zero.

## 2026-08-11 goal amendment: common path and response latency first

The user added a design constraint which applies to all following work: do not
let a long tail of extremely rare cases turn the backend into a collection of
slow fallback implementations.  Soundness remains non-negotiable, but rare or
unsupported cases should be rejected by a small bounded preflight and return
`UNKNOWN`; they should not trigger scaled, relaxed, legacy, replay, or other
computational alternatives.

The selected production runtime may hold one full constraint representation
and perform one producer capture, one primary build, one native load, and one
solve/query session.  A candidate counterexample may receive one independent
forward witness check.  It must not simultaneously retain the primary core, a
full solver sidecar, and expanded legacy constraints, and it must not perform a
second whole-program auxiliary-row replay after solving.  Legacy rollback and
an experimental route may remain separate explicit configurations, but a
single request must never build both or silently switch between them.

Performance reporting now includes the cold path from request entry to the
first usable verified or `UNKNOWN` response.  In addition to the existing
single-pair 1.50x and four-thread five-pair 2.00x/1.80x-bootstrap gates, the
candidate median first-result speedup must reach 1.50x and its p95 first-result
latency must not exceed the fair legacy p95.  RSS HWM, page faults, and zero
result conflicts remain mandatory.  Local builder or addRows microbenchmarks
cannot promote a route whose caller-visible response is slow.

The machine-readable amendment is
`artifacts/hybridz_largecls_gates/goal_amendment_commonpath_latency_20260811.json`.
The immediate research emphasis is therefore the measured common structure:
direct `A_bin.nnz == 0` ADD/RANGE blocks and Conv affine construction.  Rare
mixed numeric cases may remain `UNKNOWN` until the common solver-ready primary
path demonstrates the required full-stage latency and memory benefit.

## 2026-08-11 repaired Phase C audit: sound oracle, stopped performance route

The three frozen Phase-C red-team blocker families were repaired only in the
loader and its test.  The replacement hashes are
`4856d455c7d03eced6286dafc88747aa4832386fa851c872c0347651d906f699`
for `constraint_program_highs.py` and
`d61d2eae67d4cb5e6c8fb687f523a59fc326afe4bdb74de3630441587a5e6e14`
for its test; the production constraint core remains `d301ebf...`.

An independent bounded audit replayed hostile secondary exceptions, receipt
phase truth, graph-break ABA, and interrupted retirement cleanup.  The exact
primary exception was preserved, binding-only receipts no longer claim loaded
integrality, generic no-branching language was removed, and observed graph
breaks remained terminal after restoration.  The focused loader passed 32/32,
relevant core tests passed 32/32, Operator-sink plus loader passed 48/48, and
four separate adversarial checks passed.  The audit verdict is
`NO_BLOCKER_FOR_REPAIRED_DISCONNECTED_SCOPE`; it grants no production, proof,
verdict, solver-status, or producer authority.

This closure does not revive the route as a performance candidate.  A fresh
96-row check measured 17.182 ms versus 2.035 ms, only 0.1184x, with no HWM
gate.  Profiling also measured about 0.485 ms per full live-graph validation;
the disconnected execute path performs two at entry and one after cleanup, in
addition to legacy-HZ reconstruction and whole-program Fraction replay.  That
behavior is acceptable only for a correctness oracle and conflicts with the
new common-path/first-response goal.  No further fallback, lifecycle, or
performance machinery should be added to it.

The independent audit is recorded in
`artifacts/hybridz_largecls_gates/phasec_loader_repair_independent_audit_20260811.json`.
The next implementation should start fresh in the two-file solver-ready V2
candidate described by the design document: direct stored native blocks,
initially only the measured common `A_bin.nnz == 0` path, with rare mixed cases
returning `UNKNOWN`.  It must not construct the old core, a sidecar, and legacy
facets in the same request.

## 2026-08-11 solver-ready primary V2 result: useful but stopped

The two-file disconnected V2 was implemented without importing the old
constraint core, Operator-HZ, or the legacy solver.  It captures each input
once, stores binary-free RANGE/LE blocks directly in HiGHS coordinates, uses
bounded exact-dyadic integer work only for mixed rows with at most two binary
coefficients, and returns `UNKNOWN` for unsupported coefficients or fan-in.
There is no runtime fallback, legacy materialization, Python `Fraction` in the
candidate hot module, or post-solve whole-program replay.  A fair legacy
baseline was also corrected to skip `Fraction(bound)+0` for rows with no
binary coefficient; results below use that optimized baseline.

At the representative C89-ratio scale-64 fixture, physical rows/nonzeros are
897/97,549 versus 1,537/144,805 legacy virtual rows/nonzeros.  The warm Q=1
median was 11.589 ms versus 18.752 ms, or 1.618x, so the first 1.50x topology
gate passes.  Seven alternating fresh processes measured 13.475 ms versus
21.880 ms cold medians (1.624x); p95 was 14.444 ms versus 24.274 ms (1.681x),
and worst HWM delta was 12,918,784 versus 17,842,176 bytes (ratio 0.724).

The later gates do not pass.  The paired bootstrap 95% lower speedup was only
1.620x rather than 1.80x.  Four concurrent threads over five alternating pairs
measured 22.402 ms versus 32.923 ms, only 1.470x rather than 2.00x.  There is
also no rigorous concrete forward witness, page-fault gate, producer
authentication, or proof/verdict authority.  Accordingly the route is frozen
`NO_PROMOTION`; it authorizes no CIFAR100/TinyImageNet run.  Adding caches,
fallback stacks, or rare-case encodings to this layout is explicitly out of
scope.  The next effort should target the much larger Conv common-path cost or
a fundamentally merged producer/solver design, not further loader polishing.

The machine-readable record is
`artifacts/hybridz_largecls_gates/solver_ready_primary_v2_checkpoint_20260811.json`.

## 2026-08-11 Conv ownership-transfer v4: warm win, cold rejection

The only remaining Conv hypothesis was tested by replacing, rather than
optionally supplementing, the two duplicated common-path copies in the
disconnected compact-native candidate.  The native wrapper now reuses the
already private row-local affine source arrays and transfers fresh kernel
output ownership directly into the result.  It remains restricted to
monotone, injective, row-local sources; all other valid topologies are
ineligible/`UNKNOWN`, with no runtime fallback or expanded `W`.

Focused and adjacent tests passed 43/43, including bitwise center, canonical
generator, error, mass, grouped/strided/padded/dilated geometries, subnormal
and cancellation cases, factory identity, concurrency, and ABA.  On the fixed
`1x8x12x12 -> 1x16x12x12` fixture, raw/full-warm/cold-data medians were about
0.716/0.946/1.779 ms; the matched `W + generic affine` baseline was 3.702 ms,
so the model-warm compute ratio was 2.081x.  Traced peak fell to 2.350 MiB.
The static 7,929,856-nnz large-layer output lower bound fell from 182.063 MiB
for raw+freeze to 91.031 MiB for ownership transfer, but actual C89 apply HWM
is still unmeasured and the public output has no proof authority.

The new first-response gate decisively rejects the route.  Seven alternating
fresh processes measured 15.608 ms candidate versus 5.109 ms baseline, only
0.327x; p95 was 15.660 versus 5.204 ms.  The undeclared CFFI verification/load
still occurs inside the first request.  Four-thread five-pair warm throughput
was also only 1.601x rather than 2.00x.  Hiding this cost behind undocumented
prewarm, adding a runtime-compile fallback, or introducing a new native build
system is outside the common-path amendment.  V4 is therefore frozen
`NO_PROMOTION`, authorizes no real/large run, and should not be integrated.

The exact record is
`artifacts/hybridz_largecls_gates/conv_native_ownership_v4_checkpoint_20260811.json`.

## 2026-08-11 ADD-to-exact-ReLU common path: offline gates pass

The stored-real exact ReLU candidate no longer constructs Python `Fraction`
objects in its hot path.  Bounds, exact halves, and the `hi+lo` link expansion
now use normalized integer dyadics plus an error-free subtraction, while the
tests retain independent `Fraction` and MILP oracles.  All 17 numeric tests
and the combined 32 numeric/deferral tests pass, including the historical
`0.1 + 0.7*xi` rounded-RHS counterexample and minimum-subnormal half-free
graph.  The old deferral performance benchmark was also switched to this
numeric core so it no longer measures the known-inexact production primitive.

For a completed width-128 ADD whose sole intended consumer is exact ReLU, the
direct graph has `384C/128B/128E/256U/1024nnz`; materializing an ADD frame first
has `512C/128B/128E/512U/1536nnz`.  Single-thread medians were 2.419 versus
5.575 ms (2.304x).  Four-thread 15-pair medians were 14.331 versus 34.583 ms
(2.413x); the paired bootstrap 95% lower bound was 2.195x.  Seven alternating
fresh processes measured 2.957 versus 6.212 ms (2.101x), p95 3.053 versus
6.440 ms, and worst HWM delta 364,544 versus 471,040 bytes.  This is the first
current common-path experiment to pass the 1.50x, four-thread 2.00x,
bootstrap-1.80x, cold median, p95, and local HWM gates simultaneously.

It is not yet a production promotion.  `consumer_kinds` is still caller
metadata rather than a graph-owned sole-consumer seal, the historical build
API still contains its old fallback behavior, and production SparseHZ buffers
do not yet provide the required producer-owned immutable handoff.  The only
permitted next implementation is therefore one graph-owned path: sole
ADD-to-exact-ReLU consumes the numeric-v2 builder, while any topology mismatch
returns `UNKNOWN`.  It must not materialize ADD at runtime or silently fall
back.  No real/large run is authorized before that ownership integration and
its independent audit.

The full measurements and frozen boundaries are in
`artifacts/hybridz_largecls_gates/add_relu_numeric_commonpath_checkpoint_20260811.json`.

## 2026-08-11 goal revision and Operator ADD/ReLU stop-loss

The common-path amendment is now an implementation budget, not just a
benchmarking preference.  A selected request may execute one algorithm and
one full representation.  Rare cases must fail a bounded preflight and return
`UNKNOWN`; they do not earn a scaled, legacy, replay, or other computational
fallback.  Diagnostic-only full sparse scans are forbidden on the request
path, and a new specialization is allowed only when it replaces work rather
than coexisting with another runtime route.

The existing graph-owned Operator path was therefore tested before adding any
new ADD/ReLU code.  On a width-128 `ADD -> exact ReLU`, setting the existing
`materialize_add=False` reduced continuous factors/upper rows/constraint nnz
from `384/640/1408` to `256/384/896`, but complete build speed was only
1.444x.  A repeated width-64, eight-block residual fixture with 512 exact
unstable phases reached about 1.65x for build, native load, solve, and cleanup.

Profiling found genuinely redundant common costs: every layer recomputed a
complete cube only to emit unused fact-width metadata, every ReLU converted a
second fact frame for two unused differences, and the dominant unscaled
relation built two identity diagonal matrices before subtracting CSR rows.
Opposite exact relations now share one identical outward error audit and use
one exact sparse negation.  Those operations were deleted globally rather
than hidden behind another option.  The same work also exposed and closed a
canonical CSR producer bug for mixed active/unstable/inactive affine output.
Focused tests pass 18/18, including 64 seeded bitwise comparisons against an
independently recomputed reverse relation; combined Operator and numeric
regression passes 244 tests plus 102 subtests.

After simplification, the repeated fixture improved from 36.922 ms
materialized to 21.930 ms fused, or 1.684x, with p95 38.573 versus 22.663 ms.
Four concurrent requests measured 134.544 versus 95.539 ms (ratio of medians
1.408x); paired median speedup was 1.483x with bootstrap interval
[1.307, 1.705].  Absolute latency improved substantially, but this still
misses the 2.00x and 1.80x-lower gates.  The
numeric-v2 compact graph is not integrated as a second encoding: at width 128
it has the same total constraint-row count and more continuous variables and
nonzeros than the already fused Operator exact graph.

The result is a strict stop-loss, not a promotion.  Keep the removal of the
non-semantic scan and the CSR canonicalization repair, but add no ADD/ReLU
mode, fallback, or side representation.  CIFAR100 and TinyImageNet real/large
runs remain unauthorized.  The exact checkpoint is
`artifacts/hybridz_largecls_gates/operator_add_exact_relu_commonpath_stoploss_20260811.json`.

## 2026-08-11 stable-active elimination census: rejected before code

Stable-active elimination was the next large structural hypothesis.  The
existing CIFAR100 receipt contains 6,348 stable-active versus 2,177 unstable
ReLU rows, but the dominant active rows are not local factors: the first ReLU
has 4,951 active rows after a Conv whose generator matrix has exactly 27
nonzeros per row; the second has 1,320 active rows after an even denser Conv.
Deleting their local factors and equality rows therefore composes adjacent
convolutions across the identity ReLU.

A small structural replay retained the same first-two-layer 3x3 channel
pattern and the archived active/unstable fractions.  It compared the next-Conv
local-factor nnz plus both active equality directions against direct active
Conv composition plus unstable local factors.  Across nine fixed masks the
candidate used 1.146x--1.224x as many total nonzeros, median 1.160x.  A
row-local-only eligibility rule would avoid this loss but would exclude the
dominant CIFAR rows and introduce exactly the rare-case branch the current
goal forbids.

No candidate or fallback was implemented.  This is a pre-code stop-loss, and
no real/large network was executed.  The exact record is
`artifacts/hybridz_largecls_gates/stable_active_elimination_census_20260811.json`.

## 2026-08-11 common-path copy deletion and last-use stop-loss

The active objective now explicitly forbids spending request latency on rare
backend cases.  One selected request may build one full representation, load
it once, and run one solver/query session.  Unsupported shapes return
`UNKNOWN` at bounded preflight; they do not trigger a second encoding, scaled
retry, legacy rebuild, or auxiliary full-row replay.  Cold first response and
p95 are first-class gates rather than warm microbenchmark footnotes.

Four redundant common-path operations were removed from the existing
Operator path without adding a mode.  `_prepare_upper_block` no longer
boolean-slices both CSR matrices when every row is retained.
`_inflate_nonnegative` handles an all-active row vector without three boolean
gather/scatter copies.  The all-positive source-mass path no longer builds a
boolean sparse matrix and performs a second SpMV.  Finally, row-L1 mass is
reduced directly from canonical CSR data instead of cloning data, indices and
indptr through `abs(CSR)`; 256 focused randomized cases and 20,362 stress rows
matched the old reduction bitwise.

On the width-64, eight-block, 512-exact-phase fixture, 15 alternating
single-request pairs reduced build median from 17.034 to 14.296 ms (ratio of
medians 1.192x; paired median 1.153x).  Nine alternating four-request pairs
reduced group median from 99.913 to 86.773 ms (ratio 1.151x; paired median
1.201x), while request p95 moved from 106.178 to 87.866 ms.  Counts stayed
exactly `512 binary / 1536 upper rows / 4480 constraint nnz`, with zero result
conflicts; 245 tests plus 102 subtests pass.  A separate direct build-plus-solve
recheck measured only 1.403x, so this remains an absolute-response improvement,
not promotion authority, and real/large execution remains closed.

The proposed topological last-use release was tested and then fully reverted.
On the exact residual fixture, traced peak moved only from 2,784,018 to
2,745,619 bytes; on a dense affine stress fixture it moved only from
80,893,661 to 80,864,893 bytes.  Operator temporaries and final constraint
assembly, not retained `exprs`, dominated the peak.  Extending the idea to the
optional suffix/dual path would also require a second layer-bound snapshot and
would violate the new simplicity budget.  No last-use bookkeeping, suffix
snapshot representation, or fallback remains in production code.

The machine-readable measurements are in
`artifacts/hybridz_largecls_gates/operator_add_exact_relu_commonpath_stoploss_20260811.json`,
and the active design constraints are in
`artifacts/hybridz_largecls_gates/goal_amendment_commonpath_latency_20260811.json`.

### 2026-08-11 scalar roundoff and affine topology deletion

The next profile showed 232 outward-inflation calls and 280 gamma evaluations
on the same eight-block exact-ReLU build.  Fixed scalar operation counts were
being broadcast to complete row arrays before two finite/range reductions.
The selected path now validates the scalar once and preserves the identical
binary64 formula.  In affine error propagation, `abs(W)` now reuses the
already-canonical CSR indices and indptr, allocating only absolute data when
the operator contains a negative stored bit.  This is a transient topology
view, not another constraint representation.

Across 31 alternating single-request pairs these two deletions reduced median
build time from 13.563 to 12.234 ms (`1.109x` incremental), and nine
alternating four-request pairs moved from 101.268 to 90.849 ms (`1.115x`) with
zero structural/result conflicts.  A 167,772-nnz mixed-sign CSR micro-census
reduced the absolute-operator peak from 2,096,162 to 1,408,478 bytes while
both products remained bitwise equal.  Focused tests are 21/21; the broader
matrix is 247 tests plus 102 subtests.

This improves absolute response time but not the algorithmic promotion gate.
On the same topology the current fused path is only `1.394x` faster than
materialization for one request and `1.404x` for four concurrent requests,
below the required `1.50x/2.00x`.  CIFAR100/TinyImageNet real/large execution
therefore remains prohibited.

A diagnostic complete build-plus-solve scale sweep shows why the fused graph
is still worth retaining: single-request speedup rose from `1.583x` at width
64 to `1.673x` at width 128 and `1.761x` at width 256.  The improvement did
not survive concurrent contention.  Four concurrent width-128 requests
reached only `1.513x`, with paired-bootstrap 95% lower `1.492x`; width 256 was
noisy and lower at `1.376x`, with lower bound `1.263x`.  All paired verdicts
agreed, but the required `2.00x` median and `1.80x` lower bound remain closed.
These are synthetic diagnostics, not a cold-process RSS/HWM promotion run.
The next admissible candidate must replace a full shared load/solve or
representation cost; another small arithmetic specialization is stopped in
advance.

### 2026-08-11 exact-ReLU RHS deletion: synthetic pass, real topology stop

The exact-ReLU MILP loader was still constructing `Fraction` objects for
every upper row while applying `xi_b = 2*z - 1`.  Exact ReLU contributes zero
or one binary coefficient per row, so the selected path now uses vectorized
IEEE-754 TwoSum to obtain an error-free `rounded + residual` expansion.  A
positive residual advances an inequality RHS once toward `+inf`; an equality
accepts only zero residual.  Rows with multiple binary terms keep the existing
exact path.  No solver mode, constraint representation, retry, or fallback was
added.

The Operator build also stopped constructing a dictionary for every exact
phase when both property suffix replay and micro-RLT are disabled.  One scalar
counter preserves the receipt count; detailed records are allocated only for
an explicitly requested consumer.  Independent exact arithmetic covered
100,121 random and extreme input pairs with zero wrong expansions or inward
roundings.  The scoped regression is 142 tests plus 35 subtests, with compile
and whitespace checks green.

On the width-256/eight-block fused synthetic, 15 paired requests moved from
44.354 to 26.542 ms (`1.671x`).  Fifteen four-request pairs moved from 199.462
to 98.639 ms (`2.022x`), with paired-bootstrap 95% lower `1.825x`.  Seven
fresh-child request timings were 48.315 versus 29.723 ms (`1.625x`); p95 fell
from 49.508 to 30.985 ms, median minor faults were equal at 1,295, major faults
were zero, and worst HWM deltas differed by only 4 KiB.  These request timings
start after interpreter imports and the fixed model fixture are ready; full
process startup was also recorded and is not claimed as a per-request gain.

This alone did not authorize a real run.  A strict single-path replay with
`materialize_add=true` reached only `1.463x` for one request and `1.779x` for
four concurrent requests; its bootstrap lower bound was `1.652x`.  More
importantly, a read-only ONNX census found that both medium models are
pre-activation ResNets.  A one-hop count finds no direct `ADD -> RELU`, but the
Operator route is longer: all eight ADDs in each model have exactly one
`Conv/BN -> ReLU` or `Flatten/Gemm -> ReLU` main route, while the additional
consumer is the residual skip.  CIFAR100-medium has 55,460 ReLU outputs per
sample and TinyImageNet-medium has 172,296.  The fused topology is therefore
relevant, but its real first response still needed a bounded gate.

CIFAR100-medium iid2 was run once in the declared `act-py312` environment with
all unstable ReLUs exact, fused ADD, one worker/thread, no property/dual path,
and a 30-second outer timeout.  ONNX conversion/spec generation took 1.310 s,
synthesis 0.002 s, and torch-to-ACT 0.118 s; analysis/solver replay began at
1.674 s but produced no usable result before the worker was reaped at
30.193 s.  The receipt is `DIAGNOSTIC_PARTIAL/timeout`, GPU free memory was
restored, and no TinyImageNet instance was attempted.  An earlier base-Python
attempt failed before model construction because that environment lacked the
declared `onnx2torch` package; its failure receipt is retained separately.

The retained code is thus an absolute latency improvement, not a solved-count
breakthrough.  Extending the timeout or layering another fallback would hurt
the front-end responsiveness goal.  The next candidate should profile and
delete work inside the single real analysis/build stage before the first full
model exists; unsupported requests should return `UNKNOWN` before expensive
work.

The machine-readable record is
`artifacts/hybridz_largecls_gates/solver_exact_relu_rhs_commonpath_checkpoint_20260811.json`.

### 2026-08-11 live-affine stored-center prescreen

The 30-second CIFAR100 trace separated the request cleanly.  ONNX/spec setup
and Torch-to-ACT took about 1.5 seconds, interval analysis took 0.19 seconds,
and the Operator build then spent about 12 seconds in repeated SciPy sparse
`W @ G`.  Three residual affine layers alone consumed 7.26 seconds.  The
front end was not the bottleneck.

The selected optimization adds no mode, representation, solver, retry, or
fallback.  Before composing `W @ G`, the builder propagates one outward
radius around the stored affine center with `abs(W)`.  A row whose resulting
independent cube recheck is nonpositive has a sole ReLU successor and can skip
generator composition because that successor maps it to exact zero.  All
other rows keep the established sparse composition and collapsed-row recheck.
This is one high-frequency screen, not a menu of special cases.

An early prototype incorrectly reused radius about the interval midpoint.
It was rejected immediately because that radius need not enclose the stored
affine center.  The retained implementation uses
`max(outward(c-lower), outward(upper-c))`.  Sixty-four seeded sparse cases
(448 output rows) checked the screen against both full composition and an
independent Fraction exact-real upper oracle.  With both old and new paths
given the same 60-second candidate budget on CIFAR100 medium iid2, the final
`c/b/ub/Gc/Gb/Ac/Ab/Auc/Aub` shapes, CSR structure, and float64 bits were all
identical.

Under that equal decision budget, build time fell from 10.663 to 2.761
seconds (`3.862x`).  Under the normal eight-second live-affine budget, the
observed build fell from about 11.964 to 2.967 seconds (`4.032x`).  Saving the
early composition time also allowed three later residual layers to finish
the already-existing compression instead of materializing: continuous
factors fell from 25,735 to 19,591 and upper rows from 46,979 to 34,691.

TinyImageNet medium iid6 showed the same common-path effect.  Its build fell
from 30.150 to 16.771 seconds (`1.798x`); continuous factors fell from 135,660
to 60,359 and upper rows from 254,004 to 103,358.  No full TinyImageNet solve
was attempted.  The CIFAR100 iid2 30-second full request was repeated: it now
reached a 7,730,031-nnz solver model roughly nine seconds earlier, but the
first large HiGHS feasibility/objective solve still produced no usable result
before the worker was reaped.  The verdict therefore remains
`DIAGNOSTIC_PARTIAL/timeout`, not a solved-count or promotion claim.

The updated goal constraint is explicit: optimize only measured common
CIFAR100/TinyImageNet work; keep one selected runtime path; reject unsupported
rare inputs at bounded preflight; never hide a second full encoding, scaled
retry, legacy rebuild, or auxiliary replay behind a request.  Current code is
kept as a real build-latency candidate, while RSS, four-thread, and first
usable-result gates remain closed.  The next admissible step is only a single
authenticated reuse of the common solver prefix or constructive base witness;
if that cannot remove the redundant first solve, stop rather than add another
backend alternative.

The exact checkpoint is
`artifacts/hybridz_largecls_gates/operator_live_affine_prescreen_checkpoint_20260811.json`.

### 2026-08-11 common-path native RANGE load

After the affine prescreen, a real CIFAR100 row/nnz census showed that one
ordinary constraint family dominated the first property model.  The
`affine_chain_cut` forward and reverse blocks contributed about 63% of all
constraint nonzeros.  They are the same exact band stored as two one-sided
rows.  HiGHS accepts that conjunction directly as one ranged row.

The selected change is deliberately narrow.  Tags locate only contiguous
`affine_chain_cut:<layer>:forward/reverse` blocks.  Before removing a reverse
row, the loader independently checks the complete continuous and binary CSR
row structures and requires every stored float64 coefficient to differ by
exactly the sign bit.  The forward RHS becomes the upper bound and the
outward-negated reverse RHS becomes the lower bound.  Any missing, reordered,
or non-negated pair fails closed to `UNKNOWN`; there is no legacy fallback,
second solver backend, retry, or second solver model.

The binary `xi=2z-1` shift remains exact/outward.  Fraction tests, enumerated
source-versus-range membership, malformed-pair rejection, and a production
`hz_objbound_decide` entry test cover the new path.  The scoped adjacent
regression is 131/131 tests.  Matrix validation, native load, and solve also
share one deadline; a bounded native-return tail prevents the solver from
being deliberately given the entire caller budget after model loading.

The real structural reduction is substantial:

- CIFAR100 medium iid2: 34,691 to 26,499 rows (`-23.6%`) and 7,730,031 to
  5,295,727 nonzeros (`-31.5%`), with 8,192 ranged pairs and 21,444 columns.
- TinyImageNet medium iid6: 103,358 to 78,270 rows (`-24.3%`) and 23,581,758
  to 16,187,454 nonzeros (`-31.4%`), with 25,088 ranged pairs and 62,215
  columns.

These are production-entry censuses after real ONNX/VNNLIB construction, not
synthetic extrapolations.  The unchanged cold 30-second request was then run
once for each instance.  Both were still hard-timed-out without a usable
verified/`UNKNOWN` worker receipt.  Thus the range load is a real model-size
improvement but not yet a solved-instance or front-response breakthrough.
Timeouts were not extended, and no alternative solver path was added.

The next admissible work is restricted to the first native model load and
first property query.  A candidate must replace work in that single path and
improve the unchanged cold first-response gate.  Otherwise the request should
return `UNKNOWN` earlier; do not add another representation, cache, retry,
auxiliary replay, or rare-case compute branch.

The machine-readable checkpoint is
`artifacts/hybridz_largecls_gates/operator_commonpath_range_checkpoint_20260811.json`.

### 2026-08-11 stable-active RANGE extension and native-query stop-loss

The same RANGE loader now also recognizes `relu_active:<layer>` equality
bands.  This is not a second encoding or a tag-trusting shortcut: tags only
locate a contiguous forward/reverse block, while complete continuous and
binary CSR indices and every float64 sign bit must still match.  A malformed
present tag frame fails closed to `UNKNOWN`; unrelated HZ objects with no tag
frame keep their established solver route.

This one extension materially reduces both target models again:

- CIFAR100 medium iid2: the original 34,691 rows / 7,730,031 nnz now load as
  20,125 rows / 4,862,325 nnz, with 14,566 RANGE pairs.  This is a 42.0% row
  and 37.1% nnz reduction.
- TinyImageNet medium iid6: 103,358 rows / 23,581,758 nnz now load as 54,463
  rows / 13,669,073 nnz, with 48,895 RANGE pairs.  This is a 47.3% row and
  42.0% nnz reduction.

The affected regression is 132/132, with compile and whitespace/JSON checks
green.  Both unchanged cold 30-second requests still time out.  The first
CIFAR100 native cutoff call moved from roughly 17.2 seconds on the
affine-only RANGE model to roughly 14.8 seconds on the active-RANGE model,
but that is not enough to produce a usable result.

Several alternatives were deliberately evaluated only in memory and then
discarded.  A joint rival OR-MILP took 14.95 seconds; an exact-ReLU auxiliary
factor formulation reached only 1.33x; disabling the persistent LP and using
four solver threads both still timed out.  Directly composing the residual
projection nearly halved both real solver matrices (CIFAR100 to 11,933 rows,
13,252 columns and 2,559,621 nnz; TinyImageNet to 29,321 rows, 37,092 columns
and 6,530,310 nnz), but its paired complete Operator build was 0.994x and the
CIFAR100 30-second result remained `UNKNOWN`.  It therefore was not added to
production code.  No alternative query mode, projection branch, retry, or
solver option remains.

Profiling now shows the dominant latency inside native `Highs.run`, not the
Python row loader.  The next admissible algorithmic change is consequently a
single reusable native model lifecycle: the persistent LP prefilter and the
first exact cutoff must share one loaded model and one deadline.  It may
replace the duplicated lifecycle, but it may not coexist as another runtime
path or silently rebuild the model.  If that design cannot cross the 1.5x
first-response gate, stop and return `UNKNOWN` rather than adding more
backend machinery.

### 2026-08-11 reusable HiGHS lifecycle stop-loss

The single allowed follow-up was implemented as a bounded experiment: the
persistent LP prefilter retained its already-loaded HiGHS model, changed the
binary columns to integer in place, added one temporary exact-cutoff row, and
independently checked any incumbent against the original stored-float rows.
The path used one HiGHS instance, one deadline, and no hidden model rebuild;
missing or damaged session state returned `UNKNOWN`.  A focused binary toy
produced the expected exact witness with exactly one native model, and the
pre-real regression passed 122 tests plus 299 subtests.

The unchanged CIFAR100-medium iid2 cold request in the established
`act-py312` environment nevertheless timed out after 30 seconds without a
usable result.  TinyImageNet was not run: it is the larger instance, and the
common CIFAR100 stop-loss had already failed.  A separate base-Python attempt
failed before model construction because that environment lacked
`onnx2torch`; it is archived only as an environment diagnostic and is not an
algorithm result.

The reusable-session implementation and its test were then removed.  Exact
restoration was checked against the prior frozen hashes:

- `solver_hz.py`: `d4fe5491aec1d41819a00a6638d2ca7fd3b2d128939667f53623d8c89db4300d`
- `test_solver_hz_binary_shift.py`: `894af568e360d4d30df1588b02733db4c9e95b43fc07e20c17eb26fddef64071`

The valid progress therefore remains structural rather than solved-count:
the retained common RANGE path removes 42.0%/37.1% of CIFAR100 rows/nnz and
47.3%/42.0% of TinyImageNet rows/nnz, but both targets still have zero new
usable results in the unchanged 30-second gate.  Do not add more solver modes,
rare-case branches, retries, or side representations.  The next change must
replace measured work in the retained native run and cross the cold
first-response gate before it is integrated.

## 2026-08-11 common-case-only exact-MIP census

The goal is now stricter: rare-case optimization is not a production goal.
Unsupported numerical or topological cases receive one bounded preflight and
`UNKNOWN`; they do not receive a second encoding, repair pass, solver retry,
or background computation.  A selected request may build one primary
representation, load it once, and follow one solver path.  Cold request entry
through the first usable response is the latency boundary visible to the
caller.

The additional real-model census did not run a TinyImageNet solver.  It only
stopped immediately before native solve and measured the common exact model.
CIFAR100-medium iid29 has 1,061 phase binaries and compacts to 17,489 rows,
19,600 columns, and 3,498,525 constraint nonzeros.  TinyImageNet-medium iid17
has 1,919 phase binaries and compacts to 54,327 rows and 12,127,453 nonzeros.
The exact-ReLU dense branches remain the dominant nonlinear source; front-end
parsing and the Python row loader are not the measured bottleneck.

Two cheap-looking alternatives are now closed without production code.  In
CIFAR100 iid29, every one of the 99 property objectives reaches all 17,489
native rows and all 3,498,525 nonzeros, so connected-component slicing removes
nothing.  Deterministically rounding 87 persistent-LP candidates yielded zero
base-feasible points; the best still violated a base row by 14.64569.  No
rounding repair or component-specific path will be added.

One non-swept deadline reallocation was tried because it deletes LP work
rather than adding a new algorithm.  Reducing the existing persistent-LP
fraction from 0.90 to 0.20 on CIFAR100-medium iid2 still hit the unchanged
30-second wall limit (`30.292985` seconds at the parent); the analysis call
returned only at `28.640063` seconds and no usable result was serialized.  No
other fraction will be scanned and no setting was retained.

The current dataset verdict is therefore honest: the structural common path
is materially smaller and builds faster, but CIFAR100 and TinyImageNet have no
new solved instance in the formal cold budget.  The next admissible idea must
replace the LP-to-exact-MIP native work itself and first pass a theoretical
stop-loss plus a single-path `>=1.5x` toy gate.  It may not coexist with the
current solve as a fallback.

Machine-readable evidence is in
`artifacts/hybridz_largecls_gates/commonpath_exact_mip_census_20260811.json`.

### 2026-08-11 exact-phase MIP-start stop-loss

One final low-overhead replacement candidate was tested without a parameter
sweep.  The exact-ReLU builder already has each stored preactivation center
when it allocates the corresponding phase binary, so its sign can be supplied
as a binary-only partial start to the same HiGHS cutoff MILP.  This requires no
extra network forward pass, model representation, solver, retry, or proof
path.  The hint has no authority: HiGHS must still satisfy every original row,
and any returned incumbent keeps the existing independent validation.

The candidate crossed the isolated same-topology gate: nine paired width-128
requests improved from 4.498 ms to 2.406 ms (`1.869x`), and width 256 improved
from 8.036 ms to 3.911 ms (`2.055x`), with matching witness outcomes.  It did
not survive the response-concurrency gate.  Nine groups of four simultaneous
width-128 requests had only `1.256x` median paired speedup and a paired
bootstrap 95% lower bound of `1.223x`, below the required `2.00x` and `1.80x`.

The implementation and focused test were therefore removed before any real
CIFAR100 or TinyImageNet run.  The retained Operator and solver hashes were
restored exactly to `0f3ea83b...f5707` and `d4fe5491...4300d`.  No alternate
phase heuristic, threshold, start repair, or retry will be explored.  The
machine-readable stop-loss is
`artifacts/hybridz_largecls_gates/exact_phase_mip_start_stoploss_20260811.json`.

### 2026-08-11 validation-only, no-sampling rule

The project goal now explicitly forbids concrete input sampling as a way to
find counterexamples.  The verifier must not execute ONNX at an input-box
center, boundary point, random point, grid point, or any other sampled input;
PGD and related concrete attacks are also forbidden.  Solved-count progress
must come from the verifier itself: its abstract domain, exact ReLU phase
semantics, constraint representation, certified bounds, and exact solver.

An experimental center precheck briefly evaluated during this work is fully
withdrawn.  Its verifier and gate changes, tests, receipts, and solved-count
credit were removed.  It must not be cited as CIFAR100/TinyImageNet progress.
The retained formal dataset verdict therefore remains unchanged: structural
RANGE compaction is real, but neither target family has gained a new solved
instance under the formal cold budget.

Concrete ONNX execution is permitted only as independent, zero-tolerance
validation of a candidate already produced by the verifier's authoritative
exact solve; it may not generate, select, rank, search, or repair candidates.
No sampling/attack path may be added as a fallback when verification returns
`UNKNOWN`.

### 2026-08-11 objective-guided exact-MIP stop-loss

One common-path replacement was evaluated after the no-sampling rule was
frozen.  The existing exact phase/binary MILP was kept, but HiGHS minimized
the property objective instead of solving a zero-objective model with the
property encoded only as a cutoff row.  Solver infeasibility, objective bounds,
and optimality had no SAFE authority; only an incumbent replayed against the
original constraints and cutoff could leave the solver as a witness.  This
was one solver session with no retry or fallback, and it used neither sampled
ONNX inputs nor PGD.

On CIFAR100-medium iid29, two unchanged 30-second cold runs still produced no
usable result.  The second run recorded `UNKNOWN` with no counterexample before
result serialization; its analysis/solver/replay body took 26.470 s and the
outer request timed out at 30.107 s.  TinyImageNet was not run after the smaller
common-path stop-loss failed.  The candidate and its temporary status logging
were removed; `solver_hz.py` and the gate were restored to SHA-256
`d4fe5491...4300d` and `45eb6536...7b5a1`.  The machine-readable receipt is
`artifacts/hybridz_largecls_gates/exact_mip_objective_search_stoploss_20260811.json`.

This changes no formal solved count.  Retained progress remains structural:
native ranged rows materially reduce rows and nnz, but both CIFAR100 and
TinyImageNet still lack a new conclusive instance inside the formal cold
budget.

### 2026-08-11 strict phase-equivalence sharing stop-loss

Strict sharing of exact ReLU phase variables was checked directly on the
formal Operator-HZ builds, with every unstable ReLU exact and ADD
materialization disabled.  The solver entry was replaced in-process by an
immediate `UNKNOWN`, so the census ran abstract propagation and HZ construction
only.  It did not execute a concrete ONNX input, sample the input box, invoke
PGD, or run HiGHS.

For phase sharing to preserve the complete feasible set, two preactivations
must have the same stored-real affine value over the same stable factors and
compatible error semantics.  A nonzero row-local roundoff interval is
independent freedom: two otherwise equal affine centers/generators may take
opposite signs inside their separate error intervals, so their binary phases
cannot be identified.

All 1,061 CIFAR100-medium iid29 exact rows and all 1,919 TinyImageNet-medium
iid17 exact rows carry nonzero independent error.  The eligible zero-error
subset is therefore empty in both models; strict identical classes, exact
positive-scalar classes, and potential binary savings are all zero.  No
runtime signature table, eligibility branch, shared-binary implementation, or
fallback was added.  The machine-readable stop-loss is
`artifacts/hybridz_largecls_gates/strict_phase_equivalence_sharing_stoploss_20260811.json`.

### 2026-08-11 one-session aggregate exact-MIP stop-loss

The next diagnostic replaced the per-rival search by one fixed objective: the
sum of all 99 original CIFAR100 rival margins. It retained the same exact
phase/binary HZ, used one HiGHS session and one thread, and provided no retry or
fallback. This was verifier-internal candidate generation only; there was no
concrete ONNX execution, input sampling, PGD, or raw replay.

The compact model contained 19,600 columns (1,061 binary), 17,489 rows, and
3,498,525 original nnz. After a 15.351 s native solve, HiGHS reached its time
limit without a valid original-model primal. The returned column vector
violated an original HZ row by as much as 7.323, so it was rejected before any
property or ONNX check. TinyImageNet was skipped under the common-path
stop-loss. This shows that the immediate bottleneck is finding a feasible
exact phase-model incumbent, not merely choosing a better rival objective.
No code path was retained. See
`artifacts/hybridz_largecls_gates/aggregate_exact_mip_witness_stoploss_20260811.json`.

### 2026-08-11 forward-roundoff zero counterfactual stop-loss

A build-only causal census then asked whether improving the forward arithmetic
error envelope could remove the exact phase bottleneck. For each preactivation,
the stored center, generator matrix, stable factor order, and outward cube-bound
arithmetic were retained while its independent arithmetic-error allowance was
set to zero. This is an intentionally impossible best case, used only to bound
the maximum possible benefit of error tightening. No solver or concrete ONNX
execution ran, and no input sampling or PGD was used.

The result was zero stabilization in both targets: all 1,061 CIFAR100-medium
iid29 exact rows and all 1,919 TinyImageNet-medium iid17 exact rows still
crossed zero. Typical stored error was between about `1e-14` and `1e-10`, while
the remaining median distance to a stable phase was about `1e-2` through `4`.
No error-tightening implementation or runtime branch was added. The receipt is
`artifacts/hybridz_largecls_gates/forward_roundoff_zero_counterfactual_stoploss_20260811.json`.

### 2026-08-11 fixed-center-phase continuous-LP stop-loss

One verifier-internal replacement then removed integer search entirely. Every
exact ReLU phase was fixed once from the sign of its stored preactivation
center, and one continuous LP maximized the fixed sum of all original rival
margins. It replaced the MIP in the diagnostic: there was one solver session,
no retry or fallback, no sampled input or PGD, and no concrete ONNX execution.

The CIFAR100-medium iid29 LP had 19,600 columns, 1,061 fixed phase columns,
17,489 ranged/native rows, and 3,498,525 nnz. HiGHS reached the 15-second limit
without a valid primal even though all discrete choices were removed. The
candidate was therefore stopped and TinyImageNet was not run. This narrows the
bottleneck further: the current exact-HZ model has a base feasibility/numerical
problem, not only an integer branching or property-objective problem. No code
path was retained. See
`artifacts/hybridz_largecls_gates/fixed_center_phase_property_lp_stoploss_20260811.json`.

### 2026-08-11 exact-HZ feasibility preconditioner stop-loss

Three common-path causes were then checked without adding runtime modes. First,
2,082 of 17,489 native CIFAR100 rows contained 29,454 coefficients at or below
HiGHS' minimum matrix threshold. Every affected row admitted an exact
power-of-two scale (median exponent 5, maximum 25), which retained every nnz
and changed `addRows` from `kWarning` to `kOk`. Nevertheless, the fixed-phase
LP still returned no primal after 15.266 seconds, and total time slightly
worsened from 18.270 to 18.449 seconds.

Second, intersecting the current cube with already-computed forward interval
facts stabilized 0 of 1,061 exact rows. Third, one sparse FBBT pass cost 0.489
seconds and tightened continuous bounds, but fixed 0 of 1,061 phase binaries.
None of these diagnostics executed ONNX, sampled the input, used PGD, or ran a
second solver path. No implementation was retained. The combined receipt is
`artifacts/hybridz_largecls_gates/exact_hz_feasibility_preconditioner_stoploss_20260811.json`.

This closes further parameter-level tuning: no more phase-start variants, LP
fractions, row scalings, fact intersections, FBBT pass counts, objective
schedules, or HiGHS option scans should be added. A future candidate must
replace general exact-MIP feasibility search rather than precede or supplement
it, and must remain one forward-only exact path.

### 2026-08-11 complete constructive-primal stop-loss

A verifier-internal construction clarified the earlier apparent base-HZ
infeasibility. The incomplete diagnostic had left the single 8,192-column
`affine_chain_cut:20` materialization at zero; those missing factors caused
8,267 stored rows to violate their upper bounds, with maximum residual 1.2355.
Assigning the materialized factors from the same forward construction removed
every positive stored-binary64 row residual. The resulting CIFAR100-medium
iid29 point covered all 18,539 continuous factors and all 1,061 exact phase
factors; continuous factors stayed in `[-1,1]`, phases were exactly `-1/+1`,
and the largest row residual was `-4.2081e-20` over 31,795 upper rows.

This was not input sampling and did not execute ONNX or PGD. It propagated one
deterministic value inside the verifier's existing ACT/Operator-HZ
construction solely to assign local HZ factors. Reusing already-built affine
operators reduced the incremental numeric work to about 57.7 ms (56.7 ms of
sparse forward products and 1.0 ms of local-factor fills), rather than the
121-second cost of repeatedly multiplying full HZ expression matrices.

The point did not itself reach the unsafe abstraction: all 99 property rows
remained on the safe side, with maximum margin `-1.1953`. It was then submitted
exactly once as a complete continuous-plus-binary start to the unchanged exact
solver. HiGHS accepted the start (`kOk`), but the persistent LP exhausted its
budget with all 99 rivals unresolved; the solver returned `UNKNOWN` after
44.330 s and total elapsed time was 48.463 s. No candidate was available for
raw replay, TinyImageNet was skipped under the common-path stop-loss, and no
production code was retained.

This removes base-feasible-primal discovery as the explanation for the current
failure, but it does not replace the property search. Further partial or full
phase-start variants are closed. The machine-readable receipt is
`artifacts/hybridz_largecls_gates/constructive_full_primal_stoploss_20260811.json`.

One final replacement gate fixed all 1,061 phases to that constructively
feasible assignment and removed integer search entirely. It used one
continuous aggregate-property LP, not a MIP warm-start fallback. The exact
RANGE model contained 19,600 columns, 17,489 rows, and 3,498,525 nnz. The
already-audited power-of-two row normalization scaled 2,082 rows, retained
every nnz, and made native loading return `kOk`. Nevertheless HiGHS ended with
`Unknown/kWarning` and no valid primal; its exposed placeholder violated an
original HZ row by `1.876e36` and was rejected before property or ONNX replay.
The full diagnostic took 6.747 s, but formal solved-count gain remained zero.
This closes the constructive phase-cell LP as well as further warm-start
variants.

### 2026-08-11 direct sparse forward-graph stop-loss

A structurally different single representation was then tested without
building a solver model.  It retained one explicit local value variable for
each nonconstant Conv, Dense, or ADD output, aliased stable-active ReLUs,
constant-folded stable-inactive ReLUs, and encoded every interval-unstable
ReLU with one output variable, one binary variable, and the three non-box
rows of the exact ideal big-M graph.  Conv connectivity was counted directly
from stored nonzero weights and geometry; no expanded convolution matrix was
materialized.

The common-path stop-loss failed decisively.  After structural constant
pruning, CIFAR100-medium iid29 would require 138,452 columns, 136,634 rows,
and 35,426,210 nonzeros, versus the retained HybridZ native model's 19,600
columns, 17,489 rows, and 3,498,525 nonzeros.  The ratios are 7.064x columns,
7.813x rows, and 10.126x nonzeros.  TinyImageNet-medium iid17 would require
414,350 columns, 406,925 rows, and 119,074,953 nonzeros, versus 62,016,
54,327, and 12,127,453; the ratios are 6.681x, 7.490x, and 9.819x.

The census used ACT interval propagation and stored model weights only.  It
did not execute the concrete ONNX model, sample any input, run PGD, invoke a
solver, or create a candidate/production path.  The large gap in rows alone
closes this direction even if affine coefficient handling were made
unrealistically cheap.  A future exact algorithm must therefore preserve the
current compact HybridZ expression structure rather than replacing it with a
local layer-value graph.  Machine-readable evidence is in
`artifacts/hybridz_largecls_gates/direct_sparse_forward_graph_census_stoploss_20260811.json`.

### 2026-08-11 TinyImageNet iid17 all-exact-phase Gate-1

After the structural census, TinyImageNet-medium iid17 received one formal
Gate-1 attempt with `operator_exact_budget=-1`, ADD materialization disabled,
and the unchanged 100-second shared cold deadline.  This was the sole run: no
parameter scan, retry, longer timeout, sampling precheck, PGD, triangle
fallback, BaB, or backward pass was enabled.  The unchanged gate did retain
its pre-existing persistent LP/Lagrangian certificate stage, so this run is
diagnostic and is not evidence for the later, explicitly dual-free target.

The parent hard deadline fired at 100.694 seconds.  The child log reached the
end of the combined analysis/solver/replay call at about 98.002 seconds, but
the worker did not finish constructing and serializing its authoritative
receipt before the parent cutoff.  The only admissible result is therefore
`timeout`, not a verifier verdict; no candidate or proof may be recovered
from the log.  The run was nonconclusive, gained zero formally solved
instances, and produced no CUDA peak-memory receipt.  Ground truth was not
loaded.  Any ONNX validation remains gated behind a verifier-produced
candidate; there is no concrete sampling precheck in this path.

This also resolves the old TinyImageNet ambiguity.  The historical iid17
`CERTIFIED` result at 67.48 seconds used the earlier legacy/relaxed diagnostic
scope and remains non-authoritative.  It cannot be promoted into an
all-exact-phase proof.  The exact run will not be repeated or granted a longer
budget.  Evidence is stored in
`artifacts/hybridz_largecls_gates/gate1_exactphase_tinyimagenet_medium_iid17_once_20260811.jsonl`
and its adjacent `.summary.json`; their SHA-256 values are respectively
`f4b8fcb591739065aeb2667d1c117bf8e23869c11092cefe19f89b2188171798`
and `3a075ebb91609ee649cde08e1833bca53a21eb49f4d88ccc3c8a2cc88e9805a8`.

### 2026-08-11 projection-skip affine-chain deletion

The next retained common-path change removes the last large intermediate
`affine_chain_cut` in both medium ResNets.  The graph-owned source ADD has the
standard downsample shape: its main route is `Conv -> ReLU -> Conv`, its skip
route is a `1x1`, stride-2 projection, both meet at the next ADD, and that ADD
has one already-supported `Conv -> exact ReLU` consumer.  The builder now
keeps the source affine expression live through the projection and requires
that existing downstream fusion to consume it.  Admission creates no second
representation and, once admitted, failure is fail-closed rather than a
runtime materialization fallback.

On the real abstract builds this removes the layer-20 materialization of
8,192 continuous factors in CIFAR100-medium iid29 and 25,088 in
TinyImageNet-medium iid17.  CIFAR native rows fell from 17,489 to 9,279
(`46.94%`) and native nnz from 3,498,525 to 1,556,040 (`55.52%`).  TinyImageNet
native rows fell from 54,327 to 29,226 (`46.20%`) and native nnz from
12,127,453 to 5,838,020 (`51.86%`).  Every unstable ReLU remained exact;
there were no triangle rows or materialization events on the selected path.
The scalar graph test independently enumerates the exact dyadic phase graph,
and the focused Operator regression is 46/46.

This is a structural breakthrough, not yet a solved-count breakthrough.  The
outward cube prefilter certified none of 99 CIFAR rivals and none of 199 Tiny
rivals, so every output class still reaches expensive constrained reasoning.
One unchanged 100-second TinyImageNet iid17 formal attempt timed out, gained
zero solved instances, and produced no authoritative worker receipt.  That
historical run also retained the existing persistent LP/Lagrangian stage, so
it is diagnostic rather than compliant evidence for the now-explicit
dual-free target.

The next admissible step is therefore one exact output-query session over the
single compact model.  It must replace, not supplement, the current
cube/persistent-dual/per-rival stack; it may not add a second representation,
retry, solver fallback, sampled ONNX precheck, PGD, triangle relaxation, BaB,
backward propagation, or dual tightening.  Full measurements and artifact
hashes are in
`artifacts/hybridz_largecls_gates/projection_skip_chain_candidate_20260811.json`.

### 2026-08-11 single-session exact-query stop-loss

The requested replacement was then tested on CIFAR100-medium iid29 without
adding production code.  Three bounded formulations were tried, each as one
HiGHS session with a 15-second solve limit and no retry or fallback: one
combined exact-OR cutoff model, the same 99-rival maximum with a deterministic
constructive HybridZ-feasible start, and that maximum after one fixed exact
power-of-two scaling rule for continuous columns.  None used an input sample,
PGD, concrete ONNX execution, triangle relaxation, BaB, backward propagation,
or a dual stage.

The constructive start satisfied all stored abstract rows, but its largest
property margin was `-1.1952774497`.  The solver timed out without improving
that incumbent.  Exact power-of-two column scaling reduced coefficients at
or below the backend tiny threshold from 3,285 to two, but again left the
incumbent and margin unchanged.  The two surviving tiny coefficients were
not silently accepted as proof evidence.  There was no valid unsafe abstract
candidate, no verifier verdict, and no formal solved-count gain.  TinyImageNet
was skipped because the smaller common-path instance had already failed the
stop-loss.

For completeness, the earlier single-property objective was retested once on
the reduced projection-skip model, rather than on its old 17,489-row parent.
The verifier selected rival 69 solely because it had the largest outward cube
upper bound (`36.5731`).  Its one dual-free exact MIP contained 9,279 native
rows and 1,556,040 nnz.  It returned `unknown` after 13.56 seconds of matrix
preparation and bounded solve, with no abstract candidate.  There was no
second rival, retry, fallback, parameter scan, or TinyImageNet run.

These results close the exact-OR, max-epigraph, constructive-start, column
scaling, and projected single-rival objective variants.  They must not be
integrated as alternate runtime paths.
The retained result remains the projection-skip structural deletion; before
another real solve, a candidate must demonstrate by bounded structural census
that it deletes a material number of live phase factors or constraints while
preserving every unstable ReLU exactly.  Full details are stored in
`artifacts/hybridz_largecls_gates/single_session_exact_query_stoploss_20260811.json`.

### 2026-08-11 property-connected-component stop-loss

One final build-only structural census checked whether the exact query could
delete a disconnected factor/constraint component instead of changing the
solver.  The solver entry returned immediate `UNKNOWN`; the census used only
the verifier's stored sparse matrices and all 99 property rows.  It did not
execute concrete ONNX, sample an input, run PGD, or invoke HiGHS.

Only 289 of 10,347 continuous columns were absent from all constraints.  Of
those, 100 still occurred in property rows and 189 were fully unused.  More
importantly, every one of the 1,052 phase binaries, every one of the 15,402
constraint rows, and all 1,994,878 constraint nonzeros belonged to a component
touched by the property.  Thus even the maximum free-column deletion would
remove only `2.54%` of total factors and no phase variable, row, or constraint
nonzero.  This is too small to justify another production branch, and the
larger TinyImageNet census was skipped.

This closes free-column and disconnected-component elimination before
implementation.  The next restart should work from the retained
projection-skip deletion and must first show a material exact phase/constraint
reduction in a bounded structural census.  Evidence is in
`artifacts/hybridz_largecls_gates/property_connected_component_elimination_stoploss_20260811.json`.

### 2026-08-11 uniform exact-ReLU residual-coordinate stop-loss

A different semantic representation was checked build-only after the solver
variants closed.  Every unstable exact ReLU was uniformly rewritten as
`y = slope*x + rho`, using the existing Fraction-audited secant residual
range.  The fresh-factor count stayed at one per row, and the original phase
binary plus all three exact Big-M rows were retained.  Thus this was an exact
coordinate proposal, not a triangle relaxation or phase deletion.  It was
installed only by a process-local method replacement and never written into
production source.

The proposal failed the frontend/common-path budget decisively.  Carrying the
preactivation support through `rho` caused sparse expressions to expand across
later affine layers.  CIFAR100 reached only four of six exact-ReLU layers and
837 exact rows before the 30-second build deadline; the retained path completes
all 1,052 exact rows in roughly 2.5 seconds.  The solver was never reached.
TinyImageNet was therefore skipped, and no runtime mode or fallback was added.

This closes uniform exact-ReLU residual coordinates.  The artifact is
`artifacts/hybridz_largecls_gates/uniform_exact_relu_residual_coordinate_stoploss_20260811.json`.

### 2026-08-11 local sparse exact-ReLU residual stop-loss

The residual-coordinate idea was then restricted to one simple structural
rule rather than a layer list: only exact-ReLU preactivations with at most 32
stored generator nonzeros were eligible.  The build-only census showed one
clear shared population.  CIFAR100 had 316/1,052 eligible rows and TinyImageNet
605/1,914; every eligible row was in the first exact ReLU and had exactly 27
nonzeros.  Later-layer minima were already 73 and 86, rising to 2,407 and
7,133 in the last exact ReLU.

The single CIFAR100 candidate rewrote all 316 eligible rows.  It made the
model worse: later unstable phases increased from 1,052 to 1,078, continuous
factors from 10,347 to 10,367, source rows from 15,402 to 15,468, and source
nnz from 1,994,878 to 2,530,908 (`+26.9%`).  The independent `rho` range
expanded subsequent cube bounds more than the retained correlation helped.
The TinyImageNet candidate was therefore not run, and no sparse eligibility
branch or production code was added.  Evidence is in
`artifacts/hybridz_largecls_gates/local_sparse_exact_relu_residual_stoploss_20260811.json`.

### 2026-08-11 terminal-ReLU and rival-dominance stop-loss

One final deletion-only census checked the last ReLU and the output query.
Directly retaining a stable-active row as `y=x` at the terminal ReLU cannot
weaken any later ReLU cube because there is no later ReLU.  On
CIFAR100-medium iid29, however, layer 40 contains only nine stable-active
rows, one inactive row, and 90 unstable exact rows.  The maximum saving is
therefore nine continuous factors and 18 source inequality rows, too small
relative to the retained 10,347 factors and 15,402 source rows.

The same final HybridZ was used for a deterministic correlated-cube dominance
census over all 99 rival margins.  A rival was eligible for deletion only if
the verifier-internal affine generators proved its margin no greater than a
retained rival everywhere.  No rival passed.  The closest pair still had a
strictly positive upper bound of `14.749701359535209`.  TinyImageNet was
skipped after the smaller shared gate failed.

The installed HiGHS 1.14 Python API was also inspected locally.  It exposes
ordinary and semi-continuous/integer column types but no native indicator or
SOS constraint interface, so there is no compact native exact-ReLU primitive
to substitute for the current three-facet Big-M graph.  A simulated second
encoding would violate the one-representation rule and is not being added.
No solver, concrete ONNX execution, input sampling, or PGD was used.  The
evidence is in
`artifacts/hybridz_largecls_gates/terminal_relu_and_rival_dominance_stoploss_20260811.json`.

### 2026-08-11 stable-active forward-DAG structural pass / performance fail

The earlier stable-active stop-loss used only a fixed two-layer sparsity
replay.  A process-local full CIFAR100 build therefore tested the stronger
common-path hypothesis: stable-active ReLUs alias their preactivation exactly,
stable-inactive rows remain zero, and every remaining unstable row keeps the
existing local output factor, binary phase factor, and three Big-M facets.
The projection-skip deletion remained active, and affine-chain cuts were
forbidden.  No production source was changed.

The structural result is material.  Exact phase binaries fell from 1,052 to
699 (`-33.56%`), source rows from 15,402 to 2,097 (`-86.38%`), and constraint
nnz from 1,994,878 to 952,293 (`-52.26%`).  Omitting the 6,249 now-unused
stable-active allocations would reduce continuous factors from 10,347 to
3,871 (`-62.59%`).  All 699 remaining unstable rows were still exact and no
triangle row or materialization event remained.  The tighter forward
correlation itself stabilized 353 previously unstable phases.

A separate control forced every layer to reuse the retained independent-cube
phase classification.  It therefore kept all 1,052 exact binaries, yet still
reduced source rows to 3,156 (`-79.51%`) and potential continuous factors to
4,224 (`-59.18%`); constraint nnz became 1,715,618 (`-14.00%`).  This isolates
the base deletion from the optional tighter-correlation effect: a valid first
DAG need not claim any phase stabilization or change a downstream cube.

The flat implementation nevertheless failed the mandatory performance gate:
the retained build is about 2.394 seconds, while the CPU candidate took about
10.169 seconds (`0.235x`).  Profiling attributes the cost to forward SpGEMM
over residual intermediates with 7--20 million nonzeros, plus repeated CSR
hashing and sorting.  Reusing the existing survivor prescreen restored speed
only by reintroducing six old chain cuts and 16,998 rows, so that route is not
the desired representation.  A PyTorch/CUDA sparse-product feasibility run
reduced the candidate to 6.472 seconds but remained much slower and reached
about 22.04 GiB peak allocated memory; it is also rejected.

This is the first new structural lead that crosses the phase/row/nnz census,
but it is not a promotable implementation and produced no solved-count gain.
The only admissible continuation is one forward affine DAG which replaces the
flat intermediates rather than coexisting with them, followed first by exact
affine/ReLU/residual toys and the same-topology 1.50x cold-build gate.  No
solver or TinyImageNet candidate was run.  Detailed evidence is in
`artifacts/hybridz_largecls_gates/stable_active_forward_dag_census_20260811.json`.

### 2026-08-11 GPU factor-stream stop-loss

The remaining stable-active DAG idea was implemented once as a disconnected,
single-representation CUDA factor stream.  It propagated fixed batches of
factor columns through the verifier graph, retained only exact preactivation
rows and final output rows, aliased stable-active ReLUs, zeroed inactive rows,
and kept one binary plus all three Big-M facets for every interval-unstable
ReLU.  Center, roundoff-error, and mass propagation were then merged into the
same GPU operator schedule to remove an initially duplicated CPU pass.

The symbolic gate passed 3/3 without evaluating model inputs: exact
`Fraction` phase elimination on a residual toy recovered the inactive and
active affine forms and their Jacobians, while the stable-active ReLU added no
factor or constraint.  The real C100 build likewise had no triangle rows,
fallback, solver, sampled input, PGD, or concrete ONNX candidate search.

On CIFAR100-medium iid29, the candidate produced 4,426 continuous factors,
1,254 exact phase binaries, 3,762 source rows, and 3,453,014 constraint
nonzeros.  It used about 1.22 GiB incremental CUDA allocation.  Against the
current projection-skip production build in five alternating same-process
pairs, median time was 1.571 s versus 2.195 s, or only `1.400x`.  A fixed
batch-128 diagnostic reached only `1.413x` and independent batch-64/batch-128
builds were not bitwise identical.  Both are below the mandatory `1.50x`
frontend gate.

The candidate and its test were therefore removed rather than retained as a
second backend.  TinyImageNet and the solver were not run; formal solved-count
gain remains zero.  The complete receipt, including pre-removal file hashes,
is
`artifacts/hybridz_largecls_gates/gpu_factor_stream_stoploss_20260811.json`.

### 2026-08-11 retained Operator hot-path stop-loss

A profile of the retained projection-skip build then isolated Conv CSR
construction as the largest remaining frontend cost.  The already-audited
direct canonical constructor was exercised process-locally, and the repeated
`abs(W)` scan between each live-affine prescreen and its row chunks was removed
in one combined candidate.  This changed neither the HZ structure nor any
stored bit.

The direct constructor alone reached only `1.460x` median paired speedup.  The
combined direct-plus-single-`abs(W)` candidate reached `1.494735x`; its cold
ratio was `1.480611x`.  Both are below the strict `1.50x` first gate.  A
two-column nonnegative SpMV was also rejected before implementation because it
was slower than the two existing reductions (`0.0889` versus `0.0730` s),
despite bitwise-equal results.

All tentative production edits were reverted, restoring `operator_hz.py` to
SHA-256 `2502c009...90fd8`; the focused suite is again 46 tests plus 17
subtests.  The four-thread, TinyImageNet, and solver gates were correctly
skipped.  Evidence is in
`artifacts/hybridz_largecls_gates/retained_operator_hotpath_stoploss_20260811.json`.

### 2026-08-11 exact-MIP structural replacement stop-loss

The next bounded census moved past frontend representation changes and
inspected the retained projection-skip exact MIP itself.  The native C100
model has 10,347 continuous factors, 1,052 binary phase factors, 9,279 ranged
or upper rows, and 1,556,040 constraint nonzeros.  All 1,052 phase binaries
have degree exactly two.  There is no property-free continuous leaf column:
the nine degree-one continuous columns all touch a property row.  Therefore a
no-fill existential leaf projection removes zero columns and zero nonzeros.

Eliminating the 6,123 stable-active equalities is algebraically tempting, and
all 6,123 pivot-box redundancy checks pass in exact dyadic arithmetic.  It is
not a viable stored-binary64 transformation: the pivot coefficients range
from about `1.9985e-7` to `6.1446`, are not unit/power-of-two pivots, and a
naive substitution would attempt about 137.4 million insertions into a matrix
that currently has 1.556 million nonzeros.  This is the solver-side form of
the already observed stable-active alias densification, so no implementation
was added.

The installed PySCIPOpt 6.2.1 / SCIP 10.0.2 stack was then tested as a true
replacement, never as a fallback.  The same Big-M model returned `UNKNOWN`
after 15.90 seconds on verifier-selected rival row 69.  A native-indicator
formulation kept every phase binary, replaced the two conditional Big-M rows
by 2,104 SCIP indicators, and retained 7,176 linear rows including the cutoff;
it reached only the root node, found no solution, and timed out after 15.07
seconds.  Ordinary SCIP status is not treated as proof authority.

A second exact extended formulation retained every phase binary and introduced
one nonnegative `r=y-x` slack per unstable ReLU.  It replaced the repeated
dense branch by `r+(-l)z<=-l`, reducing nnz to 1,001,121 (`64.34%` of the
baseline), but increased columns from 11,399 to 12,451.  Its exact query still
returned `UNKNOWN` in 13.52 seconds, with no solve-time improvement, so it too
was stopped before code or TinyImageNet.

SCIP 10 documents a numerically exact MIP mode and optional independently
checkable VIPR proof logging, but this environment's SCIP binary was compiled
without exact-solve support: `enableExactSolving(True)` fails before a problem
is created.  Consequently no floating infeasibility result was promoted and
formal solved-count gain remains zero.  No ONNX concrete execution, input
sampling, PGD, triangle fallback, ACT BaB, backward pass, or dual tightening
was used.  Detailed evidence is in
`artifacts/hybridz_largecls_gates/exact_mip_structural_replacement_stoploss_20260811.json`.

### 2026-08-12 common-path follow-up stop-loss

One final common-path Conv/affine follow-up combined the audited direct Conv
CSR constructor, a single reused `abs(W)` snapshot, and removal of a duplicate
finite scan performed in the same callback-free stack.  On the fixed
CIFAR100-medium iid29 abstract build, the complete HZ remained bitwise equal.
Seven alternating single-request pairs reached `1.643x`, crossing the first
`1.50x` gate.  The required concurrency gate failed decisively: five groups
of four requests reached only `1.451x` median, with bootstrap 95% lower bound
`1.345x`, below the required `2.00x/1.80x`.  The tentative edit was therefore
reverted and `operator_hz.py` returned exactly to SHA-256
`2502c009...90fd8`.

Two apparent escape routes were also closed before any model solve.  Gurobi
is locally importable, but its restricted license rejects a synthetic model
at 2,001 variables (`10010: Model too large`), far below the retained C100
exact MIP.  The deleted GPU factor-stream candidate was recoverable only as a
read-only Python bytecode cache; inspection confirmed that it already used
the direct Conv builder and selected phases from independent interval facts,
which explains its 1,254 binaries versus the retained correlated 1,052.
Recovering the tighter phase schedule would require a second factor pass or
the previously rejected full flat representation, so the backend was not
restored.

The disconnected cross-layer residual facet was also not generalized: its
audited contract is a fixed scalar toy, not a model-owned CIFAR/Tiny binding.
Turning it into production branches would violate the common-case/low-
redundancy constraint, while the already completed K2/K3/PCOH work covers the
general phase-hull direction without a real verdict gain.

No TinyImageNet run, solver run, concrete ONNX execution, input sampling,
PGD, triangle fallback, ACT BaB, backward pass, or dual tightening occurred.
Formal solved-count gain remains zero.  The exact receipt is
`artifacts/hybridz_largecls_gates/commonpath_followup_stoploss_20260812.json`.

### 2026-08-12 proof-capable exact SCIP/VIPR checkpoint

The local PySCIPOpt library could not enable exact solving, so the official
SCIP 10.0.2 and SoPlex 8.0.2 sources were rebuilt once with
`EXACTSOLVE=ON`, `LPSEXACT=spx`, GMP, and MPFR.  The official VIPR completer
and checker were built from source against the same stack.  This is intended
as one replacement query backend, not an additional runtime fallback.

On SCIP's rational mixed-integer `flugpl` instance, the exact solver returned
identical exact primal and dual bounds of `1201500`.  The default separator
configuration produced a certificate that `viprchk` rejected, so it is not
admissible.  With the single fixed `separating off` configuration, SCIP again
proved the same exact optimum; `viprcomp` completed the proof and `viprchk`
ended with `Successfully verified optimal value range [1201500, 1201500]`.
This closes the previously missing proof-capable solver dependency.

No verifier integration or real C100/TinyImageNet solve was performed in this
phase.  Consequently the dataset status is unchanged: the retained
projection-skip path still cuts C100 native rows/nnz by `46.94%/55.52%` and
TinyImageNet by `46.20%/51.86%`, while new formal solved-count gain remains
zero.  The next and only admitted action is a C100-medium iid29
build/export-only stop-loss on the reduced 9,279-row / 1,556,040-nnz model,
measuring exact-rational export wall time, bytes, RSS, and frontend latency.
TinyImageNet remains gated behind that smaller test.

The full source SHAs, build flags, binary/certificate/log hashes, negative
control, restrictions, and restart point are recorded in
`artifacts/hybridz_largecls_gates/exact_scip_vipr_capability_checkpoint_20260812.json`.
The binaries live under a temporary `/tmp` build root and may not survive a
reboot; the receipt therefore binds the official source revisions and build
configuration needed to reproduce them.  At checkpoint time there were no
background solver/test processes and no unfinished runs.

### 2026-08-12 constraint-local preactivation factor checkpoint

The live-row GPU stream was rebuilt as one disconnected representation, with
no production dispatch or second HZ.  Its new rule is deliberately small: a
dense stored preactivation generator row appears once, in an equality to a
normalized constraint-local continuous factor.  The exact-ReLU Big-M facets
and refined stable-active bands then use that sparse factor.  The local factor
is allocated only after the GPU generator stream and is never propagated
through a later affine layer.  Every remaining unstable ReLU still owns one
exact binary phase and all three Big-M facets; triangle relaxation and runtime
fallback remain absent.

The exact gates now cover equality elimination over `Fraction`, a three-ReLU
residual graph, refined active/inactive rows, stored Big-M scale, and the proof
that the outward row-L1 scale keeps every local factor in `[-1,1]`.  In the
`act-py312` environment the candidate and its exact Conv dependency passed
31/31 tests.  The candidate is frozen at SHA-256 `308728c6...2ef89`, its test
at `81b4bb47...9f447`, and production `operator_hz.py` remains unchanged at
`2502c009...90fd8`.

On CIFAR100-medium iid29, the retained HZ builder has 10,347 continuous and
1,052 binary factors, 15,402 upper rows, and 1,994,878 constraint nonzeros.
The disconnected candidate has 5,311 continuous and 788 binary factors, 885
local equalities plus 2,927 upper rows, and 977,465 nonzeros.  Relative to the
immediately preceding streamed form, its nonzeros fell from 1,945,117 to
977,465 because the dense preactivation appears once instead of twice.

The required frontend gates pass with margin.  Five alternating single pairs
gave `2.5636x` median build speedup.  Five groups of four simultaneous builds
gave `2.7496x` median paired speedup, bootstrap 95% lower `2.6979x`, and zero
stable-ID conflicts.  Fresh-process build was 1.541 s versus 3.090 s
(`2.005x`).  Full parse-plus-build was 3.493 s versus 4.945 s (`1.416x`, not
misreported as 1.5x).  The build-stage host HWM increment was about 222 MB
versus 382 MB.  The fixed six build-only cases all completed: both C100
medium cases, both C100 large cases, and TinyImageNet medium iid6/iid17.  Tiny
builds were about 4.27--4.46 s with about 277--306 MiB incremental CUDA peak.
No concrete ONNX input, sampled center/boundary/random point, PGD, ACT BaB,
backward bound pass, dual tightening, or solver was used by those gates.

The result is a real frontend/representation improvement but not a verdict
breakthrough.  One exact-rational SCIP cutoff on C100 iid29 loaded the new
3,812-row / 977,465-nnz model in 0.264 s, then remained at zero nodes and
timed out with `UNKNOWN`.  The existing verifier returned `UNKNOWN` on both
iid29 and iid2.  Direct HiGHS observation on iid2 showed two cutoff attempts,
each at zero MIP nodes with no valid primal or dual bound: root processing is
the bottleneck, rather than rejection of a candidate witness.  Exact equality
substitution did not help.  A power-of-two local scale had no solver benefit
and weakened the local box, so it was removed.  Constraint connectivity also
does not offer a useful split: one component contains all 788 binaries, 5,019
continuous variables, and 3,809 of 3,812 rows, and every rival reaches all
source rows.

Accordingly, new formal solved-count gain remains zero for both CIFAR100 and
TinyImageNet.  Do not repeat solver-option scans, add solver backends, multiply
rival workers, revive equality substitution, or use input sampling/PGD.  The
next admissible research step is one new root-node algorithm that consumes the
compact exact representation and produces a rigorously replayable verifier
result.  Before any production dispatch, the candidate's full soundness and
the disconnected Conv dependency's authority boundary must be reviewed.
Machine-readable evidence is in
`artifacts/hybridz_largecls_gates/live_row_constraint_local_factor_checkpoint_20260812.json`.

### 2026-08-12 exact phase-projection production checkpoint

The next bounded root-node experiment produced the first new formal
CIFAR100/TinyImageNet verdict gain of this work period.  It is intentionally
one path rather than a menu of heuristics.  Starting from the established
forward interval facts, the verifier fixes the center phase cell, selects one
TOP1 row analytically, computes one inward factor-domain corner, updates every
unstable ReLU phase exactly once, solves one continuous input-factor LP, and
replays every selected phase against outward stored-affine error envelopes.
The decoded input is checked against the raw BOX.  A verifier-owned
zero-width forward interval replay and exact `Fraction` evaluation then prove
that the selected TOP1 margin lower bound is strictly positive.

This is not input sampling.  No center, boundary, random, or other candidate
is executed through ONNX; there is no PGD.  There is also no triangle ReLU,
ACT BaB, backward bound propagation, or dual tightening.  An independent
certificate may still be used as an auxiliary audit, but it is absent from
this verdict path and is not the source of authority.  The public candidate
receipt remains non-authoritative; the production verifier callsite owns the
raw-BOX, zero-width forward, and exact-property proof obligation.

The path is default-off and admitted only with
`engine=operator_hz_objbound`, `operator_exact_budget=-1`, and a positive
`operator_phase_projection_time_limit`.  Configuration rejects simultaneous
phase-clique, preactivation, property, micro-RLT, query-dual, or GPU-dual
enhancements.  When enabled it is now a true single path: a successful proof
returns `FALSIFIED`; inapplicability, resource exhaustion, or failed exact
replay returns `UNKNOWN` immediately.  It never starts the old root MILP as a
fallback, avoiding redundant backend work and frontend latency.

The focused suite passes 10/10 and explicitly proves that no `model_fn`,
external replay, or root MILP is used.  The adjacent Phase-0, ADD, residual,
preactivation, and micro-RLT suite passes 95/95.  The previously registered
performance gate also passes: `4.0279x` in the single paired gate;
`2.1237x` median across fixed four-concurrent by five paired groups; paired
bootstrap 95% lower bound `2.0566x`; zero result or stable-ID conflicts.

Six fixed-manifest successes were rerun through the production
`verify_once` entry and are now formal verifier results:

- CIFAR100: medium iid2, medium iid11, medium iid64, and large iid118;
- TinyImageNet: medium iid193 and medium iid1.

All six return `FALSIFIED`.  Their exact positive singleton margin lower
bounds are respectively `0.0929015978`, `0.0002534557`, `0.0146032404`,
`0.5550047360`, `0.0352310707`, and `0.7018047322`.  CIFAR100-medium iid29 is
the negative control: it returns `UNKNOWN` in about 2.55 seconds without
launching the legacy root solver.  Thus the confirmed new formal solved-count
gain is at least six: four CIFAR100 and two TinyImageNet.  The full fixed-40
production-entry rerun has not been repeated, so this checkpoint does not
inflate that result into a 40-case aggregate claim.

Current content hashes supersede the earlier live-row hash paragraph:

- phase projection candidate: `615b4551...a04993ba`;
- focused test: `2a5cae3e...b99214`;
- live-row dependency: `be00bd11...9004e86`;
- live-row dependency test: `a94b37a9...c61601`;
- verifier: `eb3dfc86...dbc3afd`;
- config Python/YAML: `8d94960c...fddf21` / `8f3e4935...b9e77`.

The exact receipts and full hashes are in
`artifacts/hybridz_largecls_gates/phase_projection_production_integration_checkpoint_20260812.json`.
The next admissible scale step is the preregistered fixed-400 formal verifier
gate using this same frozen one-update/one-row/one-LP path.  It must retain the
sampling/PGD/ONNX-execution prohibition and must not add fallback algorithms.

### 2026-08-12 fixed-400 formal verifier result

The fixed-400 gate is now complete.  It ran every row in the official fixed
CSV ranges in deterministic order: 100 CIFAR100-medium, 100 CIFAR100-large,
and 200 TinyImageNet-medium instances.  Every instance ran in a fresh process;
each result was fsync'd immediately and the summary was atomically replaced,
then the completed JSONL was independently revalidated.  There were no worker
errors, result conflicts, or residual background processes.

The verifier itself formally falsified 43/400 instances:

- CIFAR100-medium: 12/100, iids
  `2,11,31,43,48,54,61,64,65,69,83,97`;
- CIFAR100-large: 10/100, iids
  `118,119,124,148,155,164,181,183,194,199`;
- TinyImageNet-medium: 21/200, iids
  `1,7,29,38,46,62,65,69,77,85,87,90,103,116,120,125,150,155,181,189,193`.

The other 357 instances are `UNKNOWN`, not unsound negatives.  The dominant
failure bucket is exceptionally concentrated: 345 cases reach the sole
continuous LP but its candidate does not pass the exact phase-and-positive-
margin replay.  Seven LPs are infeasible, two hit the LP time limit, two fail
closed in live-row construction, and one exhausts the deadline at singleton
replay.  This identifies one common next target: improve the deterministic
phase-cell/property objective shared by those 345 cases.  It does not justify
adding a collection of rare-case fallbacks.

The median complete fresh-process time was about 5.27 s for successful
CIFAR100-medium cases, 10.39 s for successful CIFAR100-large cases, and
11.15 s for successful TinyImageNet cases.  The earlier same-topology
performance promotion remains valid (`4.0279x` single, `2.1237x` fixed
four-concurrent median, bootstrap lower `2.0566x`).

Every one of the 43 successes has exactly one phase update, zero phase
retries, one selected property row, zero property-row retries, one continuous
input LP, all unstable ReLUs represented exactly, all phase rows replayed,
and zero triangle rows.  Across all 400 cases there is no sampled input,
center/boundary/random ONNX execution, PGD, ACT BaB, backward propagation,
dual tightening, external certificate authority, or root-solver fallback.
Independent certificates remain optional audit aids only.

Machine-readable results and their freeze hashes are in:

- `artifacts/hybridz_largecls_gates/phase_projection_fixed400_20260812.jsonl`;
- `artifacts/hybridz_largecls_gates/phase_projection_fixed400_20260812.summary.json`;
- `artifacts/hybridz_largecls_gates/phase_projection_fixed400_complete_20260812.json`.

### 2026-08-12 phase-objective stop-loss

The concentrated 345-case failure bucket was followed by a bounded set of
single-path experiments.  None improved the formal solved count.  Reselecting
the rival after the phase update preserved all 43 existing fixed-400
successes, but added 0/34 on the preregistered fixed-40 unknown subset and
performed extra property-matrix work.  Removing the intermediate projected
margin gate also added 0/34: 24 candidates then failed the mandatory
zero-width formal forward check, so the existing gate is useful latency
avoidance rather than lost verifier power.  A second deterministic phase
projection, singleton-interval phase assignment at the analytic corner, and
checking all TOP1 rows for the same LP candidate each added 0/34 while
increasing deadline pressure or downstream work.

Two replacement objective rules were worse.  Selecting by center margin kept
only 34/43 existing successes and added none; a smooth multi-rival objective
kept only 18/43 and added none.  All were removed.  The production candidate
is restored bit-for-bit to SHA-256 `615b4551...a04993ba`, and its focused
10/10 suite remains green.

The diagnostic is structural, not a rounding corner case.  On
CIFAR100-medium iid29 all 1,254 exact phase rows replay consistently, while
the chosen affine objective is about `-0.37482743`; its rigorous accumulated
arithmetic envelope is only about `5.03e-8`.  Tightening singleton rounding
cannot bridge that gap.  The next admissible improvement therefore needs one
materially different global phase-cell search mechanism.  Do not stack the
rejected rival selectors, phase retries, or late replay fallbacks into the
backend: they add latency without capability gain.  These experiments used
only fixed official manifests and verifier-internal formal quantities; no
input sampling, center/boundary/random ONNX execution, PGD, ACT BaB, backward
bounds, or dual tightening was used.

The exact stop-loss record is
`artifacts/hybridz_largecls_gates/phase_projection_objective_stoploss_20260812.json`.

A diagnostic-only split of the fixed-14 UNKNOWN cases shows that the old
combined replay message hid three distinct common outcomes.  Three candidates
replayed every exact phase but had materially nonpositive margins
(`-0.3748`, `-0.2250`, and `-1.1702`).  Three TinyImageNet candidates missed
only one or two of roughly 1,700--2,000 phase rows, with worst signed phase
safety between `-3.55e-9` and `-1.96e-8`; the remaining five were LP or
resource failures.  Increasing the uniform phase-interior guard from 16 to
128 made all three near-tolerance phase sets consistent and retained three
positive sentinels, but their margins were still negative, so it added zero
formal verdicts.  A single principled 32x check on the closest case still
missed its phase; no multiplier scan or per-case fallback followed.  The
diagnostic messages and guard change were reverted.

The original LP margins for those same three near-tolerance phase cases were
already negative (`-0.3676`, `-1.1654`, and `-0.001071`).  Thus even a
hypothetical zero-cost exact repair of the one or two offending rows would
add no formal verdict on this sentinel; no primal-correction pass was built.

This means the dominant combined bucket must not be treated as one numerical
tolerance bug.  Numerical phase repair alone does not raise solved count; the
main unresolved cases still require a globally useful exact cell/property
direction.  Machine-readable evidence is in
`artifacts/hybridz_largecls_gates/phase_projection_failure_census_20260812.json`.

An exact power-of-two row-normalization experiment then tested whether one
uniform HiGHS conditioning improvement could address both the near-tolerance
rows and LP limits without changing the mathematical cell.  Each phase row
and RHS was divided by one binary power chosen from its infinity magnitude,
then the same single LP and mandatory replay were used.  It added 0/11 fixed-
14 UNKNOWN results.  Under the four-worker census it retained two of three
positive controls, lost CIFAR100-large iid118 to the bounded live-row path,
and moved four TinyImageNet cases to an earlier deadline.  The extra full-CSR
pass was therefore removed.  Exact equivalence alone is not enough when the
implementation worsens the stated response-time/resource goal.

The remaining zero-structure-cost numerical option was also tested once:
HiGHS' primal feasibility tolerance was tightened uniformly from `1e-9` to
its supported `1e-10` minimum, with the same cell, matrix, objective, and one
LP.  Fixed-14 again added zero verdicts and retained only two of three
positive controls under four workers.  It was reverted without a solver-
option sweep.  Numerical tolerance tuning is therefore closed as a solved-
count direction for this path.

One batched global selector was tested without another graph replay.  The
first stream's complete preactivation rows evaluated all 99 TOP1 analytic
corners in one sparse-dense batch; the intended rule selected exactly one
positive-potential rival with the fewest predicted phase changes before the
usual single cell and LP.  CIFAR100-medium iid2 remained formally falsified
with roughly 0.1 s extra candidate work.  On the target negative control
iid29, however, none of the 99 rows had even a positive first-cell affine
upper.  Thus no ranking among those directions can address that case; the
batch selector was removed before any larger gate.  This closes all-rival
ranking over the same center-cell linearization, rather than encouraging a
larger property-candidate menu.

The last forward-only global signal available without another graph replay
was tested directly.  The exact-HZ stream propagated every unstable ReLU's
local output factor to the final output, and the selected TOP1 row's output
coefficient sign chose each active/inactive phase before the usual one fixed-
cell LP.  This uses genuine forward-mode downstream influence and no sampling,
backward pass, dual, branching, or retry.  Nevertheless all six sentinels
produced an infeasible input LP, including all three positive controls.  The
independent local-factor signs ignore their coupled input and binary phase
constraints, so their coordinatewise optimum is generally not a realizable
phase cell.  Enforcing those couplings during selection would reintroduce the
forbidden combinatorial solve.  The prototype was removed in full.

An exact fixed-cell redundancy pass then targeted LP structure rather than
cell selection.  A phase row was omitted only when its already-established
outward row-l1 bound was no greater than the stored inequality RHS for every
input factor in `[-1,1]`; this leaves the fixed-cell feasible set unchanged.
On CIFAR100-medium iid2 it reduced LP rows `2177->1140`, nnz
`2,465,565->560,712`, and LP time `0.916->0.168` s while preserving the exact
singleton margin.  iid11 similarly changed `1510->787` rows and
`1,721,676->340,557` nnz.  TinyImageNet iid61 advanced from an LP timeout to
mandatory replay, but remained UNKNOWN; large iid101 still timed out.  There
were no new fixed-14 formal verdicts.

The headline gate is the full candidate, not its LP substage: iid2 improved
only `3.592->3.052` s (`1.177x`), below the required `1.50x`.  The elimination
was therefore reverted despite its attractive LP-only numbers.  This result
is useful for a future fused builder, but cannot be promoted alone or quoted
as an end-to-end breakthrough.

The elimination was also paired once with a build-local memo for repeated
affine structure: immutable affine snapshots, device weights, fan-in counts,
and exact support propagation.  On the same network and facts, seven paired
single-thread measurements gave a median full-candidate speedup of `1.871x`
(`1.691x` minimum, `1.976x` maximum), with bitwise-identical decoded inputs,
unchanged margins and replay counts, and zero stable-ID conflicts.  This was
not sufficient for promotion.  Under the required four-worker, five-pair
protocol the paired ratios were `1.725x, 1.552x, 1.699x, 1.571x, 1.532x`:
median `1.571x` and paired-bootstrap 95% lower `1.532x`, below the required
`2.00x` and `1.80x`.  Both optimizations and their focused tests were removed,
and all four affected source/test files were restored byte-for-byte to their
frozen SHA-256 values.  This closes per-build Python memoization as a common
path: the next performance attempt must reduce the shared concurrent work or
eliminate a required traversal, not retain extra caches or add fallback
branches.

A final single-neighbor rule was checked after the structural census.  When
the projected cell's strict affine upper bound was already nonpositive, it
replaced that hopeless cell with exactly one adjacent cell: the phase having
the strongest rigorously signed disagreement at the same objective corner.
Only the replacement cell was sent to one LP; there was no second solver or
candidate list.  Fixed-14 nevertheless stayed at 3/14 and one large case
exhausted its LP deadline earlier.  The rule was removed before fixed-40.
Ranking the sole rival by only its controllable affine upper
(`center + ||G||_1`), while retaining arithmetic error in every proof replay,
also left every fixed-14 outcome unchanged at 3/14.  It was reverted as a
no-op rather than changing a frozen path without capability gain.

Two further single-path rules were stopped on a six-case sentinel before any
fixed-14 expansion.  Solving the one center-selected exact cell retained only
two of three baseline successes, lost CIFAR100-medium iid2, and added none of
the three baseline unknowns.  A materially different analytic-line rule then
traced ReLU breakpoints continuously along the sole property direction,
selected one best cell over that line, and retained the same single
full-dimensional LP.  It did not execute ONNX at sampled inputs and did not
use PGD, branching, backward bounds, dual tightening, or a fallback.  Even so,
all six sentinels exhausted the ten-second candidate budget inside repeated
full-graph line traversal, including all three baseline successes.  The rule
was removed immediately.  Varying its breakpoint cap would be a parameter
scan over an already disqualified latency shape, so it was not attempted.

The frozen production candidate is again byte-for-byte SHA-256
`615b4551...a04993ba`.  These results close center-cell substitution and
full-graph analytic line traversal: any next global cell mechanism must avoid
both repeated graph replay and a collection of rare fallback paths.

### 2026-08-12 strict phase-sharing and refinement stop-loss

The remaining exact-phase population was checked before implementing binary
sharing.  Neither model contains two ReLUs owned by the same preactivation
node.  In the constraint-local live-row representation, CIFAR100-medium iid29
has 788 remaining exact phases and TinyImageNet-medium iid0 has 1,428; hashing
the complete stored generator row, center, arithmetic error, and lower/upper
bounds found zero bitwise-identical groups in both models.  Thus strict
graph-owned or full-affine phase equivalence cannot remove one binary on the
target topology.  No numeric-nearness sharing was attempted, because it would
change the feasible phase set.

A second experiment reused the live-row representation's full correlation
refinement before phase projection.  On CIFAR100-medium iid29 it correctly
proved 466 of 1,254 interval-unstable rows stable, leaving 788 exact phases.
This did not add a success on the fixed-40 unknown subset.  More importantly,
the required extra full generator stream exhausted the bounded live-row path
on the large/Tiny cases: only the 12 CIFAR100-medium members of the original
43 successes survived the 77-case comparison, while all CIFAR100-large and
TinyImageNet successes were lost.  The experiment was removed and the
production phase-projection SHA returned to `615b4551...a04993ba`.

This result closes phase sharing and a separate full refinement pass as
uniform common paths.  It does not invalidate correlation stabilization
inside the already-audited constraint-local representation; it shows that
paying for an additional representation-sized stream before the fast phase
LP violates the large-model latency/resource requirement.  A future design
would have to obtain the same stable facts inside an already-required stream,
not coexist with it or dispatch by model.  No sampling, concrete ONNX input
execution, PGD, BaB, backward bounds, or dual tightening was used.  The exact
census is in
`artifacts/hybridz_largecls_gates/strict_phase_sharing_refinement_stoploss_20260812.json`.

A bounded common prefix was then tested rather than assuming that a fusion
would help.  Both target graphs used the same rule: propagate correlation
only through the third ReLU and release it there.  This reduced iid29 from
1,254 to 1,010 phase rows and passed the synthetic performance gates
(`3.502x` single; `2.174x` four-worker/five-pair median; bootstrap lower
`2.142x`; zero ID conflicts).  It nevertheless left the fixed-14 formal count
unchanged at 3/14.  TinyImageNet iid1 phase rows fell from 2,488 to 2,261 but
candidate latency rose from about 8.25 to 10.88 seconds.  Since the explicit
goal is formal solved-count gain under responsive budgets, not phase-count
reduction alone, the prefix was also removed.  A fused implementation was not
added after its capability gate produced zero gain.

### 2026-08-12 common-path cache, residual, and boundary stop-loss

Three compact follow-ups were tested without adding runtime alternatives.
Hoisting loop-invariant work from the two required generator streams improved
the full candidate by only `1.009x` in one thread and `1.031x` with four
workers.  A broader build-local cache for immutable affine snapshots, device
weights, fan-in counts, and support propagation reached about `2.10--2.26x`
in one thread but only `1.74--1.77x` with four workers, below the required
`2.00x` median and `1.80x` bootstrap lower bound.  It also has no legitimate
cross-query reuse population in the fixed-14 set: all 4 CIFAR100-medium, all
4 CIFAR100-large, and all 6 TinyImageNet property topologies had distinct
fingerprints.  A persistent cross-query cache would therefore optimize a
repeated benchmark invocation, not the target verifier workload.  Both cache
ideas were discarded.

The first-to-second phase-cell change is locally sparse at the ReLU masks on
CIFAR100-medium iid2 (`104,54,21,1,0,5,0,0,0,2` changed phases), but it does
not remain sparse enough through the affine graph.  The restricted affine
delta contains `32,839,860` nonzeros, `44.19%` of the full affine weights;
the selected residual stream contains `3,389,854` nonzeros.  A real CUDA
prototype spent about `0.089` s constructing this residual and `0.170` s
streaming it, for a `3.913` s full candidate.  It was slower than the frozen
path and changed the last binary64 bits of the projected margin because it
changed reduction order.  No residual representation or fallback was kept.

Finally, the mathematically distinguished boundary value of the uniform LP
interior multiplier was tested once, rather than scanned.  Setting it from
`16` to `0` added no result on the closest TinyImageNet iid143 UNKNOWN and
regressed both positive controls, CIFAR100-medium iid2 and TinyImageNet-medium
iid1, from FALSIFIED to UNKNOWN.  The original value was never changed on
disk.  This closes zero-interior relaxation and confirms that the existing
guard is part of numerical replay stability, not merely unused conservatism.

All tests above used only fixed official instances and verifier-internal
formal quantities.  They performed no input sampling, center/boundary/random
ONNX execution, PGD, BaB, backward bounds, or dual tightening.  The frozen
formal result remains 43/400 FALSIFIED (CIFAR100-medium 12, CIFAR100-large 10,
TinyImageNet-medium 21), and the production candidate remains byte-for-byte
SHA-256 `615b4551...a04993ba`.  The detailed stop-loss record is
`artifacts/hybridz_largecls_gates/phase_projection_commonpath_residual_stoploss_20260812.json`.

The remaining cheap cell-selection variants were then closed under the same
single-path rule.  Moving along the existing property affine to its unique
linearized zero crossing retained three positive controls but added none of
three fixed UNKNOWN sentinels.  Sending the ordinary LP result directly to
the verifier-owned zero-width forward proof, without requiring the heuristic
cell's phase or margin replay, added `0/34` fixed-40 results and caused extra
TinyImageNet deadline pressure.  Thus the intermediate replay gate is not a
source of false UNKNOWNs; it prevents wasted formal-forward work.

Choosing the initial phases by the larger side of each formal interval lost
CIFAR100-medium iid2 and added none of three UNKNOWNs.  Crossing only the
first ReLU boundary made all six sentinels correctly phase-ambiguous under the
outward envelopes, while choosing the opposite property corner lost all
three positive controls.  Neither rule was retained or repaired with a step
parameter.

Finally, the earlier analytic-line idea was reimplemented as scalar,
event-driven forward dual-number propagation rather than repeated generator
streams.  Even this compact form encountered 64 ReLU events after reaching
only `t=0.194` on iid2 and `t=0.276` on iid29; tracing alone cost about 4.2 s
per case, and both results were UNKNOWN.  Increasing its cap would reproduce
the previously observed frontend timeout, so the event-line prototype was
removed.  Static corners, one-neighbor cells, and piecewise line traversal
are now closed as common-path directions.  The detailed entries are appended
to `phase_projection_objective_stoploss_20260812.json`.

The solver-ready primary representation was also revisited only as an
optimistic performance upper bound.  The native RANGE schedule was treated as
already built for free, excluding its construction, authority, and validation
costs.  With two warmup groups and five paired four-worker groups, the
speedups were `2.0015`, `2.0123`, `1.8470`, `1.7760`, and `2.0199`; the median
was `2.0015x`, but the paired-bootstrap 95% lower bound was only `1.7760x`,
below the required `1.80x`.  Since a real implementation can only add work,
single-query (`Q=1`) producer/solver integration is closed.  It may be
reconsidered only if the real verifier uses multiple objectives per identical
model and both candidate and legacy baselines receive the same reuse.

A global complementarity formulation was then screened before implementation.
The feed-forward ReLU identities can be written as complementary pairs, but
with boxed inputs, arbitrary signed residual operators, affine equalities, and
the TOP1 feasibility objective the resulting problem is a general LPCC/mixed
complementarity problem, not a monotone square LCP with a polynomial pivot
guarantee.  A complete solver must still make combinatorial active-set choices
or invoke a mixed-integer/disjunctive search; a custom sparse basis engine
would also duplicate substantial backend machinery.  This conflicts with the
no-BaB, single-common-path, and frontend-latency requirements, so no prototype
was added.

The stop decision is supported by verifier-internal diagnostics on one
official UNKNOWN from each family.  CIFAR100-medium iid0 replayed `2030/2030`
exact phases but had margin `-0.03057`; CIFAR100-large iid100 replayed
`4527/4527` with margin `-0.72798`; TinyImageNet-medium iid0 replayed
`1865/1867` and also had margin `-0.36755`.  Thus the dominant issue is that
the selected exact cell has no violating point, not that the replay check is
too strict.  No sampled input, concrete ONNX point execution, PGD, branching,
backward bound propagation, or dual tightening was used.  The machine-readable
record is `phase_projection_global_cell_stoploss_20260812.json`.

One bounded continuous selector was still admissible and was tested once.
It replaced the analytic corner rather than becoming a fallback: inside the
first affine cell, one LP required the sole positive property direction to
reach zero while maximizing a normalized minimum signed phase-safety scalar.
Its point selected exactly one new phase cell, after which the unchanged exact
cell LP and mandatory verifier replay ran.  CIFAR100-medium iid0 remained
`UNKNOWN`, while fresh diagnostic time rose from about `4.83` to `5.04`
seconds.  The first target therefore triggered the stop rule; TinyImageNet,
fixed-14, multiplier scans, and alternate-cell retries were not run.  No code
was retained.  See `phase_projection_phase_balance_stoploss_20260812.json`.

The analytic line was also retested as an optimistic GPU event engine rather
than the earlier Python scalar traversal.  A two-row tensor propagated each
cell's intercept and slope through the stored Conv/Dense/ADD topology; events
advanced only to analytic ReLU zeros.  Full-network recomputation reached
`t=0.327` after 64 cells in `2.18` seconds.  Caching the current cell and
recomputing only from the event ReLU completed `t=1` in 186 cells, but still
cost `6.00` seconds and 3,497 affine kernel calls.  Most events occurred in
the first two ReLUs (104 and 54), so suffix caching deleted little work.

This is already a generous lower bound: it used raw GPU layer tensors and
excluded outward-error envelopes, the selected exact-cell stream and LP, and
the verifier-owned singleton proof.  Nevertheless the trace alone is slower
than iid2's complete retained phase-projection path (`3.59` seconds).  Its
positive raw affine margin therefore has no formal authority or solved-count
credit.  TinyImageNet and fixed-14 were skipped, and no event engine or cache
was retained.  See `phase_projection_gpu_event_path_stoploss_20260812.json`.

A final representation change removed the per-event kernel launch bottleneck.
Instead of replaying cells sequentially, it carried all current analytic line
segments as one CUDA batch, introduced every raw ReLU zero once per graph
layer, and released branch tensors at their last DAG consumer.  On iid2 this
reproduced all 186 segments using only 21 affine kernels in `0.0818` seconds,
with about 180 MiB peak CUDA allocation.  This is a genuine selector-level
performance result.

The formal capability gate nevertheless failed immediately.  On the first
CIFAR100 UNKNOWN target, iid0, the complete 159-segment line took `0.0763`
seconds but its best raw affine margin was still `-0.07122`; the selected cell
was the final interval `[0.9884,1]`, so the full traversal found no better
phase region than the existing endpoint direction.  The disconnected module
was deleted, no exact LP/verdict path was added, and the experiment was not
expanded to fixed-14 or TinyImageNet.  The performance technique may be useful
only if a future verifier derives a materially different formal line; it is
not itself a solved-count improvement.

The same batch partition was applied once to the only other already-computed
formal direction that required no new solver: the line from the represented
center to the existing exact-cell LP candidate.  iid0 again had 159 segments;
the best raw margin was `-0.0305664` in the final interval
`[0.99999977,1]`, matching the rigorous endpoint margin `-0.03056648`.
Thus neither the property corner nor the exact LP point hides an intermediate
positive cell.  Combining line traversal with the existing LP as a late
fallback is closed.

For completeness, the batch engine checked the one remaining parameter-free
formal endpoint already constructed in this work: the max-min phase-safety LP
subject to the first-cell property reaching zero.  Its optimum phase safety
was itself `-0.0590`; the center-to-endpoint path contained 154 cells and its
best raw margin was `-0.0660`, again in the final interval.  These three
natural formal lines all fail on the same first UNKNOWN target.  No additional
line family, endpoint menu, or direction scan is permitted or retained.

The corresponding natural two-dimensional plane was screened once without
turning it into another runtime option.  Its triangle was defined entirely by
three verifier-owned formal points: the represented center, the analytic
property corner, and the existing exact-cell LP endpoint.  Network values were
propagated as affine plane coefficients and polygons were cut only at analytic
ReLU zero lines; no ONNX input point was sampled or executed.

The first ReLU had 97 relevant zero lines and 1,468 arrangement regions.  A
streaming second-layer census produced only 2,784 regions (at most 23 children
from any parent), so geometric explosion was not the immediate blocker.  The
full-network bounded run nevertheless reached the ten-second frontend cap
after 89 of 92 groups.  It had processed 3,710 final regions, used about
137 MiB of CUDA allocation, and its best observed raw margin remained
`-0.030566416`, the same negative value as the existing exact-LP endpoint.
The three unprocessed groups are not claimed negative.  Rebuilding or retaining
a second execution path merely to inspect that 3.3% tail would violate the
single-common-path and responsiveness requirements, so the plane was closed
without TinyImageNet or fixed-set expansion.  No prototype was retained and
the formal total remains 43/400.

A single fixed batching follow-up ruled out kernel-launch overhead as the
escape route.  Increasing the first-region group width from 16 to 64 reduced
the affine kernel count from 1,780 to 441, but the propagation again reached
ten seconds with one of 23 groups left.  Peak CUDA allocation rose to about
342 MiB, and the best observed margin was still exactly `-0.030566416` at the
old LP endpoint.  Polygon partitioning and live-state replication, not affine
launches, dominate.  Larger batches were not tried because they principally
increase memory and do not address that cost.

One final global selector was screened because it replaced the analytic corner
rather than adding a retry.  In the initial affine cell it imposed a
nonnegative lower envelope for the selected property and minimized the sum of
normalized hinge violations of all 2,030 exact phase rows.  This single LP
selected 137 changed phases without an ambiguous row.  The resulting exact
cell LP replayed all 2,030 phases consistently and improved iid0's rigorous
margin from about `-0.03056648` to `-0.02979118`, but it remained strictly
negative.  The selector itself added 2,131,487 nonzeros and 0.67 seconds; the
whole disconnected diagnostic took 5.36 seconds.

The hinge optimum was also sent directly—once—to ACT's verifier-owned
zero-width interval proof, rather than relying on the selected affine cell or
an external certificate.  The decoded point was inside the original BOX, but
its exact property lower bound was `-0.06433955`.  Thus neither the candidate
point nor its selected exact cell violates the property; the result is not an
artifact of an auxiliary certificate or a strict replay gate.

Because the first UNKNOWN capability gate still produced no formal result,
the total-hinge selector was not expanded to TinyImageNet, fixed-14, alternate
normalizers, or objective aggregations.  No implementation was retained.  The
receipt is
`artifacts/hybridz_largecls_gates/phase_projection_total_hinge_stoploss_20260812.json`.

### 2026-08-12 global phase-affine structure census

The first UNKNOWN target's complete verifier-owned phase-affine matrix was
then inspected to determine whether a genuinely global algorithm could exploit
hidden low dimension or small separators.  The 2,030 by 3,072 matrix has
2,126,574 nonzeros and structural rank 2,030/2,030.  It uses 2,883 input
columns; each active column participates in 458--978 phase rows (median 742).
All 2,030 rows are in one bipartite connected component, and there are no
bitwise-identical or sign-opposite row pairs.

The row interaction graph joins two phases whenever their affine supports
share an input factor.  It contains 1,363,842 of 2,059,435 possible edges
(`66.2%` density); degree ranges from 622 to 2,029 with median 1,268, and an
interacting pair shares a median 147 input factors.  Since the whole graph
already has minimum degree 622, its degeneracy—and therefore its treewidth
lower bound—is at least 622.  This rules out a compact common path based on
strict phase sharing, low-rank phase coordinates, connected components, or
low-treewidth dynamic programming.

SCIP's SOS1 API was also checked before implementation.  Pure complementarity
would remove the explicitly required phase/binary representation; retaining
the binary and using native disjunction reproduces the already-failed SCIP
indicator path and still invokes general branching.  No fifth solver encoding
was added.  The evidence is in
`artifacts/hybridz_largecls_gates/phase_affine_global_structure_stoploss_20260812.json`.

The remaining research frontier is consequently a genuinely global nonconvex
exact root algorithm over a dense, high-separator phase system.  It cannot be
obtained by another corner, line, hinge normalization, local phase table,
component split, or wrapper around the same generic solver.  The production
candidate and the formal 43/400 result remain unchanged.

The verifier control flow was also checked for a cheaper completeness gap.
An unsuccessful phase projection currently returns `UNKNOWN` before the
generic interval SAFE test.  This is sound but could in principle miss SAFE
instances.  On one fixed UNKNOWN from each family, however, ordinary interval
property uppers remained strongly positive.  Repeating the check with the
constraint-local HybridZ output cube and exact-dyadic property accumulation
also failed: maximum uppers were `136.43` (C100 medium iid0), `42.73` (C100
large iid100), and `180.00` (Tiny iid0).  Since the first three-family gate
showed no capability, no SAFE side path was added.

Finally, the complete center-to-exact-LP analytic line was evaluated against
all 99 TOP1 rivals rather than only the selected row.  Across all 159 cells,
the best row was still row 53 at the final endpoint, with margin
`-0.030566416`.  Thus the first target is not failing because a positive
alternate rival was ignored.  See
`artifacts/hybridz_largecls_gates/phase_projection_safe_and_all_rival_stoploss_20260812.json`.

### 2026-08-12 support-kernel and dual-pivot stop-loss

A candidate-only profile on the same official CIFAR100-medium iid0 UNKNOWN
separated frontend work from model parsing.  The two exact-cell builds took
about 1.45 seconds in aggregate; their two generator streams accounted for
about 0.60 seconds.  Repeated compact Conv support propagation was a real hot
spot: 105 forward-support calls accounted for about 0.84 seconds.  This also
corrects the dependency name in earlier shorthand: the frozen file is
`forward_exact_relu_live_row_stream_candidate.py`, SHA-256
`be00bd11...9004e86`.

Several implementations were screened without changing production.  CUDA
float64 support convolutions were bitwise equal but reached only `1.450x`
with forward and backward combined.  Float32 appeared faster (`1.623x`) but
was invalid: even with TF32 disabled it produced 512 false-positive support
rows in one Conv call.  A purpose-built Triton Boolean-OR forward kernel was
exact and reached `1.505x` median, but one of five paired runs was only
`1.486x`.  That margin is inadequate for the required concurrency gate and
duplicates the already-closed build-local support-cache direction, so no
kernel or second runtime route was retained.  Building the audited full Conv
CSR and filtering afterward was worse (`0.864x`), because it materialized
74,489,216 nonzeros to retain only 4,960,430.

An intentionally impossible upper bound then replayed all 105 support results
from precomputed truth and charged no computation, sealing, validation, or
cold-population cost.  Five paired full-candidate speedups were `1.499x`,
`1.497x`, `1.495x`, `1.482x`, and `1.518x`: median `1.497x`, minimum `1.482x`.
Even this unrealizable best case misses the first `1.50x` gate.  Support
underflow special cases, persistent caches, and another kernel path are
therefore closed without implementation; a real cold request can only be
slower than this bound.

The stronger combined upper bound also failed.  On the known-positive
CIFAR100-medium iid2 control, all 105 search-stage support results were
precomputed for free and the complete terminal singleton proof was likewise
replayed at zero cost.  Five groups of four simultaneous requests produced
paired speedups `1.317x`, `1.344x`, `1.342x`, `1.297x`, and `1.356x` (median
`1.342x`).  This is far below the `2.00x` concurrency gate even though no real
implementation can match its zero-cost assumptions.  The verifier-owned
singleton proof therefore remains intact; its authority surface is not
widened merely to save work.

There is one genuine formal near miss, but it is not counted.  The fixed-400
run stopped TinyImageNet-medium iid60 immediately before singleton replay.
With a diagnostic-only extended deadline, the same verifier-generated
candidate replayed all `2926/2926` exact phases, had projected margin lower
`0.5646356418`, and was proved by ACT's zero-width interval replay with exact
margin lower `0.5646358087`.  One measured run spent about `1.124` s in the
first stream, `0.964` s in the second, `3.191` s in the LP, and `1.117` s in
the terminal proof, for `11.408` s total.  This demonstrates latent verifier
capability, but extending the common deadline or adding a candidate-ready
grace path would worsen response latency and fails the combined four-worker
stop-loss above.  It remains `UNKNOWN` in the formal 43/400 total.

One materially different cell-selection rule was also stopped at the first
capability target.  At the exact-cell LP optimum it flipped only the active
phase inequality with the largest-magnitude HiGHS marginal, then rebuilt and
solved that single adjacent exact cell.  Both cells replayed all `2030/2030`
phases with no ambiguity.  The rigorous margin improved from
`-0.0305664810` to `-0.0302519995`, but remained strictly negative while
adding one complete stream and LP.  No multi-pivot path, retry list, fixed-14
expansion, or TinyImageNet run followed.

No input sampling, ONNX point execution, PGD, BaB, backward bounds, or dual
tightening was used.  Scratch code was deleted and the disposable probe was
restored to SHA-256 `8295e504...0eaca81`.  The formal count remains 43/400;
the detailed record is
`artifacts/hybridz_largecls_gates/phase_projection_support_dual_pivot_stoploss_20260812.json`.

### 2026-08-12 admissible global-root algorithm screen

The remaining root-search space was screened against the updated project
contract before adding another backend.  A feasibility pump is not a new
mechanism here: binary-only MIP starts, deterministic rounding of persistent-
LP candidates, a complete constructive start, a second phase projection, and
fixed-cell LP variants have already failed either capability or concurrency
gates.  Reluplex/DPLL(T), exact-star enumeration, and learned phase clauses
obtain completeness by activation case splits; under the current contract
that is the forbidden combinatorial search under a different name.

Convex LP/SDP variants violate the no-relaxation rule when used for a result.
The theoretical exact completely-positive formulation does not provide a
tractable escape: completely-positive membership remains hard and the lifted
matrix is quadratic in the full forward-graph variables.  General mixed
complementarity is likewise nonmonotone on these signed residual networks and
has no complete root-only pivot guarantee; retaining a local numerical solve
would create another nonauthoritative search path and custom backend.

This negative screen is also supported by the local structure already
measured on iid0: 2,030 phase rows have full structural row rank, one dense
interaction component, minimum interaction degree 622, and a treewidth lower
bound of 622.  There is no hidden small separator on which a compact dynamic
program could replace the global solve.  Consequently no feasibility-pump
wrapper, fifth solver encoding, relaxation path, or complementarity engine was
added.  The restart condition is intentionally strict: a future proposal must
give a genuinely new exact root representation for this dense system and
cross the existing single/common-response gates before any formal expansion.
The machine-readable screen is
`artifacts/hybridz_largecls_gates/global_exact_root_algorithm_screen_20260812.json`.

Two final common-path upper bounds make the remaining division of work
explicit.  Reusing the first-cell analytic corner directly in the updated
exact cell, while retaining every phase/margin/BOX/singleton check, lost all
three positive controls: CIFAR100-medium iid2, CIFAR100-large iid118, and
TinyImageNet-medium iid1.  The candidate LP is therefore capability-essential,
not optional cleanup.  Conversely, precomputing the correct iid2 LP solution
and charging zero time for the entire LP still gave only `1.177x`, `1.230x`,
`1.246x`, `1.248x`, and `1.248x` across five four-request groups (median
`1.246x`).  Native loading, warm bases, or a GPU row-action implementation
cannot cross the `2.00x` common-response gate by optimizing the LP alone.

The definitive local-optimization floor made all three identified stages
free simultaneously: every search support result, the correct LP solution,
and the complete singleton proof.  Only the two generator streams, selected
affine matrices, strict phase/margin replay, and host/device orchestration
remained.  Five four-request groups achieved `1.837x`, `1.864x`, `1.952x`,
`1.846x`, and `1.796x` (median `1.846x`) with identical results and zero cache
mismatches.  Even this impossible implementation misses the `2.00x` gate.
Consequently support kernels, caches, native LP loading, warm starts, proof
reuse, and combinations thereof are jointly closed.  A future promotable
candidate must eliminate a complete generator stream or replace the core
representation; optimizing another local stage cannot suffice.

The first of two fixed-400 `unexpected_fail_closed:ExactReLULiveRowStreamError`
large cases was also diagnosed once.  C100-large iid103 completed in 8.868 s
under a single extended diagnostic and returned the ordinary negative
phase/margin replay outcome.  Its fixed-400 label came from the live stream
observing the ten-second deadline under concurrency, not an RSS cap or
last-use failure.  No memory-specific branch or deadline exception was added.

### 2026-08-12 active-goal blocked audit

The active objective was audited requirement by requirement after the same
blocking condition recurred for three consecutive goal turns.  The retained
path remains sound and focused regressions are green (`22/22` across the phase
projection and live-row suites).  The fixed-400 record is complete and has no
forbidden-method flag.  The representation, solver, memory, structural
sharing, residual, cache, and root-selector censuses are also present.

The requested end state is nevertheless not complete: the formal total is
still 43/400, and the remaining 357 cases require a materially different
global exact phase-root mechanism.  The phase system is dense and high-
separator, all admissible local work is below the response gate even under an
impossible zero-cost bound, and known complete global methods require a
currently prohibited case split, relaxation/bound propagation, backward
sensitivity, or slower multi-path search.  Continuing to add parameter
variants or fallback branches would contradict the explicit low-redundancy
and frontend-response requirements.

Work can resume if any one of the following becomes available: a new exact
root representation that avoids every prohibited mechanism; authorization
for a narrowly bounded complete split; permission to use backward/convex-root
information only for candidate selection while retaining exact terminal
proof; or a changed response/deadline contract for rigorously verified near
misses such as TinyImageNet iid60.  Until then the goal is formally blocked,
not completed.  The machine-readable audit is
`artifacts/hybridz_largecls_gates/active_goal_blocked_audit_20260812.json`.

### 2026-08-12 resumed goal amendment: fixed-HZ warm phase walk

The goal was resumed without removing any restriction after an external
review identified a valid logical gap in the global-root screen.  A candidate
phase search does not need a completeness theorem: it may stop at any resource
limit and return `UNKNOWN`, provided that it has no result authority and every
successful decoded input is still discharged by the raw BOX, verifier-owned
zero-width forward replay, and exact stored-binary64 property check.  Thus
"no complete pivot guarantee" is not by itself a reason to reject a
deterministic phase walk.  The no-sampling, no ONNX point execution, no PGD,
no activation/input split, no BaB, no triangle/convex verdict fallback, no
backward bound, and no dual-tightening rules all remain in force.

The proposed implementation was tested at the smallest discriminating gate.
Every exact ReLU binary in the existing constraint-local live-row HZ was fixed
to the already established projected cell.  HiGHS was loaded as a pure LP:
there were no integer columns, nodes, branches, or phase enumeration.  A
neighbor changes only one fixed binary bound, and the initial IPM crossover
produces a valid basis for subsequent dual-simplex solves.

One correction to the proposal was necessary.  Raw Big-M branch duals are not
phase-boundary marginals: the branch equality is active throughout a phase,
even when the phase boundary is unreachable.  Ranking those raw values picked
an empty cell and spent 15.66 seconds proving it infeasible.  The exact
constraint-local analogue of the earlier composed-LP rule is instead:

- for an inactive phase, the lower sign row must be at its basis upper bound;
- for an active phase, the normalized ReLU-output factor must be at its basis
  lower bound;
- only these boundary-active coordinates are eligible, after which the
  largest absolute dual is selected with binary-position tie breaking.

This corrected rule reproduced the old independently recorded pivot exactly:
binary position 148 is ReLU `(layer 3, row 2389)`.  The bound flip took about
0.175 seconds and changed the constraint-local HZ objective from
`-0.0305663401` to `-0.0302518606`, matching the old reconstructed-cell result
`-0.0302519995` up to the different outward candidate representation.  A
strictly monotone continuation made five distinct flips and improved the
margin to `-0.0289411387`; the sixth solve did not strictly improve and the
single path stopped.  It produced no new formal verdict on CIFAR100-medium
iid0, but it confirms that the warm bound-flip mechanism itself is real.

The route nevertheless fails the preregistered response gate before any
fixed-14 expansion.  The complete Operator-HZ is the wrong representation:
on CIFAR100-medium iid2 it took 11.97 seconds and stored 10.50 million
constraint nonzeros.  The intended one-stream live-row HZ is much smaller
(6,853 continuous factors, 1,413 binaries, 6,598 rows and 1.61 million
constraint nonzeros) and took about 1.48--1.77 seconds to build.  However, its
first fixed-cell IPM still took about 2.23 seconds.  On the same positive
control, just `live-HZ build + first IPM` cost 3.715 seconds while the retained
complete projection path cost 3.880 seconds.  This deliberately impossible
candidate lower bound omits the analytic projected-cell seed, all walk steps,
input decoding, raw-BOX replay, singleton forward proof, and exact property
check, yet its maximum possible speedup is only about `1.044x`, below the
required `1.50x` first gate.  A real implementation can only be slower.

The fixed-binary HZ is also not claimed to be bit-identical to the composed
input-factor cell polytope.  It retains outward numerical-error factors, and
the HiGHS proposal copy explicitly filters coefficients at its supported
small-matrix threshold.  This is acceptable only because the LP and its
duals remain candidate generators and final ACT replay remains authoritative.

Accordingly the idea is recorded as a useful correction and a successful
mechanism experiment, but it is stopped rather than added as a slow fallback.
There was no fixed-14/fixed-40/fixed-400 run, no production integration, and no
new formal solved case.  The audited score therefore remains `43/400`:
CIFAR100-medium `12`, CIFAR100-large `10`, and TinyImageNet-medium `21`.  The
357-UNKNOWN ceiling census was not expanded after the single-pair performance
gate and the first official UNKNOWN capability gate both failed; such a census
could not authorize this implementation.  It may be reopened only if a new representation can
produce both the fixed-cell model and its first usable basis cheaply enough to
pass the existing single/common-response gates.  Full measurements are in
`artifacts/hybridz_largecls_gates/phase_projection_fixed_hz_warm_walk_stoploss_20260812.json`.

### 2026-08-12 resumed external proposal: rank-one walk elimination

The external proposal was resumed and its central criticism of the earlier
global screen was accepted.  Rejecting a pivot walk merely because it lacks a
completeness guarantee was stronger than the contract: exhaustion or a local
stall may return `UNKNOWN`, and only the unchanged terminal replay has verdict
authority.  The high-separator result rules out a cheap decomposition; it does
not rule out deterministic movement between exact cells.  This closes a logic
gap in the screen, but does not by itself change a verdict or relax a rule.

The first elimination experiment attacked the measured per-neighbor rebuild
cost in the existing composed input-factor cell LP.  A single phase flip has a
rank-one downstream effect, so the base HiGHS model can stay resident while one
normalized auxiliary factor and one native `RANGE` definition row are added.
The downstream influence and its outward floating-point error are propagated
through the same forward graph; there is no second neighbor matrix rebuild.
For the established CIFAR100-medium iid0 pivot `(layer 3, row 2389)`, the
audited one-update path reduced the scratch materialized rank-one update's
roughly 1.10-second assembly to about
0.166 seconds in its best complete diagnostic confirmation (delta-error pass,
assembly, model mutation, warm solve, solution/dual readback, and
exact-fraction definition postcheck).  The candidate's complete evaluation
radii covered the stored-affine residuals against an independently rebuilt
neighbor on all `2131/2131` checked rows, and both candidate and neighbor
replayed all `2030/2030` ReLU phases.  This one-pivot diagnostic does not
establish a whole-suite soundness result.  The candidate margin was still
strictly negative, approximately
`-0.0302519970`, so this is a genuine mechanism improvement but not a new
formal result.

An independent formula audit established the intended normalized actual-value
`RANGE` as a sound outer candidate model, not as cell or verdict authority.
Any multi-pivot extension would have to append updates sequentially, recompute
the downstream influence under the new phase mask, and carry all cross terms;
parallel reuse of influences from the frozen base would be unsound.  The
terminal input-only replay remains mandatory regardless of the LP margin.

The decisive cost is now before that cheap update.  Eliminating the initial
second generator stream requires representing the 187 iid2 seed-cell phase
changes at once.  The measured auxiliary DAG reproduced the fully expanded
generators to about `2.17e-17`, but a production-shaped 187-auxiliary first LP
needed `4.635` seconds for its first solve and put the end-to-end initial
response floor near `6.09` seconds.  It therefore fails even the zero-pivot
single-response gate before any walk step.  Pre-eliminating the auxiliary DAG
back into input coordinates made the LP smaller, but its optimistic initial-response
floor was already at least `1.784` seconds before the still-missing sound
outward-bound construction and authoritative replay.  Although that partial
zero-pivot sum alone is below the `2.2145`-second limit implied by the frozen
`3.3218`-second reference and the `1.50x` gate, it leaves too little room for
the preregistered maximum-five-attempt walk: after the measured per-step cost,
the zero-pivot portion had to stay below about `1.495` seconds (`1.428` under
the conservative warm-solve shape).  The diagnostic also borrowed the full
second-stream RHS as an oracle, so it cannot be promoted or used to claim that
the missing stream has been eliminated.

One disposable iid2 timing receipt accidentally stamped its
`scope.fixed_manifest_id` as `cifar100_medium_iid0`.  The 187-change signature,
saved iid2 capture, and invocation identify the run as iid2; only the durable
machine-readable record below corrects that metadata, and the temporary output
must not be treated as a canonical receipt.

The intended walk was a single preregistered rule with strict monotone margin
improvement, deterministic tie breaking, fail-closed deadline/stall handling,
and a fixed maximum of five pivot attempts; only strict improvements would be
committed.  The complete bounded-walk response budget, together with the
unfinished outward construction, stopped the route before that rule reached
an official sentinel.  No
end-to-end positive-control gate, fixed-14 run, or real-suite expansion was
performed; no candidate code was retained in production.  The diagnostic work
used no input sampling, no execution of ONNX at center, boundary, random, or
LP-decoded points, no PGD, no BaB, no activation/input split or enumeration, no
backward bound or dual tightening, and no triangle/convex verdict fallback.
The full-neighbor computations mentioned above were verifier-internal
forward-affine diagnostics only and had no result authority.

The formal score remains `43/400`: CIFAR100-medium `12`, CIFAR100-large `10`,
and TinyImageNet-medium `21`.  This route should be reopened only when a new
initial representation can construct the projected exact cell, sound outward
row bounds, and its first usable LP basis without a diagnostic second-stream
oracle, and when the measured complete bounded-walk response (including setup,
every allowed pivot, soundness work, and replay rather than an optimistic
partial sum) first clears the existing `1.50x` single-pair gate and then the
common-response gates.  A faster later pivot or a zero-pivot-only partial sum
is not a restart condition.  The machine-readable
record is
`artifacts/hybridz_largecls_gates/phase_projection_rank_one_warm_walk_elimination_stoploss_20260812.json`.

### 2026-08-13 single-stream float64 candidate breakthrough

The next experiment applied the proposed separation literally: approximate
arithmetic is used only to generate a candidate, while the terminal verifier
proof is unchanged.  The disconnected probe uses one float64 generator
stream, a single deterministic analytic-corner rival/phase selector, and a
topologically triangular batch of phase-change auxiliaries pre-eliminated into
the input-factor coordinates.  It intentionally does not construct candidate
outward error bands and does not replay intermediate phases or margins.  One
`highs-ds` LP with presolve disabled is the only candidate path; there is no
algorithm menu or retry fallback.

This weaker candidate layer has no authority.  A positive LP margin is useful
only as a proposal.  The decoded input must still lie in the raw input BOX and
must pass the existing verifier-owned zero-width interval forward pass and the
exact stored-binary64 `Fraction` property lower-bound check.  A negative,
infeasible, nonfinite, or terminally rejected candidate remains `UNKNOWN`.
The experiment used no input sampling, ONNX input-point execution, PGD, BaB or
split, backward bounds, or dual tightening.

The ability result is materially positive.  All 43 cases reported
`FALSIFIED` by the frozen fixed-400 production run were replayed in fresh child
processes and all 43 again passed the unchanged terminal proof.  Separately,
the official VNN-COMP 2025 result repository identified 24 public-SAT cases
that were `UNKNOWN` in the frozen ACT record; public labels were used only to
choose this diagnostic set, never at runtime or for a verdict.  Sixteen of
those 24 now passed ACT's terminal proof: CIFAR100-medium iid26 and iid39, plus
TinyImageNet-medium iid3, iid6, iid10, iid23, iid32, iid53, iid54, iid60,
iid73, iid108, iid146, iid161, iid164, and iid192.  The other eight remained
honest `UNKNOWN`.  Therefore the disconnected probe has a measured lower
bound of `59/400` terminal-proved instances: the old 43 are retained and 16
are new.  The production/formal score remains `43/400` until integration and
the remaining gates are complete.

The result also exposes a performance boundary.  A same-process warm
CIFAR100-medium iid2 sentinel took a median about `1.4734` seconds versus the
registered `3.3218`-second retained reference, a diagnostic `~2.25x` speedup.
That is not a performance-gate verdict: the 43-case fresh-child retention run
accumulated substantial system slowdown, and some Tiny/large candidates took
8--25 seconds.  No four-worker wall/RSS gate has run.  Production promotion is
therefore blocked on focused algebra/fail-closed tests, an independent audit,
an unlabeled fixed-400 candidate run, and paired isolated single/four-worker
wall and memory gates.  The prototype remains a single disconnected scratch
file rather than another backend fallback.

The implementation is `scratch_phase_projection_float64_probe.py`, SHA-256
`66fd1a47692ba786beed27af880a0f087f01dac450014f769f12fe62e190f169`.
Its focused algebra/scope test is
`test_scratch_phase_projection_float64_probe.py`, SHA-256
`8155c4c77cb6aeaab8f1115c232a017edbdbfeadf05dae9245995d4be25bf856`
(`6/6` tests passed).  An attempted five-pair AB/BA run was explicitly
non-authoritative: another user's `abcrown.py` process occupied about 46.7 GiB
on the same GPU at 97--98% utilization.  Its `0.973x` median paired ratio is
recorded as a failed/contaminated gate, not as the candidate's intrinsic
speed; the foreign process was not modified or terminated.  The exact gate
must be repeated only when the GPU is uncontended.
The machine-readable record is
`artifacts/hybridz_largecls_gates/phase_projection_single_stream_float64_candidate_20260813.json`.

### 2026-08-14 GPU-emitted selected CSR passes the disconnected promotion gates

A third representation restart retained the existing ordered-CSR generator
stream but removed its dominant host construction.  Two deterministic Triton
passes count and emit the selected Conv CSR directly on the GPU; Dense rows use
binary64 indexed tensors.  The admitted domain requires every stored affine
weight to be finite and nonzero.  A zero weight returns `UNKNOWN`; the old
builder is not retained as a fallback.  Candidate arithmetic still has no
authority, and the only terminal proof remains raw BOX membership, ACT's
zero-width interval forward pass, and the stored-binary64 `Fraction` property
lower bound.  No input sampling, ONNX point execution, PGD, BaB/split,
backward bound, or dual tightening was used.

On CIFAR100-medium iid2, selected-CSR build plus schedule fell from `0.51887`
seconds to a `0.02600`-second median, with byte-exact indptr, indices, and data.
The same exact comparison and unchanged terminal margin passed on
CIFAR100-large iid118 and TinyImageNet-medium iid1.  A focused grouped,
dilated, strided, padded, batched Conv test was then added; the focused suite
is `9/9` green.

The frozen response gates passed without a rerun for luck.  The five
single-request paired speedups were `3.9319, 2.0287, 1.3798, 2.8848, 1.9675`,
with median `2.0287x` versus the required `1.50x`.  The decisive four-thread
speedups were `1.8009, 2.1718, 1.9269, 2.3608, 2.0391`; median `2.0391x` passed
the `2.00x` gate and the exact five-point paired-bootstrap lower bound
`1.8009038x` passed the `1.80x` gate.  There were no result conflicts.  This
bootstrap pass is narrow and is frozen rather than repeatedly measured.

The fresh-process four-worker memory gate also improved total resource use.
Host HWM fell from `4,948,779,008` to `3,844,440,064` bytes.  Median CUDA peak
rose from `517,303,296` to `829,996,032` bytes, but the host-plus-CUDA diagnostic
fell from `5,466,082,304` to `4,674,436,096` bytes.  All measured outcomes were
terminally verified.

Fresh-child capability gates then completed: fixed-14 was `4 FALSIFIED / 10
UNKNOWN / 0 errors`; the retained-59 set was `59/59`; and the complete
fixed-400 manifest was `59 FALSIFIED / 341 UNKNOWN / 0 errors`.  The candidate
breakdown is CIFAR100-medium `14/100`, CIFAR100-large `10/100`, and
TinyImageNet-medium `35/200`.  The minimum exact terminal margin was
`0.00025346796228209456`.  Thus the disconnected lower bound remains a fully
measured `59/400`, exactly retaining the previous 43 and adding 16.  The formal
production score is still `43/400`: production integration and a bounded
hostile audit have not yet completed, and no official score claim is made.

The implementation is `scratch_phase_projection_float64_probe.py`, SHA-256
`f23a36dbe58f41011d989eac8823723949eb63561b7ed85815ae3748adcc4bf4`.
The focused test is `test_scratch_phase_projection_float64_probe.py`, SHA-256
`14f8b23cb5ea4440830d1e4803d36815f1c4ffac591fd23e61fc8bef888f1777`.
The full fixed-400 JSONL SHA-256 is
`57c856ba09cfe16d8e5401707ad093f4d71443149fb6bd92ca3aeb39e7592984`.
All gate details and receipt hashes are in the machine-readable artifact cited
above.  The next admissible step is to replace the existing phase-projection
implementation with this sole representation, not add another backend option;
deadline or unsupported-domain failures must remain immediate `UNKNOWN`.

The uncontended gates were subsequently completed.  Five single-request
AB/BA pairs all favored the new candidate; their median paired speedup was
`1.8279x`, above the required `1.50x`, with identical margins on every run.
The four-request profile then isolated a complete redundant operation: every
cold request spent about half a second scanning `weight != 0` during both
directions of live-row support analysis, even though every stored affine
weight in all three benchmark models is nonzero.  The disconnected path was
narrowed to that exact domain and now computes the same support relation by
ones-convolution/transpose-convolution.  A zero stored weight fails closed;
there is no old-algorithm fallback.  CIFAR100-medium, CIFAR100-large, and
TinyImageNet each reproduced the original `live_rows` and `possible_rows`
exactly.  On iid2 the isolated support stage fell from `0.2896` to `0.0546`
seconds (`5.31x`).

This structural deletion was still insufficient for promotion.  Under the
unchanged two-warmup, five-pair, four-thread protocol, paired speedups were
`2.1486, 2.1091, 1.7819, 2.1906, 2.0909`.  The median `2.1091x` passes the
`2.00x` gate, but the deterministic 20,000-resample bootstrap 95% lower bound
was `1.7819x`, below the required `1.80x`.  There were zero result conflicts;
median process RSS decreased from about 4.95 to 4.58 GB, while median CUDA
peak increased from about 521 to 834 MB.  By the preregistered stop-loss, the
candidate is frozen with `NO-PROMOTION`; it will not be tuned or rerun for
luck to close the approximately `0.0181x` shortfall.  The measured
terminal-proof ability lower bound remains 59/400, while the formal production
score remains 43/400.  A restart requires a different structural
representation that removes another complete per-request traversal or
materialization, followed by all gates again.

One such replacement was screened once before implementation.  Because the
float64 candidate has no authority, the live-row support analysis and all
selected CSR matrices could in principle be replaced by full-layer native
float64 Conv/Dense generator batches.  On CIFAR100-medium iid2 the native
generator reproduced every stored preactivation and output coefficient
bit-for-bit and retained the terminal exact margin, but took `1.3685` seconds
versus `0.3603` seconds for the selected-CSR generator (`0.2633x`, about 3.8
times slower).  Even granting the entire measured setup cost for free cannot
make that full-native representation faster end to end.  It was therefore
rejected before any file change or suite expansion; it is not a fallback.

### 2026-08-13 selected-CONV representation restarts: both concurrency stop-losses

Two further representation restarts were evaluated under the same strict
scope: no input sampling or ONNX point execution, no PGD, no BaB/split, no
backward bounds or dual tightening, no runtime fallback, and no authority for
candidate arithmetic.  Both retained the unchanged raw-BOX, zero-width ACT
forward, and stored-binary64 `Fraction` terminal proof.  Neither changed the
formal `43/400` score or the disconnected `59/400` ability lower bound.

The first restart removed selected-CSR construction entirely.  A single
runtime-geometry Triton kernel read the original binary64 CONV weights and
computed only the established live rows.  On CIFAR100-medium iid2 it avoided
`5,178,905` selected CSR nonzeros, reproduced preactivation/output generators
bit-for-bit, and reduced the isolated representation path from `0.9741` to
`0.4961` seconds (`1.9637x`).  Against the correctly frozen phase-projection
baseline, the five-pair single-request median was `2.4701x`.  An earlier
`7.76x` number was rejected because its harness accidentally compared against
the ten-second HZ `UNKNOWN` path rather than the frozen baseline.

The direct kernel failed the decisive concurrency gate.  Four-thread paired
speedups were `1.6727, 1.8067, 1.7259, 1.6572, 1.4888`; the median was only
`1.6727x` and the paired-bootstrap lower bound only `1.4888x`, versus required
`2.00x/1.80x`.  A stage profile showed each request's first generator stream
expanding to about `1.68--1.77` seconds under GPU contention.  Fusing its
selected temporary and scatter did not materially improve the warm path, so
the implementation was removed rather than retained or tile-tuned.

The second restart returned to the concurrency-friendlier ordered-CSR stream
but exploited the already-audited all-nonzero-weight domain to emit each layer's
selected CSR in one vectorized operation.  A zero weight still mapped directly
to `UNKNOWN`; there was no fallback.  It retained exact CSR data/order and the
terminal margin, but the large temporary topology arrays caused request setup
to fluctuate between about `1.18` and `1.90` seconds even in the single gate.
That gate barely passed at `1.5052x` median.  Under four threads, setup expanded
to `2.54--3.52` seconds per request and paired speedups collapsed to
`1.1448, 1.0887, 1.2651, 1.0371, 1.0524`; median/bootstrap were
`1.0887x/1.0371x`.  It was removed without chunk-size, allocator, or parameter
scans.  The frozen implementation and its six focused tests were restored
byte-for-byte.  The durable measurements are in
`artifacts/hybridz_largecls_gates/phase_projection_single_stream_float64_candidate_20260813.json`.

### 2026-08-14 production integration and formal fixed-400 result

The GPU-emitted selected-CSR candidate has now replaced the old two-stream
phase-projection implementation behind the existing verifier configuration.
It is still default-off and lazily imported; enabling it remains mutually
exclusive with the optional phase/property/dual enhancements.  The old
two-stream algorithm was removed from the production module rather than kept
as a fallback.  Explicit converter-produced `SCALE` and `BIAS` layers are
handled in the same forward path, and their zero-width terminal intervals use
directed outward rounding.  A zero scale or any zero affine weight remains
outside the admitted domain and maps to `UNKNOWN`.

The production `verify_once` fixed-400 run completed with `59 FALSIFIED`, `341
UNKNOWN`, and `0 ERROR`: CIFAR100-medium `14/100`, CIFAR100-large `10/100`, and
TinyImageNet-medium `35/200`.  Its FALSIFIED set is exactly equal to the
disconnected 59-case set, so all old 43 are retained and all 16 new cases now
have production verifier receipts.  The minimum stored-binary64 `Fraction`
terminal margin is `0.00025361815719049385`.  This updates the formal enabled
path from `43/400` to `59/400`; the gain is 16.

Across the 400 fresh workers, production candidate time had median `1.9768`
seconds, p95 `3.4711` seconds, and maximum `10.4946` seconds.  Whole-worker
time including model parsing/conversion had median `3.9814`, p95 `5.4779`, and
maximum `12.4236` seconds.  Deadline expiration and all unsupported/numerical
failures still map directly to `UNKNOWN`; no root solver fallback exists in
this mode.

The frozen production hashes are:

- phase projection: `4b66470df55edebb595e0e06c6b8a2de5c65496b8671c4d2f2552003d01ea306`
- live-row/GPU-CSR dependency: `d53c2335c43905097e78bef8311175d7151d7e98293a6152fce62dba00d37511`
- focused test: `a59d08ad648bf8d91387f7f2d6934cf318c7d00c355c1bc2276f6f6eb4441991`
- fixed-400 harness: `d3e1825b9a8f8c8bb7f83cb8f08bdabd68f0c3fa32c379cac8ce08bb4c1e24c1`
- fixed-400 JSONL: `749db4e400329598c23c3dd7c9b9863c291eb3d1ba556cdcfcfa879c58487b43`
- fixed-400 summary: `036a87f7005033ad8478af5dbecfce8657d3f02457f654a9b0e63e5b47e2ab41`

The focused/adjacent suite is `34/34` green, including grouped+dilated Conv,
explicit SCALE/BIAS enclosure, zero-scale rejection, terminal rejection, and
single-path verifier integration.  The restrictions remain unchanged: no
input sampling or ONNX point execution, no PGD, no BaB/split/enumeration, no
backward bounds, and no dual tightening.  Candidate LP values have no proof
authority.  A subsequent independent bounded hostile audit reported
`NO-BLOCKER` for these frozen hashes and the existing release gates.  The mode
remains default-off because changing the default is a separate release-policy
decision, not because another implementation audit is pending.  No additional
fallback or selector menu should be introduced.

A bounded self-audit was subsequently completed without changing either
production hash or rerunning the narrowly passing concurrency gate.  The two
production-focused suites passed `25/25`; five direct fail-closed checks covered
malformed GPU row selection, zero affine weights, an expired terminal replay
deadline, and an unexpected candidate exception with the root fallback held
forbidden.  All 400 durable receipts were reparsed: case IDs were unique,
counts remained `59/341/0`, every FALSIFIED record carried a positive terminal
margin and verifier-owned authority, every candidate receipt denied authority,
and all prohibited-feature flags were false.  Static inspection confirmed one
LP call, no prohibited runtime import/autograd/basis menu, clean compilation,
and no trailing whitespace.  This is deliberately recorded as a self-audit,
not an independent release audit.  The later independent audit completed that
requirement, so the current artifact now records `audit_complete=true`; the
default-off state remains intentional.  The updated machine-readable artifact is
`artifacts/hybridz_largecls_gates/phase_projection_single_stream_float64_candidate_20260813.json`,
SHA-256 `a1323cf69ac5e9f2f7189b8ce4ac96e67cd6e0b354f1c6a044a3a5433b7e16c9`.
The same bounded audit also compared GPU-emitted and CPU-reference CSR arrays
bit-for-bit across 128 deterministic Conv geometries and four Dense geometries
with zero mismatch.  A separate exact-`Fraction` DAG oracle enclosed all eight
checked outputs through Dense, SCALE, BIAS, ReLU, Dense, and ADD.  Fresh-process
checks confirmed that the mode defaults to zero seconds and importing the
verifier does not load the candidate module.  Finally, the disconnected and
production 59-case positive sets had an empty symmetric difference.

The receipt-only UNKNOWN census contains `333` strictly negative candidate
margins, seven infeasible LPs, and one LP time limit.  There are no positive candidates rejected by
the terminal proof, so the current gap is cell/objective ability rather than a
terminal-validation bottleneck.  The closest negative margins are TinyImageNet
iid143 at `-0.00107108`, CIFAR100-large iid110 at `-0.00453403`, and
CIFAR100-large iid153 at `-0.00546486`; the median negative margin is about
`-0.42616`.  This census only parses verifier receipts and performs no input
sampling or point execution.  Any next ability experiment should therefore be
one deterministic cell update with a preregistered latency stop, not another
selector/backend menu.  The previously rejected five-pivot warm-walk cannot
simply be revived: its required zero-pivot budget was below the current path's
measured latency.

The full durable histogram is
`artifacts/hybridz_largecls_gates/phase_projection_margin_ceiling_census_20260814.json`,
SHA-256 `0ed6b4980dc52f3852551f2afded26c18802f90cf200fc454ee37dd99a971551`.
Its mutually exclusive bins from `(-inf,-1]` through `(-0.005,0)` contain
`48, 93, 146, 25, 13, 4, 2, 2` cases.  Strictly inside `0.01`, `0.02`, and
`0.05` of zero are only `4`, `8`, and `21` cases.  The historical single-pivot
gain of about `0.00031448`, if unrealistically transferred unchanged to every
instance, crosses no case; the historical five-attempt cumulative gain of
about `0.00162520` crosses only iid143.  These are scale scenarios, not solved
predictions or verdict evidence.

The performance-reference question was also recomputed from the frozen raw
gate records rather than from cross-family fixed-400 medians.  On the registered
CIFAR100-medium iid2 gate, the current-path wall median is `1.438896` seconds
and the terminal-proof median is `0.429867` seconds, not `1.77/1.1`.  A ratchet
comparison against the current path gives a next total budget of `0.959264`
seconds and leaves `0.529397` seconds after the measured terminal median.  Thus
an additive capability step cannot pass, but the data do not prove that terminal
authority alone makes every future structural replacement impossible: a new
path would have to add ability while deleting about `0.479632` seconds of the
current nonterminal work.  Under a fixed old-43 reference, isolated single
headroom is `0.908779` seconds, but four-thread median headroom is only
`0.118624` seconds and the tightest observed pair has only `0.001755` seconds
before its `1.8x` allowance is lost.  Consequently a universal `+0.15`-second
step is not already proved compliant under either complete gate; a real changed
path would need fresh paired measurement.  A step invoked only after a negative
margin must additionally be timed on a fixed negative control, because the
positive iid2 gate would bypass it.  The durable arithmetic record is
`artifacts/hybridz_largecls_gates/phase_projection_performance_reference_semantics_20260814.json`,
SHA-256 `96956ba0f9628f2aa4f110c9230c560fd822859482a7f886051ac4a282bd42ca`.

For handoff, the objective has also been expanded into a machine-readable
requirement/evidence matrix at
`artifacts/hybridz_largecls_gates/phase_projection_gpu_csr_completion_audit_20260814.json`,
SHA-256 `40a755f225cfe9f6b0e66d665fae225094225d18db46e7d6df23c524edcf7a37`.
It binds all production, test, verifier, harness, fixed-400, and supporting
artifact hashes and records the evidence status for formal gain, terminal
authority, prohibitions, fail-closed behavior, both response gates, memory,
structural deletion, and production integration.  Every objective item is
proved or bounded-tested.  The authorized independent read-only review is now
complete and that row is `PROVED_WITH_BOUNDED_INDEPENDENT_AUDIT`.  It matched
nine frozen hashes, passed `25/25` focused/dependency tests and `5/5` hostile
fail-closed injections, found zero bit mismatch across 128 Conv plus four Dense
CSR geometries, enclosed `8/8` exact DAG outputs and `48/48` grouped/dilated Conv
outputs, and revalidated all 400 receipts plus 403 input hashes.  Its performance
check did not rerun the narrowly passing experiment; it exactly recomputed the
locked five-point summaries as `2.0287406x` single-request, `2.0390894x`
four-thread median, and `1.800903776x` paired-bootstrap lower bound.  The verdict
is `NO-BLOCKER` for the frozen default-off path only; it does not authorize a
new multi-flip path, a changed performance reference, or default enabling.

### 2026-08-14 measured public and search ceilings

The official VNN-COMP 2025 manifests and raw result CSVs were matched to the
ACT fixed-400 manifest by row index, ONNX basename, and VNNLIB basename.  All
`400/400` mappings agreed after removing each result CSV's trailing `test_nano`
row.  The union of officially reported `sat` instances is `67/400`: 14
CIFAR100-medium, 15 CIFAR100-large, and 38 TinyImageNet.  ACT's 59 FALSIFIED
instances are all in that set, so the current verifier covers `59/67 = 88.1%`
of the publicly demonstrated counterexamples.  Only eight of ACT's 341 UNKNOWN
instances have a public `sat` result:

- CIFAR100-large iid110, iid114, iid160, iid161, and iid166;
- TinyImageNet-medium iid93, iid143, and iid153.

These eight are external reachability evidence, not ACT verdict authority; no
public counterexample was replayed during this census.  The remaining 333 cases
without a public `sat` result are not thereby SAFE, and 67 is not a mathematical
ceiling.  An especially useful cross-check is CIFAR100-large iid153: its ACT
margin is close to zero (`-0.00546486`), but the official result is `unsat`.
Near-zero margin alone therefore cannot establish that a counterexample exists.
Of the public-positive gap, the closest current margins belong to Tiny iid143
(`-0.00107108`), large iid110 (`-0.00453403`), and large iid160
(`-0.00833311`).

The official 2025 report also rules out the proposed “public PGD score was
zero” interpretation.  It says CIFAR100 samples verifiable by vanilla CROWN
were filtered, while roughly 18 percent of instances with known adversarial
examples were deliberately retained; TinyImageNet used a similar filtering
method.  Older 2022 material describes a PGD screen for a different, much
smaller manifest and cannot be transferred to this fixed-400 set.  The current
public record therefore supports “ACT independently certifies 88.1% of the
published counterexample set,” not “ACT uniquely found 59 cases missed by all
public attacks.”

The durable public census is
`artifacts/hybridz_largecls_gates/phase_projection_vnncomp_public_ceiling_20260814.json`,
SHA-256 `91b875c57bcebd8b784b1e1141ab727526aa44aeb3928ce7ddcdb8b887117294`.
It pins the 2025 benchmark commit
`8b7b811b78ce6a329dc96f04ae6652da3c245948`, result commit
`ea89fbc2518b6729f17c96eeec22c56c88e496a9`, four source CSV hashes, the
eight-case mapping, and the authority caveats.  This census was read-only and
performed no model execution, ONNX point execution, sampling, PGD, BaB/split,
backward bound, or dual tightening.

The performance-policy choice remains real but its numerical premise is now
precise.  Under a ratchet against the current 59-path, the next single-request
budget is `0.959264` seconds; the measured iid2 terminal stage leaves
`0.529397` seconds, so a future capable representation would have to delete
about `0.479632` seconds of current nonterminal work rather than merely append
a cell update.  Under a fixed old-43 reference there is substantial isolated
single-request headroom, but only `0.118624` seconds at the four-thread median
and `0.001755` seconds in the tightest observed bootstrap pair.  Thus neither
policy makes a universal `+0.15`-second update automatically compliant.  If
future work is authorized, the reference policy must be fixed first, the
changed path must be measured on a preregistered negative control as well as
the positive performance control, and the existing no-menu and terminal-proof
rules remain unchanged.

### 2026-08-14 post-59 ability/ratchet exploration

The active objective was reset against the formal `59/400` path, without
lifting any restriction.  A new path still had to preserve all 59 positives,
add at least one terminal-verified case, and pass the current-path ratchet:
`1.5x` single-request median, `2.0x` four-thread median, and `1.8x` paired
bootstrap lower bound.  Sampling input centers, boundaries, or random points;
ONNX point execution for search; PGD; BaB/splitting/enumeration; backward
bounds; dual tightening; per-instance runtime rules; fallback menus; retries;
and parameter scans all remained forbidden.  LP values, phases, and marginals
remained candidate-only.  Formal success still required raw-BOX membership,
an independent verifier-owned zero-width outward forward, and a strictly
positive stored-binary64 `Fraction` property lower bound.

A single deterministic multi-flip rule produced a genuine disconnected
ability result.  After a negative base LP, it selected every screened phase row
that was primal-tight under the frozen tolerance and had a strictly negative
upper-row marginal, flipped the complete set once, and solved once more.  It
used no top-k rule, magnitude threshold, subset search, or retry.  On
TinyImageNet iid143 it flipped 57 phases, changed the candidate margin from
`-0.0010710773153` to `+0.0002170254612`, and the unchanged terminal proved a
`Fraction` lower margin of `+0.0002169926152`.  The updated LP had 22
candidate-phase sign discrepancies (maximum `2.36e-8`), so the LP/cell is not
described as exact or authoritative; the terminal result is independent of
those labels.  Across nine bounded ability cases this was the only new
terminal success.  It is a real capability gain but not a formal score gain,
because it was never integrated or admitted through the response gates.

The simpler zero-correction analytic-corner replacement was also tested as a
single path, not as a fallback.  It independently proved CIFAR100-large iid166
with exact lower margin `+0.4187211701`, an instance whose existing selected
cell LP was infeasible.  It nevertheless failed retention: of the first 41
old positives checked, it retained 34 and lost seven (medium iid11/31/54/83
and Tiny iid7/69/150).  Testing stopped immediately; the remaining 18 retained
cases, fixed-14, and fixed-400 were not run.  Taking the union of the old path
and this replacement would reach 60, but that would be the prohibited
two-path/menu construction.  This branch is therefore retired.

A request-local immutable device-affine program did pass its component gate.
It presealed weights, absolute weights, and index schedules separately inside
each request, with no cross-request cache.  Generator arrays were bitwise equal
to the frozen implementation, terminal bounds were bitwise equal, and terminal
replay remained causally independent of candidate data.  Single-request
first-stream-plus-terminal median fell from `0.702328` to `0.278302` seconds;
four-thread group median fell from `1.664214` to `1.054942` seconds, saving
`0.611448` seconds.  This reopened a structural experiment but was not itself
a full-path promotion result.

The preregistered target-transaction sentinel then tested the only remaining
integration route: preserve the current 59-path semantics, form target rows in
chunks, screen them, load a request-local HiGHS model directly, and include
solve/readback.  It passed the isolated single-request budget
(`0.301222 <= 0.642884` seconds) but decisively failed the four-thread budget
(`1.183850 > 0.367046` seconds).  Concurrent chunk composition, screening,
safe filtering, and row loading alone had median `0.849837` seconds per
request.  The paired four-thread median saving was only `0.032938` seconds,
versus the required `0.922732`.  Per the frozen stop rule, no chunk tuning,
favorable rerun, auxiliary-column update, production patch, fixed suite, or
formal performance gate followed.

An independent read-only audit also downgraded that transaction artifact's
soundness claims.  The scratch tiny-coefficient compensation used
`b + sum(max(a*l,a*u))`; the general sound upper-row relaxation is
`b - sum(min(a*l,a*u))`, rounded outward, and the scratch did not assert the
exact symmetric-bound precondition under which the two coincide.  Moreover,
the current solve uses SciPy 1.17.1 with bundled HiGHS 1.12.0, while the direct
loader used highspy/HiGHS 1.15.0 with different options; matrix/objective data,
factors, and terminal margins were close but not bitwise equal.  The proposed
owner also lacked the final required BaseException-safe cleanup/deadline and
per-chunk postcondition audit.  The measured performance failure remains a
valid stop signal, but the record now explicitly says `mechanical=false`,
`loader_soundness=BLOCKED`, and `audit_complete=false`; it must not be cited as
a sound transaction-equivalence proof.

The formal result therefore remains `59 FALSIFIED / 341 UNKNOWN / 0 ERROR`.
The production phase and live-row files were not edited and remain at SHA-256
`4b66470df55edebb595e0e06c6b8a2de5c65496b8671c4d2f2552003d01ea306`
and `d53c2335c43905097e78bef8311175d7151d7e98293a6152fce62dba00d37511`.
There is no pending benchmark or GPU process.  The durable records are:

- goal amendment: `goal_amendment_post59_ability_ratchet_20260814.json`, SHA
  `eb34cf05f25c7782d4c858685351ac53d64998e4af883379b9346fc64734e4df`;
- multi-flip capability: `phase_projection_one_multi_flip_capability_20260814.json`,
  SHA `1a7dbb47467eea657dfecf552b51d24cff20adcdbad238f7e53027c86665a303`;
- device-program component: `device_affine_program_iid2_sentinel_20260814.json`,
  SHA `7c0dc7de4d08dd396c2aecf53d1499bdc6b9860d7797ce8c041a887997dceca7`;
- zero-corner retention stop: `phase_projection_zero_corner_ability_stoploss_20260814.json`,
  SHA `54e062c41aa2d20db84ffa1a951285581b7b277deb406f6e554620824abc72a4`;
- corrected transaction stop: `phase_projection_chunked_target_transaction_iid2_20260814.json`,
  SHA `8caedde58b3f97841f11d3b10684471437b47258d42c3d8bf5bd3c3e93fa41b3`;
- consolidated decision: `post59_ability_ratchet_exploration_stoploss_20260814.json`,
  SHA `2d80c448e0726d5f922a3bfc05e0bd14de970b01af8a129ee6e02e09aafb0e46`.

An independent final record audit returned `NO-BLOCKER_FOR_RECORD` for those
hashes.  That verdict authorizes the accuracy of this STOP-LOSS archive only;
it does not authorize the transaction, loader, or a production change.  The
consolidated artifact deliberately retains `audit_complete=false` because the
underlying transaction remains soundness-blocked, while the corrected
transaction artifact separately records the completed red-team findings.

The restart condition is deliberately narrow: one deterministic
representation must simultaneously retain all 59, add at least one terminal
success, implement a sound request-local loader, and pass the unchanged
single/four-thread/bootstrap/RSS gates.  The disconnected Tiny143 result is a
useful future positive control, not permission to append a second backend.

## F-prime production integration checkpoint (2026-08-14)

The response rule was subsequently clarified to the F-prime ability policy:
an ability promotion must retain the current 59, remain non-inferior on the
fixed single/four-concurrent controls, preserve every prohibition and terminal
authority boundary, and add at least one formal terminal success.  The older
1.5x/2.0x/1.8x ratchet remains applicable to pure speed promotions, not to
this ability promotion.

A single production path is now wired and frozen.  It owns one request-local
HiGHS instance, solves the base cell once, and performs at most one
simultaneous repair and warm re-solve.  An optimal negative base cell selects
all primal-tight upper rows with strictly negative row dual; an infeasible
base cell selects every exact-nonzero row in the validated upper-row dual ray.
Both branches use the same incremental low-rank representation.  The dual
information only chooses Boolean phase flips and has no proof, tightening, or
verdict authority.  The sole success authority remains raw BOX membership,
an independent zero-width outward forward, and a strictly positive
stored-binary64 Fraction property lower bound.

The production five-case sentinel completed once with 5/5 expected outcomes,
zero errors and zero retries.  The retained iid2 remained FALSIFIED.  The
large153 negative control remained UNKNOWN, and large166 read one infeasible
ray then failed closed.  TinyImageNet iid143 became a strict terminal success
after one 57-row repair, with Fraction lower margin
`0.0002169995966969651`.  TinyImageNet iid153 became a strict terminal success
after one five-row infeasible-ray repair, with Fraction lower margin
`0.029165812151973114`.  Independent source, ledger, receipt, owner, terminal,
and artifact audits all returned `NO-BLOCKER`.

These are production-path capability results, but the formal fixed-400 score
is deliberately unchanged at `59 FALSIFIED / 341 UNKNOWN / 0 ERROR` because
the retention and non-inferiority gates have not run.  The current formal
family split is CIFAR100 `24/200` (medium 14, large 10) and TinyImageNet
`35/200`.  If every remaining gate passes, the two new TinyImageNet cases
would make the conditional split CIFAR100 `24/200`, TinyImageNet `37/200`,
and total `61/400`.  No new CIFAR100 case was proved in this stage.

The next retention harness is statically frozen and independently audited,
but explicitly `NOT_RUN` with zero attempts and no runtime output.  It first
runs fixed14, then proves all 59 retained cases, reusing the four overlapping
unique attempts rather than rerunning them.  After explicit GPU authorization,
resume with:

```text
PYTHONUNBUFFERED=1 python scratch_phase_projection_fprime_production_retention_gate.py
```

Do not use the older fixed-400 runner for this checkpoint because its resume
identity does not bind the new source bundle.  The authoritative machine
handoff is
`phase_projection_fprime_production_stage_handoff_20260814.json`; the narrower
retention handoff is
`phase_projection_fprime_production_retention_gate_handoff_20260814.json`.
At pause time there was no GPU compute process or background gate process.

## F-prime production retention and performance stop-loss closure (2026-08-14)

This section supersedes only the `NOT_RUN` run-state statements in the
immediately preceding F-prime checkpoint.  No earlier paragraph or artifact
was rewritten.  The integrated production source remained frozen throughout
these gates: source-bundle SHA-256
`2c990582fe20e7e880a530557d5ca9c825c3d44f3f20758cf64df8d1845f6855`,
phase-path SHA-256
`13625f452c36a1b7844e4385b884471c8a0c82abf015bf2af417257e2c96c23a`,
device-program SHA-256
`7f0cce0e461f63ff6599ddd82ad5e61ef7c921eb489ef7bbbf4d60cda9048962`,
request-local HiGHS-owner SHA-256
`2f5678a5b3d2b098637b27558a8bdbffcc5160ca89cb8b4947f557320d03f5b7`,
and incremental-repair SHA-256
`acc1a98fa47d36c3b0bea7d10bf93af33d41ee9de108b0151ca01cd4822f997e`.
The production shape was still one request-local owner, one base solve, at
most one simultaneous repair, and at most one warm re-solve; there was no
second runtime path.

The production five-case ability sentinel completed with three FALSIFIED, two
UNKNOWN, zero ERROR, and zero retry.  It retained CIFAR100-medium iid2;
CIFAR100-large iid153 and iid166 remained UNKNOWN.  TinyImageNet iid143 became
a terminal FALSIFIED result after the one 57-row optimal-negative repair, with
stored-binary64 Fraction lower margin `+0.0002169995966969651`.
TinyImageNet iid153 became terminal FALSIFIED after the one five-row validated
infeasible-ray repair, with Fraction lower margin
`+0.029165812151973114`.  These two TinyImageNet results are production-path
ability evidence, but remained conditional rather than formal fixed-400
results.

The retention gate then completed exactly once with status
`COMPLETE_RETAINED59`.  It executed 69 unique cases: fixed14 first, followed
by the non-overlapping remainder of the retained-59 set, reusing the four
fixed14/retained59 overlaps rather than attempting them twice.  Fixed14
finished `5 FALSIFIED / 9 UNKNOWN / 0 ERROR`; all `59/59` old positives were
retained with zero regression and zero ERROR.  Across the 69 unique cases the
result was `60 FALSIFIED / 9 UNKNOWN / 0 ERROR`; TinyImageNet iid143 was the
one new FALSIFIED case outside the retained 59.  All 69 cases had at most one
attempt, and the durable event ledger has 139 lines: one run-created record,
69 attempt-started records, and 69 result records.  The independent read-only
retention audit returned `NO-BLOCKER`: one logical owner at most per request,
all owners closed and natively cleared, no `clearModel`, terminal authority
for every positive, and every declared prohibition flag false.

The frozen paired non-inferiority gate had ten preregistered case/mode jobs,
two warmup pairs and five alternating measured pairs per job, with no retry or
parameter change.  The first job, `cifar100_medium_iid2__single`, passed.  Its
five old59/new-production speedups were `1.1874614582503094`,
`1.1782268668461688`, `1.1661252248015697`, `1.1683247845050249`, and
`1.1596132014498013`; median speedup was `1.1683247845050249` and the exact
3125-resample paired-bootstrap 95-percent lower bound was
`1.1596132014498013`, above the frozen `1.0` and `0.95` minima.

The next job, `cifar100_medium_iid2__four_thread`, was the first blocker and
returned worker-emitted `ERROR(GateError)` with return code 2.  It did not time
out, it emitted exactly one isolated result, and there was no source/input
drift, parent postvalidation rewrite, stdout-isolation failure, return-code
status conflict, `ResultConflict`, or measured non-inferiority rejection.
The exact substage is
`UNRECOVERABLE_FROM_FROZEN_RECEIPT`: the frozen worker persisted only the
exception type, not its message, traceback, stage marker, partial warmup, or
stderr body.  Timing is compatible with an early warmup failure but is not
authority for that attribution.  The bounded candidate set includes
authorization or pre-timing setup, a warmup group, a later warmup/measured or
post-measurement check, and final cleanup or resource-receipt handling; the
receipt cannot rank them.  Therefore this record proves a fail-closed
performance-gate error, not a measured four-thread regression and not a
four-thread non-inferiority pass.

The parent stopped at that first blocker.  The following eight jobs are
explicitly `NOT_RUN`: CIFAR100-large iid153 single and four-thread;
CIFAR100-large iid166 single and four-thread; TinyImageNet iid153 single and
four-thread; and TinyImageNet iid143 single and four-thread.  The paired gate
ended `FAILED_CLOSED_ERROR` after 2/10 attempted jobs.  Consequently fixed40
and fixed400 are both
`FORBIDDEN_FOR_THIS_FPRIME_PROMOTION_ATTEMPT_AFTER_PAIRED_GATE_FAILED_CLOSED_ERROR`;
neither was launched, and no timing claim was promoted.  This prohibition is
scoped to this F-prime promotion attempt and records a gate error, not a
measured performance rejection or a permanent benchmark-wide prohibition.
GPU compute was released after stop-loss.  The independent read-only audit of
the raw paired summary/events returned `NO-BLOCKER` for the record while
preserving `FAILED_CLOSED_ERROR` as the performance outcome.  Its final narrow
audit also returned `NO-BLOCKER` for the revised stop-loss completion: the two
semantic corrections above are present, and the remaining hashes, recursive
references, numbers, and formal/retention/downstream boundaries did not drift.
That audit was an in-session read-only conclusion and created no separate
filesystem artifact.

The formal checkpoint therefore remains exactly
`59 FALSIFIED / 341 UNKNOWN / 0 ERROR`, not 60 or 61.  Its family split remains
CIFAR100 `24/200` (medium 14, large 10) and TinyImageNet `35/200`.  Retention
evidence plus TinyImageNet iid143 would conditionally make `60/400` and
TinyImageNet `36/200`; adding the five-case-only TinyImageNet iid153 evidence
would conditionally make `61/400` and TinyImageNet `37/200`.  Neither
conditional number was promoted because the paired performance gate failed.
CIFAR100 has no new conditional FALSIFIED case in this stage: large iid153 and
iid166 both remained UNKNOWN.

Every original restriction remained in force: no input sampling or input
point ONNX execution; no PGD; no BaB, input/activation split, or enumeration;
no backward bounds; no dual tightening; no per-instance production rule; no
runtime fallback, menu, retry, or parameter scan; no second solver; no
cross-request cache; and no external SAT label supplied to production.  Dual
rows, marginals, and infeasibility rays selected Boolean phase flips only and
had no proof, tightening, or verdict authority.  A positive verdict still
required the same stored binary64 input to pass raw BOX membership, an
independent verifier-owned zero-width outward forward, and a strictly positive
exact Fraction property lower bound.

The durable closure records are:

- five-case summary
  `phase_projection_fprime_production_five_case_sentinel_20260814.json`,
  SHA-256
  `fc1471a8c41dc548a8386a34a1996fe1fdf2d91f420accdab53803153ac8c681`;
- retention summary/events
  `phase_projection_fprime_production_retention_gate_20260814.json` and
  `.events.jsonl`, SHA-256
  `870175a30f09fd359b743f63ca480fddc707b9ef3d6959b1b434a8765faa3567`
  and
  `bfc249ed95d27d9e73c0554477ad1644d879abcb3fc640f900f45c9ce1c33b43`;
- retention completion
  `phase_projection_fprime_production_retention_completion_20260814.json`,
  SHA-256
  `aaf288bf66659105f122f26fa72fd1039111283fae50b754956e059facc15c56`;
- paired summary/events
  `phase_projection_fprime_paired_noninferiority_20260814.json` and
  `.events.jsonl`, SHA-256
  `08cc9bbf87a96b5a3be1d489105bc0b904daa6ad26dce7578dd66c4e228def52`
  and
  `b33930d37ac5ea6d95c7dd579438fd62e9945ade1ade1b985187dca954f830ac`;
- paired stop-loss completion
  `phase_projection_fprime_paired_noninferiority_stoploss_completion_20260814.json`,
  SHA-256
  `075348e25dd910a8341ae762243d0fa5f8a40d99f67eaeafd68006aa9393ec59`.
