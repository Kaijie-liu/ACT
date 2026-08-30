# HybridZ CIFAR100 / TinyImageNet exploration ledger

Date: 2026-07-27  
Branch: `hybridz-fse-20260727`  
Scope: pure HybridZ verification for the VNN-COMP 2025 CIFAR100 and
TinyImageNet families.

## Objective and non-negotiable rules

The long-term objective is zero errors/conflicts and more conclusive results
than the VNN-COMP 2025 best under the official 100-second instance budget,
ultimately approaching 400/400.  This is not permission to infer verdicts
from reference labels or numerical solver statuses.

The exploration ladder is fixed:

1. deterministic toys and independent exact/Fraction oracles;
2. one fixed real sentinel (Gate1);
3. Gate6;
4. incremental Gate14;
5. incremental Gate40;
6. all 400 only after Gate40 passes.

An error, raw-replay conflict, malformed receipt, non-finite value, loss of
coverage, or unsupported operator is a P0 stop.  The first inconclusive result
per family stops that family.  Gate14 and Gate40 consume only their new
8/26 instances, never rerun the earlier stages.

## Verdict authority

| Evidence | May certify SAFE | May expose FALSIFIED |
|---|---:|---:|
| outward cube upper bound | yes | no |
| independently recomputed long-double Lagrangian upper bound over original CSR | yes | no |
| exact stored-float point/Fraction proof | yes | no |
| HiGHS/SCIP optimal, infeasible, objective-bound, or warning status | no | no |
| solver factor incumbent | no | no |
| decoded input accepted by strict raw ONNX + raw VNNLIB replay | no | yes |
| historical/reference S/U label | no | no |

HiGHS is a candidate generator.  Coefficients at or below `1e-12` are
explicitly removed only from its solver copy and audited.  The original
canonical CSR remains the certificate and witness-validation authority.

## Historical lessons retained

- The old CIFAR100 `192/200` and TinyImageNet `175/175` tables are not proof
  evidence; the strict raw parser/replay path supersedes them.
- Uniform exact-ReLU budgets, larger caps, and all-layer LP tightening did not
  produce stable gains.  One historical tightening run consumed about 748
  seconds without useful bound changes.
- Increasing exact `K` can make the formulation much harder while targeting
  neurons irrelevant to the current rival.
- A final-layer hull cannot repair relaxation gaps created by earlier
  generalized intersections and residual joins.
- Full/global RLT, full activation-tree enumeration, dense convolution
  expansion, and blind all-400 reruns are prohibited exploration paths.
- The useful structural FCHZ cases were sparse; structure alone did not solve
  the large-classification families.

## Phase-0 foundation implemented

- strict raw VNNLIB Boolean evaluation and deterministic CPU ORT replay;
- raw model/spec SHA-256 receipts;
- Kahn DAG scheduling and stable global factor IDs;
- sibling-branch regression
  `ReLU(x) + ReLU(-x) = |x|` with exact range `[0,1]`;
- strict Conv2d contract and sparse/Fraction/PyTorch audits;
- row-wise affine/ADD roundoff envelopes and final output error generators;
- Fraction-audited ReLU triangle intercepts;
- exact-or-outward binary `xi_b -> z` mapping;
- constructive nonempty theorem token for the operator graph;
- single shared wall deadline and outer process-group hard kill;
- fixed 6→14→40 manifests, fingerprints, thread caps, GPU lock, and P0
  classifications.

Current deterministic checks:

- Phase-0: 33 tests, including certified preactivation and time-limited-dual
  regressions;
- strict replay: four audit groups;
- gate runner: 15 tests, including a one-iid diagnostic selector and an
  isolated worker import smoke test;
- property-aware residualization: 6 tests, including 64 seeded rational
  three-ReLU DAGs under every budget;
- original-frame GPU dual mapping: 5 tests;
- motif-local sharpness: 5 tests;
- fused ADD frames: 5 tests, including 16 fixed rational DAGs with exact
  phase enumeration and a `1e16+1-1e16` cancellation oracle;
- independent operator postfix fuzz: about 470k local cases plus 2,000 full
  residual DAGs;
- Conv2d: 104 synthetic cases / 22,228 nonzeros plus all 58 real convolutions.

## Gate1 experiment ledger

Fixed instance:

`cifar100_2024`, medium, iid 2,
`CIFAR100_resnet_medium_prop_idx_6232_sidx_3020_eps_0.0039.vnnlib`.

### Build-only receipt

- operator build about 22 seconds;
- `n_cont=52,657`, `n_bin=0`, `n_ub=98,970`;
- constraint `nnz=10,498,220`;
- final output roundoff envelope max about `1.06e-11`.

### First verdict attempt

Result: `UNKNOWN`, 94.93 seconds.

Every rival failed while loading the same CSR because HiGHS returned
`kWarning`.  Root cause:

- default `small_matrix_value=1e-9`;
- 669,735 coefficients at/below `1e-9`;
- 118,742 coefficients at/below `1e-12`;
- matrix was canonical, sorted, finite, and had no large-coefficient problem.

The old control flow then rebuilt the same model for each of 99 rivals and
printed one traceback per failure.  No conclusion about HybridZ tightness can
be drawn from that attempt.

### Second verdict attempt

After explicit tiny-coefficient accounting, one persistent model, unified
HiGHS threads, and a shared 90-second HybridZ deadline:

- result `UNKNOWN`, total 84.13 seconds;
- build 22.31 seconds;
- one persistent model loaded 10,379,478 coefficients;
- 118,742 candidate-only coefficients dropped, total absolute mass
  `3.41e-8`, maximum below `1e-12`;
- cube pruned `0/99`, cube upper range roughly `[89.26,120.13]`;
- the first full LP alone consumed 60.05 seconds;
- only `1/99` rival started/completed; zero certificates and zero witnesses.

This isolates the current bottleneck: one global 10.5M-nnz LP is already too
expensive.  More budget and per-rival model rebuilding are stop-loss failures.

The next scheduler gives every rival a fair time slice and independently
checks any time-limited dual iterate.  A time-limit/warning status itself
still has no authority.

### Third verdict attempt: fair slicing

The first launch exposed an infrastructure error before model parsing:
executing the nested worker by filesystem path made `act.*` unavailable on
`sys.path`.  It was recorded as `FAIL_ERROR` in an append-only receipt after
0.92 seconds.  The worker now launches with
`python -m act.pipeline.verification.hybridz_largecls_gate`; a real
subprocess import audit permanently covers this contract.

The corrected single-iid diagnostic then produced:

- result `UNKNOWN`, parent wall time 59.17 seconds;
- operator build 24.19 seconds;
- all 99 rivals completed and exact coverage was preserved;
- persistent LP elapsed 30.98 seconds with one model build;
- all 99 runs ended at a time limit; 94 returned a zero dual;
- 5 nonzero duals were independently checked, with best observed certified
  candidate upper still about `84.95`;
- zero certified rows and zero validated witnesses.

This is a successful scheduling result but a tightness stop-loss failure.
The HiGHS-only path will not receive more time.  The next real experiment may
use only a small, explicitly capped batched CUDA candidate budget, with every
candidate sparsified and checked against the original CSR.

### Property-tail upper rows and negative slopes

The safe-only final-tail fold keeps the prefix HZ and replaces only the
exclusive final `RELU -> DENSE -> ASSERT` graph by Fraction-audited affine
upper rows.  On the fixed iid2 sentinel it pruned 100 final-ReLU variables,
200 rows, and 409,800 constraint nonzeros.  The build-only cube range improved
from approximately `[89.26,120.13]` to `[63.20,91.19]`, but no rival closed.

For negative final-property coefficients, projected CUDA candidates propose
lower ReLU slopes while the exact Fraction endpoint oracle reconstructs every
retained plane.  The original alpha-zero row remains in every property group.
The fixed-step scan was stopped at the first plateau:

- 8 steps: cube-improvement sum `255.24`, maximum `4.83`;
- 16 steps: sum `300.16`;
- 32 steps: sum `310.32`, maximum `5.14`;
- 32-step build about `24.51` seconds; all 99 candidate rows retained.

An eight-second grouped LP attempt completed 19 rows, 18 with zero dual, and
certified zero groups.  This closes additional LP time for the two individual
planes; the next experiment must change the bound geometry, not the timeout.

Receipts:

- `gate1_property_tail_buildonly_20260727.jsonl`;
- `gate1_property_tail_alpha_s{8,16,32}_buildonly_20260727.jsonl`;
- `gate1_property_tail_alpha_s32_lp8_20260727.jsonl`.

### Final-ADD source-shadow experiment

The controlled candidate snapshots the final residual ADD before
materialization, propagates it through the exact exclusive
`FLATTEN -> DENSE` bridge, and appends one source-based property row per rival.
It deliberately retains the materialized ADD variables and both relation
bands, because the final-ReLU `l/u` certificate belongs to that prefix frame.

The initial real launch failed closed after 3.07 seconds: the guard allowed
only direct `ADD -> RELU`, while all three official models use
`ADD -> FLATTEN -> DENSE -> final RELU`.  This P0 receipt was preserved.  The
corrected bridge passed a nontrivial two-branch/Dense Fraction toy, constant
row relation regression, 240 seeded rational DAGs, and verifier proof-object
hash checks.

The corrected iid2 build-only receipt then gave the decisive stop loss:

- build `24.861` seconds, 198 exported rows;
- source ADD: 2,048 rows / 2,048 nonzeros / 2,048 active columns;
- shared source columns `0`, maximum column row-use `1`;
- effective cube-improved groups `0/99`;
- improvement sum and maximum exactly `0`;
- no source row certified at zero.

The extra rows therefore doubled output value nonzeros
`202,851 -> 405,702` without tightening a single rival.  This candidate stays
implemented and default-off as a sound regression mechanism, but is closed
for Gate6; it will not be combined with alpha, given LP time, or run on the
other five sentinels.

Receipts:

- P0 topology guard:
  `gate1_property_tail_addsource_buildonly_20260727.jsonl`;
- corrected stop loss:
  `gate1_property_tail_addsource_buildonly_v2_20260727.jsonl`.

### Exact dyadic envelope and live-affine/tail composition

For two sound upper planes of the same rival, every nonnegative convex
combination is another sound upper plane.  The candidate searches
`k / 2^b` mixtures only when the solver objective is exactly `C=I,t=0`.
Every stored binary64 coefficient is converted with `as_integer_ratio()` to a
common power-of-two integer scale; exact forward-difference bisection then
finds the discrete grid optimum.  The original two rows remain in their
property group.  The mixture has no proof authority: only the ordinary
outward cube or original-CSR Lagrangian checker may certify it.

The search is sparse and bounded by 250,000 union terms per pair, 1,000,000
total terms, and five seconds.  Any exceeded bound appends no rows.  Fraction
oracles cover grid quantization, flat optima, large cancellation, repeated
breakpoints, subnormals, and a 10,000,003-column/two-nonzero anti-densification
case.  A representative 99-pair, 2,048-union-nnz microbenchmark took about
0.42 seconds at 16 bits.

Before a real launch, an exact combination toy exercised the official
`ADD -> FLATTEN -> DENSE -> RELU -> DENSE -> ASSERT` topology with:

- `materialize_add=False`;
- one provably inactive and two correlated unstable final preactivations;
- negative-alpha property folding;
- the dyadic mixture enabled;
- exact property value `-3/4`.

Both the materialized and fused versions were independently verified SAFE.
The fused build retained the baseline row, passed all property proof-object
hash checks, and was strictly smaller.

The one allowed iid2 build-only composition then produced:

- worker/run wall time `31.48 / 33.40` seconds, no error or conflict;
- five live-affine fusions, `n_cont=25,733`, `n_ub=44,926`,
  `constraint_nnz=8,325,936`;
- build time `28.60` seconds;
- alpha s32 improved all 99 rivals, with improvement sum `369.58` and
  maximum `5.35`;
- the 99 original group-best cube uppers were
  `[47.3827,69.5901]`, with sum `5775.73`;
- the prior materialized alpha-s32 worst upper was `88.6687`, so the combined
  worst upper improved by about 21.5%;
- exact mixture search covered all 99 pairs / 405,702 sparse terms, but
  selected only one interior mixture;
- that mixture improved one group by only `0.000688685`, certified zero
  groups, and left all 99 groups unresolved;
- group input coverage remained complete; no LP or GPU-dual work ran.

The structural and 20% geometry gates pass for
`live-affine + property-tail alpha`, so it becomes the stronger default
experimental base.  The dyadic mixture itself decisively fails its
`20 groups / 20% aggregate` stop-loss and is closed: no LP, no Gate6, and no
larger grid.  The 8.326M-nnz frame also misses the predeclared 30% nnz
reduction required to reopen the global LP.

Receipt:

- `gate1_live_affine_tail_alpha_s32_mix16_buildonly_20260728.jsonl`.

### Property-bundled two-ReLU PairHull

PairHull V1 tested whether a small property-selected two-neuron projection
could recover final-ReLU correlation that singleton endpoint envelopes lose.
For a pair `(i,j)`, its projected preactivation set is outer-approximated by
the eight exact support halfspaces in directions
`±e_i, ±e_j, ±(e_i+e_j), ±(e_i-e_j)`.  Independent row errors contribute
`|d_i| err_i + |d_j| err_j` to every support.  The exact replacement term is
then the maximum of the joint ReLU objective over all four activation phases
and every halfspace/axis boundary intersection.  The implementation never
uses the convex hull of support points, which would be an unsound inner
approximation.

For a stored foundation row with exact singleton requirements `rho_i,rho_j`,
the replacement intercept is computed as

`Fraction(stored_K) - rho_i - rho_j + beta_pair`,

then rounded toward positive infinity.  This deliberately inherits any
outward slack already present in the stored foundation.  The full candidate
row is reconstructed through the ordinary outward affine operator; baseline
and negative-alpha rows stay in every group.

Before the real probe, the audit suite covered:

- point, line, cancellation, independent-error, subnormal, and `1e16`
  numerical cases;
- a red-team example where every original polygon vertex misses the maximum
  attained at a ReLU-axis intersection;
- 300 seeded rational cases against an independent phase/axis oracle;
- a 10,000,000-column/two-nonzero anti-densification case;
- deterministic global disjoint-pair selection, deadline/resource fallback,
  nested checksum tampering, and complete operator/verifier/Gate replay;
- a realistic `100 x 2048`, 99-rival, 8-pair probe in about `0.47` seconds.

The first iid2 launch completed the operator but was rejected by Gate after
`34.95` seconds.  This was not a tightness result: Gate's JSON sanitizer
replaced checksum-covered exact phase records below nesting depth eight with
`<depth-limit>`.  The depth guard is now 32, and regression tests require the
signed receipt to survive repeated sanitization, JSON round trips, worker
wrapping, JSONL, and summary persistence.  A separate late-timeout audit also
found and fixed stale applied-only pair/foundation fields after HZ rollback.

The one permitted corrected iid2 build-only rerun was conclusive for the
candidate mechanism:

- worker/run wall time `33.42 / 35.19` seconds, no error or conflict;
- PairHull selector plus all exact audits `0.423` seconds; complete operator
  PairHull stage `0.437` seconds;
- eight unique disjoint pairs, 99 frozen proposals, 99 exact Fraction
  evaluations, and zero rejected/incomplete audits;
- all 99 exact stored-intercept reductions were positive, but only
  `2.42e-13 .. 1.04e-11`; their sum was `2.60e-10` and the largest relative
  reduction was `2.38e-13`;
- the full outward affine guard retained `0/99` candidate rows;
- group-best cube range and sum remained exactly
  `[47.382748,69.590067]` and `5775.726535`;
- build `30.91` seconds, `n_cont=25,733`, `n_ub=44,926`, and
  `constraint_nnz=8,325,936`; zero groups certified.

PairHull therefore fails every predeclared promotion threshold
(`>=20` improved groups, `>=15%` aggregate reduction, and `>=10%` worst-row
reduction or a cube certificate).  V1 remains implemented and default-off as
a proof/soundness regression, but it is closed for Gate6 and will not be run
on another sentinel or granted LP time.  The diagnostic is stronger than
“the selected pairs were missed”: all 99 property rows chose and exactly
audited their best frozen proposal, yet the projected joint correction was
only roundoff scale.  The next candidate must attack cross-layer ReLU
relaxation, not the final two-neuron geometry.

Receipts:

- sanitizer failure, retained for diagnosis:
  `gate1_live_affine_tail_alpha_s32_pairhull8_buildonly_20260728.jsonl`;
- corrected stop loss:
  `gate1_live_affine_tail_alpha_s32_pairhull8_buildonly_rerun1_20260728.jsonl`.

### Proof-carrying layerwise query-dual diagnostic

The first bounded cross-layer diagnostic used the ordinary interval facts as
the only initial bounds.  It did **not** import `compute_forward_bounds`
results and did not give `DualSolver` proof authority.  Four ReLU
preactivation layers were queried in topological order with batched
`+/-` one-hot objectives, eight alpha steps, and frozen-bound backward
passes.  Each stage fed its candidate bounds to the next stage only for this
untrusted diagnostic:

| ReLU lid | queried unstable | unstable after | candidate seconds | unstable-width sum before -> after |
|---:|---:|---:|---:|---:|
| 10 | 603 | 196 | 0.911 | 382.337 -> 124.866 |
| 14 | 145 | 20 | 0.609 | 157.831 -> 40.379 |
| 22 | 386 | 94 | 2.219 | 337.548 -> 79.701 |
| 40 | 100 | 36 | 1.106 | 3432.429 -> 180.013 |

The final 99 property rows then took `0.924` seconds for eight alpha steps:

- `85/99` candidate uppers crossed zero;
- upper range `[-6.217664, 1.801216]`;
- sum of all uppers `-175.564335`;
- sum of only positive survivors `7.563752`.

The optimizer/transcript binding issue was then tested explicitly.  Every
stage requested the returned alpha state and recomputed the bound in a
separate frozen, non-optimizing pass.  The result remained `85/99`, with
maximum upper `1.801072`, positive sum `7.563806`, and at most about
`1.03e-3` loss from the optimizer's unpaired per-row best scalar.  Thus the
effect does not depend on trusting a mismatched optimizer receipt.

A single untrusted operator build-only injection measured whether the
verified bounds would improve the actual HybridZ formulation rather than
only a second scalar-bound path.  No solver verdict was requested.  Relative
to the PairHull baseline it changed:

- build time `30.91 -> 27.34` seconds;
- continuous factors `25,733 -> 19,076`;
- upper rows `44,926 -> 31,810`;
- constraint nnz `8,325,936 -> 5,927,806`;
- group-best sum `5775.726535 -> 829.165590`;
- worst group upper `69.590067 -> 14.686187`.

The operator cube alone still certified zero complete groups.  This fixes the
production integration boundary: independently verified intermediate bounds
must tighten the native operator ReLU graph, while independently verified
final property uppers are appended as constant alternatives in the same
99 groups.  The latter would cover the 85 negative rows; neither route may
consume the untrusted diagnostic values.

This exceeds the predeclared geometry and time gates by a wide margin, but
none of the 85 rows is a certificate yet.  A second diagnostic retained only
the newly proved phase signs and discarded the tighter numerical bounds.  It
fell back to `0/99`, maximum upper `50.704953`, and positive sum `4320.285`.
Therefore the gain is not merely phase stabilization: the full certified
`l/u` values and their topological feedback are essential.

The promoted mechanism is **Proof-Carrying Layerwise One-Hot Alpha
Feedback**:

1. CUDA produces alpha/query candidates only, with
   `proof_authority=false`, `refresh_forward=false`, and a frozen hash of all
   bounds consumed by the query.
2. Every returned alpha state is frozen and the candidate scalar is
   recomputed from that exact state.  The optimizer's historical per-row
   best scalar cannot be paired with a later whole-batch alpha snapshot.
3. A separate CPU reverse-topological implementation replays
   Dense/Conv/Add/Flatten/ReLU without calling the candidate's backward
   handlers.  ReLU endpoint envelopes are audited against exact stored-float
   Fractions; affine arithmetic carries explicit outward roundoff guards.
4. Only replayed lower/upper bounds may enter the next layer.  The transcript
   binds `target -> parent-bound hash -> alpha hash -> replay result`, so a
   later layer cannot consume an unverified predecessor candidate.
5. Baseline facts remain immutable.  Unsupported topology, non-finite
   arithmetic, hash mismatch, timeout, or non-improvement rolls the entire
   candidate stage back.

The first independent-replay red team found a proof-blocking root assumption
before production integration.  Ordinary interval facts are not themselves
an outward-rounded certificate.  On the point box
`x=(1,1,1)`, stored binary64 `W=(1e16,1,-1e16)`, and zero bias, the current
`affine_bounds` implementation returns `[0,0]`, while exact rational
evaluation of the stored floats is `1`.  Therefore `_before/after` may still
be used for candidate scheduling and diagnostics, but they cannot authorize a
query-dual receipt or a ReLU Big-M intersection.  Production replay now has an
additional mandatory root:

0. An independent CPU binary64 outward box certifier must cover every
   consumed INPUT_SPEC/Dense/Conv/Add/Flatten/ReLU edge, bind the network and
   all per-layer boxes by hash, and pass cancellation, subnormal, convolution,
   residual-DAG, tamper, and deadline toys.  Only its boxes, followed by
   independently replayed refinements, may form the parent-hash chain.

Merely recording `trusted_assumption=supplied_bounds_are_certified` is not a
certificate and is explicitly non-promotable.

The independent reverse replayer is now implemented, but remains
non-promotable until the root certifier and transactional chain are connected.
Its authority entry point rejects narrow `longdouble`, FTZ/DAZ or missing
gradual underflow (including a BLAS subnormal-dot probe), and non-nearest-even
rounding.  It uses CPU binary64 nominal arithmetic, Higham-plus-subnormal
coefficient guards absorbed on the certified predecessor box, direct audited
Conv2D adjoints, and exact stored-float Fraction audits for every required
ambiguous-ReLU upper line.  The live result array is checked against the
receipt's hexadecimal values and SHA before consumption; validating only the
receipt is insufficient.

Fifteen directed replay tests cover point boxes, one/two ReLUs, interior
starts, batched-versus-scalar execution, residual DAG joins, dense and sparse
Conv2D, `1e16` cancellation, subnormal products, all four input hashes,
live-result/receipt tampering, deadlines, unsupported operators, and numeric
platform failure.  A separate seeded campaign compared 700 randomized query
bounds against the Fraction replay oracle with zero numeric overestimates.
A representative sparse one-hot convolution block (`Q=256`, `C=128`,
`8x8`, `3x3`) took approximately `0.27` seconds.  These are replay-component
results only; no real Gate has been authorized.

The independent outward BOX anchor is also implemented.  It reconstructs the
single raw BOX lane without reading ordinary facts; hashes the complete
stored-binary64 network, raw INPUT_SPEC, implementation source, semantics,
and every live bound; exports ReLU keys as preactivations and all other keys
as layer outputs; and rejects non-binary/nonzero-bias ADD semantics.  Its CPU
binary64 Dense and grouped Conv reductions use explicit Higham-plus-eta
enclosures, per-channel-chunk guards, nextafter endpoints, and runtime probes
for round-to-nearest-even, scalar/NumPy/Torch gradual underflow, Torch
pointwise subnormal preservation, and wider `longdouble`.

Eight directed anchor tests cover the `1e16` cancellation counterexample,
subnormal/FTZ behavior, grouped+strided+padded+dilated Conv2D, residual
fanout, exact-zero/nonzero ADD bias, full coverage, raw-input/live-bound
tampering, malformed objects, hash pins, and deadlines.  Together with the
candidate and replay suites, the fixed regression count is `30/30`.

Exactly one bounded real-width performance trace was then permitted on the
same CIFAR100-medium iid2 sentinel, with no solver and no Gate:

- parse `0.9577` s, synthesis `0.00094` s, Torch-to-ACT `0.10093` s;
- outward anchor `0.22269` s (`0.22114` s in its own receipt);
- total `1.2921` s;
- 43 layers and 41 exported boxes, including 19 Conv, 8 ADD, 10 ReLU, and
  2 Dense layers; maximum width 14,400 at layer 2;
- anchor bounds SHA-256
  `5b47b383cf4efd91b1dde253da3041449aeb2c8d19a16c1246da468429c717ff`.

This passes the root-cost stop loss by a wide margin.  It does not authorize
the iid2 query result or any benchmark verdict; the transactional
candidate-to-replay chain and operator consumption checks remain mandatory.

The first production gate remains the same CIFAR100-medium iid2 build-only
instance.  Candidate generation plus independent replay must finish within
12 seconds, accept all used bounds with zero checksum/coverage error, retain
at least 20 strictly improved final rows, reduce aggregate group upper by at
least 15%, and reduce the worst upper by at least 10% or certify a row.  No
Gate6 is authorized until controlled Fraction, residual-DAG, convolution,
`1e16`, subnormal, tamper, and deadline tests pass and the independent
replayer reproduces the real improvement.

#### Transactional production probe and step-count stop loss

The complete production plumbing subsequently passed `100/100` directed
tests: 51 root/candidate/replay/pipeline/operator/verifier tests and 49 Gate
tests.  This includes a second live validation plus binary64 bit comparison
after the Operator-HZ private snapshot, negative-threshold and residual-DAG
oracles, same-object and reconstructed-object tampering, shared deadlines,
and worker-local CUDA peak-memory receipts.  A real CUDA residual toy
completed the authority transaction in `0.703` seconds with no device
fallback; its operator revalidation/snapshot cost `0.0105` seconds and CUDA
peaks were `16.3 MiB` allocated / `22 MiB` reserved.

Exactly one eight-step production iid2 build-only probe was then launched
with the promoted fused-ADD + property-tail alpha-s32 base and with LP,
GPU-dual, PairHull, and dyadic mixture disabled.  It failed closed:

- run/worker wall time `46.48 / 44.53` seconds;
- analysis/solver/replay time `43.47` seconds versus the approximately
  31-second unchanged operator base;
- no outer timeout and no OOM;
- worker CUDA peaks `2,052,084,224` bytes allocated and `2,225,078,272`
  bytes reserved out of `101,947,998,208` bytes;
- all run-end source/config/artifact/environment integrity checks passed;
- the enabled feature did not export an applied authoritative transaction,
  so Gate emitted `FAIL_ERROR` and stopped the family.

The approximately 12-second delta equals the predeclared transaction budget:
the eight-step candidate therefore misses the time stop loss and is frozen.
It must not be rerun unchanged.  Receipt:
`gate1_query_dual_txn_buildonly_iid2_20260727.jsonl`.

One cheaper, predeclared recovery probe is allowed before closing the family:

1. run only the independent query transaction on iid2 with four alpha steps,
   the same targets `[10,14,22,40]`, block 1024, and the same 12-second hard
   deadline; do not build Operator-HZ and do not request a verdict;
2. require a complete live authority receipt in at most `10.5` seconds, zero
   checksum/coverage/numeric-platform/device-fallback errors, at least 20
   strict bound improvements at the final target ReLU 40, and record all
   per-stage candidate/replay times plus CUDA peaks;
3. only if all transaction gates pass may one four-step iid2 operator
   build-only composition be run against the frozen `5775.726535` aggregate
   and `69.590067` worst-upper baseline, retaining the original `15%`
   aggregate and `10%` worst-row-or-certificate thresholds;
4. failure at either step closes the four-step candidate.  Gate6 remains
   unauthorized in every case until this two-part ladder succeeds.

The four-step query-only recovery probe also failed closed and therefore did
not enter Operator-HZ:

- transaction builder `13.8620` seconds, exceeding the 12-second hard limit
  and the 10.5-second promotion margin;
- failure stage `query_dual_transaction`, error
  `QueryDualPipelineTimeout: query-dual replay deadline expired`;
- total process work `15.0059` seconds under a separate 30-second hard outer
  timeout;
- CUDA peaks `2,011,665,920` bytes allocated and `2,195,718,144` bytes
  reserved, again ruling out memory pressure;
- input and proof-source hashes remained stable, and the atomic receipt is
  explicitly `proof_authority=false`.

Receipt:
`gate1_query_dual_only_iid2_s4_20260728.json`.

The unchanged four-step mechanism is closed.  The next candidate is a
distinct implementation/schema, **Alpha-Descriptor-Only Query V2**, motivated
by the measured time failure and a read-only call-graph audit:

1. CUDA optimization may output only the objective descriptor and frozen
   alpha tree.  Candidate-side optimized margins and the second
   `DualSolver(optimize=False)` GPU replay are diagnostics without proof
   authority and are removed.
2. The independent CPU reverse-topological replay remains mandatory and is
   the only source of every committed lower/upper/property value.  It still
   binds objective, alpha, parent boxes, network, roundoff guards, Fraction
   ReLU endpoints, coverage, and the transaction deadline.
3. Candidate V2 must never label a row improved from optimizer output.
   Improvement/intersection/status are decided only after the CPU replay.
   Candidate, pipeline, verifier, Gate, and experiment fingerprints must bind
   the new schema; all V1 promotion receipts are invalid for V2.
4. Before any real retry, controlled toys must show identical authoritative
   CPU replay bits for V1 and V2 given the same frozen alpha; candidate margin
   tampering must have no effect; empty/malformed alpha, objective reorder,
   timeout, DAG joins, convolution, cancellation, subnormal, and receipt
   tampering must all fail closed.
5. Only after those tests pass is one V2 query-only iid2 probe allowed.  It
   retains the 12-second hard limit, must have zero integrity errors and at
   least 20 strict improvements at ReLU 40.  Failure closes V2; success alone
   authorizes one operator build-only geometry probe, not Gate6.

#### Alpha-Descriptor-Only Query V2 controlled promotion evidence

The V2 implementation and its pre-real-instance audit completed on
2026-07-28:

- standalone Candidate V1 remains the default compatibility path, while the
  production transaction explicitly requests `descriptor_only=True`;
- Candidate V2 uses schema `act.query_dual_candidates.v2` and protocol
  `descriptor_only_v2`; pipeline, target-stage, and property-stage schemas are
  independently bumped to V2 and mutually bind candidate schema, protocol,
  status, ordered descriptor coverage, and receipt hashes;
- each candidate block performs exactly one optimized CUDA call, exports no
  optimizer margin, performs no `optimize=False` GPU replay, and reports no
  candidate-side improvement; only the independent CPU binary64 replay may
  intersect a box, set property uppers, or increment `strict_improvements`;
- target synchronization inherited from a non-ReLU predecessor is recorded
  separately and cannot inflate query improvement counts;
- fully rehashed attacks against candidate/stage/property/top schema,
  protocol, status, objective order, `M`, alpha, baseline placeholders,
  coverage, authority wording, transaction atomicity, property sign,
  replay chunk/workspace, and root Conv chunk all fail closed;
- the only receipt fields intentionally excluded from mathematical authority
  are explicitly enumerated as `non_authoritative_audit_fields`: candidate
  learning-rate/solver/timing diagnostics and pipeline callable/default-device
  telemetry.  Expanding either whitelist to objective or bound fields is
  rejected;
- malformed explicit live objects now become a proofless
  `error_fallback_baseline` (and trigger Operator-zero-call fail-fast when
  requested), rather than escaping as an uncaught exception;
- the fixed V2 policy is part of the experiment fingerprint and both
  `run_start` and `run_end`; enabled production runs reject V1 or mixed-version
  chains before Operator-HZ construction.

The independent full regression was `128/128`:

- 8 outward box-certifier tests;
- 15 independent CPU replay tests, including residual DAG, Conv2D,
  cancellation, `1e16`, subnormal, coverage, mutation, and deadlines;
- 9 Candidate tests, 13 transactional pipeline tests, and 19
  pipeline/Operator/verifier integration tests;
- 11 query-only probe tests and 53 Gate/fingerprint/tamper tests.

The final red-team sweep reported P0=0, P1=0, and P2=0 in this scope.  A real
CUDA residual-DAG toy then completed the V2 transaction in `0.7203` seconds
and Operator-HZ live revalidation/private snapshot in `0.0252` seconds.  It
had one target block, two property blocks, one strict target improvement,
`optimizer_margins_exported=false`, `gpu_frozen_alpha_replay=false`, and valid
pipeline/Operator authority.  CUDA peaks were `17,085,440` bytes allocated
and `23,068,672` bytes reserved.

These results authorize exactly one four-step query-only run on the fixed iid2
sentinel, still with a 12-second transaction deadline and a separate 30-second
outer kill:

`artifacts/hybridz_largecls_gates/gate1_query_dual_only_v2_iid2_s4_20260728.json`.

No Operator-HZ build, Gate6, alternate iid, or full benchmark is authorized
unless that receipt has stable input/source hashes, zero integrity error,
transaction time at most 12 seconds, and at least 20 CPU-replay strict
improvements at target ReLU 40.

The single authorized V2 iid2 query-only run then failed the time gate and
closed V2:

- atomic receipt status `error`, `proof_authority=false`, failure stage
  `query_dual_transaction`;
- `QueryDualPipelineTimeout` with
  `DEADLINE_EXPIRED: query-dual replay deadline expired`;
- transaction `13.8952` seconds, total process `15.0552` seconds, exceeding
  both the 12-second hard limit and 10.5-second promotion margin;
- setup `1.1205` seconds; source and input hashes remained stable;
- CUDA peaks `2,011,665,920` bytes allocated and `2,195,718,144` bytes
  reserved on the 101,947,998,208-byte device;
- `operator_hz_called=false`, `hz_solver_called=false`, and
  `produces_verdict=false`.

Receipt:
`artifacts/hybridz_largecls_gates/gate1_query_dual_only_v2_iid2_s4_20260728.json`
(file SHA-256
`cf2d0eb429ca6abdf583059bdced972ccfcbfa5283c720a9166fd5e5ca6d1fa0`,
internal receipt SHA-256
`b76aed00645b92415b857e2f19314df6022b598ebcef1926c2e72a4a9c2e6fc2`).

Both four-step wall times are right-censored by the same 12-second deadline
and include roughly 1.9 seconds of in-flight replay overrun, so their numerical
closeness cannot quantify the saving from V2.  The valid conclusion is only
that removing the candidate-side margin copy and second GPU replay was
insufficient, and that both attempts terminated inside independent CPU replay.
No V2 operator build or Gate run is permitted.  A later candidate must change
the independently verified replay cost structure itself, demonstrate
bit-identical authoritative bounds on controlled multi-block/DAG/Conv toys,
and receive a distinct schema and stop-loss declaration before another real
probe.

## Pre-registered V3: property-covered sealed sparse replay

The next and only authorized query-dual candidate is
**Property-Covered Sealed Sparse Replay V3**.  This declaration is frozen
before its implementation and before any further real-network probe.  V3 has
two inseparable changes:

1. the root outward certificate privately carries one owned, read-only
   binary64 frozen graph; root-certifier and replay manifests are derived from
   that same capture and are joined by an explicit crosswalk receipt;
2. each target stage optimizes and independently replays only a fixed,
   property-selected subset of unstable rows.  Every omitted row remains
   bit-identical to its immutable parent BOX.  The final property stage still
   covers every raw VNNLIB rival row.

For the fixed CIFAR100-medium topology with target ReLUs
`[10, 14, 22, 40]`, the only real configuration is the per-stage quota
`[16, 8, 24, 16]`: at most 64 selected target rows, 128 signed target
objectives, and all 99 property objectives.  This replaces V2's 2,468 target
objectives plus 99 property objectives with at most 227 total objectives.
There is no real K128 fallback and no quota sweep.  CIFAR100-large and
TinyImageNet must receive topology-role schedules established independently;
layer numbers may not be copied from this model.

Selection is diagnostic, not proof authority.  With certified preactivation
bounds `l < 0 < u`, property adjoint `nu`, and rival weight `w`, the base
benefit is

`abs(nu) * (-l*u/(u-l))`.

Each stage applies its quota independently using deterministic multi-rival
facility coverage, with stable candidate IDs and an explicitly bound
tie-break.  The adjoint, scores, ranks, and estimated costs all have
`proof_authority=false`.  Soundness comes solely from immutable parent boxes,
source-generated signed objectives, frozen alpha descriptors, independent
CPU binary64 replay, outward intersection, and raw-property sign checks.

Candidate V3 must construct only the selected objective rows at source:
all positive unit rows followed by the corresponding negative unit rows.  It
is forbidden to optimize every unstable row and slice the result afterward.
Optimizer margins are neither exported nor consumed, and no candidate-side
frozen-alpha GPU replay may alter a bound.  For each stage the authoritative
receipt binds:

- the complete ordered eligible unstable IDs and hash;
- the ordered selected IDs and hash;
- the ordered omitted IDs and hash;
- no duplicates, empty intersection, and exact selected-plus-omitted
  partition of eligible IDs;
- exactly one positive and one negative objective, alpha descriptor, CPU
  replay result, and before/after hexadecimal value for every selected ID;
- `selected_coverage_complete=true`,
  `eligible_coverage_complete=false` when rows are omitted, and
  `unselected_policy=bit_identical_immutable_outward_parent_box`;
- objective, alpha, root snapshot, parent snapshot, cone, replay chunk,
  workspace, committed-box, stage-chain, and final transaction hashes.

Unselected rows changing by even one bit, selected/omitted overlap or loss,
objective reorder, alpha substitution, stale parent state, cross-session
cache reuse, or an incomplete property row makes the whole transaction
proofless and rolls every target/property update back.

The sealed replay session freezes the network once from the exact graph used
by the root certificate, constructs each unique dependency cone once, and
freezes one bounds frame per sequential target/property stage.  Candidate
code receives only a private device clone.  The existing independent
`_replay_block` arithmetic is unchanged.  Session construction validates the
root, and atomic commit rebinds the live network to the sealed root manifest;
the Operator consumption-time TOCTOU checks remain.  Every build, snapshot,
candidate, replay, commit, and cache-hit path observes the same absolute
deadline.

### V3 controlled soundness and equivalence gates

All of the following must pass before a real ONNX/VNNLIB process may start:

- point-box and stable-affine/Jacobian checks with maximum error `1e-12`;
- legacy replay and sealed V3 replay use identical selected alpha and produce
  `numpy.array_equal` lower bounds, identical `float.hex()` values, and the
  same lower-bound SHA on Dense/ReLU, ADD residual DAG, grouped/strided/
  dilated Conv, shared-fanout, property, and multi-block cases;
- at least 1,000 deterministic Fraction/exact-phase/random-Conv/DAG
  objectives have zero under-approximation;
- the residual toy
  `h+=ReLU(x), h-=ReLU(-x), s=h++h--3/4, z=ReLU(s)` reaches
  `s_upper <= 0.2500001` and `z_upper <= 0.2500001`;
- a multi-rival distractor toy covers each rival's unique influential ReLU,
  with bound results invariant to rival and row permutations under the bound
  stable-ID convention;
- the nested toy quotas
  `[4,2,4,4] -> [8,4,12,8] -> [16,8,24,16]` are prefix-monotone, and K64
  recovers at least 80% of full-V2 tightness gain;
- duplicate, omission, objective reordering, alpha-slice substitution,
  parent mutation, unselected-row mutation, candidate-clone mutation, wrong
  cone/start/root/session, live-network mutation, `1e16` cancellation,
  subnormal, and deadline-at-every-phase tests all fail closed with no partial
  authority;
- a five-block spy records one network capture, five bounds snapshots, five
  numerical replays, and zero loop-internal full-root refreezes.

### V3 performance and real stop-loss gates

On one fixed warmed synthetic wide Conv-ResNet, V3 K64 transaction time must
be at most `0.45x` full V2, and a 32-MiB static-weight preparation toy must
save at least 60%.  The real-CUDA residual-DAG transaction must not regress
more than 10% from `0.7203334719` seconds.  Peak CUDA allocation must remain
at most 2.5 GiB, incremental CPU RSS at most 2 GiB, and replay workspace at
most 512 MiB.  CPU replay begins with one worker.  A fixed two-worker query
chunk is allowed only if CPU arithmetic exceeds 50% of the transaction and
is bit-identical to one worker; topologically dependent stages stay serial.

Only after every controlled gate passes is one real iid2 query-only V3 probe
authorized.  It keeps the 12-second hard transaction deadline, 10.5-second
promotion margin, and separate 30-second process kill.  It must have zero
integrity error, at least 20 selected-target strict CPU-replay improvements,
at least one strict improvement in every nonempty target stage, and at least
20 strictly improved property upper bounds.  Any failure closes V3; it does
not authorize K128, another iid, an Operator build, or Gate6.

If and only if that query probe passes, one Operator build-only comparison is
authorized against aggregate/worst property-upper baselines
`5775.726535 / 69.590067`.  Promotion requires at least 15% aggregate
improvement, at least 10% worst-row improvement or a certificate, and no more
than 50 seconds for V3 plus build.  Only then may the fixed configuration
enter Gate1 and the existing `6 -> 14 -> 40` progression.  Full 400 remains
last.

Persistent predicate-normal form stays frozen until V3 reduces the old
8.326-million-nonzero system by at least 30% (`<= 5.828` million nonzeros).
Residual normal form may consume only V3-certified bounds and must itself
remove at least 30% of rows/nonzeros with no group-upper regression.

### V3 implementation freeze and controlled-gate receipts

The sealed sparse implementation was frozen on 2026-07-28 only after two
additional independent closure reviews.  The final validator now recomputes
selector and candidate partitions/counts/hashes, binary64 bits including
signed zero, root-to-snapshot crosswalks, every unique cone, and every bounds
frame.  Fully rehashed root/cone/frame, candidate, stage, property, and top
receipt mutations fail closed.  Reverse target order, empty targets, and
all-zero quotas fail before the root certifier; mixed zero quotas and a
positive quota with no eligible unstable row remain valid and still cover
every property row.

The final controlled CPU receipt is
`artifacts/hybridz_largecls_gates/query_dual_v3_controlled_audit_20260728.json`:

- schema `act.query_dual_v3_controlled_audit.v2`, status `pass`, internal
  receipt SHA
  `9a7d2063ed34afef86a7c718b85e839e216aa414e4fa855418b754e1078b166e`,
  file SHA
  `ecedebd9206adbdcc28815d168d48173c79041c111cab43b881ce81fbd962568`;
- 109 in-receipt tests plus three atomic-failure-envelope tests passed, with
  1,000 deterministic exact objectives and zero lower-bound overestimates;
- the 33,562,240-byte, five-cone static toy saved `91.5635%`
  (`ratio=0.0843654`) against repeated preparation;
- the fixed wide Conv residual transaction had V2 median `7.113888s`, V3
  median `0.347416s`, and `V3/V2=0.0488364`, below the `0.45` gate;
- incremental CPU peak RSS was `140,652,544` bytes, below 2 GiB.

The final same-graph CUDA receipt is
`artifacts/hybridz_largecls_gates/query_dual_v3_cuda_toy_audit_20260728.json`:

- schema `act.query_dual_v3_cuda_toy_audit.v1`, status `pass`, internal
  receipt SHA
  `a1a88011db16f2160cbe9e6dcf02bcbbb50a9fcbbcc9529f5f4d21c20f3c78bd`,
  file SHA
  `d400893e95769495414da86ffa3aeb1041cab2236f1c714d02bf43cd4c9ced50`;
- it uses the historical ADD residual DAG, target/quota `(7)/(1)`,
  `steps=1`, `block=1`, and replay chunk 16, rather than a different serial
  graph;
- warmed median V3 build time was `0.054367s`, or `0.075475x` the frozen V2
  baseline `0.7203334719s`; build plus independent live validation was
  `0.061625s`;
- CUDA peaks were exactly `17,085,440` bytes allocated and `23,068,672`
  bytes reserved, below 2.5 GiB.  The real selector, CUDA-f64 DualSolver,
  sealed CPU-f64 replay, and public live validator were used without a mock.

Both receipts have `source_integrity_stable=true`; their internal canonical
hashes and every current source hash were independently recomputed after
publication.  Compilation and whitespace checks also passed.

The only authorized real launch is now the fixed CIFAR100-medium iid2
query-only probe below.  It cannot import or call Operator-HZ or produce a
verification verdict.  Its pre-run config SHA is
`223c03267debd8c1caa86310744f1873428cbffb8a2570104c7678c3113dd954`;
the input SHAs are ONNX
`aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4`
and VNNLIB
`33e795c8421b7b19125f32415adb9cee09b2f90cb83152c4cd3aa03810e91ec3`.
The branch/head are `hybridz-fse-20260727` /
`391785fe7fa7c02b927ac2a9240a09d293f1363c`; the proof-path freeze is:

- probe `39e739f1ada378ba4bb2676ea25e2cc244c5881e2436b3e02d0673e5fe30b846`;
- selector `ea1720ed7be06d3e8cede9838ea524cda3c5df5eeeb7912e2e60d13bb82494c8`;
- candidate `08b8525ce582781669e2446c7f45eea29e21cd4553fdd9caf34029ea798bea6a`;
- certifier `c282f22e3510bd8427daa914cd85b8a89a36974f85e283e6b04048fda2ac0708`;
- replay `6e291bdd4526518496e664c14e15664bf554c1e9f089d92f65f8097081db5d7e`;
- V2 dispatcher
  `c58a0e3dd7cd04efe2ab92018cd82738fd72e94cf42635b5b69ebab6730ea6e0`;
- V3 pipeline
  `0961e3a58ea3ebe80ec4be63d9c08c62f4b9004de1c85726f17792062225f3b6`;
- DualSolver
  `fd6eb263d05525b107f9d10d6ec6f39a5bd367808ec50184fa0c0fd4d195ea88`;
- spec creation
  `2f0df22f69bc4f384435012832725247dd70c605bec7219b5610e5d43d322e71`,
  synthesis
  `0a4104de37653f7a11321b62c8fb0a24dd98a51ba01d546446069e8c2abeffc8`,
  Torch-to-ACT
  `84ad13bfed2565649ec3431454584f9d2ba10063c44c8a4f9762135a84375a0c`,
  and device manager
  `e58c585af28e329627061216578d5dacfb3b52365c2071b15cad69befd70c55e`.

The fixed launch is:

```bash
timeout --signal=TERM --kill-after=5s 30s \
  /data1/Kane/miniconda3/envs/act-py312/bin/python \
  -m act.pipeline.verification.query_dual_probe \
  --onnx /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/onnx/CIFAR100_resnet_medium.onnx \
  --vnnlib /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/vnnlib/CIFAR100_resnet_medium_prop_idx_6232_sidx_3020_eps_0.0039.vnnlib \
  --targets 10,14,22,40 --v3-quotas 16,8,24,16 \
  --steps 4 --time 12 --block 1024 --selector-time 1 \
  --selector-max-adjoint-cells 30000000 \
  --selector-pool-per-rival 64 --device cuda \
  --output artifacts/hybridz_largecls_gates/gate1_query_dual_only_v3_iid2_s4_k64_20260728.json
```

Promotion still requires transaction time at most `10.5s`, at least 20
selected-target strict replay improvements, at least one in every nonempty
stage, at least 20 property rows strictly better than the independent root,
stable input/source hashes, CUDA peaks at most 2.5 GiB, and incremental RSS
at most 2 GiB.  Completion in `(10.5,12]` is valid but not promoted.  Any
other failure closes V3 without an alternate iid, K128 retry, Operator build,
Gate6, or full benchmark.

The single authorized V3 iid2 launch then closed V3:

- output
  `artifacts/hybridz_largecls_gates/gate1_query_dual_only_v3_iid2_s4_k64_20260728.json`,
  internal canonical receipt SHA
  `e26c32f1d1f6fd160edb7fe4ed4edd7ae739c41761af166901589e20c007e6`,
  file SHA
  `5b16d922547ebbfeffbda7bf2439b6d818a37d1c138a41ccf00829ebe4ca30be`;
- status `error`, `proof_authority=false`, failure stage
  `query_dual_transaction`;
- `QueryDualPipelineTimeout` reported
  `query-dual replay deadline expired` after `12.015105s`;
- both source and input TOCTOU checks remained stable and the receipt's
  canonical hash and current source map were independently reverified;
- CUDA peaks were only `146,308,096 / 203,423,744` bytes
  allocated/reserved, and incremental process RSS was `499,458,048` bytes,
  so memory was not the stop-loss cause;
- atomic rollback exposed no stages, property values, Operator object, solver
  verdict, or partial authority.

Consequently K128, another iid, Operator build-only, Gate6, and full benchmark
remain unauthorized.  The next research version must reduce independent
replay arithmetic on controlled topology-matched toys; merely selecting fewer
rows or rerunning the same configuration is not a permitted response.

## V4 pre-registration: bit-stable CPU replay kernels

This section formalizes the `>=2x` and bit-identical gate announced immediately
after the V3 failure, before any V4 controlled result or further real-instance
execution.  A parallel worker had begun isolated micro-profiling, but no V4
candidate had been accepted or exercised through a pipeline.  The V3 numerical
replay source is frozen by SHA-256
`6e291bdd4526518496e664c14e15664bf554c1e9f089d92f65f8097081db5d7e`.
Only the ONNX graph and tensor shapes were inspected statically after the V3
failure; no property was executed.  That inspection shows that the medium
CIFAR100 model is dominated by 3x3 residual convolutions at `128x8x8` and
`128x4x4`, whereas the old performance toy used only `4x8x8`.  A V4 result
therefore cannot be promoted on the old toy alone.

The primary V4 hypothesis is that proof arithmetic, rather than selected-row
count, can be reduced while retaining the same CPU binary64 authority:

- precompute immutable per-convolution geometry and read-only kernel views
  once in each sealed cone;
- remove repeated construction of bias broadcasts, spatial index arrays, and
  masks;
- fuse allocation passes in the long-double outward-guard helpers without
  changing their evaluation order;
- exploit known-nonnegative radius/box products only when the optimized helper
  reproduces the V3 nominal, error, penalty, and final lower-bound bytes
  exactly;
- retain the V3 direct sparse-scatter or kernel-offset channel-GEMM arithmetic
  order.  A faster reformulation that changes even one result bit is a
  different proof method and fails this V4 track.

CUDA output remains candidate-only.  It may be studied in parallel, but it
cannot replace the CPU replay unless a separate CPU certificate proves every
consumed residual with less work than the replay itself.  Benchmark speed or
agreement on samples is not proof authority.

The controlled audit must be a new, standalone command that imports no
Operator-HZ path, creates no solver verdict, and reads no official VNNLIB.  It
must freeze both implementations in the same source state and alternate their
warm-run order.  It has two mandatory classes:

1. A fixed-seed synthetic residual graph with the production channel/spatial
   regimes `128x8x8` and `128x4x4`, four overlapping start cones, and the
   unchanged V3 objective schedule `32,16,48,32` target objectives plus `99`
   property objectives.  Selection/quota reduction is forbidden.
2. Small Dense/Conv/ReLU/ADD DAGs with exhaustive ReLU phases and a Fraction
   oracle, including cancellation, ties, signed zero, boundary padding,
   stride two, groups, and subnormal-scale coefficients.

Promotion requires all of the following in one atomic, source-hashed receipt:

- byte-identical lower-bound arrays for V3 and V4 on every controlled case;
- byte-identical per-query guard arrays and identical
  `guard_total_hex`, `guard_max_hex`, guard counts, sparse/dense block counts,
  and Fraction-audit counts;
- zero exact-oracle enclosure violations and all existing replay/session/
  pipeline soundness tests passing under `unittest`;
- at least `2.00x` improvement in the warmed alternating median of the
  production-scale replay-only workload, with a 95% bootstrap lower bound of
  at least `1.80x`;
- no more than a 5% replay regression on either the sparse or small-DAG
  workload, incremental RSS at most 2 GiB, and no hidden CUDA fallback.

Any bit mismatch, soundness failure, timeout, non-finite value, receipt
instability, or speedup below the gate closes that V4 implementation.  It does
not authorize a smaller quota, another iid, or a real-network retry.  Only a
passing receipt may pre-register exactly one new query-only probe on the same
CIFAR100 iid2 input; Operator-HZ, Gate6, TinyImageNet, and full-400 remain
closed until that probe itself passes its fixed time/tightness gates.

### V4 controlled result: rejected

V4 was stopped at its first production-scale synthetic gate.  Its immutable
receipt is
`artifacts/hybridz_largecls_gates/query_dual_v4_synthetic_profile_20260728.json`,
with canonical receipt SHA
`1bb98912ac390fb639ee3ec1f288580b1f2c52b5dcd17a610ec451806b9652ef`
and file SHA
`7485d1a6933f404710863ef45515644a751a91d6487815a1ff8c9999d7e7a2a9`.
The topology-matched workload used synthetic weights, the fixed objective
schedule `32,16,48,32,99`, and no real ONNX or VNNLIB data.

All 227 lower bounds were byte-identical, but the alternating medians were
`13.302618s` for V3 and `13.069931s` for the single-GEMV absorption
specialization: only `1.017803x`, far below `2x`.  Profiling showed that
geometry/index construction was about 1% of a `128x8x8,Q=99` convolution;
the componentwise long-double error arrays dominated.  The candidate was
therefore removed from production.  The replay source is again exactly
`6e291bdd4526518496e664c14e15664bf554c1e9f089d92f65f8097081db5d7e`;
the isolated rejected-candidate tests are retained under SHA
`a2962e206c54f82259bcd30a1e893f5d2bbb8c974f30bb0089f001c0767c81a9`.
The relevant V1/V3/V4 regression set passed `28/28`.

## V5 pre-registration: support-compressed scalar roundoff certificates

This section is fixed before any production V5 implementation or additional
real-instance execution.  Two isolated, explicitly non-authoritative algebra
toys already exist:
`query_dual_gpu_certificate_toy.py` /
`test_query_dual_gpu_certificate_toy.py`, with SHAs
`a29aa9313828b7e4b1c8eefd2d17f0e401299bdaabca2e1bf71dbee8f3cedc9d` /
`6df757c1b8dd2388eff5a51830a27acf28121647a4aa044cca2d8f629beaf52a`.
Their `7/7` Fraction tests establish an algebraic candidate, not proof
authority.  An independent review rejected GPU nominal replay plus a light
CPU check: a tight deterministic residual check must itself perform the
Dense/Conv product, while a triangle-only residual makes the GPU claim no
better than the zero-claim interval bound.  Dual-GPU agreement and
Freivalds/ABFT are also not deterministic soundness proofs.

V5 therefore remains CPU-authoritative and preserves the V3 nominal operation
order.  It changes only how affine roundoff is absorbed.  For a Dense
adjoint `a`, weight `W`, and predecessor-box magnitudes
`M_j=max(abs(l_j),abs(u_j))`, the direct CPU product obeys

```
abs(fl(a W)_j - (a W)_j)
    <= gamma_d * sum_i abs(a_i W_ij) + tau_d .
```

Instead of materializing this radius for every query and coordinate, V5 must
outward-precompute

```
s_i >= sum_j abs(W_ij) M_j,       B >= sum_j M_j
P_q >= sum_i abs(a_qi) s_i
G_dense_q >= gamma_d P_q + tau_d B .
```

It then immediately applies `scalar_q = down(scalar_q-G_dense_q)` and
continues with the byte-identical nominal coefficient.  Every displayed
inequality is an implementation obligation: `s`, `B`, `P`, and `G` must be
outward, finite, and bound to the frozen layer and exact bounds frame.
The additive subnormal term is not assumed to be bare `k*eta`; V5 uses the
outward value `tau_k >= k*eta/(1-k*u)`.

Dense Conv uses a separate two-part certificate.  For each kernel offset, its
channel GEMM has a weighted dot-discrepancy guard `D`.  The explicit,
V3-order offset additions have a second guard `A`, using their actual maximum
touch count and the bound `sum M*abs(nominal_term) <= P+D`.  The consumed
guard is `G_conv >= D+A`.  Boundary padding, stride, dilation, and groups all
contribute to the precomputed support and underflow multiplicities.  Using
only `gamma*P`, using one underflow term for all offsets, or counting only
`Kh*Kw` additions in the sparse scatter path is forbidden.  Sparse Conv keeps
the V3 componentwise-radius implementation in the first V5 version.

Bias dots, ReLU slope/alpha products and intercepts, DAG merges, and input
support keep their existing independent guards.  Each affine scalar guard is
absorbed before any ReLU or DAG consumer; delaying it could select a relaxation
line from the wrong nominal sign.  The V3 radius and V5 scalar guard may never
both be charged for the same affine operation.

V5 is an explicit opt-in numeric protocol; the V3 default and its receipt
semantics remain frozen.  Its receipts must additionally bind:

- numeric protocol, CPU/BLAS identity, thread count, RN-even and gradual-
  underflow probes, including a nontrivial matrix kernel;
- each support cache to layer/weight, Conv geometry, bounds-frame content SHA,
  and pre/post-ReLU box semantics;
- Dense, dense-Conv, and legacy sparse-Conv branch counts; dot and accumulation
  operation-count maxima; support and scalar-guard hashes;
- deadline, workspace, source, and session/crosswalk data already required by
  V3.

The controlled sequence is fail-fast:

1. Prove the Dense formula on cancellation, signed zero, ties, overflow
   rejection, minimum subnormal, degenerate boxes, and at least 5,000 fixed
   random Fraction objectives.  The V5 nominal coefficient must be byte-
   identical to V3 and its lower bound no larger than the exact oracle.
2. Add dense Conv only after Dense passes.  Exhaustively cover boundary
   padding, stride two, dilation, groups, residual fanout, offset cancellation,
   and every dense/sparse threshold neighbour.  Sparse results must remain
   byte-identical to V3.
3. Run all existing replay/session/pipeline `unittest` suites plus mutation
   tests for support/frame/geometry/count/hash substitution.  Every controlled
   V5 lower bound must enclose the exact oracle and must not be lower than V3.
4. On the same production-scale synthetic topology and unchanged
   `32,16,48,32,99` schedule, alternate V3/V5 for at least three paired runs.
   Promotion requires median speedup at least `2.00x`, paired-bootstrap 95%
   lower bound at least `1.80x`, zero tightness regressions, incremental RSS at
   most 2 GiB, and replay workspace at most 512 MiB.

Any failed lemma, oracle violation, old/new branch confusion, non-finite
value, hidden CUDA use, receipt mismatch, or performance miss closes V5
without a quota change or real retry.  Only one passing atomic controlled
receipt can authorize pre-registration of the same iid2 query-only probe with
an explicit V5 protocol switch.  Operator-HZ, Gate6, TinyImageNet, and
full-400 remain unauthorized.

### V5 controlled result: rejected

V5 is closed without a real-network retry.  Its final controlled receipt is
`artifacts/hybridz_largecls_gates/query_dual_v5_controlled_audit_20260728.json`,
with receipt SHA
`fa9cf7a5800565d4a47a69cd2a609ccb4d1f52595c285f50b87fcace02fbfe26`
and file SHA
`299b7cfb61ddde3344f63e6d8396b2e6a442daf0af1259a699e8ed47c24f15ca`.
The receipt contains `100/100` passing replay/session/pipeline tests, the
fixed 5,000-objective Dense Fraction corpus, zero Fraction violations, zero
tightness regressions on its original corpus, and only about 143 MiB of
incremental high-water RSS.  Those successes do not override a failed gate.

On the unchanged production-scale synthetic topology and all 227 objectives,
the three paired times were

```
V3 = [5.330134, 4.946634, 5.324586] seconds
V5 = [3.001022, 3.324817, 3.071256] seconds.
```

The median speedup was only `1.733684x` and the paired-bootstrap 95% lower
bound was `1.487791x`, below the required `2.00x` and `1.80x`.
An earlier diagnostic run observed `2.208669x`, but it was rejected by an
over-broad CUDA-import check and cannot be substituted for the final atomic
receipt.  The difference also demonstrates that three pairs at eight BLAS
threads are too sensitive to concurrent CPU load for the next version.

An independent adversarial audit then found three stricter blockers which
would reject V5 even if its timing had passed:

- Dense cancellation with `a=[1e16,1,-1e16]`, `W=ones(3,1)`, and
  `x in [-1,1]` gives exact lower bound `-1`, V3
  `-0x1.1c37937e08012p+4`, and V5
  `-0x1.1c37937e08013p+4`: V5 is sound but one ULP looser.
- The analogous one-by-one dense Conv with three output channels is sound but
  two ULP looser than V3.  A mixed zero/dense query block also charges
  channel-dot and offset-add underflow mass to a structurally zero row.
- A caller can replace every candidate ledger field, recompute the unkeyed
  JSON hashes, and make the research-only candidate verifier return true.
  `proof_authority=False` prevents a false solver verdict, but this is not the
  semantic receipt validation required for promotion.

The rejected implementation SHAs are
`3bd0e6358705bccb42ac3d9fdaf50f357fe24537239b04d079b6473b09e58a4c`
for the integrated V5 replay,
`1c466f511f52f8f93e79bc6707d33495bf186f7958ea3fafbe5edfb8b50ed4f9`
for Dense,
`c8a7a0495bfff515dfc1fe4a3fde9cf0e0ad9cda017f4f103cd0604f926785cc`
for dense Conv, and
`140d4e723767ed9341a640927d8db31e7d92e85c65341dfcba4aebf06201ae98`
for the accounting-only sidecar.  The V3 production replay remains frozen at
`6e291bdd4526518496e664c14e15664bf554c1e9f089d92f65f8097081db5d7e`.

### V5.1 pre-registration: wide support, structural activity, sealed catalog

This section is fixed after the V5 rejection and before any V5.1 source is
created or any additional real instance is read.  V5.1 is a new isolated
numeric protocol; it may import rejected V5 research helpers, but V3 and the
recorded V5 sources are not silently redefined.  V5.1a deliberately has no
normal-valued cancellation heuristic.  If the formula below still has a
controlled tightness regression, V5.1a closes; a threshold cannot be tuned
against the failing row.

For every nonnegative support dot, V5.1a replaces the stacked binary64
enclosure by a wider deterministic enclosure.  Let `uL` and `etaL` be the
measured unit roundoff and minimum subnormal of a platform-gated
`numpy.longdouble`, and let `k=2*n+2`.  Define

```
gammaL_k = upL(k*uL/(1-k*uL))
tauL_k   = upL(k*etaL/(1-k*uL))
DotUpL(x,y) =
    ceil_f64(upL((fl_longdouble(sum_i x_i*y_i)+tauL_k)/(1-gammaL_k))).
```

Every short multiply/add/divide in this expression is directed outward in
long double.  `ceil_f64` keeps the rounded binary64 value when it is already
at least the long-double enclosure and otherwise takes exactly one binary64
successor; it is not the old unconditional successor.  The platform must
have at least eight more mantissa bits than binary64 and pass RN-even,
gradual-underflow, and exact Fraction spot checks.  Dense uses

```
s_i = DotUpL(abs(W_i), M)
B   = DotUpL(ones, M)
P_q = DotUpL(abs(a_q), s)
G_q = ceil_f64(upL(gamma64_d*P_q + tau64_d*B)).
```

The nominal coefficient remains the byte-identical CPU binary64 `a@W`.
On the fixed cancellation counterexample the pre-implementation estimate was
`0x1.1c37937e08009p+4`, below the V3 penalty
`0x1.1c37937e0800fp+4`.  The final directed implementation exposed a
one-ULP documentation discrepancy: applying the formula above literally,
including its conditional rather than unconditional `ceil_f64`, produces the
tighter `0x1.1c37937e08008p+4`.  The latter still encloses the exact error and
is the normative hard-test value; no extra successor is added merely to match
the preliminary estimate.  Preliminary normal-valued Fraction fuzz has no
regression, but this observation is not the proof.

Subnormal and potential-underflow rows use a deterministic conservative
fallback.  A layer records nonzero exponent extrema for its frozen weight and
support; a query row is marked when any operand is subnormal or those extrema
permit a product or short guard expression below the normal range.  For only
those marked rows, V5.1 streams the V3 componentwise penalty in bounded
column tiles and applies

```
G_final = min(G_wide, G_streamed_V3).
```

Both arguments must independently enclose the exact weighted discrepancy, so
their minimum is sound.  The radius tile is discarded immediately and no
`Q x predecessor_width` radius is retained.  The receipt binds the fallback
mask, reason codes, both guard hashes, tile width, and final guard hash.
This is one scalar policy and one scalar subtraction, not simultaneous
application of two guards.  If the fallback fires on more than 5% of normal
production-synthetic rows or destroys the performance gate, V5.1a closes.

Dense Conv keeps its separate `D+A` proof, but all structural-zero decisions
are corrected.  For offset `o`, predecessor support `m`, selected query
coefficient `c`, and kernel `w`, define exact Boolean activity

```
E_S(co,p) = OR_ci ((w(co,ci,o) != 0) AND (m(ci,p) != 0))
E_D(q,o)  = OR_p,co,ci
              ((c(q,co,p) != 0) AND (w(co,ci,o) != 0)
                                   AND (m(ci,p) != 0))
E_A(q,o)  = OR_ci,p
              ((m(ci,p) != 0) AND (old_f64(q,ci,p) != 0)
                                  AND (term_f64(q,ci,p) != 0)).
```

`E_S` is a true contraction-overlap mask, not the unsoundly loose outer
product of row-any and column-any.  `D` or `A` is set to exact zero only when
its corresponding Boolean is false.  The `E_A` conjunction is valid because
adding representable `x+0` or `0+x` is exact; channel-dot discrepancy is
already charged by `D`.  Signed zero is inactive, while every nonzero
subnormal is active.  Offset accumulation and final `D+A` use a
zero-preserving outward sum: a row remains exact zero iff all consumed terms
are structural zero.  A 10,000-sequence controlled audit has already checked
60,000 `D` and 60,000 `A` cases against Fraction with zero inactive-mask
violations; that audit must be reproduced from the final source.

Scalar absorption is also row-local.  Only rows with a nonzero final guard
execute a downward subtraction; zero-guard rows retain their exact scalar
bytes.  The active-mask SHA and count are mandatory receipt fields.  The
dense/sparse Conv threshold remains exactly
`8*nonzero_count <= dense_count`; all three neighboring cases, mixed masks,
signed zero, subnormal nonzero, and both cancellation counterexamples become
permanent hard tests.

V5.1 uses a new root-owned sealed session.  It must build its own immutable
cones from the checked sealed-graph bridge of the root box certificate; it
may not reflect into a V3 session.  One immutable bounds frame owns a catalog
keyed by frame content, cone/start layer, layer and predecessor, weight and
Conv geometry, source `lb/ub`, pre/post-ReLU box semantics, numeric platform,
and implementation SHA.  Conv plans and Fraction-audited ReLU lines may be
shared across the five overlapping stages only under that exact frame key;
cross-frame reuse fails closed.  Cache hits check the absolute deadline.

The V5.1 guard ledger extends each affine execution key with
`query_start`, `query_end`, query-block SHA, active/fallback-mask SHA, and
support/catalog content SHA.  Expectations are reconstructed from the sealed
cone and chunk schedule, not accepted from receipt text.  Exactly one final
policy is applied per row, all execution indices are consecutive, and commit
requires complete coverage.  Fully rehashed field substitution, query-span
gaps/overlap, branch substitution, frame reuse, copied capabilities,
double-charge, missing-charge, and deadline-after-last-record mutations must
all fail.  Dense helper deadline expiry is normalized to replay TIMEOUT, not
reported as ERROR.

The CPU contract records NumPy, BLAS vendor/version/library SHA, fixed thread
count, dynamic-thread settings, RN-even and gradual-underflow probes, and a
nontrivial matrix-kernel Fraction audit containing cancellation and
subnormal lanes.  Merely finding an ambient `torch.cuda` module is neither
failure nor evidence of GPU execution; the numeric dependency closure is
AST-audited to exclude CUDA compute calls, and all consumed arrays remain
CPU NumPy binary64.

The controlled promotion ladder is stricter than V5:

1. Reproduce both cancellation counterexamples, the disjoint-support mask,
   mixed zero/dense rows, signed zero, minimum subnormal, overflow rejection,
   and at least 5,000 Dense plus 5,000 Conv Fraction query rows.  There must
   be zero soundness violations and zero V5.1-below-V3 rows.
2. Pass all replay/session/pipeline tests and the semantic mutation suite.
   Sparse Conv must remain byte-identical to V3; every V5.1 nominal affine
   coefficient must remain byte-identical to V3.
3. Run the unchanged `32,16,48,32,99` schedule with exactly 227 objectives.
   Official timing uses four BLAS threads, disables MKL/OMP dynamic threads,
   warms both paths, pauses other audit workers, alternates at least five
   pairs, and records per-stage times.  This isolates measurement; it does
   not reduce runtime parallelism available to other pipeline phases.
4. Promotion still requires median speedup at least `2.00x`, paired-bootstrap
   95% lower bound at least `1.80x`, zero tightness regressions, incremental
   RSS at most 2 GiB, workspace at most 512 MiB, and stable source hashes.

Any failure closes V5.1a without changing the objective schedule, selecting a
friendlier iid, or running a real model.  If all gates pass, the result only
authorizes a separate pre-registration for one same-iid2 query-only probe;
Operator-HZ, Gate6, TinyImageNet, and full-400 remain closed.

### V5.1 implementation checkpoint: numeric pass, sealed session pending timing

The isolated V5.1 implementation was completed on 2026-07-28 without reading
another real ONNX/VNNLIB instance.  The frozen production V3 source remains
`6e291bdd4526518496e664c14e15664bf554c1e9f089d92f65f8097081db5d7e`.
The current V5.1 numeric sources are:

* Dense wide/streamed guard
  `dad748771e9bf8ea7c4db8fb0a163a975cfe3b3e5e4f6242b117a77d940b8e44`;
* dense Conv structural replay
  `cc58681fafd2a2a4827711164b894962f96b7931f639ffebeb03c074f0d97b56`;
* BLAS contract
  `d2bc8974ebda1f8788dc3940e16aaa395b8fc6e4beb4cb270d4ca0493d061d90`;
* integrated non-authoritative replay
  `bbf2f3ebcedcfe1c9e4d4bf8d5f29304f06243b3c9738b1c26093c59916538ac`;
* semantic authority sidecar
  `99b42d5275046e5f1195fe799409de9a69df21c416b4d15fe110e6a87f94f4a8`;
* root-owned five-stage session
  `5b81f096c1d8279bae88e62f91bcdcd78d8efa640227361ef0e7ed676ec6c172`.

The fixed Dense test audits 5,000 normal Fraction rows and 1,000
underflow-oriented rows.  The fixed Conv test audits 5,000 Fraction query
rows.  Independent fuzz added 2,376 Dense/Conv Fraction rows with random
groups, stride, padding, dilation, signed zero, subnormal, overflow-adjacent
and cancellation cases.  Across these audits there were zero soundness
violations, zero final-guard-above-V3 rows, zero lower-bound-below-V3 rows,
and nominal affine coefficients remained V3-bit-identical.  Sparse Conv
absorption also remained V3-bit-identical.

The authority audit found and closed several defects before timing:
fallback rows outside the active mask, undercharged scalar/componentwise
absorption, live-object rehashing, copied-ledger publication, mutable
container substitution, missing-span certificates, duplicate semantic
executions, record-to-commit trace mutation, deadline extension, cache-hit
deadline overrun, fork inheritance, and commit concurrency/publication
TOCTOU.  The session now has a registry-external runtime seal, uses its
externally held operation lock, requires stage indices `0..4` exactly once
on one frame, binds every pending stage externally, rechecks the live root
graph/network/BLAS/source manifests at commit, and removes every provisional
result if the final publication deadline expires.

Under four fixed BLAS threads with MKL/OMP dynamic threading disabled, the
current BLAS, authority, Dense, Conv, integrated replay, full-session and
controlled-audit suites pass 104/104 tests.  This count includes 23 session
tests covering copied/self-resealed sessions, frame/catalog transplant,
missing or double observer events, exact five-stage coverage, PID/container
binding, concurrency poisoning, normalized create/frame/Conv/BLAS/platform
deadlines and errors, failed-session catalog cleanup, and
deadline-at-publication cleanup.  An independent final review reproduced
all prior failure cases against the final session SHA and found no remaining
ordinary correctness, registry, deadline, lock-release, or GC defect.

The standalone controlled-audit source is
`bb738d531d8d0d36ed5de994b0ddaa569d30b0b793ce7e7101a56c5d8145f77a`;
its 16-test harness is
`f7f63df6a3ce2abb1570e2cbedefd92305ccaf3e9ab69f7ceeb7b475bb43dd6b`.
The immutable official configuration SHA is
`1e9505d76376bad2cf5c4ba5e9d1972b9313ff738ba6a8a26f961b84c233ab82`.
The harness obtains one actual outward BOX root certificate and passes the
same root bounds, bytes-backed query rows, and bytes-backed alpha values to
both full transactions.  Target uses replay the real preactivation
predecessor cones `(2,5,9,14)`, not the legacy toy's ReLU-output starts
`(3,6,10,15)`; both paths still execute exactly
`32,16,48,32,99 = 227` objectives.  Numeric-source SHAs, query/alpha material,
the four-thread environment, live BLAS probes, CPU-only AST closure, memory,
workspace, and host-worker state are fail-closed.  ACT package bootstrap
transitively imports solver modules, so the receipt records those ambient
modules honestly while separately proving that this audit path has no direct
solver/Operator import or call and creates no verdict.

The host gate was re-frozen after a live counterexample showed why `load1`
alone is insufficient: with `load1=3.26`, another readable process was still
measured at `1.007` CPU-core equivalents.  Official execution is now pinned
to the pre-registered physical CPUs `4,5,6,7`; they have no SMT siblings.
Every initial, pre-timing and post-timing check samples `/proc/<pid>/stat`
twice for 0.25 seconds, seals PID identity with its kernel start-time tick,
and rejects any other readable non-ancestor process at or above 0.50 core
equivalents.

That short sample is only the first line of defence.  Every warmup, each
individual timed V3/V5.1 execution, and the complete warmup-plus-timing
window now has a cross-window accounting seal.  It sums only the selected
`cpuN` rows' `user+nice+system+irq+softirq+steal` ticks, subtracts this
process's `utime+stime`, records but does not gate Linux's unstable `iowait`,
and does not double-count `guest` already included in `user/nice`.  The
start/end ordering makes the global interval conservatively enclose the
self interval.  An integer comparison requires external CPU strictly below
0.50 core equivalents, so equality rejects; affinity, PID start time,
included-field monotonicity, or clock changes fail closed.  This aggregate
seal covers processes which start, exit, or become idle during a run and
prevents one contaminated implementation from being diluted by the other
nine samples.  Deterministic distributed-load, PID-reuse, process-churn,
denied-cmdline, guest/iowait, exact-threshold, per-run-dilution, affinity and
receipt/config-substitution tests are included in the 104-test closure.
The receipt verifier binds the known configuration SHA and requires
gate/status consistency; it still grants no proof authority.

The required invocation uses `taskset -c 4-7`,
`OPENBLAS_NUM_THREADS=4`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
`MKL_DYNAMIC=FALSE`, and `OMP_DYNAMIC=FALSE`.  `OMP_DYNAMIC=0` is not used
because the installed `libgomp` rejects that spelling.  At this checkpoint
the pinned-CPU aggregate sample itself measured only about `0.08` external
core, but the all-PID seal correctly rejected an unrelated PID using about
one full core.  Consequently neither the one-pair diagnostic nor the
official five-pair audit was launched.

The standalone candidate receipt is explicitly marked
`integrity_scope=unkeyed_internal_consistency_only` and
`semantic_authority=False`: coordinated digest and receipt replacement is
not cryptographic authentication.  It is never consumed as authority.  The
full session instead reconstructs query blocks/spans, catalog aliases,
expectations and traces synchronously from sealed live material.  All V5.1
outputs still carry `proof_authority=False`.

No official five-pair topology timing has yet been accepted at this
checkpoint.  Therefore V5.1 has passed the numeric and controlled semantic
preconditions only; the unchanged `32,16,48,32,99` performance gate and every
real iid remain closed.

### V5.1a diagnostic stop loss and V5.1b material-cache pre-registration

This section is fixed before a production-topology V5.1 timing result and
before any V5.1b source is created.  It does not weaken or replace the frozen
V5.1a five-pair gate.  The first available quiet-host measurement is a single
non-official pair using the same topology, actual root certificate,
predecessor cones, 227 objectives, four BLAS threads, chunk 64, and one full
session per implementation.  It may add counters and phase timers but may not
change arithmetic.  A timeout, tightness regression, or speedup below `1.50x`
closes V5.1a immediately without spending four more pairs; a result at or
above `1.50x` still grants nothing and only permits the immutable five-pair
gate.  No real model is read in either case.

The quiet-host diagnostic was subsequently run exactly once under the
pre-registered physical-CPU affinity `4,5,6,7`, fixed four-thread BLAS,
dynamic threading disabled, chunk 64, the unchanged five-stage schedule, and
the same outward root certificate for both implementations.  All three
immediate host preflights passed; the per-implementation and complete timing
windows, as well as the postflight, also passed the external-CPU gate.  The
non-official result was:

```
root certificate       0.038637 s
V3 full session        4.833535 s
V5.1a full session     3.784443 s
V3 / V5.1a speedup     1.277211812x
catalog entries        20
tightness regressions  0
```

This result produced no official receipt and read no real benchmark.  Because
`1.277211812x < 1.50x`, the pre-registered stop loss closes V5.1a immediately:
there is no V5.1a five-pair audit and no real-instance retry.  The zero
tightness regression preserves the numerical result, but the performance is
insufficient.  Only a separately reviewed V5.1b implementation may receive a
new one-pair diagnostic after all of its soundness gates pass.

Static inspection plus the immutable rejected-V5 receipt already gives a
specific performance hypothesis.  The fixed cones contain `1,2,4,6,7`
affine layers and the chunk schedule creates 27 observer events, including
25 dense-Conv executions.  There are only six unique Conv layers and one
unique Dense support, but the current stage-bound catalog builds 19 Conv
plans.  Replay, property cache hits, and commit together call the exhaustive
Conv-plan validator 50 times.  That validator recomputes long-double support
matrices and sums; it is not an O(1) identity check.  Therefore V5.1b tests
one architectural change before considering a new numerical formula:
An untimed two-channel instance with the same DAG, cone overlap and chunk
pattern then reproduced these counts exactly: plan builds by layer were
`{2:5,4:4,7:3,8:3,11:2,13:2}` (19 total), full validations were
`{2:12,4:10,7:8,8:8,11:6,13:6}` (50 total), and the frame held 20 aliases.

1. A frame owns a private physical-material registry.  Its key contains the
   frame content SHA, layer and predecessor IDs, operator/branch, weight SHA,
   Conv geometry SHA, source `lb/ub` SHAs, pre/post-ReLU semantics, numeric
   platform SHA, and implementation SHA.  It deliberately excludes stage and
   cone IDs.
2. The physical core is bytes-backed and immutable.  Conv may share the
   existing numeric plan.  Dense must split its numeric support, box mass,
   and exponent data from its current cone-bound diagnostic wrapper; the
   wrapper itself is never reused across cones.
3. Every stage still receives a distinct frame-owner-minted alias containing
   `stage_use_sha256` and `cone_start_lid`.  An alias points to the physical
   core content SHA; it does not grant another stage's authority.
4. Admission performs the full mathematical validation once.  A private
   session capability may use an admitted core only after an absolute
   deadline check and an external identity/content-seal check.  Public helper
   calls retain their current exhaustive validator.  Commit reconstructs
   every physical key from the root/frame and performs one full validation
   per unique core, then verifies all per-stage aliases and ledger coverage.
5. Cross-frame, different source-box semantics, changed `lb/ub` despite equal
   max-abs, weight, geometry, platform, implementation, fork/PID, copied
   capability, or mutable-container reuse all fail closed.

The controlled topology must report exactly seven physical builds, 20
stage-local aliases, and 13 cross-stage hits; the six Conv cores are fully
revalidated exactly once each at commit.  Permanent toys cover two
overlapping cones on one frame, a second frame, equal-maxabs/different-box
inputs, ReLU pre/post semantics, plan/offset/activity/support-sum mutation,
alias transplant, cache-hit deadline, fork, copy, and commit mutation.
All 5,000 Dense plus 5,000 Conv Fraction rows, nominal binary64 bit equality,
V5.1-at-least-V3 tightness, sparse-Conv V3 bit equality, the semantic
mutation suite, CPU-only AST closure, and the 104-test frozen V5.1a baseline
remain mandatory.  Only after those gates may V5.1b run the same one-pair
stop loss and, if promising, its own newly frozen five-pair audit.  BLAS
commit re-probing is not weakened in this track.

The sixth isolated material-cache toy is now frozen as design evidence:

```
source SHA256  faaf583bdf5bdc64fbbdcfa41e2fa50416db24a51ff316a44a2b706c04f0f369
test SHA256    ed89938a7489c35e593c3fe1322d6143463e95a2dd7a9c9a675e0499f65c140a
scope          isolated toy; proof_authority=False
tests          31 cache tests + 10 Conv numeric tests = 41/41
counts         6 physical builds / 19 aliases / 13 hits / 6 commit validations
hit work       0 array digests / 0 mathematical validators
```

The frozen four-thread regression command then passed `135/135` in
`7.735 s`: the complete 104-test V5.1a closure plus all 31 cache-specific
tests.  This run used CPUs `4-7`, four OpenBLAS/OMP/MKL threads and disabled
dynamic threading; it was a correctness run, not a timing receipt.

Its root-owned bytes-backed snapshot survives caller mutation, while exact
`_FrozenLayer`, `_Box`, `ndarray`, tuple and scalar types reject subclass
dispatch before dynamic reads.  A `MappingProxyType` is accepted only when
backed by an exact `dict`.  Coordinated public seal/snapshot/signature/key
redirection from material A to an existing material B is rejected at commit.
An independent exact-SHA review found no accepted wrong binding or TOCTOU
case.  This freezes the isolation pattern only; it does not authorize replay,
session integration, a production certificate, or any timing run.

The first production-registry scaffold is explicitly rejected despite its
native 18/18 tests and correct `7/20/13/27` counters.  Directed review showed
that layer/box/stage tuple subclasses, ndarray subclasses and dict-subclass
mapping backings crossed admission and commit.  A dynamic box getter was
executed during an accepted transaction.  Its lookup path also reconstructed
the full root identity on every one of 27 execution lookups, so it was not
O(1), and commit derived work from all 20 occurrences instead of the seven
unique physical locators.  Finally, module-global Conv preparation and
validation callables were not anchored to the opened root: replacing both
could admit six altered plans which the original validator later rejected.
Consequently green native tests and correct counters are not promotion
evidence.  The replacement must bind exact input types, closure-owned
dependency implementations and their receipt digests, perform no full-tree,
array-hash or mathematical-validation work on a hit, and reconstruct exactly
seven unique physical materials at commit.

The reviewed private-execution design copies only the frozen Dense/Conv
numeric body into a new source-hashed, factory-local lexical closure.  It
does not recursively clone the public Python function graph: captured
functions still resolve helpers through mutable `function.__globals__`, so
that approach has an unbounded dependency audit and a TOCTOU gap.  The
registry alone retains the private kernel port and opaque locator table; no
API returns a support, plan, core or callable.  It performs alias resolution
and execution under one transaction boundary.  Runtime coefficients are
rejected unless they are exact native C-order binary64 arrays, then captured
once into bytes-backed storage so counting, GEMM and guards observe one
snapshot.

The private Dense path must preserve the exact nominal GEMM and underflow
fallback, while the private Conv path preserves offset order, both
`np.take` operations, transpose/reshape order, GEMM, accumulation and the
exact `8 * nonzero_count <= size` sparse boundary.  Private execution emits
only bytes-backed numeric arrays and masks: no helper receipt, diagnostics,
physical array hash or mathematical validator.  The surrounding ledger
hashes authoritative outputs once.  Admission and unique-core commit retain
their independent full builders/validators and dependency receipts.
Promotion requires bit-differential agreement with V3/V5.1a, the existing
5,000 Dense plus 1,000 underflow plus 5,000 Conv Fraction gates, zero old
validator/hash calls during private execution, deadline checks around every
uninterruptible BLAS/offset/tile boundary, and the integrated
`7/20/13/27`, `7/0/7` counts.

The first standalone private-kernel candidate is rejected even though its
fixed suite passed `22/22`:

```
source SHA256  7523d04d13a9cb1e87d743bef9fbd8ee51f9264672ebb71b92a542c6a6eb19ed
test SHA256    c5454c5b7169d08cd86174b7a3a20a5fa357597dcd246d7e7d5eadeb6c41e836
```

It independently captured runtime coefficients and reproduced the public
V5.1a arithmetic, but it still cloned the trusted public preparer's support
or plan into its private core.  Exact result types were therefore not enough:
a same-type Conv plan with the correct weight but an empty offset tuple was
accepted and produced an incorrect all-zero result.  Public Dense
`_validate_support` is likewise a metadata/hash validator, not an independent
mathematical reconstruction.  The replacement must derive Dense support,
box mass, exponent/underflow flags and Conv geometry, every offset index,
support, channel DotUp, activity and support sum directly from the single raw
snapshot using its own lexical arithmetic.  Only those independently rebuilt
values may enter a private core.

The next production-registry candidate is also rejected:

```
source SHA256  eb4503a6cc79d95d3cf740c4b8be2808acdde4c16fb52cc8b52c38858f888149
test SHA256    634f7f8d5848492fdaccce81297f25734e959c083d8b038531732dfeea20eed4
native tests   23/23
```

It fixed exact types, made all 27 alias lookups perform zero root/tree/hash/
derive/validator work, derived exactly seven commit specs, and captured the
four top-level prepare/validator callables.  That last fix was insufficient:
captured Python functions retained their shared module `__globals__`.
Replacing only Conv `_dot_up_l_matrix` after open with a zero-valued helper
was invoked 18 times by the supposedly captured functions.  The registry
still emitted a `7/20/13/7` certificate, while the original validator,
restored after the transaction, rejected all six Conv plans.  The dependency
digest did not change.  Therefore callable names, top-level function identity
and module file hashes do not bind a Python function's transitive execution
graph.  The replacement must use factory-local isolated globals or a fully
lexical implementation, bind function-code digests, and retain an external
binding-change gate only as defence in depth; a check-call-check gate alone
cannot close an ABA concurrency window.

A later independently rebuilt private-kernel candidate is rejected before
integration:

```
source SHA256  36f745dc637659f07f9aedc937e90f3220e6752e12666ba99813d7517425fd8d
test SHA256    4a4e29034694b4f39a879ac91df35d490f8474d9aeb4965bdf72152aa6194748
```

It rebuilt Dense support and every Conv offset from raw snapshots and rejected
same-type forged public plans, but it still delegated the declared
transposed-Conv output-shape contract to the public preparer.  If that
preparer was already replaced when the private factory captured it, an exact
forged plan could carry an illegal output padding into the private core.  The
replacement must lexically recompute
`base=(output-1)*stride-2*padding+dilation*(kernel-1)+1` in both dimensions
before the first public call and require `0 <= input-base < stride`.

The following replacement fixed that geometry gap and passed `25/25`, the
combined `160/160` four-thread regression, 192 systematic legal Conv geometry
bit differentials and 192 systematic Dense bit differentials, but it is also
rejected:

```
source SHA256  497df0eb02029df5c57a6a06feac887a4de16f54d7d5742a4a5a8f7df8bdf51c
test SHA256    74c805bf09420a98a232dcd26f6a6edf5578eb62c8bebebffbe4179f8eba4626
```

Its arrays were bytes-backed and read-only, but the returned custom result
objects stored those arrays in writable Python slots.  An external
`object.__setattr__(result, "final_guard", writable_zero_array)` replaced a
positive guard without touching the protected bytes.  Thus array immutability
does not imply binding immutability, and even a green independent review is
not sufficient without a direct field-replacement adversary.  A replacement
must use an immutable private ABI such as an exact built-in tuple, or keep the
entire result behind an opaque port-owned locator; it must also test
instance-slot, class/property, copy, transplant, pickle and GC paths before
any session consumes the result.

The first exact-tuple replacement is rejected as well:

```
source SHA256  69cf1e057e2a51456a26a3804b092d1934192ed20230f6a3975e45d7f9a0c7c1
test SHA256    c9c45f8cdf72202f6c471bb55e93a77a5a0a338c5bb2424085eb599c2975bba1
focused        29/29
combined       164/164
```

Its normal execution emitted only exact tuple/bytes frames and passed 192
systematic Conv plus 192 Dense bit differentials.  However, the factory
captured the dynamically resolved builtins `bytes`, `int` and `tuple`, and its
later exact-type tests resolved the same mutable names.  Replacing
`builtins.bytes` with a subclass before factory creation therefore caused
payload and dtype-tag fields of that subclass to cross the supposedly exact
builtin ABI while all 29 tests passed.  The next candidate must derive every
boundary type from literals (`type(b"")`, `type(0)`, `type(())`, and the
corresponding float/bool/string forms), remove dynamic builtin constructors
from the factory, and test persistent plus ABA substitutions before and after
factory creation in an isolated process.

The first dependency-sealed private-kernel replacement was initially frozen
for the isolated-kernel phase:

```
source SHA256  6588de61df436f0dbc63bd6f005ffaa95089bd1ca5abd4b7f8e843eebf4b71a0
test SHA256    f94531e588163db28fb4d5f23dba66bc3793709f785437b3f009bd26bfb00d4d
focused        33/33 in 3.510 s, fixed four-thread command
combined       168/168 in 11.452 s, frozen V5.1a/toy/private closure
systematic     Dense 100 + Conv 80, all result fields bit-exact
scope          isolated candidate; proof_authority=False
```

The implementation derives exact boundary types from literals, constructs the
factory implementation with a private builtin dictionary, and resolves all
nested arithmetic through import-time anchored builtin, NumPy, math, clock,
PID, lock and weak-reference dependencies.  Recursive disassembly of the
sealed implementation contains no `LOAD_GLOBAL`; its five
`LOAD_BUILD_CLASS` operations resolve through the private builtin dictionary.
Persistent and ABA dependency-disturbance tests after factory creation call
none of the changed public bindings and leave every output bit unchanged.
The previously failing exact case now has
`required = 5/72057594037927936`, an active row, and a stored final guard that
encloses that Fraction requirement.  Independent review also confirmed the
raw rebuild of Dense support and all Conv offsets.

This acceptance does not authorize a session or timing run.  Production
integration must keep the port and locators closure-private, capture the bound
execute methods immediately, consume and strictly decode their direct tuple
return while holding the same session operation lock, and bind exact expected
request/core shapes.  No caller-returned tuple may re-enter that boundary.

That `6588...` freeze is subsequently rejected.  A NumPy ufunc retains a
mutable instance dictionary: assigning a delegating callable to
`np.logical_and.reduce`, `np.logical_or.reduce`, `np.maximum.reduce` or
`np.minimum.reduce` left the guarded module binding unchanged, so the factory
accepted it and the private Dense path called it.  The same late lookup
affected the decoder.  The kernel also retained the mutable cached singleton
objects returned by `np.finfo`; changing `f64.eps` or `wide.nmant` changed its
factory-time platform decision while every module binding still matched.

The corrected isolated kernel and decoder are re-frozen with narrower,
explicit scope:

```
kernel source SHA256  db367fd398aad62a6366fda60f18fd860ccc16f0f7be115a88ab12b785964275
kernel test SHA256    bc16f20b40acda10fb1519437b46deec198d48435be41ec85658d9227b1ab993
decoder source SHA256 dec1271ce0f63d18c44a95ac724aeb5d73260a35a9d7f2a64086f46761d4e8cf
decoder test SHA256   0f7df3d2158bd2364a19c77a2fcebe8a82528d2cd2d5ff0a6e418fe557a294db
focused               67/67 in 9.458 s, fixed four-thread command
complete closure      202/202 in 17.417 s
scope                  isolated candidates; proof_authority=False
```

All seven used reductions are now bound from the immutable
`numpy.ufunc.reduce` descriptor at import time.  Factory construction first
checks the exact ufunc type, descriptor identity and complete per-instance
dictionary fingerprint.  Persistent pre-factory overrides are rejected with
zero changed calls; simultaneous post-factory overrides leave Dense and Conv
kernel/decoder outputs bit-exact with zero changed calls.  The kernel no
longer retains either `np.finfo` singleton and carries only exact immutable
copies of the required mantissa, epsilon and tiny values.  Recursive
disassembly covers 85 kernel and 32 decoder code objects and contains no
global load/store/delete or runtime import.

This repair does not turn the isolated kernel into an integrated authority.
Its ordinary admission still calls the public Dense/Conv preparers, whose
Python functions, result classes and cached platform state are the explicitly
unsealed outer boundary.  For example, holding a changed `np.finfo` singleton
through ordinary Dense admission can make the public preparer fail even
though the private kernel's own platform gate is lexicalized.  The prepared
integration must bypass that boundary under a separately reviewed hidden
runtime seal.

A host-contaminated synthetic stop-loss check was run only to decide whether
session integration remained worth implementing.  It used CPUs `4-7`, four
fixed BLAS threads and included the strict test decoder in the private timing.
For a `64 x 128` Dense query against a `128 x 256` weight, the public/private
execution medians were `0.000684565/0.000474121 s` (`1.44386x`).  For a dense
batch of eight through a `32 x 8 x 8`, 3x3 padded Conv, they were
`0.011613034/0.006307660 s` (`1.84110x`).  Private Conv admission took
`0.0100761 s` versus `0.00507214 s` for one public plan build because it also
performs the independent raw rebuild.  These are neither a quiet-host result
nor a promotion receipt.  They justify continuing the integrated session and
identify duplicate setup work as a measured risk; they do not open the
same-topology timing gate.

A non-timing prototype then established a viable way to remove that duplicate
public preparation without changing the frozen kernel.  Its public factory
has the exact closure ABI `("implementation", "sealed_dependencies")`;
the dependency tuple has five entries and its corrected 42-entry
direct-dependency
tuple stores the Dense and Conv public preparers at indices 7 and 8.  A new
adapter can copy those tuples, replace only indices 7 and 8 with one-shot
factory-local dispatches that return the registry's already-built exact
support/plan, and invoke the same sealed implementation.  On controlled Dense
and Conv examples each dispatch was consumed exactly once and every returned
field matched ordinary frozen-kernel admission/execution bit for bit, while
the `db367...` source remained unchanged.

This does not bypass the independent numerical rebuild: after dispatch the
frozen Dense path still reconstructs support, box mass, exponents and flags
from raw bytes, and the Conv path still reconstructs geometry, weight,
max-abs support and every offset field before creating its private core.
It also does not prove physical-key provenance by itself.  Numerically
equivalent Dense material can omit provenance-only fields, and Conv bounds
with equal max-abs support can be equivalent.  The prepared adapter must
therefore be independently frozen as a one-shot lifecycle/closure boundary,
while the integrated registry separately binds root, frame, layer,
predecessor, raw hashes and physical key.

A subsequent physical-registry candidate is still rejected despite passing
its own `41/41` suite, the production `7/20/13/27` and `7/0/7` counts, and a
100,000-lookup micro-check:

```
source SHA256  dbe3a884f3d01ac0ba595123b5575599e2f6921bdf00bf9aa4917eec7ec412a9
test SHA256    de0fc0769dcb966fc6ff0852703acb649967b747540bb8f435f675af64d6e3e4
```

Its isolated imported helpers and public slot checks were insufficient for the
factory's own nested hot functions.  After open, adding a module-global
`enumerate` that returned an empty iterator caused `fast_fields_equal` to skip
all field comparisons.  A deliberately changed alias `layer_id` was then
returned by `lookup_execution_alias` instead of being rejected.  This is a
verifier-integrity failure even though the registry remains
non-authoritative.  The replacement must give the complete hot lookup body
factory-private fixed dependencies, add both pre/post-factory persistent and
ABA dependency-disturbance regressions, and retain zero root/tree/hash/math/
derive/validator work on all 100,000 lookups.

The immediate replacement fixed that exact lookup dependency and passed
`47/47` with the required `7/20/13/27`, `7/0/7` counts.  Its 100,000 lookup
check completed in `1.239856 s` (`12.399 us/lookup`):

```
source SHA256  68214865a8a7d9ba55e08e0376b8dd3d4acef78f993927e9e8bd357d2a53c23b
test SHA256    f0888d60e54db2d4caeb063bbdfd0ab36c337a7e875971d4e36f5cfc944ba4ab
```

It is nevertheless rejected.  The factory-local physical, frozen, Dense and
Conv function clones still retained the actual mutable NumPy module.  After
opening a registry, replacing `numpy.asarray` with a counting wrapper allowed
admission to succeed and invoked that changed wrapper 126 times.  Therefore
cloned Python functions and fixed top-level helper identities are not enough
when a cloned globals dictionary still contains a live module object.  The
next registry must replace every module-valued numerical or digest dependency
with a factory-owned fixed namespace, recursively close Python helper globals,
and prove by bytecode plus pre/post-open tests that no actual NumPy, math,
JSON, hash or HMAC module dispatch remains.

The next `51/51` candidate replaced those live module references, but is also
rejected.  Every runtime view shared the public mutable
`_CapturedModuleNamespace` class.  Changing its `__getattr__` after open
allowed admission to finish and invoked the changed `numpy.asarray` wrapper
125 times.  A private namespace object is not private when its class remains a
public mutation point.

The corrected production-shape diagnostic scaffold is:

```
source SHA256  1d310330ba76ebb599d0b2dce077856ba555a8788e73ec600782b0a3d963b31c
test SHA256    12131d477dc86c3ae8ac0dd87ebbbfec39d2ecafa637e4d56d1914799830e3bc
focused        54/54 in 10.339 s, fixed four-thread command
100k lookup    1.4589745901 s
topology       7 physical / 20 aliases / 13 hits / 27 lookups
validation     7 admission / 0 hot / 7 commit
scope          non-authoritative diagnostic scaffold
```

Its runtime views now use a factory-private type.  The public legacy class is
inert; changing it calls neither the changed method nor `numpy.asarray`.
Persistent changes to the private type's `__getattribute__`, `__getattr__`,
`__dict__`, `__setattr__`, `__reduce_ex__`, property getter code or method code
are rejected before dispatch with zero trap calls and poison the registry.
The post-operation gate also prevents a persistent mid-operation change from
being published.  Recursive bytecode/profile checks find no live module
object in the sealed function graph and no root/tree/hash/derive/validator
work in 100,000 hot lookups.

This is deliberately not the formal registry.  Pure Python pre/post
fingerprints cannot detect a same-operation
`change -> private dispatch -> restore` cycle.  Therefore this candidate is
accepted only as the production-count diagnostic and performance scaffold.
The formal integrated path must use a hidden runtime seal and individual
closure-local numerical callables with zero module or custom-view dispatch;
the diagnostic scaffold cannot authenticate a solver verdict.

The V5.1b session integration boundary is fixed before implementation.  One
session-private frame bundle owns the registry port, all five stage admissions,
the private numeric port, one private locator per physical key, and captured
bound lookup/execute/close methods.  None is stored on or returned through the
public bounds-frame value.  Frame setup, replay lookup, numeric execution and
direct result decoding all occur while the existing one-shot session
operation lock is held.  For an affine execution the order is fixed:

1. resolve the pre-admitted stage/layer alias with the O(1) registry lookup;
2. require identity equality with the session-owned alias and select the
   private locator through the closure-owned physical-key table;
3. call the previously captured bound Dense or Conv execute method with the
   session's exact coefficient snapshot;
4. decode that direct return immediately, with no intervening callback or
   caller-visible value; and
5. bind the decoded arrays, exact expected shapes and semantic masks into the
   existing query-span/ledger record before continuing replay.

The Dense decoder requires the exact builtin tuple/bytes ABI, native binary64
and boolean tags, exact `(batch,input_width)` nominal shape and `(batch,)`
guard/mask shapes, finite nonnegative guards, `active == (support_mass != 0)`,
`fallback` contained in `active`, zero inactive rows, and bit-exact
`final = fallback ? min(wide,streamed) : wide`.  The Conv decoder requires the
exact `(batch,input_width)` nominal shape, three `(batch,)` guards and masks,
finite nonnegative guards, component masks equal to their nonzero guards,
`active = channel_active OR accumulation_active`, and the exact
zero-preserving directed sum for the scalar guard.  A decoder is not a public
API and no copied, pickled or caller-reconstructed tuple may be accepted.

The first strict decoder candidate passed its submitted `25/25` suite but was
not frozen.  Independent raw-value mutations exposed two omitted Dense
relations: an active row could retain nonzero support while all three guards
were changed to zero, and a non-fallback streamed guard could differ from the
wide guard by one ULP.  A lifecycle probe also showed that the public decode
methods dynamically called `self._check_self()`, so a changed class method
could bypass the PID-before-lock check.  The corrected candidate was frozen
for the non-authoritative hidden-decoder phase before the later ufunc-instance
audit:

```
source SHA256  e9203e0da9e26f4a822b100c461f9371864b314e3be6ed46db336f80403d7966
test SHA256    5d958c488769e1d626ae62790965341aecfb0fe028744557c6b2d946200865a5
focused        30/30, fixed four-thread command
combined       63/63 with the frozen private kernel
systematic     5,007 Dense rows + 5,005 Conv rows, all fields bit-exact
scope          isolated candidate; proof_authority=False
```

Active Dense wide, streamed and final guards are now strictly positive;
non-fallback streamed and wide guards must match bit for bit; and all public
decode entry points call a closure-private port/PID/deadline check directly.
The deterministic fork test holds the decoder operation lock in another
thread before forking and still obtains `FORKED_PROCESS` without entering the
inherited lock.  Independent review additionally passed all 41 pre/post
factory dependency bindings, 108 malformed tag/rank/shape/dtype/payload
cases, 383 extreme Dense outputs including a legitimate active guard of
`5e-324`, and 256 wide-exponent Conv outputs.  Recursive disassembly of 34
code objects remains free of global load/store/delete and runtime imports.
The complete fixed-four-thread V5.1a, controlled-audit, material-cache,
private-kernel and decoder closure then passed `198/198` in `13.599 s`.
Production integration must additionally keep the port type and functions
unreachable, bind the exact error-class and complete closure function/code
fingerprints before and after each atomic operation, and translate any
unrecognised exception to an integrated fail-closed result.

The later `ufunc.reduce` finding reopens this `e920...` decoder baseline as
well.  Its replacement is the `dec1271...` source recorded above: the two
logical reductions are descriptor-bound, persistent pre-factory instance
changes reject before dispatch, and post-factory changes are inert.  The
resulting complete closure is the `202/202` run, not the superseded
`198/198` run.

Sparse Conv remains on the frozen V3 componentwise path.  The first integrated
candidate may pre-admit all six Conv physical materials because the controlled
topology exercises 25 dense-Conv events; the one-pair stop loss will measure
whether that setup cost is justified.  It may not respond to a poor timing by
weakening validation.  If setup dominates, the next separately reviewed
candidate must remove duplicate public preparation between registry and
private-kernel admission while retaining an independent raw rebuild; it may
not make admission query-dependent or reuse material across frames.

The generic abstract-domain operator audit is instantiated for V5.1b as an
additional, non-substitutable promotion ladder:

1. Degenerate-box Dense, grouped/strided/padded/dilated Conv and residual-DAG
   toys must reproduce the real operator and frozen V5.1a bit for bit.
2. Every non-degenerate layer must be compared with Fraction/exact enumeration;
   the first width or lower-bound divergence is reported by layer rather than
   hidden in a final aggregate.
3. Stable-ReLU toys must preserve the exact affine Jacobian through Dense,
   Conv, ADD, reshape and residual fanout.
4. Small exact MILP phase enumeration must match the HybridZ encoding and
   independently enclose every candidate result.
5. The optimization may not change unstable-neuron or exact-binary counts;
   material aliases and physical-cache counts are audited separately.
6. Canonical output rows are checked against the raw VNNLIB Boolean asserts;
   cached stage or rival identities may not redirect a property.
7. The complete frozen V5.1a, strict replay, property/residual and mutation
   suites form the blast-radius test, with gains and regressions both recorded.
8. Only after steps 1--7 pass may one same-topology diagnostic pair run.
   A speedup below `1.50x` closes the candidate immediately; no real dataset is
   read until the stronger five-pair gate also passes.

## Innovation tracks and gates

### A. Proof-carrying constrained ReLU bounds

Current ReLU `l/u` values come from the unconstrained factor cube.  They ignore
already accumulated residual/equality/triangle rows.

Required toy:

```
h1 = ReLU(x)
h2 = ReLU(-x)
s  = h1 + h2 - 0.75
z  = ReLU(s),             x in [-1,1]
```

The true constrained upper bound of `s` is `0.25`; the cube gives `1.25`,
and the current triangle admits `z=0.625`.  A bound may replace the cube value
only when an original-CSR long-double certificate verifies it.  The hard gate
is `z_upper <= 0.2500001`, zero Fraction/phase under-approximation, and no
false improvement on tie/cancellation cases.

Real-network policy: GPU/solver candidates rank rows; CPU certificate checking
authorizes at most top `16 -> 32 -> 64 -> 128`.  Each expansion must close at
least 20% of the remaining gap or remove 20% of survivors.  Maximum targeted
tightening time is 20 seconds.

### B. Property/residual-aware graph reduction

The reduction target is rows and nonzeros, not merely neuron count.  Removed
neurons use a shared continuous factor across all fanout so residual
correlation is preserved.  Selection uses current-rival sensitivity,
triangle gap, residual fanout, and cost:

`score = sensitivity * triangle_gap * fractionality * fanout / added_cost`.

The toy gate requires exhaustive-phase enclosure, budget monotonicity, and
unchanged shared-fanout relationships.  A real build-only candidate must
reduce rows or nonzeros by at least 30% before it may consume a Gate1 verdict
run.

The first production-scale structural candidate is fused `ADD -> ReLU`:
`materialize_add=False` carries the shared affine expression directly into
the following ReLU predicate instead of inserting a redundant normalized ADD
frame.  On the baseline receipt, ADD materializations account for 81,920 of
98,970 rows; a controlled soundness suite and one build-only Gate1 receipt
must verify the actual nnz/runtime change before this switch can be frozen.

The controlled suite now passes: the residual toy reduces rows/nonzeros
`4/8 -> 2/4`, shared-fanout factor identity survives, and every exact ReLU
phase is enclosed without a tightness regression.  A fixed four-thread run
passed `41/41` in `3.678 s`: six residualization tests, eleven property-target
tests, eight operator residual-normal-form tests with 32 rational DAGs, and
sixteen ADD-fusion tests with 64 rational DAGs plus the
`1e16 + 1 - 1e16` oracle.  The real build-only comparison remains the
promotion condition; these toys do not authorize it by themselves.

The local HZ/CNN reduction literature supplies a useful score,
`column_L1(next_weight) * current_width`, together with an explicit interval
correction in the next layer.  A global application is not accepted here
because it discards residual/fanout correlation and can recreate the earlier
generator-retention regression.  The admissible candidate combines that cost
with the existing rival sensitivity and triangle gap, removes only
property-covered coordinates, and represents the correction with one shared
residual factor across every fanout.  Point boxes must add no correction;
Fraction DAG enumeration must enclose every phase; and a production build-only
probe must remove at least 30% of rows or nonzeros without widening any
registered rival before it can receive verdict time.

The sharp-HZ RLT result is also deliberately localized.  Its published global
complexity is exponential in the number of binary factors even at low order,
so full/global RLT remains prohibited.  A future `micro-RLT` toy may cover
only one property-relevant `ADD -> ReLU` residual join with at most four
selected binary factors.  It must retain the original relaxation, add at most
16 auxiliary rows, match exhaustive phase enumeration, and close at least 20%
of the measured local relaxation gap per added-cost tier.  The first tier
missing that threshold closes the track; it is never enabled merely because
the resulting LP is feasible.

### C. Persistent predicate-normal form

The network graph is immutable; CIFAR100/TinyImageNet differ only by 99/199
final rival rows.  Build/presolve once, change one objective/predicate, retain
the basis, and cover every rival exactly once.  A spy test already verifies
one `Highs()` construction for nine rivals.

For continuous HZ, the persistent LP is the only non-box proof engine.
Multiple-survivor cutoff rebuilding is skipped because its infeasibility
status has no independent SAFE authority.  A primal candidate must still pass
original-CSR validation and strict raw replay.

The historical cost evidence is unusually strong but is treated only as a
hypothesis source.  The sound6 run ended CIFAR100 at
`37 verified / 26 ADV / 2 UNKNOWN / 135 TIMEOUT` and TinyImageNet at
`156 verified / 36 ADV / 7 UNKNOWN / 1 TIMEOUT`; the prior three-certificate
admission experiment changed many formerly verified CIFAR cases into
timeouts, indicating proof-cost explosion rather than a new parser error.
In a later, separate implementation, one persistent HiGHS model with basis
reuse kept five fixed CIFAR verdicts unchanged while reducing their Tier-3 LP
loop by `7.8x` and end-to-end wall time by `4.68x`.  Those old paths are not
current authority.  They justify re-testing persistent multi-objective
execution only after the current root/session and independent original-CSR
certificate gates are frozen; they do not justify importing historical
verdicts or running a full benchmark.

### D. Motif-local sharpness repair

The controlled motif

```
h1=ReLU(x), h2=ReLU(-x), h3=ReLU(h1-h2), r=h1-h3
```

has exact `r=0` but independent triangles give `[-1/2,1/2]`.  The strict
symbolic motif identity adds one equality row/two nonzeros and closes the
range to `[0,0]`; an obvious intermediate complement identity gives zero
improvement and is rejected.  Production matching requires exact topology,
coordinates, coefficients, and biases—never approximate similarity.

Static stop-loss scan result: all three unique official ONNX models have zero
complete motifs, zero direct or BN/Gemm-folded complement pairs, zero exact
opposite affine channel pairs, and zero exact `+1/-1/0` affine weights.  The
nearest approximate opposite pair is still far outside roundoff scale.  This
track is therefore closed for production and retained only as a soundness
regression toy.

### E. Time-sliced dual candidates

A time-limited HiGHS dual is allowed as an arbitrary multiplier candidate.
Every candidate is projected to legal row signs and recomputed on the
original CSR with residual box correction.  Fair slices prevent one rival
from consuming the other 98/198 rivals' budget.  Promotion requires actual
certificate coverage; “solver reached time limit” remains UNKNOWN.

### F. Batched original-frame CUDA dual candidates

The CUDA candidate tensor is indexed directly by
`[rival, original_constraint_row]`, avoiding an unsound reinterpretation of
layerwise DualSolver `nu` values.  Projected multipliers minimize the
Lagrangian support in float64, but have no proof authority.  The existing
long-double residual-box checker alone may prune a rival.

The implementation is disabled by default and exposes explicit step, wall
time, learning-rate, and per-rival dual top-k caps.  An expired deadline
returns the zero candidate; unavailable CUDA is an explicit error, never a
silent CPU fallback.  The real stop-loss probe is limited to 3--8 iterations
and must fit inside the same 100-second end-to-end budget.

## Compute policy

- GPU: interval/dual backward scoring, batched rival sensitivity, optional
  attack candidates, and sparse candidate optimization.
- CPU: original-CSR certificate verification, strict parser/replay, small
  exact LP/MILP toys, and final receipt assembly.
- One worker uses at most four row workers and twenty total solver threads.
  All HiGHS instances in the process use the same five-thread global
  scheduler setting.
- No background full benchmark run; no hidden CPU fallback when CUDA is
  required.

## Promotion checklist

Before Gate6:

- all deterministic soundness/tightness tests pass;
- Gate1 total wall time below 100 seconds;
- all 99 rows covered by cube or independently checked receipts, or one
  strict-replayed counterexample;
- zero non-finite values, errors, replay conflicts, missing rows, or fallback;
- candidate mechanism has a frozen config and source/environment fingerprint.

Gate6, Gate14, Gate40, and full-400 results must be appended here with exact
receipts.  Reference S/U labels remain diagnostics only.
