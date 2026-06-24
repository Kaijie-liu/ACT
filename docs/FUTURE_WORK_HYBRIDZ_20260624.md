# HybridZ Future Work After 2026-06-25 Soundfix

Canonical soundfix artifact:

`/data1/Kane/ICSE/act_hybridz_soundfix_20260625`

Sound headline:

`1763/2213 = 977 CERT + 786 ADV`, `P0=0`.

The preserved 2026-06-24 artifact copy is historical provenance:

`/data1/Kane/ICSE/act_hybridz_clean_20260624_cora25/FUTURE_WORK_HYBRIDZ_20260624.md`

Soundfix note: the old metaroom `100 CERT / 0 ADV` row is superseded by
`94 CERT / 1 ADV / 5 TIMEOUT`. The package runner now applies the correct
split-disjunct rule: any disjunct `ADV` makes the instance `ADV`, and `CERT`
requires every split disjunct to be certified.

This local copy records the same research direction in a more actionable
roadmap.  These are future directions only.  They are not part of the frozen
count unless a new full clean artifact is generated with `P0=0`.

## Ground Rules

Future work must keep the strict pure-HybridZ boundary:

- no input splitting;
- no random or adversarial input sampling counted as verification;
- no LP-relaxation witness promotion;
- no CROWN/backward rescue after HybridZ returns `UNKNOWN`;
- no ReLU triangle decision in the counted HybridZ path;
- no Gurobi-counted proof, though Gurobi may be used as a diagnostic oracle;
- no per-sample special configuration that cannot be applied benchmark-wide.

ORT/model replay remains an audit mechanism for returned ADV witnesses.  Its
wall time is audit time, not HybridZ verifier time.

## Current Sound Ranking Snapshot

| Bench | N | CERT | ADV | V+A | Unsolved | Rank | Gap To Best |
|---|---:|---:|---:|---:|---:|---:|---:|
| `safenlp_2024` | 1080 | 432 | 647 | 1079 | 1 | #2 | 1 |
| `metaroom_2023` | 100 | 94 | 1 | 95 | 5 | #1 | 0 |
| `sat_relu` | 100 | 50 | 50 | 100 | 0 | #1 tie | 0 |
| `malbeware` | 150 | 131 | 19 | 150 | 0 | #1 | 0 |
| `cersyve` | 12 | 5 | 6 | 11 | 1 | #1 | 0 |
| `acasxu_2023` | 186 | 86 | 34 | 120 | 66 | #5 | 66 |
| `linearizenn_2024` | 60 | 39 | 1 | 40 | 20 | #5 | 20 |
| `dist_shift_2023` | 72 | 70 | 0 | 70 | 2 | #1 | 0 |
| `tllverifybench_2023` | 32 | 5 | 12 | 17 | 15 | #3 | 13 |
| `cora_2024` | 180 | 19 | 6 | 25 | 155 | #1 | 0 |
| `relusplitter` | 220 | 41 | 2 | 43 | 177 | #3 | 70 |
| `cgan_2023` | 21 | 5 | 8 | 13 | 8 | #2 | 6 |

The most attractive future work is therefore not the same for every benchmark.
`dist_shift` is mostly an operator-tightness story.  `acasxu`,
`linearizenn`, and `relusplitter` are mixed-integer/exact-ReLU compression
stories.  `cgan` and `tllverify` are sparse representation and formulation
stories.

## Highest-Value Directions

### 1. Sound S-Curve Tightening

Target benches: `dist_shift_2023`, any future sigmoid/tanh-heavy benchmark.

The clearest non-binary wall is still S-curve over-approximation.  The frozen
result already shows that a benchmark-wide sound sigmoid/tanh policy can move
`dist_shift` strongly, and only two rows remain unresolved.

Promising work:

- deterministic `k` selection from input bounds, with `k=2` as the current
  conservative default unless a full clean rerun proves a better default;
- graph-domain cuts that are selected by a fixed global policy, not by iid;
- pair-correlation cuts only under an explicit nnz and wall-time budget;
- separate reporting of CERT gain, MILP nnz growth, and wall-time change.

Acceptance test:

- full `dist_shift_2023` clean rerun;
- no P0;
- same config for the whole benchmark;
- explicit comparison against the frozen `70/72`.

### 2. Selective Exact-Valid Cuts

Target benches: `cgan_2023`, `relusplitter`, `acasxu_2023`,
`linearizenn_2024`.

Blanket cuts are not a valid default for large sparse rows.  cGAN showed that
all ReLU valid cuts can be exact but matrix-negative, growing the sparse MILP
from about `25.99M` nnz to about `76.50M` nnz before substitution.  The right
direction is a budgeted cut scheduler:

- estimate local support and nnz growth before adding a cut;
- accept cuts by bound improvement per added nnz;
- cap total added rows and total added nnz per benchmark profile;
- log accepted and rejected cuts for reproducibility;
- keep the rule benchmark-wide, not per-sample.

This keeps the method pure-HybridZ because the cuts must be valid for the exact
HZ formulation, but avoids making the sparse system too dense to solve.

### 3. Exact ReLU Compression

Target benches: `acasxu_2023`, `linearizenn_2024`, `relusplitter`,
eventually CIFAR-style models.

The useful path is not ReLU triangle.  It is exact compression of the HZ/MIP
representation:

- remove stable phases using sound forward HZ bounds;
- eliminate redundant exact ReLU auxiliaries when equality structure proves
  equivalence;
- project or merge exact-ReLU variables only when the represented set is
  unchanged;
- reduce `ng/nb/nc` before final MILP construction;
- add toy exact-MILP parity tests for every compression rule.

This is the main principled path for binary-wall benchmarks where the current
solver sees too many exact phase decisions.

### 4. Formulation-Aware Sparse HZ Presolve

Target benches: `cgan_2023`, `tllverifybench_2023`, large residual/sparse
models.

Generic connected-component splitting is not enough when the hard objective
row connects the system into one component.  Future presolve should use HZ and
operator structure:

- preserve sparse equality form when substitution would densify the model;
- use Schur/block elimination only when row density stays controlled;
- detect local convolutional or residual blocks before flattening;
- simplify generator/equality rows algebraically without changing the HZ set;
- compare every presolve rule against a small exact dense oracle.

The goal is to make large sparse systems smaller without losing the sparse
structure that made them solvable.

### 5. Open-Source Solver Portfolio At The HZ Layer

Target benches: `acasxu_2023`, `linearizenn_2024`, `relusplitter`,
`tllverifybench_2023`, `cgan_2023`.

HybridZ should stay open-source in counted results.  Gurobi can diagnose
whether a row is formulation-hard or solver-hard, but the accepted method
should use open solvers:

- root LP and presolve profile first;
- choose HiGHS, SCIP, or another open MIP backend by integer count, nnz, row
  density, and equality structure;
- share HZ-derived incumbents or phase hints only if they are produced inside
  the counted pure-HZ solve;
- count the full portfolio wall as verification time;
- keep portfolio choices benchmark-family-wide.

This turns "try another solver" into a reproducible HybridZ scheduling policy.

### 6. GPU-Resident Sparse Propagation

Target benches: large sparse/cnn-like cases, especially cGAN/TLL/CIFAR-style
probes.

The solver itself is mostly CPU-side today, but propagation and row
materialization can benefit from GPU batching:

- batch sparse affine/conv row construction on GPU;
- transfer only final margin rows and needed linking rows back to CPU;
- use chunked transfers so VS Code and system memory stay safe;
- keep exact semantics identical to CPU sparse propagation.

This is an engineering speed direction, not a mathematical relaxation.

### 7. Lazy / Matrix-Free Exact HZ For CIFAR-Style Models

Target benches: CIFAR and other high-dimensional CNNs.

CIFAR remains a useful long-term stress test for HybridZ itself.  The correct
direction stays exact HZ:

- represent the same 6-tuple lazily;
- materialize only selected preactivation rows, final margin rows, and linking
  equality rows;
- make Conv/Dense/Add/ReLU row-only where possible;
- return `UNKNOWN` on exact-capacity failure instead of silently dropping to
  interval;
- first use CIFAR as a census of representation wall versus binary-MIP wall.

This is future work because the frozen benchmark win should not be destabilized
by a large representation rewrite.

### 8. Sparse Operator Coverage

Target benches: future VNN-COMP families, CIFAR-style probes, models with
residual/gather/concat/constant-matmul patterns.

Good next operators:

- exact constant-side `MATMUL` as sparse affine propagation;
- exact residual/add/sub/concat/gather handling with metadata invariants;
- exact average-pool and supported max-pool handling, or explicit
  `UNKNOWN/unsupported` if exact semantics are not implemented;
- robust row materialization tests for each operator.

Operators to keep out of counted results until re-audited:

- var-var `MATMUL` product relaxations;
- softmax simplex/ratio relaxations;
- any non-affine helper whose sound HZ semantics are not proven.

### 9. Mainline Reproducibility And Data Plumbing

Target: productization, not new counts.

Some useful behavior still depends on local orchestration or environment
details.  Future cleanup should make the normal ACT entrypoints enough:

- `--solvers hybridz` should run strict HybridZ only;
- `--verify hybridz-benchmark` should reproduce frozen profiles;
- benchmark profiles should live in backend code;
- external VNN-COMP benchmark roots should be discoverable without copying
  datasets into the ACT tree;
- frozen-match mode should fail loudly on missing data or profile drift;
- reusable script behavior should migrate into mainline tests before scripts
  are deleted.

## Benchmark-Specific Backlog

- `safenlp_2024`: frozen artifact has one remaining UNKNOWN (`iid454`).
  The earlier frontend/productization drift on `iid844` has been recovered in a
  2026-06-24 package-frontend recheck: benchmark-wide `normal_pscost1` proves
  `CERT` with `hz_timeout_s=19.0` and `wall_s=16.53s`.  Future work here is
  engineering acceptance, not a new algorithm: run a clean complete frozen-suite
  `--hybridz-require-frozen-match` pass and keep the profile/env metadata
  stable.
- `cersyve`: one remaining UNKNOWN.  Useful small test for open solver
  portfolio and exact cut scheduling.
- `dist_shift_2023`: highest-return operator-tightness target; remaining
  rows are S-curve tail cases.
- `acasxu_2023`: mixed-integer wall; target exact ReLU compression and open
  solver scheduling.
- `linearizenn_2024`: binary/MIP wall; target exact ReLU compression and
  stronger forward HZ bounds.
- `relusplitter`: binary/MIP wall; projected/compressed exact-ReLU formulation
  is the most principled path.
- `tllverifybench_2023`: large sparse HZ/equality system; target
  formulation-aware sparse presolve.
- `cgan_2023`: sparse representation/MIP wall; avoid blanket cuts and dense
  substitution.
- `cora_2024`: mostly official-wall timeouts; classify model families and
  keep any policy family-wide, not per-instance.

## Directions To Avoid As Defaults

- blanket ReLU cuts on large sparse rows;
- equality substitution that densifies sparse HZ systems;
- LP incumbents counted as ADV;
- ORT replay counted as verification;
- input split;
- sampling/PGD counted as verification;
- CROWN/backward rescue after HybridZ returns `UNKNOWN`;
- per-sample hand-tuned configurations;
- post-HybridZ rescue counted as HybridZ capability.

## Acceptance Checklist For Future Improvements

Accept a future improvement only with:

- full clean rerun for the affected benchmark;
- `P0=0`;
- per-row detail CSV and benchmark summary CSV;
- manifest/checksum and source snapshot;
- explicit CERT/ADV/UNKNOWN/TIMEOUT deltas against this freeze;
- wall-time and memory notes;
- statement whether the change is operator-side, solver-side, or
  orchestration-only;
- benchmark-wide configuration, not iid-specific rescue.
