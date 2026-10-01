# Source integration with checked rowwise HybridZ enclosure

The [frozen source integration](hz_lifted_source_design_20261001.md) is complete
within its finite scope. Exact source-step checks now precede an independently
checked binary64 enclosure, actual SparseHZono instantiation and fresh endpoint
proofs. This closes the integration gap left by the
[given-reference kernel](hz_binary64_enclosure_20261001.md), not real-model
capacity, external competitiveness or native floating execution.

## Implementation and guarantee

The existing local kernel admits at most eight referenced factors and at most
four rows each of outputs, equalities and inequalities. The old source controls
already exceed those row counts. The new
[row producer](../scoped_source/hz_row_enclosure.py) projects each output or
constraint onto exactly its nonzero original factors, invokes that unchanged
kernel, then assembles one global HZ. It keeps **all** original factors, even
unused ones, and allocates distinct compensation factors per affected row.

The [independent composition checker](../scoped_source/check_hz_row_enclosure.py)
reconstructs local projections and the complete global matrix, including zero
entries of new columns outside their own row. Therefore the per-row assignment
extensions coexist for one shared original input. This is decomposition of the
same L1 enclosure rule, not copying the original input into independent row boxes
or claiming a tighter relaxation. On all eight old kernel patterns, the entire
resulting target equals the old kernel's target, including deterministic IDs.

A nine-original-factor sparse control with six inequalities verifies that global
width and local row width are distinct. A nine-factor **individual row** refuses.
Global reference and target still obey the old 128-factor/128-output/256-constraint
source limits. No cap was raised, no dense row was split into a new relaxation.

The [source producer](../scoped_source/hz_lifted_source.py) and
[source checker](../scoped_source/check_hz_lifted_source.py) connect:

1. Checked input enclosure and declared factor ownership.
2. Actual sparse affine nominal, original exact affine-error reference, then lift.
3. Exact registered ReLU equations, explicit blocked-row permutation, then lift.
4. Exact conditional pair guards from the checked router, then lift once into the
   common expert entry. These are conditional guards, not globally redundant rows.
5. Private expert factors, sign-derived gate intervals and all fresh endpoint duties.

ReLU here uses the exact reference construction followed by actual SparseHZono
instantiation; it does **not** run the old floating ReLU kernel and declare its
coefficients exact. The checker never calls the source/lift producer or numerical
optimizer. All unordered pairs and all class properties remain; no route is
excluded. The support layer explicitly relaxes binary factors as before.

## Results on the unchanged declarations

The four sources and their hashes are unchanged from the prior representation
controls. Each produces a new request identity and fresh 128-step CPU candidates;
no old matrix certificate is imported. The acceptance threshold remains `1e-7`.

| Declaration | Positive / required properties | Checked endpoints | Checked row lifts | Nonzero lift compensations |
|---|---:|---:|---:|---:|
| weighted_sign | 3 / 3 | 6 | 169 | 0 |
| tied_partial_reuse | 18 / 18 | 18 | 426 | 0 |
| unsafe_tied | 0 / 6 | 6 | 167 | 0 |
| unresolved_sign | 6 / 6 | 12 | 219 | 0 |

These are historical fixture names, not promised outcomes. The incomplete
derived package remains `UNKNOWN_MISSING_EVIDENCE`. The complete nonpositive
control does not establish network unsafety. All 33 property-level aggregate
results match the old source connection; this is a regression comparison, not
reuse of its certificates or a new coverage gain.

All four full declarations happen to have representable references: **zero
nonzero lift compensations**. Their success establishes source-chain integration,
not an observed real-model failure repaired by rounding. Three separately frozen
operator controls exercise the missing cases:

- Affine: original exact error `2^-104+2^-200` is checked, then its storage lift
  adds one new compensation factor.
- ReLU: binary64 input coefficients give nonrepresentable exact output/equality
  coefficients; the checked reference lift adds two factors.
- Guard: the exact RHS `1−2^-54` is weakened outward without a new factor.

These start from supplied operator inputs. They are not three additional complete
MoE source proofs. Old proof matrices are neither modified nor retroactively fixed.

## Controls and independent recheck

Final R3 passes **18/18 frozen groups** and the saved-evidence audit. It independently
checks five source packages (four complete, one specified partial), eight local
kernel differentials, the sparse global composition and three operator bridges.
Twenty-two negative queries are reconstructed by fixed key from the bound normal
objects; their concrete input hashes and actual refusal reasons are rechecked.
One rejected query cannot replace another merely by updating an archive hash.
Only the explicitly expired control may count TimeoutError as its intended refusal.

Runtime mutation, producer expiry and snapshot-over-deadline injections are test
controls, not independently replayed static mathematical proofs. The distinction
is explicit. The old mathematical, support, device, propagation, intake and
documentation regressions pass **128/128**.

R1 is preserved before deadline-tail, failure-accounting and dependency/negative
inventory hardening. R2 is preserved before the final unexpected-timeout refusal
rule. Their compact checks describe their then-current audits, not the stronger
R3 audit. All four normal source packages are identical across R1/R2/R3; no source,
range, objective, candidate algorithm or positive threshold was tuned.

Reports: [R1](hz_lifted_source_20261001_r1.json),
[R2](hz_lifted_source_20261001_r2.json), [final R3](hz_lifted_source_20261001_r3.json).
Raw directories remain under `/data1/Kane/MOE/baseline_runs/` with matching stems.
The three directories total 8,296,383 logical file bytes; no evidence was deleted.

## Cost and trust boundary

R3's complete control suite took about 3.253 seconds. The four generation,
serialization and independent-check intervals were approximately 0.229, 0.351,
0.161 and 0.209 seconds respectively. The archive separates source creation,
construction, candidate generation, serialization and checking. Imports and test
orchestration are outside these per-source intervals but inside the suite clock.
Failure records retain total elapsed time, the open stage, completed components
and the unallocated remainder (which can also include orchestration overhead).

These are tiny, cooperative-deadline controls, **not** end-to-end production timing,
a new hard-budget supervisor or GPU speedup. There were zero native optimizer
calls, zero CUDA calls and zero real-model requests. This checker still uses the
repository import environment; it is not a new portable/clean-environment claim.

Accepted positives concern the fixed declared real graphs. Declaration-to-intended-
program correspondence and checker correctness remain trusted; native PyTorch/CUDA
execution is not proved. Real-model dimensions/operators and dense rows remain
outside admission. Historical external losses and source-gap findings are unchanged.
All G1–G6 remain OPEN.

## Next decision

Do not rerun these fixtures to seek another positive count, or add a wrapper as a
substitute for capacity. The next algorithm question is shared/private block
representation and support capacity: preserve a single input/router interface,
keep expert-private factors separate, and identify which repeated row/constraint
work can be reduced without weakening the proof contract. Freeze this as its own
factor before any new capacity execution. A dense row exceeding the current local
bound cannot be admitted by relabeling it sparse. The new source path also needs
separately bound full-budget/portable integration before any real-model use.

Audit command, using the existing environment:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -B -m scripts.run_hz_lifted_source audit /data1/Kane/MOE/baseline_runs/hz_lifted_source_20261001_r3
```
