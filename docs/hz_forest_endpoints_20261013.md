# Checked source references now reach complete output obligations

Protocol `4ad52becf` was committed and pushed before implementation. Code
`77784b5f5` was tested, committed and pushed before the once-only archive.
The [frozen protocol](hz_forest_endpoints_design_20261013.md) and
[compact results](hz_forest_endpoints_20261013.json) bind the exact sources,
code, complete proof and independent audit. Raw archive:
`/data1/Kane/MOE/runs/hz_forest_endpoints_20261013_r1`.

## Result and scope

The opt-in HybridZ consumer now uses the independently checked source forest
directly. It does not expand that forest into an eager Store or rebuild a
coefficient matrix for every pair. The four small captured sources cover the
complete source→affine/ReLU→separate guards→all output endpoint chain.

| Source / tile width | Checked endpoints | Positive duties / required | Full result | Parent-observed seconds | Cumulative work / 8M |
|---|---:|---:|---|---:|---:|
| small40 / 8 | 2 | 0 / 1 | UNKNOWN_NONPOSITIVE | 1.814430 | 3,605 |
| mixed / 8 | 6 | 3 / 3 | checked captured-real-source positive | 1.888089 | 11,981 |
| guarded / 8 | 24 | 12 / 12 | checked captured-real-source positive | 1.870634 | 8,201 |
| half / 8 | 5 | 3 / 3 | checked captured-real-source positive | 1.928173 | 4,193 |
| guarded / 1 | 24 | 12 / 12 | width differential, same result | 2.132978 | 11,162 |

The last row repeats a control; it is not another independent positive source.
There are 37 endpoints in the four distinct requests and 61 including that
differential. Every candidate's exact global residual and lower bound equals
the test-only old expanded representation. No missing endpoint was discarded.
The nonpositive result is not a counterexample or proof of unsafety.

The `half` router is constant [0,1,0]; pair (0,2) has an empty guard. Its large
positive bound is vacuous for that pair, not an improved nonempty safety margin.
The other required pairs close. These are continuity/functionality controls,
not new real-model gains or a comparison against an external verifier.

## What changed inside the support algorithm

The [CPU candidate operator](../act/back_end/moe/forest_endpoints.py) stores one
sparse local coefficient bank L. The checked source format permits a reference
only from an unstable ReLU equality to an earlier, reference-free output. This
is checked explicitly, giving C=(I+R)L with R²=0. Forward products use Lx and
reference scatter; transpose products use Lᵀ(I+Rᵀ)α. No expanded C is built.
This is a restricted source operator, not a shortcut for arbitrary DAGs.

Here “one bank” describes the source representation and one rational-to-native
conversion, not one physical tensor allocation. A post-run CPU storage inspection
on torch 2.9.1+cu128 found that sparse `transpose` copies its value storage.
The archived implementation therefore holds local L **and** its native transpose;
the source forest still has one local bank and no expanded constraint C. The
misleading “index/value view” code comment is corrected after execution, with no
operation changed or numerical rerun. No single-allocation memory claim is made.

The [independent exact consumer](../scoped_source/forest_endpoints.py) keeps
each original rhs and row identity. A query includes common ancestry once,
one pair guard, and the two expert suffixes; guards for different pairs are
never conjoined. Shared factors retain one global coordinate, while private
expert factors remain distinct. Intermediate and inactive source inventory is
still checked even when it is not a final objective root.

For a property objective with node weights β and original constant d, equality
duals z and inequality duals y≤0, inject the negative constraint duals into
the node weights α. The checker computes

$$L_{\rm checked}=d+b^Ty+h^Tz-
\left\|L^T(I+R^T)\alpha\right\|_1.$$

All coefficients and residual arithmetic in this expression are rational;
shared residuals combine **before** the absolute value. For every feasible
factor vector in [-1,1], the dual terms plus this box support are a lower
bound. Relaxing binary factors to that box enlarges the domain. Distinct rhs
are not merged merely because coefficient roots alias.

Every unordered pair and every non-label classification property is covered.
The shared router difference gives a sound sign interval for the selected
softmax weight. Both distinct endpoints are checked; linearity in that weight
then covers its interval. Missing evidence is UNKNOWN. Acceptance still needs
every required minimum strictly above 1e-7; negative bounds never imply UNSAFE.

Only `check_proof` constructs the full plan and issues the checked-positive
terminal. An untrusted aggregation helper cannot do so. The source handle is
privately owned and immutable, including deletion refusal; portable verification
requires an external bundle anchor, not a caller-supplied receipt or `None`.

## Validation and cost accounting

82 tests pass: 20 new consumer controls, 17 source controls, 3 source-archive
controls, 20 coefficient-reference controls and 22 tensor-row controls.
They cover mutation/identity, complete pair/property inventories, private
factors, independent forward/transpose oracle, same-candidate residuals,
zero fallback, weak nonpositive duals, missing evidence, deadline and work
refusal, partial publication, finite native inputs and bounded proof bytes.

The initial two unit runs had one invalid expectation: the mixed source was
assumed to remain UNKNOWN but actually checked positive. Its source and result
were not altered. The negative-evidence test now supplies fixed weak duals.
The archive retains that development correction and pre-execution review
repairs; no frozen execution was replaced or rerun.

All five numerical calls use one cumulative 8M counter and one at-most-300s
deadline per call. Parent time includes startup, imports and receipt publication.
Source capture/build/check, planning, candidates/exact selection, serialization,
full endpoint check and the test-only eager differential are separately recorded.
Native affine and candidate details are **nested**, not added twice.

The full seven-call control archive took 12.530461 seconds before terminal
publication, including the 82 tests and relocation. The independent offline
audit took another 0.057285 seconds, reported separately. Logical archive size
including audit is 946,645 bytes (about 2.4 MiB allocated blocks). About 87 GiB
was free on the shared filesystem; no scientific evidence was deleted.

The width-1/8 results match exactly, but these single tiny calls do not establish
a production speedup. Their parent costs are dominated by imports; the eager
differential is a control cost, not part of a proposed deployment algorithm.

Four copied bundles passed `python -I -S` source-plus-endpoint verification using
only copied checker code and source/proof files. No checkpoint, old run path,
NumPy, SciPy, torch or solver was loaded in that check. The offline auditor then
rechecked all five original bundles, frozen anchors, original/relocated code,
partial inventories, complete bounds and costs. This is an independent exact
arithmetic implementation check, not an independent human review.

## Boundary and next decision

These are synthetic captured real-arithmetic declarations. Correspondence to a
trained checkpoint/intended deployment remains an assumption, and native floating
execution is not proved. No convolution, real model, CUDA or external comparison
ran. All G1–G6 gates remain OPEN; the historical 23 source-incomplete gains remain
unchanged. The GPU subgoal is still required, not satisfied by the CPU operator.

Stop these small-source and width controls. Existing dense-prefix records already
show the next eager `live_parent` expansion needs at least 965,318 counts, exceeding
the remaining 797,330; following affine reference arithmetic is also too costly.
Do not rerun it, increase caps, or declare this small complete chain a large one.
The next bounded mechanism question is whether later affine layers can retain
source-weight/coefficient references without expanded intermediates, while
checking their compensation and a sound range norm. Overlapping references need
a valid norm contract, not an assumed exact sum. Full source/query capacity must
be assessed before a new control freeze. No GPU retry or real solve is authorized.
