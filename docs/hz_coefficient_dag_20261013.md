# Coefficient references and exact adjoint support — bounded result

2026-10-13. [Protocol](hz_coefficient_dag_design_20261013.md) committed/pushed as
`a05a586ba6aae975fa204a7e178db127bad4fa1c` before implementation/execution.
**Given-row algebra and an actual small SparseHZono interface pass. No new
trained model, route-changing certificate, external win or GPU run. G1–G6 OPEN.**

## What changed in HybridZ support

[Native opt-in interface](../act/back_end/moe/coefficient_support.py) accepts an
actual SparseHZono plus explicit factor identities and earlier-row compression
hints. Hints do not authorize dropping constraints: an independent forward-path
bridge checks every original row, rhs and coefficient against the resulting
immutable representation. Continuous and binary local columns have an explicit
global mapping. Source equality is bound to the actual comparison traversal,
not inferred only from equal before/after hashes.

The [stdlib support component](../scoped_source/coefficient_dag.py) represents a
coefficient vector as a local sparse row plus scaled earlier vectors. Original
root identities, kinds and rhs remain distinct. It injects objective and dual
weights, propagates scalar adjoints backwards, and consumes each surviving local
vector once. Shared-factor residuals combine before one box support operation;
expert-private coordinates remain distinct. Domain ancestry and guard scope are
checked before use, including subgraphs whose weights later cancel.

This is a representation/support-operation change, not merely a cache or a new
wrapper. It is not claimed as a new general abstract domain, a novelty-priority
result or a stronger relaxation. For the same candidate it must return exactly
the expanded bound. Default production entrypoints remain unchanged.

## Executions and review repairs

Local archives (not committed raw objects):

- `/data1/Kane/MOE/runs/hz_coefficient_dag_20261013_r1`
- `/data1/Kane/MOE/runs/hz_coefficient_dag_20261013_recheck_r2`

R1 ran 16 stdlib controls, 4 actual-SparseHZono tests, small and fixed dense
objects, and a relocated `-I -S` check. Four subprocesses completed within their
300-second caps. The mathematical objects and all R1 receipts/code stay intact.

Subsequent read-only review found and led to repairs of mutable query/candidate
intake, CSR column/pointer validation, native whole-row copying/expanded ancestor
caches, pre-multiplication hint bounds, and the native comparison-pass identity.
The final bridge uses one row accumulator with charged recursive paths, not a
cache of expanded ancestors. The producer reads current/reference native rows
on demand. It still builds bounded local graph records; it is not a fully
streaming captured-network converter.

R2 retains new code hashes and rechecks the unchanged R1 bytes. It performs
20 stdlib tests, 7 native tests and 42 relevant existing regressions: **69 pass**.
Added controls include query/dual alias pollution, mutation-and-restore during
native comparison, stale and uncached malformed CSR metadata, actual adjoint
bit growth, deadline/work exhaustion during accumulation, and early hint refusal.
An independent stdlib program, which imports neither producer nor DAG checker,
recomputes every dense coefficient and all three saved residuals/bounds.
Final-code relocation also passes without NumPy, SciPy, Torch or a solver.
Read-only review of the final repairs found no additional blocking issue; this
AI review is not the required independent human technical review.

## Exact observations

The fixed dense given-row duplication object has 256 vectors of width 3072,
their 256 negative references, and separate rhs values. The pinned R1 executable
uses all output weights `1/8`, offset zero, ordinary equality multipliers
`(r % 3 + 1)/16`, and cancellation multipliers `-1/8`. These were in its captured
code before execution; the prose freeze described the candidates but did not
spell out their numbers. No post-result selection or candidate optimization ran.

| Quantity | Observation |
|---|---:|
| Distinct active factors | 3072 |
| Nodes / original roots / reference edges | 512 / 512 / 256 |
| Stored local coefficients | 786432 |
| Original expanded coefficient occurrences | 1572864 |
| Serialized dense graph | 11410675 bytes |
| Ordinary accumulation: local / edge visits | 786432 / 256 |
| Cancellation accumulation: local / edge visits | 0 / 256 |
| Validation + both checker calls, cumulative work units | 3164417 / 8000000 |
| Ordinary exact lower bound | `-19195/64` |
| Cancellation exact lower bound | `-261/2` |

The cancelled call skips accumulation only after all 786432 local coefficients
have been checked. Both results equal direct expanded-row computation exactly.
The small three-expert controls cover all three unordered pairs, nonzero-center
ReLU-style equality signs, stable aliases, diamonds, shared cancellation and
private-factor separation. The actual SparseHZono control includes the exact
unstable-ReLU auxiliary equation and distinct rhs on shared vectors.

**Important control limitation:** the large object's equalities are infeasible.
`W[r+62]=W[r]`, while the corresponding rhs differ by `93/16`. This was found in
review, independently checked and disclosed without changing the frozen recipe.
It does not invalidate residual/constant equivalence or occurrence counts, but
it cannot establish capacity for a nonempty real HZ, a correct ReLU/source
lowering, or output safety. Both recorded candidate bounds are negative anyway;
no network UNSAFE conclusion follows. The actual small native object is a
separate control and cannot be spliced into a large feasible/source claim.

## Costs and preservation

R1 dense component: generation 0.818 s; JSON encoding 0.142 s; graph validation
3.625 s; file write 0.005 s; ordinary exact accumulation 2.145 s; cancelling
accumulation 0.008 s; independently integer-scaled references 0.300/0.285 s.
Dense component total was 7.334 s. R1 parent observed 9.489 s before terminal;
R2 parent observed 9.016 s before terminal. R2 combines about 11.8 MB of retained
stage files, below 64 MiB. No evidence or prior failed/refused attempt was deleted.

These are component/control costs, not end-to-end acceleration. In particular,
the independent reference exploits a known common integer denominator; its
timings are not a same-algorithm baseline. Its operation counts are reported
separately, not mixed into the support checker's 8M counter. Both share the
worker's absolute deadline. Each given object has one cumulative checker budget;
there is no reset per row, candidate or phase.

## Continue / stop decision

Stop this given-row width/candidate timing study. A finite coefficient-sharing
mechanism now exists, with unchanged exact supports and an actual small HZ bridge.
Next connect captured tensor rows to an explicit source-checked expression
graph, starting with a small complete source and the fixed large affine/ReLU
prefix, preserving layer-wide error/lift allocation and independent rhs. Do not
route it back through the old full-layer list or eagerly expand all references.
Source operators must independently check the coefficient-reference construction;
a graph hash/native given-row bridge alone does not establish network lowering.

The old Store nnz, old source affine cap, byte-aware paging, later-layer factor
growth, all route/property endpoint consumption and convolution contracts remain
separate admission questions. Do not silently raise caps, assume this small
native adapter accepts full trained sources, or start real solving/GPU retries.
No existing scientific result or submission guarantee is upgraded.

See [compact record](hz_coefficient_dag_20261013_r2.json) for archived identities.
