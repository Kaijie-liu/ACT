# Later-affine coefficient references — bounded V2 controls

Frozen before implementation or execution, 2026-10-13. Parent execution is
`76991c194276cec805ee53429be027e1ef8f1998`. This is a new source-representation
contract, not a rerun of the closed endpoint/width or dense-prefix experiments.

## Decision and capacity gate

The existing dense prefix consumes 7,202,670 of 8,000,000 work counts, leaving
797,330. Replacing later affine expansion with references does not by itself
admit the complete dense query. The unchanged native preparation already needs
at least 796,896 coefficient conversions plus 3,726 node visits = 800,622.
Independently, the fixed zero-dual expert-0 endpoint reaches 157 dense stable
ReLU outputs: even excluding all other terms, 157×3072×3 = 1,446,912 counts
under the current exact pullback. These are static lower bounds, not measured
new calls. Do not run the dense source or query, increase caps, remove checking,
or adopt the older 96M endpoint composition budget.

This stage admits only implementation and finite correctness controls for a
source-driven representation that eliminates later affine expansion. It does
not admit another complete small-output optimization experiment. Full-query
capacity, convolution and trained-source correspondence remain separate gates.

## Mathematical contract

For parent outputs `(c_j,n_j)`, a node denotes one global factor coefficient
vector, not its center or a constraint rhs. For each exact source row `(w,b)`:

\[
c^*=b+\sum_j w_jc_j,\qquad s_n=\sum_{j:n_j=n}w_j.
\]

Create a new node with sorted nonzero `(n,s_n)` references. Compute the center
from every original row **before and independently of** merging references;
equal coefficient nodes may have unequal centers. Store the nearest binary64
center, and fresh private error/lift factors covering its exact rounding error.
All row errors precede all lifts. Referenced coefficients use exact source
weights: there is no intermediate float coefficient multiplication or its
roundoff-compensation tail. First input-diagonal affine layers retain V1's
checked native calculation and coefficient compensation. No default API changes.

For any factor in [-1,1], the recursive radius bound

\[
B_n=\|l_n\|_1+\sum_p |s_{np}|B_p
\]

is sound by the triangle inequality, even with shared ancestors. It need not
equal the expanded L1 norm. ReLU uses this bound with the existing checked
construction; V2 may change ReLU stability, factor inventory and relaxation
precision relative to V1. Therefore V1 is not a coefficient/bound-equality
oracle. The control oracle expands **the same V2** with exact arithmetic.

Each guard row references `+router_other - router_selected`, with exact merging,
and an outward-rounded center difference rhs. All unordered tie-legal pairs and
every selected-versus-unselected inequality remain separate obligations. Shared
input/router factors and expert-private ownership remain distinct. The checker
independently rebuilds references from the captured source and rejects arbitrary
equivalent-but-unregistered substitutions, extra factors, rows and cross-expert
references. Binary factors are still relaxed to [-1,1] for support.

With local matrix L and strictly lower triangular reference matrix R,

\[
C=(I-R)^{-1}L.
\]

A separate CPU operator evaluates positive topological levels for `Cx` and
negative levels for `C^T alpha`, propagating the **accumulated** adjoint. The old
one-level `R²=0` scatter is not generalized by merely relaxing its input check.
The operator is a candidate component, not an accepting proof checker; no GPU
is initialized and no numerical positive result is reported as a certificate.

## Frozen controls and execution

Use one existing small mixed captured fixture plus one finite deep fixture.
Deep fixture: two inputs, three experts, two output classes. Router dimensions
2→3→3→3; each expert 2→2→2→2→2. Include a ReLU after the second Linear and
consecutive affine layers on both sides. Weight rows use alternating signs and
dyadic `(1+(row+2*column+layer)%4)/4`; biases use `(row-layer)/10` stored as
binary64. Input center zero and radius one; existing selected-softmax top-2,
ANY_LEGAL_TOPK semantics remain. This source is synthetic, not a model-quality
or positive-output selection. Recipe/source hashes must be archived.

Required controls:

- same-node/different-center cancellation; zero and negative source weights;
- diamond shared-ancestor cancellation; triangle bound strictly above exact L1;
- nonzero center compensation and independent rhs; all-error-then-lift order;
- multiple consecutive affine/ReLU reference levels;
- source, expert, factor, guard, pair, reference sign/order/parent tampering;
- omitted/extra nodes and duties, source suffix/identity rejection;
- same-V2 exact expansion vs recursive forward and adjoint; CPU numeric operator
  vs that exact matrix on fixed bounded vectors (rtol/atol 1e-12), not acceptance;
- deadline, cumulative 8M work refusal and immutable checked-handle controls;
- stdlib-only relocated source checking with `-I -S`, no producer/native import;
- unchanged V1 source and endpoint regressions.

Per archived call: one CPU thread, ≤300s hard parent deadline, cumulative 8M
work; include fixture creation, construction, independent checking, operator
preparation, differential, serialization and terminal cost. Archive ≤16 MiB,
new directory `runs/hz_affine_references_20261013_r1`; preserve all failures.
Tests can use temporary files within /data1/Kane/MOE. Archive code before the
finite execution; no hidden rerun if a registered source fails. An offline
audit rechecks unchanged captured bytes independently and reports its own cost.

No new SAFE count, native-execution proof, GPU benefit, external victory or goal
completion can follow from these controls. Retain sealed requests and all G1–G6
as OPEN. Next work must address the complete-query resource bottleneck rather
than repeat the same small oracle or merely enlarge this control's dimensions.
