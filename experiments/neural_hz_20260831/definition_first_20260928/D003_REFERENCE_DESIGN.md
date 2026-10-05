# D003 — Default-off exact reference for phase-dependent Neural-HZ

Date: 2026-09-28; branch `redu-hz`, commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Design fixed before implementation. Execution requires a separate final
preregistration and source freeze. No benchmark or production admission.

## Mathematical scope

Implement D001/D002's exact continuous-affine/Boolean-dependent reference
language on ordinary mixed affine/ReLU/residual graphs, retaining the whole
input/phase/output relation. This is a semantic reference, not a claim of a
new useful abstraction, faster solver or improved benchmark score.

The only expression nodes are original continuous input, original free binary
input, rational affine combinations, and a new ReLU gate `beta*f`. Every ReLU
introduces and retains one distinct binary with guard `(2*beta-1)*f>=0`, even
when stable. All original EQ/LE predicates and every original input are kept.
No products of continuous expressions, arbitrary activation, phase search,
binary deletion, approximate identity detection or hidden solver is supported.

A builder owns one shared input frame and an append-only expression DAG.
Branch handles must belong to the same builder. Sealing produces an immutable
element containing ALL nodes, all phase records, all explicit predicates and
the selected output handles; it closes the builder. Unused original guards
and nodes are not filtered out. Original input binary factors have no invented
ReLU guard. The native compiler retains them all.

## API contract for this isolated version

`make_builder(n_inputs, n_binary=0, *, enabled=False)` returns None unless
strictly opted in with True. The builder exposes `inputs()` and `binary_inputs()`
(the latter are 0/1, not signed), `affine(vector, weights, bias)`, `relu(vector)`,
`add(left,right)`, `concat(*vectors)`, `constrain(vector, relation, rhs)`, and
`freeze(outputs)`. Relation is `eq` or `le`; each vector component is compared
to its corresponding rhs. Coefficients/inputs are exact ints or Fractions;
floats and bool-as-number are rejected. Original continuous inputs are [-1,1].

The frozen element exposes `evaluate(inputs,bits)`, returning all node values,
selected outputs, original input, and a feasibility flag checking every box,
bit, guard and predicate. Wrong arity or invalid scalar types reject. Feasible
assignments are diagnostic mathematical witnesses only, never network ADVs.
It also exposes `lower()` to the exact finite-binary linear reference below.

## Complete terminal reference

Native variables are all p original continuous inputs plus ONE continuous
value per ReLU gate, followed by ALL original/free and ReLU binary variables.
Affine nodes are flattened as exact sparse linear forms; each unique gate
already has a named value, avoiding polynomial expansion. Bounds are computed
soundly by rational interval propagation, not a solver or phase enumeration.

For every gate with preactivation f and certified l<=f<=u emit four rows:

    r>=0; r>=f; r<=u*beta; r<=f-l*(1-beta).

These retain both legal phases when f=0 and work for stable ranges as well.
They are equivalent to r=beta*f plus its full sign guard. Append every EQ/LE
predicate, including predicates on intermediates. Expose full linear forms
for all node values and outputs, input bounds, binary identities and gate
mapping, so independent tests can reconstruct and check all original values.
Use exact sparse rows `(terms, rhs, relation)` with <= / = conventions.

Lowered object methods: `assignment(evaluation,bits)` returns the full native
assignment; `satisfies(assignment)` checks all bounds/integrality and rows;
`output_values(assignment)` evaluates its output forms. Counters must include
p, all bits, all gates, all rows and all row coefficients, without refunds for
an unconsumed guard. No runtime solver or claim of an ACT-native accepted HZ.

This complete reference intentionally restores the D002 rank bill. It supplies
a fair unnormalized shared-circuit baseline for later quotient rules; a pass
only means the candidate definition's reference semantics works.

### Exact relation argument (not inferred from finite tests)

Fix the same original continuous input and the same ordered vector of all
original/free and gate bits. Inputs and affine nodes have one determined value
by topological induction. For a guarded gate, beta=0 enforces f<=0 and r=0;
beta=1 enforces f>=0 and r=f. Thus the ordinary max-ReLU value is obtained
without identifying the two legal phase assignments at f=0. Affine, residual
and Concat use only values from the same frame, so the induction also preserves
every intermediate predicate and original input. With no gates, an ordinary
rational HZ embeds as c+Gc*xi+Gb*(2*beta-1), with its original predicates.

Sound interval propagation bounds every feasible preactivation, even when
correlations are ignored. In the linear reference, beta=0 makes the first and
third rows force r=0; the second forces f<=0 and the fourth f>=l. Beta=1 makes
the second and fourth force r=f; the first forces f>=0 and the third f<=u.
This proof also covers l>0, u<0 and f=0. Every feasible guarded assignment
therefore extends to the native gate values, and every feasible native
assignment reconstructs the unique original DAG values on the SAME input and
bits. Appending all original predicates gives equality of the complete
input/phase/output relation, not just equality of projected output sets.

This is a mathematical exactness theorem for the reference language. The
bounded implementation may reject a construction, lowering or evaluation
when a declared size/rational limit is exceeded. Such rejection is not a
verified verdict, a completeness claim outside the bounds or an approximate
domain substitution. The theorem does not establish an advantage over the
ordinary linear reference, which is deliberately retained in full.

## Boundaries and expected complexity

Reference limits: p<=16, original free binaries<=8, total binaries<=64,
nodes<=256, vector/output handles<=256, explicit predicates<=128, affine fan-in<=32, total affine edges
<=4096. All stored/derived reduced rational numerator/denominator <=512 bits.
These are bounded-reference rejection limits, not claims of large-CNN support
or changes to any existing run cap. Reject before exceeding a limit; exceptions
do not authorize using partially built state or returning a verified verdict.

Let N be nodes, E affine edges, Q ReLU gates, B total binaries, P predicates,
D=p+Q+B. D<=144 under these limits. One evaluation costs O(N+E+Q+P) exact
scalar operations. Flattening all affine forms and bounds costs O((N+E)*D),
with O(N*D) retained coefficients; rows cost O((4Q+P)*D). Rational bit growth
is guarded, not assumed away. No Boolean monomial expansion or complete
equivalence algorithm is used. Python-object/allocator sizes need measurement;
asymptotic counts do not qualify physical memory gates.

Validation uses ordinary small mixing blocks with both signs of weights,
residual reconvergence, free input binary factors, EQ and LE predicates,
shared ancestors, and exact zero boundaries. It retains inherited3660 tests
unchanged, adds focused new tests, and uses the original combined collection+
execution <=60s gate. No omitted tests/skips or retries after a failed frozen
version. CPU1/GPU0, AS16GiB and existing transient restrictions remain where
applicable. A separate actual prototype-memory diagnostic will include the
complete builder/frozen/native/independent evidence roots. No full HZ-source
component certification is claimed by this semantic test stage.

## Fair claims and next use

Both a direct ordinary affine/ReLU evaluator and independently constructed
four-row linear constraints are comparators, not just the candidate checking
itself. Test fixtures must retain all original factors and visible predicates.
Point checks are regression evidence; the exactness argument is the nodewise
induction and four-row case proof, not finite sampling. No benchmark/property
SAT/UNSAT claim follows. Existing-HZ simplifier comparison and real Conv/CNN
lowering remain required before any domain benefit/admission claim.

All source/results go to new D003 files/directories. Old sources, D001/D002,
archives, goal, production defaults and scores stay unchanged. No model/HZ
payload decoding, new target/property solve, family replay, commit or push is
authorized by D003. The D003 implementation and standalone diagnostic use no
solver. The unchanged inherited regression suite includes existing tiny
LP/MILP tests; retaining it must not be reported as globally zero solver calls.
Streaming original input bytes for identity authentication is not decoding or
executing the original model.
