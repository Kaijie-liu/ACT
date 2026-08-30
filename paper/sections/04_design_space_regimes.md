# A design space for route-conditioned verification

Routed neural networks do not present one verification problem. The cost of
reasoning about a routed program moves among the router, the set of reachable
paths, the expert networks, and the gate that recombines expert outputs. A
method evaluated in only one corner of this space can appear either stronger or
weaker for reasons unrelated to its central abstraction. We therefore organize
the evaluation around three regimes that place the bottleneck in different
parts of the program.

The first axis is route combinatorics. With (E) experts and unordered top-
(k) dispatch, as many as (inom{E}{k}) route sets can be legal. The second
axis is router geometry. An affine router admits exact support and, for hard
top-1 dispatch over a pixel box, a closed-form boundary oracle. A shallow
piecewise-linear router can remain exact in a hybrid-zonotope frame, whereas a
deep convolutional router generally requires a relaxation-and-attack bracket.
The third axis is path structure. Routing may select independent output-level
experts or specialize a shared deep network into a small number of static
paths. Finally, weighted dispatch introduces gate semantics after route
selection; hard routing does not.

| Regime | Representative | Dominant difficulty | Route-conditioned response |
|---|---|---|---|
| combinatorial weighted output | bal010, (E=8,k=2) | feasible route sets, retained correlation, weighted output | exact candidate pruning, guarded support, staged gate elimination and F0 |
| affine hard routing at official scale | RT-ER, (E=4,k=1) | route-invariance applicability and expert scale | closed-form route oracle plus static expert specialization |
| learned deep router with two shared paths | AdvMoE, (E=2,k=1) | router relaxation and artifact semantics | optional router pruning plus router-independent verification of both static paths |

## Regime I: combinatorial weighted output routing

The verification-scale bal010 model isolates the route-set and weighted-gate
problems. Its router is nonlinear, its eight experts are independent output-
level MLPs, and the selected-softmax top-2 output depends on both expert
predictions. Requiring route invariance discards precisely the regions of
interest. Exhaustively treating all pairs is possible at (E=8), but it does
not explain how candidate correlation, path constraints, or gate modelling
affect verification.

This regime exercises the complete Route A stack. An unrelaxed reachable
router HZ eliminates infeasible candidates while preserving correlations lost
by IBP and ordinary zonotopes. Feasible unordered route sets are checked with
tie-inclusive guards. Those guards remain in the shared factor frame while
experts are propagated, so constraint-aware support can remove unstable ReLU
binaries before they enter a solver. Gate elimination proves a convex linear
property when all selected experts satisfy it. When that sufficient condition
is inconclusive, F0 introduces only property-directed expert disagreement and
a guarded scalar gate range through McCormick envelopes.

The regime establishes mechanism evidence rather than scale by itself. The
independent cohort shows candidate reduction, conditional binary-width
separation, guard-dependent coverage, and route-changing certificates. Lazy
no-good-cut enumeration and support-derived big-M tightening extend the same
semantics beyond exhaustive (E=8) proposals; their timed (E)-scaling study
remains a separately identified endpoint.

## Regime II: affine hard routing at official scale

RT-ER moves the difficulty. Its released CIFAR-10 model uses a four-way affine
hard router and ResNet-18 experts. Router membership and the minimum pixel-box
top-1 boundary are therefore exact and cheap. The challenge is not estimating
the route geometry but confronting what it implies: the released training
script leaves the router outside the optimization path, and the route-
invariance applicability set becomes nearly empty at the paper's evaluation
radius over the distribution of official-construction initializations.

Route conditioning uses the exact router analysis as a dispatch layer and
delegates each static expert to a commodity neural verifier. The comparison is
designed to keep that expert backend and its budget identical. A route-
invariance baseline can invoke it only after proving a single route; Route A
can cover every feasible route and aggregate only after every corresponding
property is established. At verification scale, this change alone accounts
for 56 additional solved samples and all 36 route-changing certificates. The
official-scale B3 table is the required external validation and remains
explicitly pending until the 130-epoch reproduction and downstream expert
checks are complete.

This regime also exposes certificate applicability as a measurable object.
The exact (A(\epsilon)) curve reports how many inputs satisfy a route-
invariance precondition before any expert verification occurs. It separates a
vacuous theorem premise from a weak expert backend and prevents zero
certificates at a large radius from being misdiagnosed as solver failure.

## Regime III: learned deep routing with a small path family

AdvMoE uses a learned convolutional image router and one hard route shared by
16 MoE convolutions. Once a route is fixed, all corresponding slices specialize
into an ordinary static ResNet-18-like network. Because (E=2), verifying both
static paths over the full input box is a sound fallback that does not require
any router bound. Router certification is an optional pruning optimization,
not a soundness dependency.

The initialization audit demonstrates why this distinction matters, but not
for the initially suspected reason. At the released default seed, eval mode
with default BatchNorm running statistics sends all 10,000 ordered CIFAR-10
test inputs to expert 0. Across 20 official-construction-order initializations,
13/20 eval-mode routers and 8/20 train-batch-statistics routers are exactly
collapsed on the same stream; their median maximum expert shares are 100% and
99.305%, respectively. Batch statistics weaken the signed global offset but
do not remove the load collapse. Collapse targets also change with the seed.
Local margin-to-gradient estimates that appeared to be roughly 130 times the
RT-ER boundary scale are therefore confounded by an initialization-dependent,
semantics-dependent near-constant router on this distribution. We retire the
architecture interpretation rather than converting the ratio into a headline.

Strong PGD does not find a route flip on the frozen 20 inputs even when the
clipped epsilon-1 boxes span the full pixel cube. This remains attack non-
discovery, not global constancy. Conversely, IBP and sparse CROWN produce
negative router bounds many orders away from the observed attack drop. Moving
from IBP to CROWN reduces the dimensionless inflation diagnostic by about
5.2 times but leaves an approximately (10^{11}) residual. The trained-
checkpoint evaluation must determine whether the supervised router objective
breaks the initial load collapse; it records route share and signed offset at
every checkpoint under both eval/current-running-statistics and registered
train-batch-statistics semantics.

The contrast with RT-ER is therefore about how training interacts with a
degenerate start, not about one router architecture being inherently stable.
RT-ER permanently preserves its random initialization because its released
training script has no optimization path to the router. AdvMoE exposes an
explicit router objective, so training must first escape its highly imbalanced
initial partition. The checkpoint trajectory tests whether and when that
escape occurs; initialization alone cannot answer it.

## Backend composition rather than backend replacement

Across all three regimes, route conditioning determines which static programs
must be checked and what path constraints accompany them. It does not mandate
one expert verifier. Exact HZ/MILP propagation is valuable where a small model
and retained generator identity expose correlation. CROWN is appropriate for
official-scale convolutional experts. A frontend rejection of dynamic gather
or dispatch is not treated as a failure of CROWN's neural bound propagation:
specialization removes the unsupported operation and presents the same backend
with an ordinary static network.

This separation fixes the comparison boundary. We claim neither that HZ
dominates CROWN nor that routing makes every expert property tractable. We
claim that an explicit, tie-inclusive path analysis converts dynamic dispatch
into a finite family of sound static obligations and that its cost and benefit
depend predictably on the regime. Timeouts remain timeouts, relaxation failures
remain unknown, and any unsafe verdict must replay through the full dynamic
model.

## Limits of the current design space

The three regimes do not cover token-level transformer MoE, capacity dropping,
stateful routers, or unbounded expert families. The current top-(k) result
assumes normalized non-negative selected gates, and the exact label applies
only while the reachable router HZ has not been relaxed. Large-(E) evidence
requires the pending lazy-enumeration scaling study. AdvMoE's official trained
checkpoint and RT-ER B3 are pending; initialization diagnostics cannot stand in
for either result. These boundaries are part of the method definition rather
than post-hoc threats.
