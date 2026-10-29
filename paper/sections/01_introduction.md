# Introduction

A small perturbation of a mixture-of-experts (MoE) input can change which
experts execute. This creates two coupled verification problems: determining
which routes are legal and proving the property of the resulting output.
A route change is not itself an output violation. In models whose experts
share an output space, different routes can produce the same correct decision;
with normalized weighted routing, a selected expert can even violate a property
while the weighted output satisfies it. A verifier that first requires an
unchanged route therefore uses a sufficient premise, not a necessary condition
for output robustness.

Enumerating possible routes alone does not resolve the problem. An expert
output that is possible somewhere in the input box may be impossible where that
expert is selected. Similarly, two expert outputs may each be individually
possible but not jointly realizable at the same input. Independently bounding
the router and experts can discard precisely the constraints needed to prove
the weighted result. Conversely, retaining every relation in one joint mixed-
integer formulation can be expensive. The research question is how to retain
the useful relations while organizing the resulting proof obligations within a
finite request budget.

We address this question on top of ACT and its HybridZ representation. The
analysis shares input factors and route constraints while preserving distinct
expert activation choices. It accounts for every tie-legal unordered route,
conditions expert analysis on membership and pair guards, and first tries a
convexity-based expert-wise sufficient condition. Where that condition is
inconclusive, a property-directed weighted stage analyzes an anchor expert and
scalar expert differences instead of constructing a separate nonlinear gate
expression for every output coordinate. Facts already proved on an expert's
membership domain can discharge the same property on a contained pair domain.
Residual obligations are then scheduled according to legal-route complexity
and the real remaining budget.

The method is not premised on decomposition always outperforming joint solving.
An earlier fair comparison found that monolithic F0 obtained more safety
results overall, while staged solving had lower observed cost and different
successful cases. We used this development evidence to freeze a structural
scheduler: a single legal pair goes directly to weighted solving, whereas
multiple pairs receive a bounded expert-wise phase followed by residual F0.
A matched monolithic path independently computes and pays for the same cheap
property facts. A legacy monolithic configuration is also retained, preventing
the comparison from relying only on a newly configured opponent.

An independently frozen 100-input experiment on three existing checkpoints
then obtained 179 safety results versus 156 for the matched path and 141 for
the legacy path, across 300 model-input requests per method. The primary
comparison has 23 gains and no losses; every gain is in a region with multiple
recorded legal routes. These are policy-accepted HZ/HiGHS results on a selected
clean-correct cohort, not whole-test-set certified accuracy. A later exact audit
of saved input boxes and the frozen input-HZ formula finds inward-rounding
gaps, including on all 23 gains. The results are not independently established
real-box certificates; downstream source conversion remains unverified for
these historical outputs. This finding does not exhibit a network
counterexample or change the observed comparison; it does not establish that
correcting the enclosure would preserve the result sets or gains. The input-clustered
descriptive interval for the mean primary gain is [4.67, 11.00] percentage
points. A separate relationship ablation yields three additional safety
results when shared expert input factors are retained, including two cases
where an independent outer product completes but remains relaxation-undecided.

The external comparison provides an important counterweight. On ten observed
inputs and the same three models, an ACT route frontend followed by plain
CROWN on variable-weight static pairs produces 13 numerical positive filters,
versus 11 HZ-policy safety results, at substantially lower observed cost.
There are three ACT-only and five CROWN-only positives. These evidence levels
are not interchangeable, but the cheaper path is a real usability challenge,
not a frontend-rejection straw man. This historical comparison supports
complementarity with that particular plain-CROWN path, not domination of
generic neural network verifiers. A separate 20-call comparison on the MetaMoE author
checkpoint is more adverse: ACT obtains four policy acceptances, all shared
with nine author-backend numerical sufficient filters. ACT has no unique
positive. The author path includes a disclosed compatibility wrapper and
strict route-invariance adapter; this is not unchanged author execution.
The formerly crashing request now times out, establishing compatibility but
not a new certificate. On the four common positives ACT is also slower
(10.55 versus 7.17 seconds on average). Thus the earlier external-path
complementarity must not be generalized to every author-model comparison.
See the [complete repaired comparison](../../docs/metamoe_las_followup_result_20260923_r1.md).

Later strong-backend experiments materially narrow the competitive claim.
The registered compatibility-adapted GenBaB path obtains numerical positives
on all 60 registered historical-success requests (43 distinct inputs),
including 51 recorded multi-pair cases.
It also closes 17 of 41 requests in a separate historical-failure cohort; the
remaining 22 UNKNOWN and two ERROR are retained. These studies select by old
ACT outcomes and execute at different times and versions, so their timings
are not a new paired speedup and their counts are not a population success rate.
They leave no ACT-only positive in the selected success cohort and stop the
registered single-layer advantage-region direction. See the
[success-region diagnosis](../../docs/advantage_region_results_20261021.md) and
[failure-cohort results](../../docs/b2_observed_results_20261021.md).

The subsequent multilayer experiment completes 600 requests. On its 180
multilayer objects, prefix-pruned GenBaB obtains 108 numerical positives,
while ACT v1 and naive enumeration obtain none. ACT retains 178 TIMEOUT,
one ERROR and one UNSUPPORTED, not 180 homogeneous solver timeouts. The
many-history mechanism criterion stops the route-union direction; this
supplement was registered after a 189-request prefix and is explicitly not
an outcome-blind rule. It does not prove that route combinations can never be
a bottleneck or that every unresolved history is limited by ReLU difficulty.
See the [completed comparison](../../docs/multilayer_stage1_20261022.md).

This evidence motivates testing optional strong-engine mechanisms rather than
assuming that an HZ implementation is necessary for competitive verification.
The original guard-dual and first-margin partition diagnostics add no history
closures and fail their frozen admission rules. Their rejection and the
partitioned bounds' observed precision/cost tradeoff are reported in the
[current evaluation](08_evaluation.md#multilayer-mechanism-diagnostics-no-admitted-m-b-or-m-c-benefit).
Because the original guard comparison used independent optimization trajectories,
a revised diagnostic retains the entire optimized baseline before adding
guard multipliers. It closes one of the same eleven histories, while an
extra-alpha continuation closes none. This local benefit still falls below the
registered five-closure threshold and stops the conditional follow-up; it is
not evidence of complete-request superiority.
The subsequent 600-call primary comparison tests gate decoupling with a coupled
fallback alone. On 300 paired model-input objects it gains nine numerical
positives and loses three, for a net two percentage points with an input-block
95% interval of [0,4] points. This fails the registered five-point practical
threshold; mean request cost rises from 189.287 to 192.983 seconds. All nine
gained requests close through decoupling without fallback, establishing a
local computational effect but not the required overall benefit. The separate
double-radius cohort gains two of 120 paired objects, with no gain among its
thirteen observed-multiple-history objects. The single-layer queue remains
pending. See the
[primary result](08_evaluation.md#complete-request-cascade-primary-comparison-practical-gate-not-met).

Finally, we separate evidence consistency from independent bound checking.
A rational checker validates supplied HZ-to-LP projections, direct McCormick
construction and dual lower bounds without trusting solver objectives. One
fixed request has a complete conditional positive proof through this path.
Transferring a frozen fresh-generation procedure to the three ACT-only external
cases leaves unresolved obligations in all three. These results delimit the
current proof pipeline rather than retrospectively invalidating the original
policy-accepted results. Network-to-HZ propagation, guard lowering and route
exclusions remain explicit trusted assumptions. A later convolutional control
also yields a complete portable conditional proof, but the separate 30-input
comparison produces no HZ-policy SAFE and the 20-input general evidence mode
produces no complete positive request. Postselected proof feasibility therefore
does not establish transferable certificate coverage.

A subsequent source-checked construction validates input containment and the
complete expert enclosure for the declared convolutional graph, but does not
obtain a positive complete request on its new matrices. It cannot inherit the
old supplied-HZ proof's bounds. The latest bounded diagnosis closes that local
follow-up without identifying a new isolated intervention. Independent checking
coverage and empirical positive coverage are separate accomplishments.

The contributions are a relation-preserving organization of complete weighted
MoE obligations, property-directed weighted verification with scoped fact
reuse, and an evaluated implementation with an explicit numerical and evidence
contract. Hybrid zonotopes, convexity, McCormick envelopes and proof reuse are
not claimed as individually new. The historical HZ production comparison covers
eval-mode output-layer selected-softmax top-2 models on CPU/float64. The later
strong-engine frontend handles the explicitly registered multilayer MLP and
residual configurations; this does not extend the old HZ result to arbitrary
internal or token-level routing. The mathematical reductions and negative
experiments are established separately from the still-unconfirmed competitive
mechanism claim. Cross-family strict certificates remain open.
