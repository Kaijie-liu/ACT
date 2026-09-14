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
legal routes. These are policy-accepted HZ/HiGHS results on a selected
clean-correct cohort, not whole-test-set certified accuracy. The input-clustered
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
not a frontend-rejection straw man. Our claim is therefore scoped net benefit
and complementary verification capability, not domination of generic neural
network verifiers.

Finally, we separate evidence consistency from independent bound checking.
A rational checker validates supplied HZ-to-LP projections, direct McCormick
construction and dual lower bounds without trusting solver objectives. One
fixed request has a complete conditional positive proof through this path.
Transferring a frozen fresh-generation procedure to the three ACT-only external
cases leaves unresolved obligations in all three. These results delimit the
current proof pipeline rather than retrospectively invalidating the original
policy-accepted results. Network-to-HZ propagation, guard lowering and route
exclusions remain explicit trusted assumptions.

The contributions are a relation-preserving organization of complete weighted
MoE obligations, property-directed weighted verification with scoped fact
reuse, and an evaluated implementation with an explicit numerical and evidence
contract. Hybrid zonotopes, convexity, McCormick envelopes and proof reuse are
not claimed as individually new. The executable version studied here covers
eval-mode output-layer selected-softmax top-2 models on CPU/float64; the broader
normalized top-k algebra is distinguished from that implementation scope.
Cross-architecture benefits and high-accuracy deep-model strict certificates
remain open empirical goals.
