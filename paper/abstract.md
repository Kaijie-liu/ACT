# Abstract

Robustness verification of a mixture-of-experts model must account for both
data-dependent routing and the output of the selected experts. Requiring route
invariance is sufficient but can exclude regions whose outputs remain safe
across routing boundaries. We develop an ACT/HybridZ analysis for output-layer,
normalized weighted MoE models that retains the relationship between the input,
legal route guards and expert outputs. It combines guarded expert-wise proofs
with property-directed weighted obligations, reuses facts under explicit domain
containment, and schedules residual solving according to route complexity.
On 100 new verification inputs and three fixed same-family top-2 models, the
scheduled implementation gains 23 safety results over a matched monolithic
configuration without losing any, under equal request budgets; all gains involve
multiple legal routes. A separate shared-input ablation supports the role of
retained correlation. An executable ACT-fronted CROWN comparison is cheaper and
produces more numerical positive filters overall, while leaving complementary
ACT-only results. We also reconstruct weighted LP obligations in rational
arithmetic and independently check their bounds, explicitly retaining trust in
upstream network-to-HZ and route lowering. The results establish scoped benefits
of relational verification and expose limits in coverage, proof cost and model
generality; they do not establish universal backend superiority or strict
certification of high-accuracy deep MoE models.
