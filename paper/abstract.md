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
scheduled implementation gains 23 policy-accepted safety results over a matched monolithic
configuration without losing any, under equal request budgets; all gains involve
multiple recorded legal routes. A retrospective input-containment audit finds
small inward-rounding gaps, so these counts are empirical policy outcomes,
not source-complete real-network certificates. A separate shared-input ablation supports the role of
retained correlation. An executable ACT-fronted CROWN comparison is cheaper and
produces more numerical positive filters overall, while leaving complementary
ACT-only results. A separate repaired MetaMoE author-checkpoint comparison
favors the author sufficient path: nine numerical filters versus four ACT
policy acceptances, with no ACT-only positive. Compatibility repair does not
establish a new certificate. We also reconstruct weighted LP obligations in rational
arithmetic and independently check their bounds; the positive stored proofs
retain explicit upstream assumptions. A separate declared-source checker
covers a complete convolutional enclosure but yields no positive request on
its new matrices. These proof links cannot be combined across sources.
A separately fixed convolutional
family yields no additional HZ-policy safety results, and a new-input evidence
mode fails to reproduce a postselected complete proof. The results establish scoped benefits
of relational verification and expose limits in coverage, proof cost and model
generality; they do not establish universal backend superiority or strict
certification of high-accuracy deep MoE models.
