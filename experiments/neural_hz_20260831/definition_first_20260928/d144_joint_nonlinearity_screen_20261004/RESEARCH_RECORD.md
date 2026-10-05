# Neural-HZ: joint nonlinear structure, not another helper

The saved pilot offers very little support for the old fixed consecutive-channel
common-anchor grouping: only **8 of 975 groups** contain two gates whose saved
intervals cross zero. This is a structural selection result, not a new domain or
verification gain. It redirects the definition work; it does not justify more
implementation around the existing common-anchor formulas.

## What was measured

The preregistered sign-only audit processed all three authenticated D120 complete
artifacts, with no model load, new bounds, solver or candidate execution. It
finished once with exit 0 under the 60-second timeout; tool-reported wall time
was 0.073194768 seconds. This is audit overhead, not verifier performance.
All five input hashes matched before and after. Full output is [RESULTS.json](RESULTS.json).

Independent static review noted two output/scope clarifications: absent histogram
keys mean zero occurrences (the script does not materialize empty bins), and the
validated references are group-to-source references. Downstream rows are counted,
not independently rebound to their original consumers. These limitations do not
change the sign counts; they must not be read as a broader reference audit. The
frozen script and preregistration were not edited or rerun after seeing results.

| Saved model pilot | Source forms | Zero unresolved gates | One | Two | Zero-touch groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR100 large | 1600 | 316 | 8 | 1 | 0 |
| CIFAR100 medium | 1600 | 277 | 43 | 5 | 0 |
| TinyImageNet medium | 1600 | 301 | 22 | 2 | 0 |
| Total | 4800 | 894 | 73 | 8 | 0 |

Here unresolved means `lower<0<upper` in the saved outer interval; it does not
prove that both signs are attainable. The 89 such source forms are 10, 53 and
26 respectively. All other 4711 forms have strict certified sign in that old
mathematical source evidence. Native HZ phase-column binding was not qualified
by D120 and is not qualified here.

The population is the old five-window pilot in each model, not all spatial
positions, all layers, all properties or a representative random sample. The
groups are the old fixed groups of consecutive channels, not all possible groups.
None of their 2850 saved interpolation-residual intervals is exactly [0,0];
2739 exclude zero and 111 straddle it. This does not measure approximate
alignment quality, because no residual magnitude was evaluated.

## Why the sign screen matters

Let the complete shared source be z in P, and g_i(z) be affine. If all gates
except at most one j are strictly stable, their original labels sigma_i are
fixed and q_i(z)=sigma_i*g_i(z). Let G_j be the remaining labelled ReLU graph
over the **complete same source**, retaining original bits, shared coordinates
and decoder identity. Adding every other gate and every affine live consumer
is an affine embedding T. Thus

    G_group = T(G_j),        conv(G_group) = T(conv(G_j)).

If all gates are strictly stable, replace G_j by P. The affine-lifting identity
does not require P convex, but the single-gate relation must include the actual
nonconvex source and ancestral bits; independently convexifying that source
first does not inherit the equality. Any additional predicate involving group
outputs must be pulled back through T **before** taking the hull. Convexification
and intersection cannot be interchanged without proof.

This standard fact means 967/975 saved groups cannot exhibit extra *within-group
multi-hinge hull* strength over that complete source-labelled single-gate
reference. The reference is not the scalar triangle. Touching-zero gates would
need separate handling because their original labels are not necessarily fixed;
there were none in these artifacts. The remaining eight groups only pass a
necessary screen, not a sufficiency test for gain.

The result does not rule out cross-group source correlations, subsequent
nonconvex phase propagation, or better consumption of existing affine relations.
It also does not prove the cause of D120's zero downstream-ReLU improvements.
That older experiment lost source dependence in multiple places; no counterfactual
implementation was run here. See the original [source-loss analysis](../d120_mixed_consumer_source_20261002/SOURCE_LOSS_ANALYSIS.md).

## Common-anchor idea checked before the screen

For an original anchor h with q_h=ReLU(h), original label alpha, and
g_i=k_i*h+r_i with k_i>=0, let a_h=2q_h-h=|h|. The identity

    q_i = k_i*q_h + alpha*r_i + e_i,
    e_i = q_i-alpha*g_i = ReLU((1-2alpha)*r_i-k_i*|h|)

is exact, including both legal zero labels. With the original member label beta_i,

    e_i + k_i*|h|*|beta_i-alpha| = |r_i|*|beta_i-alpha|,
    |h|*|beta_i-alpha| = q_h-beta_i*h.

Alpha and beta here use 0/1 notation for the original signed bits through the
bijection beta=(signed_bit+1)/2; no bit is added, deleted or continuously relaxed.

These are existing anchor/phase-gap relations, not a new definition. D020's
[phase-difference record](../d020_phase_difference_20260930/D020_PHASE_DIFFERENCE.md),
D118's [shared-curvature theory](../d118_shared_curvature_transfer_20261002/THEORY.md)
and D121's [definition audit](../d121_source_coherent_definition_audit_20261002/RESEARCH_DECISION.md)
already cover the essential alternatives. Exact mixed products still require
representation/query cost. Summing them into a source-side aggregate does not
make an integer beta an integer product variable; D143's [counterexample and
strong comparison](../d143_aggregate_gate_gap_20261004/THEORY.md) remain applicable.

Two ordinary paper controls further delimit the anchor rewrite:

- A shared anchor does not imply a single fixed error direction. Set h=x+1/4,
  r_1=1/4+y/10, r_2=1/4-y/10 and k_1=k_2=1. On
  -1/8<h<-1/16, |y|<1/4, alpha=0 and both members are active. Then
  e=(x+1/2+y/10,x+1/2-y/10), whose Jacobian has rank two. This only rules out
  an exact one-fixed-generator error model, not all structured relations.
- A total mismatch budget is not an exact replacement for individual hinges.
  Take h=x+y/5+1/10, g_1=(3/4)h-x/10+y/5+1/4 and
  g_2=(5/4)h+x/5-y/10+1/5 on [-1,1]^2. At (x,y)=(-1/4,0),
  alpha=0, beta=(1,0), h=-3/20, r=(11/40,3/20), g=(13/80,-3/80).
  The true e is (13/80,0). The false e=(1/5,0) still satisfies epigraph,
  sign/off-mask conditions, e_i<=ReLU(r_i), and
  sum e_i+|h|*sum k_i*|beta_i-alpha| <= sum ReLU(r_i), because
  5/16<=17/40. Keeping the omitted exact hinge rows would recover the old
  graph, not establish this budget as a new domain.

These are algebraic controls, not executed benchmark regressions or ADV.

## Research decision and next mathematical obligation

Do not implement another fixed consecutive-channel common-anchor candidate on
the strength of the old formulas or this audit. The main question is how a domain
element can preserve **joint source, phase and amplitude dependence** through
mixed Conv/affine readouts, a live residual, and the next nonlinearity, with a
complete cost advantage. It must not be just old HZ plus helper inequalities or
the unsimplified network graph with a new name.

A future grouping/closure rule must be uniform in observable source and nonlinear
structure, never chosen by model identity, labels or solver status. It must apply
to all matching structures, not just these eight old groups. Explicit original
bits, zero labels, latent identity and concrete input reconstruction remain in
the semantics. Source-coupled regrouping is a research question, not a newly
proved representation or a license to split phases.

The next substantive deliverable must therefore be a proposed domain element,
concretization and compositional operator theorem with a strong comparison and
full query cost. If it reduces to known symbolic generators or extra cuts, record
that and change the hypothesis. Do not substitute more audit/test infrastructure
for that deliverable. A literal claim of greater exact set expressiveness than
HZ on PWA networks is not required: useful domain/operator structure and tractable
verification strength, rather than merely renaming an exact encoding, need proof.

No new domain, novelty, native source, GPU, full-cost, shadow or full replay
qualification is claimed. Formal gain is zero; baseline remains 1870/2413 and
separate E0 remains CIFAR100 25 + TinyImageNet 36 =61/400. The goal remains active.
The documentation skill was used only to save this bounded research record;
it did not introduce an external Page or alter any research permission.
