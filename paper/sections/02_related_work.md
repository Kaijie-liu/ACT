# Related work and comparison boundary

Our comparison separates routing analysis from expert verification. This is
important because several neighboring systems answer different questions: some
prove a property only after the route is invariant, some provide analytical
conditions without an executable instantiation, and general-purpose neural
verifiers operate on static graphs rather than dynamic dispatch programs.

## MoE robustness certificates

Zhang et al.'s RT-ER work gives analytical certified-robustness theorems and an
official CIFAR-10 architecture with an affine hard router and ResNet-18
experts. It is the closest official-scale target for our Route A evaluation.
The released artifact does not numerically instantiate the theorem constants,
publish certificate code or checkpoints, or update the router in its CIFAR-10
training script. We therefore label our model as an official-code,
paper-configuration reproduction rather than the authors' checkpoint. We
evaluate theorem applicability per input, expose the choice of constants
provider, and keep empirical constants diagnostic-only. This is an artifact-
centered comparison, not a claim that static routing is intrinsically invalid.

MetaMoE motivates a route-invariance style of composition: first establish one
route throughout the perturbation region, then verify the selected expert. No
executable artifact was available in our audit, so our baseline is explicitly
a `MetaMoE-style reimplementation`. Its applicability is definition-level and
uses the same exact router feasibility analysis as Route A. On applicable
inputs, both methods invoke the identical downstream verifier with the same
budget. This controls for expert-backend strength and isolates the cost of the
route-invariance precondition. At verification scale Route A resolves 56
additional samples and produces 36 route-changing certificates; the official-
scale B3 comparison remains required before we generalize that difference.

Other paper families, including SpecSphere, enter the reproducibility case
series when source or checkpoints cannot be executed. They are evidence about
the state of certification artifacts, not silently omitted numerical
baselines. Our survey distinguishes author code, reimplementations, analytical
claims, and unavailable artifacts; it does not compare runtimes across those
categories.

## Empirically robust and large sparse MoE

Robust-MoE-CNN/AdvMoE provides a complementary official third-party target. Its
hard top-1 router is learned with a supervised objective and a straight-through
gradient path, and one route selects weight slices shared across 16 convolution
layers. This differs from our output-level selected-softmax model and from
RT-ER's affine router. We run the repository independently because it has no
license permitting source incorporation. Its checkpoint, if reproduced, is
labelled by our exact dependency environment and is not redistributed without
permission.

V-MoE represents the soft, weighted, token-routing regime at larger scale. Its
router and load losses participate in ordinary full-model gradients, making it
the third training-semantics point in our source audit. We use it to establish
generality and provenance rather than promise end-to-end exact verification of
a full vision transformer. Robust-MoE-CNN and V-MoE do not become certified-
robustness baselines merely because they use routing; empirical robustness and
formal certification remain separate endpoints.

Static or random routers are also established designs, including Hash Layers
and THOR-like randomized routing. Their existence calibrates our RT-ER finding:
the issue is not that a fixed router is prohibited, but that theorem premises,
training narrative, and released optimization path must identify the same
artifact.

## General-purpose neural verifiers

Alpha-beta-CROWN and related CROWN systems are strong backends for static neural
networks. We compose with them rather than compete with their bound propagation.
The installed frontend rejects the dynamic `GatherElements` form used by both
our selected-expert export and RT-ER. After route specialization, the same
expert or deep path is a static graph the backend accepts. The rejection-to-
specialization-to-acceptance sequence is therefore a program-analysis result:
route conditioning extends an existing verifier's input language.

We study three ways to carry a route condition to such a backend. Retaining the
linear condition in HZ/MILP preserves it exactly at verification scale.
Replacing the guarded cell by its coordinate hull is sound but recovers almost
no information in high dimension. Compiling the implication into an augmented
output is semantically sound only with a positive tie margin and remains too
loose in our CROWN cohort. These negative controls prevent us from treating all
guard representations as equivalent.

Monolithic MILP is the direct exact-encoding baseline for verification-scale
models. It uses the same support-derived bounds as the decomposed encoding. Its
role is to test simultaneous binary width and runtime, not to stand in for a
published MoE verifier. For official-scale ResNet experts, CROWN is the primary
commodity backend and HZ/MILP is limited to an explicitly labelled exactness
reference subset.

## Property compilation and path-sensitive analysis

Property-reduction systems such as DNNV show how specifications can be compiled
into network outputs consumed by existing verifiers. Routing adds a discrete
path semantics that makes tie handling part of that compilation. Our eta
counterexample demonstrates that a one-line max reduction can be unsound at a
legal tie even when its tensor implementation is correct. Route A instead
enumerates feasible legal memberships, retains their guards where the backend
can use them, and aggregates only after every obligation is proved.

The broader connection is to path-sensitive abstract interpretation and
symbolic execution. Route guards are path conditions; candidate pruning and
conditional support decide how long their correlations survive. Our novelty is
the combination specialized to learned sparse dispatch: exact or bracketed
route analysis, gate-family-independent aggregation, a weighted top-k fallback,
and executable audit semantics across output-level, affine official-scale, and
deep shared-path regimes.

## What we do not claim

We do not claim a universal verifier, a dominance result over CROWN, or a
formal prevalence estimate for papers without complete source-native survey
retrieval. We also do not yet claim superiority on the official RT-ER model:
that statement is gated on the frozen B3 table. The current evidence supports a
narrower statement: route-invariance excludes certifiable route-changing
regions by construction, while a staged path-conditioned layer can expose
those regions to the same expert backend without weakening unsafe replay or tie
semantics.
