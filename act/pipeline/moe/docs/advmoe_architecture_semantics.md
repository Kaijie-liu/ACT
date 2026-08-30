# AdvMoE architecture and training semantics

## Audited result

The pinned official `robust-moe-cnn` repository at commit
`c50796fb8284512b6f6ad8e843f95182cec527cf` implements AdvMoE as a deep
route-conditioned CNN, but **not** as a hidden-state router. The router consumes
the CIFAR image tensor before the ResNet stem. One shared hard top-1 decision
then controls every MoE convolution in the network.

This corrects the earlier planning assumption that the router consumed an
intermediate feature and could reuse a prefix HZ as its input domain. That
prefix construction does not apply to this released CIFAR-10 architecture.

The raw result is
`data/moe/results/advmoe_architecture_audit_20260830/raw.json`. The tracked,
independently replayed result is
`act/pipeline/moe/results/advmoe_architecture_audit_20260830.json`; its audit
reports zero issues over 34 hashed source anchors.

## Official CIFAR-10 configuration

The repository README's example selects ResNet-18, two experts, and width ratio
0.5. The corresponding instantiated architecture has:

| Item | Audited value |
|---|---:|
| Router input | `[B, 3, 32, 32]` image tensor |
| Router | CIFAR-ResNet20-style CNN: `3 -> 16 -> 32 -> 64 -> E` |
| Router output | `[B, 2]` scores |
| Router parameters | 269,202 |
| ResNet BasicBlocks | 8 |
| Routed MoE convolutions | 16 |
| Selected widths | 4 each at 32, 64, 128, and 256 channels |
| Distinct router objects used by those layers | 1 |
| Full attached-model parameters | 5,834,652 |

The dense input stem and projection shortcuts are not MoE convolutions. Each
BasicBlock's two main convolutions is routed, giving 16 routed convolutions.
The selected expert is a contiguous channel group at every such convolution.
Because the same score tensor is broadcast to all routed modules, the model has
two global route-specialized pathways in the official `E=2` configuration, not
`2^16` independently routed layer combinations.

## Tie semantics

Literal execution uses PyTorch `argmax`; an equal-score test selects index zero,
so its deterministic behavior is first maximum / lowest index. ACT's
tie-inclusive `ANY_LEGAL_TOPK` interpretation is a sound conservative
overapproximation only if all tied routes are checked. It must not be described
as bit-exact reproduction of the released tie breaker.

## Router training semantics

The router is learned by a separate, explicit objective. The precise schedule
matters:

1. the main optimizer is created before the router is attached, so it contains
   zero router parameters;
2. hard selection exposes a straight-through backward, and the classification
   loss creates nonzero gradients on all 59 router parameter tensors;
3. the main optimizer changes zero router tensors;
4. the subsequent router-optimizer `zero_grad` clears those classification STE
   gradients;
5. supervised router cross-entropy plus clean/adversarial router consistency is
   backpropagated, and the separate router optimizer changes all 59 router
   tensors in the synthetic control.

Accordingly, the defensible classification is **hard top-1 routing learned by
an explicit supervised/robust router objective**. It is inaccurate to claim
that the released classification STE itself updates the router.

Two additional released-training details are recorded for reproducibility:

- a distinct router loader is constructed but occurs only once in the training
  entry point and is not passed to the trainer; both update phases consume the
  same minibatch there;
- router-specific optimizer, learning-rate, and schedule command-line options
  are declared, but router construction calls the generic optimizer/schedule
  helpers that read the main optimizer settings.

These are artifact-semantic observations, not accusations about intent. A
future reproduction must either execute the source literally and label it so,
or explicitly label any correction as a modified configuration.

## Verification consequences

AdvMoE remains the preferred official third-party learned-router target because
it closes a genuine structural gap: dispatch changes computation throughout a
deep network rather than only combining output experts. The required adapter is
therefore:

1. propagate a pixel-domain abstraction through the nonlinear CNN router;
2. determine feasible global hard top-1 routes with conservative tie coverage;
3. specialize all 16 MoE convolutions consistently to one global route;
4. verify the resulting full ResNet pathway for every feasible route.

This is not the previously proposed hidden-prefix construction. The cheap
feature-space margin may still be reported as a diagnostic, but input-space
route feasibility and certification require propagation from pixels. Any
sampled census must freeze its deterministic input ranks before execution and
must be labelled separately from the affine RT-ER full-test census.

## Scope, licensing, and next gate

No AdvMoE training was started in this stage. No source was copied into ACT:
the external repository has no located license, so the committed evidence
contains only identities, hashes, line hashes, semantic classifications, and
independently computed module/tensor facts. Checkpoint redistribution remains
unresolved; the conservative artifact policy is to publish hashes and exact
reproduction instructions and distribute weights only after legal/institutional
clearance.

Official sources: [paper](https://arxiv.org/abs/2308.10110) and
[repository](https://github.com/optml-group/robust-moe-cnn). Training remains
queued after RT-ER B1/B3 and cannot run concurrently with B1.
