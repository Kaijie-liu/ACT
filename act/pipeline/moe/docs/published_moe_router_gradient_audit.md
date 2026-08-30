# Router gradient paths in three official MoE pipelines

## Result

The pinned source audit is complete and the independent audit reports zero
issues. It follows router parameters from their definition through a
differentiable objective to the optimizer update. It does not install baseline
dependencies or train a new model.

| Official pipeline | Released routing path | Router update classification | Evidence level |
|---|---|---|---|
| ICML 2025 RT-ER | output-level hard argmax/top-k indices | released training path does not update router | source audit plus existing epoch-10/20 tensor and Adam-state evidence |
| ICCV 2023 robust-moe-cnn | shared convolutional hard top-1 | explicitly learned | pinned source path only |
| Google V-MoE | hidden-layer noisy weighted top-k token routing | end-to-end learned | pinned source path only |

The raw manifest is
`data/moe/results/published_router_gradient_audit_20260830/raw.json`. The tracked
independent result is
`act/pipeline/moe/results/published_moe_router_gradient_audit_20260830.json`.

## robust-moe-cnn

At official commit `c50796fb8284512b6f6ad8e843f95182cec527cf`, the convolutional
router ends in a trainable expert-score layer. Hard top-1 selection uses an
explicit straight-through backward. More decisively, the released trainer:

1. creates and attaches a router separately from the main model optimizer;
2. creates a separate router optimizer;
3. forms a supervised router cross-entropy term and a clean/adversarial router
   KL term directly from router scores;
4. backpropagates that loss, steps the router optimizer, and saves its state.

This is an official third-party learned-router pipeline and therefore a strong
external-validity candidate. It is not isomorphic to output-level weighted
top-k Route A: its router and experts are convolutional and its hard route
selects shared channel-expert computation. Any later execution must be reported
in a separate column.

No license file was located. ACT stores only repository/commit identities,
hashes, line predicates, and semantic conclusions. No source was copied into
ACT, and any later run remains external to the ACT source artifact.

## V-MoE

At official commit `c07681241f81ba11421ba98e523e1499b2738a79`, the published
ImageNet-21k configuration is `E=8, K=2`. Each MoE block creates a Dense gate,
normalizes its scores, uses selected gate values to combine expert outputs, and
adds positive importance and load losses. The trainer differentiates main plus
auxiliary loss with respect to the full parameter tree and applies those
gradients. The audited paper configuration has no router freeze pattern.

V-MoE therefore supplies official evidence that a hidden-layer weighted top-k
router is learned. It is not currently an end-to-end exact-verification
baseline: its token routing, capacity behavior, and model scale are distinct.
It remains relevant to external validity and to the bounded hidden-layer MoE
extension.

## Cross-pipeline conclusion and claim boundary

The new result rules out a field-wide claim that released MoE routers are
generally static. Instead it sharpens the RT-ER case study: the pinned RT-ER
release is the static outlier among these three audited official pipelines,
while robust-moe-cnn and V-MoE expose two different learned-router mechanisms.

Only RT-ER currently has dynamic tensor-change and optimizer-state evidence in
this project. The other two conclusions are source-level. No accuracy,
robustness, checkpoint, or certificate result is inferred, and no new baseline
training was started while B1 is active.
