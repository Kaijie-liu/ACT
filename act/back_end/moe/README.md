# ACT Route-Conditioned MoE Verification

This package implements the Route A foundation for output-level mixture-of-experts
verification on top of ACT/HyZor. It deliberately keeps the router and experts as
separate ACT `Net` objects so the solver never instantiates every expert's unstable
ReLU binaries in one monolithic model.

`RouteAEngine` is the public **Tier-1 gate-elimination** engine. It does not
silently invoke the restricted selected-softmax top-2 fallback. That Tier-2 F0
fallback has a separate frozen soundness contract in
[`../../pipeline/moe/docs/weighted_top2_fallback.md`](../../pipeline/moe/docs/weighted_top2_fallback.md).
The complete staged experiments invoke the lower-level
`build_weighted_top2_f0` and `solve_weighted_top2_f0` APIs only after Tier 1
returns the registered semantic-incompleteness reasons. There is not yet one
gate-family-general public engine that hides both stages behind `run()`.

## Implemented scope

- hard top-1, selected-softmax top-k, normalized-sigmoid top-k, and Switch-style
  selected-expert probability scaling as concrete PyTorch semantics;
- program-level MoE IR composed of one router `Net` and one `Net` per expert;
- tie-safe membership semantics: expert `i` may occur in any legal top-k set;
- sound generator support bounds for safe pairwise `M[j,i]` values (constraint-
  optimized tightening remains a performance improvement, not a soundness
  requirement);
- exact top-k membership intersection with at most `E-1` new binaries;
- exact candidate feasibility through the open SciPy/HiGHS MILP path;
- propagation of the route guard back to the input HZ and into a separately
  analysed expert;
- a shared-input, tie-safe Lagrangian compiler for hard-top1 guarded
  properties (a sufficient reduction, not an exact guard encoding);
- output-level gate-elimination aggregation with the correct incomplete result
  for weighted MoEs;
- hard top-1 counterexamples are reported only after the concrete full model
  selects the assumed expert and violates the output property.
- exact affine hard-top1 route-boundary geometry, including input clipping,
  explicit folding of input normalization, concrete upper-bracket witnesses,
  and a vectorized full-dataset API with a validated uint8-grid fast path.

Route B (intermediate weighted layers, history-budgeted union/merge, and
route-preserving merge tests) is intentionally not included in this first slice.

## Train a controlled public-data model

ACT's existing TorchVision data root is reused. The training command downloads
the public dataset when needed and automatically uses CUDA when available.

```bash
# Fast semantic/debug benchmark
python -m act.pipeline.moe \
  --dataset MNIST --num-experts 4 --top-k 1 --gate hard_top1 \
  --epochs 10 --device cuda --output data/moe/checkpoints/mnist_hard4.pt

# Route-changing weighted top-2 benchmark
python -m act.pipeline.moe \
  --dataset CIFAR10 --num-experts 8 --top-k 2 --gate selected_softmax \
  --expert-hidden 256 128 --epochs 50 --device cuda \
  --output data/moe/checkpoints/cifar10_top2_8.pt

# Experiment 0: distinguish route instability from output failure
python -m act.pipeline.moe.route_flips \
  --checkpoint data/moe/checkpoints/cifar10_top2_8.pt \
  --objective combined --epsilon 0.0313725 --steps 20 --device cuda
```

Supported by this controlled training CLI:

| Dataset | Input | Role in the study |
|---|---:|---|
| MNIST | 1x28x28 | solver and semantic regression |
| FashionMNIST | 1x28x28 | harder low-cost verification set |
| CIFAR-10 | 3x32x32 | primary paper benchmark |
| SVHN | 3x32x32 | distribution/route specialization check |

ACT already exposes more TorchVision datasets and VNN-COMP categories. TinyImageNet
is a sensible later scaling experiment, but should not gate Route A because it makes
the expert binary wall dominate the routing contribution.

## Programmatic Tier-1 verification

```python
import torch
from act.back_end.moe import (
    RouteAEngine,
    build_act_moe_program,
    load_output_moe_checkpoint,
)
from act.front_end.specs import OutKind, OutputSpec

model, payload = load_output_moe_checkpoint(
    "data/moe/checkpoints/mnist_hard4.pt"
)
model = model.double().eval()
x = torch.zeros(1, 1, 28, 28, dtype=torch.float64)
eps = 0.01
program = build_act_moe_program(
    model,
    center=x,
    lower=(x - eps).clamp(0, 1),
    upper=(x + eps).clamp(0, 1),
    output_spec=OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=torch.tensor([0])),
)
report = RouteAEngine(
    program,
    concrete_model=model,
    expert_models=tuple(model.experts),
).run_tier1()
print(report.result.status, report.router.candidates.candidates)
```

For selected-softmax top-2, a Tier-1 `UNKNOWN` caused by gate-sufficiency or an
expert witness that cannot be lifted is eligible for the separately audited F0
portfolio. A negative F0 relaxation is still `UNKNOWN`; only a concrete input
replayed through the full weighted model can be `UNSAFE`. The current staged
orchestration is visible in `act.pipeline.moe.experiment1_confirmatory` and
`act.pipeline.moe.experiment1f0` rather than being implied by this example.

Run the regression suite with:

```bash
python -m unittest act.back_end.moe.test_moe
```

## Reproducible experiments

Experiment protocols, frozen artifact hashes, result schemas, and cohort rules are
documented with the pipeline code. See
[`act/pipeline/moe/EXPERIMENTS.md`](../../pipeline/moe/EXPERIMENTS.md) before
running or interpreting the Route A studies. In particular, Experiment 1C keeps
the original ranks 0--99 as a development cohort and leaves ranks 100--199
untouched for the later confirmatory run.

## Current boundaries

- one verification lane per Route A engine invocation;
- output properties currently use ACT's existing linear/classification specs;
- weighted gate elimination rejects `UNSAFE_LINEAR` because its safe set is
  non-convex; hard top-1 remains exact for that specification;
- router minimality is exact only when the propagated router HZ is exact and all
  candidate MILPs finish;
- for weighted MoEs, a failing expert branch yields `UNKNOWN`, not a false full-model
  counterexample;
- no claim is made yet for token-level Transformer MoEs or large public LLM checkpoints.
