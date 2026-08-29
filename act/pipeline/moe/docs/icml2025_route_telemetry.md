# ICML 2025 RT-ER Route-Geometry Telemetry Protocol

## Scientific question

Theorem 5.4 motivates hard sparse routing through the claim that robust
training tends to separate selected router scores. The reproduction will
measure that geometry throughout training instead of inspecting only the final
checkpoint:

```text
Does official-code RT-ER move the exact hard-route boundary outward, and for
what fraction of CIFAR-10 test inputs does it exceed 8/255?
```

This is training-process telemetry, not an attack result and not a certificate
of the expert output.

## Frozen provenance label

Every model and result is labeled:

```text
official-code, paper-config reproduction
```

It represents a model family produced by the authors' public training pipeline,
not the unpublished author checkpoint and not a reproduction of an
author-reported certificate result. Three seeds `0, 1, 2` are mandatory. Seed
failures remain in the manifest and are not silently replaced.

## Checkpoint schedule

The paper specifies 130 CIFAR-10 epochs. The released script evaluates and
overwrites one file every 10 epochs. The reproduction wrapper must therefore
copy each completed checkpoint to an immutable, epoch-qualified path before the
next evaluation:

```text
seed{seed}/epoch{epoch:03d}.pt, epoch in {10,20,...,130}
```

Telemetry runs immediately after each immutable copy. It never waits until the
end of training to reconstruct intermediate geometry.

## Exact affine telemetry

The official router is `Flatten -> Linear(3072,E) -> argmax`, with `E=4` in the
CIFAR script. Fold the script's raw-uint8 normalization constants into the
affine weight and bias, retain the input in `[0,1]`, and evaluate all 10,000
official CIFAR-10 test images with:

```text
affine_top1_route_boundary_batch(
    ..., input_lower=0, input_upper=1,
    compute_device="cuda", capacity_grid_steps=255)
```

The grid path accepts only float64 values exactly derived from uint8 pixels. It
fails closed if a capacity is not within `1e-6` of `k/255`. The general
`sort+cumsum` path is the reference. On every software/hardware change, at
least 100 deterministic inputs must match the reference radii within `1e-12`
and have identical clean/boundary experts before the fast path is admitted.

The current synthetic engineering benchmark is 0.712 seconds for 10,000 inputs,
3,072 features, and four experts on the available RTX PRO 6000 Blackwell GPU.
This number is not reported as a model result.

## Per-checkpoint outputs

Save raw per-input rows as a compressed array artifact and summary JSON with:

- checkpoint SHA-256, official source SHA, seed, epoch, and complete command;
- Torch/Torchvision/CUDA versions and device identity;
- CIFAR raw-data checksum, ordered test indices, and normalization constants;
- clean expert, exact boundary competitor, radius point estimate, and outward
  lower/upper bracket for every input;
- route-load counts, probabilities, entropy, and `exp(entropy)`;
- directed and unordered clean-expert/boundary-competitor counts;
- radius median, IQR, p10, p25, p50, p75, p90, p95, and finite count;
- for `epsilon in {0.25,0.5,1,2,4,8}/255`, counts proven route-stable
  (`epsilon < radius_lower`), proven route-reachable
  (`radius_upper <= epsilon`), and numerically undecided;
- telemetry wall time and backend (`numpy_sort`, `cuda_sort`, or
  `cuda_uint8_histogram`).

Primary longitudinal plots use epoch on the x-axis and show median/IQR route
radius, `Pr[radius_upper <= 8/255]`, load entropy, and boundary-pair composition.
All seeds are shown; the aggregate curve reports median and range across seeds.

## Witnesses and downstream reuse

The scalar and optional batch oracle return a concrete perturbation at the
outward upper bracket. Replay requires:

```text
lower <= x + delta <= upper
||delta||_inf <= radius_upper
score_competitor(x + delta) >= score_clean(x + delta) - tolerance
```

Witnesses are not stored for all 10,000 inputs by default because the dense
array is large. They are regenerated deterministically for selected boundary
cohorts and used as route-change seeds. They do not establish an output
counterexample.

## Failure and interpretation rules

- A route radius beyond `8/255` establishes hard-route invariance only; expert
  output robustness still requires a verifier.
- A radius within `8/255` establishes route-boundary reachability, not output
  failure.
- Missing epochs, failed seeds, non-grid inputs, checksum changes, or checkpoint
  overwrite are audit failures and remain visible.
- Test accuracy is observed alongside telemetry but never used to select an
  intermediate checkpoint for the primary route-geometry curve.
- No theorem radius is computed until a sound Lipschitz-constant method is
  separately defined.

## Execution gate

This protocol and the batch oracle are ready, but reproduction is not launched.
The authors' pinned FFCV/Torch environment requires separate dependency
authorization. No environment is created and `act-py312` is not modified by
this stage.
