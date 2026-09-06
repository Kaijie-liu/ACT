# Strict high-accuracy AdvMoE path verification

## Purpose and scope

The accepted AdvMoE numerical-compatibility checkpoint has 85.67% ordered
CIFAR-10 clean accuracy and 61.14% released-path PGD-10 accuracy.  Existing
two-path CROWN results are numerical margin filters, not formal SAFE results,
because the installed CROWN execution has no outward-rounding contract.

This stage evaluates a separate strict backend path.  Dynamic dispatch is
removed by specializing all 16 MoE convolutions to route 0 or route 1.  Each
specialized network is exported to ONNX and checked against the frozen PyTorch
path.  PyRAT then verifies the top-1 property in CPU NumPy float64 mode with
`--sound true`, which its CLI defines as directed rounding toward minus and
plus infinity.  ACT does not copy or modify PyRAT.

The endpoint rule is deliberately stronger than route invariance:

```text
both complete static paths SAFE under directed rounding
    => dynamic hard-routed model SAFE for every router behavior
```

Every path must be covered, including tie-legal paths.  A static-path UNSAFE
result is not promoted to dynamic-model UNSAFE unless a concrete input is
replayed through the full dynamic model.  Timeouts, parser failures, missing
paths, and ambiguous logs all return UNKNOWN/TIMEOUT.

## Evidence identity

Every run records and hashes the checkpoint, CIFAR-10 archive, configuration,
two ONNX paths, every VNN-LIB property, every complete command line and log,
and the append-only row stream.  ONNX Runtime must match the specialized
PyTorch path within the frozen tolerance and the graph must contain no
`ArgMax`, `TopK`, `Gather`, or `GatherElements` operation.

PyRAT is an independently installed proprietary CEA tool.  Therefore it is an
external strict checker, not a redistributable ACT dependency and not an
open-source baseline.  The generated ONNX/VNN-LIB evidence remains portable;
the paper and artifact must disclose this reproducibility limitation.  This
stage does not weaken or relabel the historical CROWN result.

## Gates

The first pilot is restricted to the two previously observed 0.5/255
two-path-positive CROWN controls (ordered indices 1 and 13).  It is a backend
feasibility check, not a prevalence estimate and not evidence of improvement
over route invariance.  At least one strict two-path SAFE result is required
before any new route-changing cohort is frozen.

The frozen pilot configuration is
`configs/advmoe_strict_pyrat_seed0_compat_pilot_r1.json`.  It uses only the
DeepPoly-style `poly` domain, a 300-second per-path timeout, four CPU threads,
and the exact final-checkpoint and CIFAR-10 archive hashes already accepted by
the AdvMoE training audit.

Attempt r1 stopped before any PyRAT call because the fixed-batch ONNX semantic
check supplied the two probes as one batch.  The partial ONNX and failure
record are preserved.  R2 changes only this check to two batch-1 replays; all
scientific inputs and strict-backend settings remain unchanged.

R2 then stopped at the same pre-solve gate: one path had a `2.861023e-6`
ONNX-Runtime/PyTorch discrepancy, above the frozen `2e-6` sanity tolerance;
all probe predictions agreed.  R3 keeps BatchNorm as explicit ONNX nodes
instead of folding rounded affine constants, uses the direct comparison form
accepted by PyRAT's VNN-LIB grammar, and sets the concrete sanity tolerance to
`4e-6`.  That tolerance cannot create SAFE: formal acceptance still requires
PyRAT to return SAFE for both paths under directed rounding.

R3 completes all four registered path calls. Every call reaches the external
checker and is stopped by the 330-second outer deadline after the 300-second
PyRAT budget; consequently both dynamic endpoints are `TIMEOUT`, strict SAFE
is `0/2`, and the backend-feasibility gate is not met. The independent audit
reconstructs both properties, replays both 81-node ONNX graphs (21 BatchNorm
nodes each), verifies the complete directed-rounding command contract, and
recomputes the result with zero issues. Its first attempt is preserved: it
incorrectly compared raw version output containing a timestamped ONNX Runtime
warning. The accepted audit compares the semantic `PyRAT 2.0` token.

Per the frozen gate, no route-changing cohort is selected and no PyRAT domain,
timeout, or radius is tuned after this result. This negative result does not
show that either input is unsafe; it only shows that the external strict
backend did not finish the already-positive numerical controls in the frozen
budget. The compact result is
`results/advmoe_strict_pyrat_seed0_compat_pilot_20260906_r3.json`.
