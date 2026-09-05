# AdvMoE two-path verification

## Frozen semantics

The accepted subject is the final epoch-100 checkpoint from the explicitly
labeled numerical-compatibility training run. It is not relabeled as unchanged
official training. The released source remains read/execute-only because it has
no license file, and the checkpoint is identified by hash rather than copied
into ACT.

For each deterministic clean-correct input and radius, the runner constructs
the clipped unit-pixel box and evaluates three compositions with the same plain
CROWN backend:

1. route invariance: a numerical positive lower bound for the selected router
   margin plus a positive property bound for the selected static path;
2. router-independent Route A: positive property bounds for both static paths;
3. a tie-safe eta implication for each path as a bounded guard-representation
   ablation.

AdvMoE has one global hard route shared by all 16 MoE convolutions. Replacing
each routed convolution with the selected contiguous weight slice therefore
produces exactly two static networks, not `2^16` paths. The CROWN adapter also
replaces the final fixed-shape adaptive average pool with `AvgPool2d(4)`. The
literal and lowered paths must agree within `1e-7`; their raw maximum error and
prediction agreement are retained. The nonlinear router uses the already
validated fixed-shape adapter.

The installed CROWN backend is not outward rounded. Consequently, positive
lower bounds are reported only as `CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE`.
Negative or incomplete bounds are `UNKNOWN`, never `UNSAFE`. Only a concrete
input inside the registered box whose prediction flip replays through the full
dynamic model may be labeled `UNSAFE_FULL_FORWARD_REPLAY`.

## Correctness smoke

The first run is frozen at the first ordered clean-correct test input and
`0.5/255`. It exercises one router bound, both static paths, both eta-compiled
implications, a two-restart 20-step full-model attack, lowering equivalence,
incremental row persistence, and artifact hashing. It is an engineering smoke,
not a prevalence or certification result. Only a zero-error, zero-conflict
independent audit may unlock the 20-sample five-radius execution.
