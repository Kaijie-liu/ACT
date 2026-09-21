# Author execution follow-up (2026-09-22)

## Frozen before real certification / conversion controls

Dual RS training landed at epoch90 with33,300 updates. Final checkpoint SHA256
`5bcf3fd0ad3b28cab728fa3e56c77f3300cae572f864ff8b3d78bcf342ad07cc`.
The saved-only90epoch re-audit is compared with the original final audit by
`archive_dual_rs_landing.py`; no training, checkpoint selection or Monte Carlo
resampling occurs in that check. The first read-only re-audit was stopped
because default Torch CPU threading oversubscribed the machine; the identical
check is rerun with OMP/MKL/OpenBLAS capped at2. This is not a training retry.
The35.692% epoch90 test metric is sigma-label accuracy, NOT class accuracy.

The new `dual_rs_certification_execution_r1.json` binds final weights and the
existing recipe. Raw inputs0,1, N0=100/N=10,000 per executed stage, alpha=.0005
per stage, native precision, GPU gate24GiB, shared7200s unchanged. Execution
starts only after this binding is committed. Statistical smoothed-function L2
results are separate from ACT deterministic HZ-policy results.

MetaMoE R3 is a separate conversion variant on exactly the old two component
requests, NOT a paper-table or full-model comparison. No BN parameter folding,
ONNX simplification or runtime graph optimization; original eval BN statistics
and dtype retained. Three export controls pass, including train-mode and
batch-statistics rejection;13 existing spec/terminal tests pass. Source/ONNX
probe tolerance remains1e-4. Same300s backend/360s outer component budgets,
same checkpoints, dataset indices and property grammar. Original R1/R2 files
and failures are untouched. Positive finite probes would not establish domain
equivalence or a deployed-float guarantee. The execution freeze is
`metamoe_component_control_r3.json`; original backend and environment unchanged.

## Other work lines

## Executed outcomes and next conversion freeze

Dual RS real pilot completed in43.6797s (43.7190s with source postflight).
Both raw inputs classified correctly; selected sigma.5/1 and composed L2
radii0.8304935215/1.1060614554. Independent saved-count audit agrees.
See `dual_rs_certification_archive_20260922_r1.json`; this is statistical,
native-numerical smoothed-function evidence, NOT deterministic HZ SAFE or CRA.

MetaMoE R3: CIFAR BACKEND_POSITIVE, MNIST ERROR. Even WITHOUT folded BN,
the original1e-4 gate fails at1.6701221466e-4. No tolerance change or erased
failure: `metamoe_component_control_review_20260922_r3.json`.
R4 now freezes the pinned author's backend `Customized` PYTORCH loader, bypassing
ONNX entirely. Same original float32 component checkpoint, spec, backend,
settings, two inputs and budgets. Wrapper-removal probes must be exactly equal.
This is a named front-end adaptation, not a claim to have fixed or reproduced
the author's original exporter. No extra backend parameters are searched.

Robust Experts: exact complete-state training continuation must precede a long
training freeze; R3 one-batch workflow alone is not this control.
RoME: three-norm empirical evaluation remains separate from ACT full-domain
support. Full-size compiler or forward compatibility is never certification.
All six author reproductions and ACT comparisons require separate status
columns; unavailable code and incompatible guarantees cannot be numeric wins.
