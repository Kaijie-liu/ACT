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

R4 EXECUTED: both original float32 components BACKEND_POSITIVE, exact wrapper
probes [0,0,0], independent saved-record audit PASS. See
`metamoe_native_control_review_20260922_r4.json`. This resolves the deployment
blocker by bypassing ONNX, NOT by proving the old conversion equivalent.
Formal complete-model matched execution remains a separate freeze.

## Robust Experts and RoME follow-up freezes

Robust Experts600s control: native R3 source, E4/k1, original PGD7/SGD and
augmentation, two epochs with one batch2 each. Fresh process resumes epoch00;
epoch01 model/optimizer/PolyLR/global step/RNG AND next augmented batch must
match exactly. A named callback adds/restores Python/NumPy/Torch RNG to native
Lightning checkpoints. Not automatic long training; failure retains both
states. Three RNG/equality controls PASS. Configuration:
`robust_experts_resume_r1.json`.

RoME four-input/three-norm control: author's seed0 randperm selects
[6044,2890,9399,1917], no clean-correct/route filtering. All12 combinations
of Linf8/255, L1=12, L2=.5, native standardAutoAttack, per-input/norm600s,
CPU2threads, original public MAX prediction checkpoint/code defaults retained.
Separate process/model load charged each time. Timeout is incomplete, never
robust; errors stop new requests; union robustness requires all three completed
without attack. Two roster/timeout controls PASS. Batch freeze:
`rome_multinorm_batch_r1.json`. No automatic100-input expansion or ACT SAFE.

Robust Experts continuation has EXECUTED and passed in21.29s with postflight.
Independent saved-state reread matches all model/SGD/PolyLR/RNG components and
next augmented batch. Long training is still not launched. RoME batch is live;
inspect its terminal ledger, not this preparation text, for current progress.

MetaMoE full-request comparison implementation is prepared separately:
same full public20-class model, materialized float64 normalized2/255 boxes,
global margin1e-7 and300s per complete request. Author route-invariance arm
uses the native pinned backend with its original alpha/beta/BaB settings;
strict route dominance, selected-score nonzero AND all19 global output margins
include zero-filled other-domain classes. It is an explicitly adapted sufficient
path, not the author's unchanged component table. ACT covers tie-legal routes.
Source float32 execution is NOT claimed equivalent to this explicit snapshot.
Three analytic semantic controls pass (zero blocks, negative selected score,
tie/zero rejection). First freeze is OLD index0 CIFAR/MNIST smoke (4calls),
not the20 new-input experiment. The latter requires the smoke audit gate.

## Full-model smoke R1 failure and limited R2 interface repair

R1 terminal audit passes as a RECORD audit; execution gate FAILS. ACT CIFAR0
replays a real center violation (label3/pred5, independently reread), author
CIFAR0 times out. Author MNIST0 verifies its full20-class sufficient obligations;
ACT MNIST0 raises unsupported functional F.relu in TorchToACT. This is an
intake implementation blocker, not evidence of a negative HZ bound. Archives:
`metamoe_paired_smoke_review_20260922_r1.json` and
`metamoe_paired_replay_20260922_r1.json`. Do not run the formal cohort on R1.

R2 uses a separately versioned adapter: F.relu(x,inplace=False) becomes an
nn.ReLU call with the identical input edge. No parameter, normalization, pool,
BN statistic or bound acceptance is changed. Original model remains the witness
target. Reject inplace/training rewrites; exact output/gradient/source-state
controls pass. Both arms now share the cheap ORIGINAL full-model center check;
neither wastes its budget on an already witnessed violation. Runtime inventories
and original/backend/LiRPA Git identities are checked inside charged execution.
Same four OLD-input calls, same300s/epsilon/margin, new R2 identity; no effect-
based sample substitution. Formal selection remains gated by completed smoke.

## Robust Experts scientific long-training recipe (NOT executable freeze)

`robust_experts_paper_training_recipe_r1.json` resolves the scientific choices:
paper200epochs/lr.01 versus source100/.1, dense ResNet18 and ConvMoE E4/k2/layer4,
native batch640/SGD/PGD7/PolyLR/augmentation, finalepoch200 checkpoint only.
This follows the source-versus-paper ledger in `recent_moe_comparison_protocol_v1.md`.
No early stopping or verifier-based model selection; native attack/eval semantics
are retained under the explicitly named R3 compatibility variant.
CPU exact continuation has now passed; GPU-capable isolated workflow, BOTH final
architectures' full-batch update/save control, resource/deadline supervision and
resolved local output bindings remain REQUIRED before execution freeze/launch.
Do not claim CPU resume proves GPU equivalence, or that long training has begun.

RoME: three-norm empirical evaluation remains separate from ACT full-domain
support. Full-size compiler or forward compatibility is never certification.
All six author reproductions and ACT comparisons require separate status
columns; unavailable code and incompatible guarantees cannot be numeric wins.

## R2 outcome and bounded representation diagnosis

R2 removes the functional-ReLU intake error and both arms independently replay
CIFAR0's center violation. Author MNIST0 again BACKEND_POSITIVE. ACT MNIST0
fails because no joint HZ reaches the router output. A separately supervised90s
propagation-only diagnostic (NO candidate/property solve) completes2.26s:
the first router ReLU,20480 outputs, triggers `sparse_relu_size_limit` under
the existing64,000,000 affine-cell guard. All later layers lack the shared HZ.
`metamoe_hz_intake_diagnostic_20260922_r1.json` preserves per-layer evidence.
This is a precise representation-capacity blocker, NOT a completed negative
bound or proof that the model is unsafe. No cap increase, model reduction,
budget extension, sample change or formal cohort execution is performed.

## Robust Experts GPU deployment gate

A dedicated writable overlay inherits the existing Blackwell CUDA stack
READ-ONLY; base package inventory is unchanged before/after installation.
Torch2.11/cu130, Lightning1.9.5, NumPy1.26.4 and the named R3 compatibility
dependencies pass pip check. This is not the original author environment and
not a self-contained copied CUDA installation; inventory binds the inheritance.
No dependency was installed/upgraded in act-py312 or the CUDA base.

`robust_experts_gpu_step_r1.json` freezes600s each for dense and ConvMoE E4/k2:
batch640, native PGD7/SGD, one actual update, native full checkpoint save/reload;
no smaller-batch fallback, no evaluation/accuracy claim, no long training.
The200epoch scheduling horizon remains, max_steps=1 bounds this deployment
control. The resource gate requires48GiB currently free and caps own Torch
allocation at40% of total GPU memory; no other tenant is interrupted.
CPU recipe controls verify these invariants before execution. Exact GPU
continuation is NOT implied by successful one-step serialization.
