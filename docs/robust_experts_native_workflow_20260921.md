# Robust Experts: native workflow and eval-state boundary

Author source ed22e81fabc3c3196b6bcd352ee83042473cdfbf and all old runs remain
unchanged. These are deployment controls, NOT paper accuracy or HZ certificates.

## R1 and R2

R1 stopped before training: Lightning1.9 references np.Inf, removed in NumPy2.
R2 uses a separate NumPy1.26.4/OpenCV4.11/tifffile2024.8.30 environment; pip check
and native configuration controls pass. Same600s outer, source-default E4/k1
layer4 ConvMoE, original transforms/SGD/PGD7, batch2 and one batch per phase.
R2 executes training, clean test, native PGD20 and APGD20 in13.1597s including
postflight;74 parameter tensors changed, finite state, exact tensor-only reload.
All test accuracies are0/2: NOT meaningful paper metrics or a failed model search.

The optional remote-checkpoint logging branch warns about absent matplotlib;
local Lightning last.ckpt and tensor-only trained_state.pt exist. Network
connections are prohibited. No installation is needed to make remote logging
work, and this warning is not a local checkpoint-save failure.

Independent saved-only audit finds60 changed BN buffers between training-end
last.ckpt and post-evaluation trained_state.pt:20 means, variances and counters.
Each counter increments once; parameter tensors remain equal. Thus the native
workflow PASS does NOT imply evaluation of one frozen checkpoint. Preserve the
separate `SAVED_CHECKPOINT_PREDICTION_MISMATCH` result; no threshold was relaxed.
The audit was added after R2 and first binds last.ckpt's hash then, not retroactively.

## Bounded source diagnosis and R3 variant

`src/utils/attack.py:auto_pgd` creates a fresh WrappedModel whose root training
flag defaults True, even when its shared child is eval. torchattacks saves that
root flag and calls model.train() on restoration. Subsequent prediction then
updates the shared BN buffers. A miniature native APGD control reproduces this:
unchanged attack tensor, but one BN update after the next original-model call.

R3 separately sets only `wrapped_model.training = model.training` immediately
after construction. It does not recursively change children at construction,
change gate normalization, loss, optimizer, PGD/APGD iterations or input data.
The eval control has no buffer drift; a train-mode caller still restores train.
This is an explicit eval-mode semantic compatibility variant, NOT a claim of
byte-identical original execution or reproduced paper results. Its source copy
and combined patch are separate; R2 stays immutable.

R3 native control uses the identical600s budget and mathematical configuration,
a new output root and the unchanged frozen worker. Freeze before execution.
Required follow-up audit: all147 prediction tensors must equal the actual
training-end checkpoint, SGD buffers finite, scheduler position1. This audit is
not exact training continuation; long training still requires a real full-state
resume control and a separately bound training recipe.

## Evidence

- `robust_experts_workflow_archive_20260921_r1.json`: preserved initial failure.
- `robust_experts_workflow_archive_20260921_r2.json`: native deployment completion.
- `robust_experts_saved_state_audit_20260921_r2.json`: strict mismatch and60 fields.
- `tests/test_robust_experts_apgd_mode.py`: actual native attack mode controls.
- `configs/recent_moe/robust_experts_workflow_r3.json`: separately frozen variant.

This work does not implement complete intermediate-route HZ lowering. Original
model execution intake remains distinct from whole-domain verification support.
