# Robust Experts: bounded native training workflow

This stage adds execution infrastructure, not a new model/training recipe or
ACT certificate. The paper-recipe JSON remains unchanged. Neither the prior
CPU E4/k1 continuation control nor one GPU update establishes a full run.

`robust_experts_supervised_pipeline.py` executes, separately for dense and
E4/k2 layer4 ConvMoE, fresh subprocesses for native training, final-weight
evaluation, and saved-state auditing. One outer process-group watchdog covers
all three stages. It may kill only its own child process group. An exception,
missing audit, changed identity, or late output cannot become a success.
Partial stdout/stderr, stage journals and completed epoch checkpoints survive.
The two architectures are serial; failed first architecture stops the batch.

Training uses the frozen native SGD/PGD7/augmentation/PolyLR recipe, seed12345,
batch640, workers2. Only the completed final epoch is selected. Every completed
epoch retains a full native checkpoint with model, optimizer, scheduler,
CPU/Python/NumPy/CUDA RNG and request binding. No validation-based selection,
automatic retries or automatic resume. Exact GPU continuation is NOT claimed;
checkpoint presence alone is not a continuation-equivalence experiment.

Fresh-process evaluation strictly loads the final checkpoint and executes native
clean, PGD20, APGD20 paths, checking their shared full-model identity and no
prediction-state/BN drift after each call. Clean uses batch640; attacks retain
the author configuration batch256. Full production evaluation is the ordered
10,000 CIFAR100 test inputs. These are EMPIRICAL results, not certification.

The current R1 control instead runs TWO native full640 updates over two limited
epochs, one validation batch each, then one640 clean and one256 batch per attack.
The PolyLR horizon stays200. Each architecture has900 seconds including startup,
imports, data/source/environment checking, training, save, evaluation and saved
audit. The launch guard and final supervisor postflight are separately charged
in the batch end-to-end total, not hidden inside worker timings. Resource gates:
48GiB free GPU, own allocator cap40%, local-only AF_UNIX IPC, no model/batch/dtype
fallback. Controls require5GiB disk; any future200epoch execution requires128GiB.

Local tests cover deadline/partial output, exceptions/non-overwrite, identity
mutation, incomplete stage roster, late completion rejection, serializable
state identity and retained native recipe. Actual R1 execution must follow the
committed configuration freeze. Its result is not known at this protocol entry.

Full training mode is explicitly gated by a SEPARATE execution configuration.
Passing this control does not automatically start200epochs or alter ACT's
unsupported full-size intermediate-router certification boundary.

## R1 result and R2 storage repair

Both R1 architectures completed the two updates, final-checkpoint loading,
three native evaluations and saved audit within900s. The independent saved
review is `robust_experts_pipeline_archive_20260922_r1.json`. However, the native
CSV logger reused `local/pipeline` during the fresh evaluation process and
overwrote its earlier training CSV. Epoch journals, checkpoints, stdout and
separate evaluation JSON survived; lost CSV entries are NOT reconstructed.

Separate R2 changes ONLY CSV version to `train` / `evaluate`, requires both
files at terminal, and retains all source settings, sample limits, budgets and
R1 outputs. One exact configuration-difference test confirms the only changed
field. The eight deadline/exception/identity controls remain unchanged.
Commit the new R2 configuration before execution. No long training launch is
implied by either control.

## R2 pass and separate long-training freeze

R2 completed at8b6f06b12: dense20.402s, ConvMoE27.464s including startup, native
updates, final-weight evaluation and in-budget saved audit. Fresh independent
saved review passes; both train and evaluation CSV files remain. See
`robust_experts_pipeline_archive_20260922_r2.json`. No short metric is presented
as trained-model RA. One further control verifies that production configuration
has200epochs and NO short-control batch/step limits, preserving the scientific
recipe and stage-isolated logging.

`robust_experts_paper_training_execution_r1.json` is a NEW long execution
identity: native full train/val, final200epoch checkpoint, full10k clean/PGD20/
APGD20 and saved audit,24h total per architecture, dense then ConvMoE. A failed
arm stops the sequence; no automatic retries or restarts. Launch requires clean
committed branch,48GiB free GPU and128GiB disk. Source/environment identities are
checked again in each subprocess. The detached launcher writes only outside
the checkout; no automatic Git commit/push. Results require separate final
archive after the terminal exists. A freeze/launch record is NOT completion.

## Full training landed and separately reviewed

Both arms now completed200epochs/12600updates and all10k clean/PGD20/APGD20.
See [landing review](robust_experts_landing_20260922_r1/README.md), including all
trajectory data, final weight identities, independent saved-file checks and
the paper/source/actual-recipe comparison. Dense37.27/18.99/18.74%; ConvMoE
22.18/12.88/11.70%, empirical only. No trained full-domain ACT result follows.
The source `entropy` includes a per-sample entropy term absent from the printed
paper objective. This run remains source-entropy; it is not relabelled as a
paper-formula reproduction. Native top2 also disables the requested STE flag.
No new training, attack, checkpoint choice or frozen-source edit was made.
