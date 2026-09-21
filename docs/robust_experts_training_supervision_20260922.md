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
