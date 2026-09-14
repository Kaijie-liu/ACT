# Second model family: convolutional output top-2, R1

This is a new model-family scope authorized by Advice/bb.md, not a reopening of
AdvMoE, a modification of bal010, or a new seed selected for easy verification.
The architecture and training rule are fixed **before training or verification**
in `configs/conv_family_training_r1.json`.

The model has four separate convolutional experts. Each uses 3x3 stride-2
convolutions with 16 and32 channels, ReLU after each, 2x2 average pooling,
flattening, a64-unit ReLU hidden layer and ten output logits. The router is an
affine map of 4x4 average-pooled pixels. Output semantics remain variable-weight
selected-softmax top-2, including all tie-legal unordered pairs in verification.
There is no BN, dropout, shared internal dispatch, frozen gate weight or new
unsupported max-pooling abstraction. Fewer parameters do not imply easier
verification: the spatial ReLU population is substantially larger than bal010's
MLP experts. Architecture changes after observing endpoint outcomes require a
new protocol with the failed attempt retained.

Seed17 determines initialization and the 45k/5k train/validation split. Training
is100 epochs, batch128, AdamW at1e-3, weight decay5e-4, cosine decay to zero,
cross-entropy plus0.01 Switch balance loss. Train-only random crop (padding4)
and horizontal flip (p=.5); validation and test use unaugmented [0,1] tensors.
Select highest validation accuracy, with the earliest exact tie. No verification
result or test accuracy selects a checkpoint. Retain the run even if accuracy
or routing diversity disappoints; improved accuracy is a goal, not an assertion.

The new versioned checkpoint format is `act-output-conv-moe-v1`; the old
`act-output-moe-v1` construction is unchanged. Correctness controls must establish
factory determinism, exact concrete checkpoint replay, ACT conv/pool lowering
and static-pair concrete parity before training/verification deployment. Small
controls are correctness tests, not candidate architecture screening on CIFAR.

The later evaluation will minimally compare frozen ACT adaptive, a reasonable
monolithic reference and ACT-fronted external variable-weight static pairs.
The external setting must pass a separate small conformance control before
selection of verification endpoints. Three-arm evaluator, exact selection
manifest and resources still need their own prelaunch freeze; this document
does not authorize calling any resulting model a successful cross-architecture
verification experiment before those runs and audits exist. Do not silently use
the old eight-expert runner assumptions for this four-expert family.

Initial stage: model construction and registration. See the later prelaunch
results and supervision protocol below; the three-arm evaluation remains pending.

Compatibility controls: four tests pass for RNG preservation, versioned exact
checkpoint replay, weighted concrete semantics, dyadic point lowering and a
nondegenerate sparse conv/pool box. Initial non-dyadic singleton testing exposed
inconsistent independent floating bounds at a ReLU; sparse propagation failed
closed. No tolerance or production rule was relaxed. This remains a numerical
limitation, not evidence that every full-sized CNN request will be consumable.
The dyadic control checks arithmetic/layout separately; it does not quantize the
registered training model. Full-shape compatibility is still a prelaunch gate.

## Full-shape gate frozen before execution

`conv_compatibility` constructs the registered seed17 model in float32, serializes
it, and supplies the identical checkpoint and float64 represented box to both
environments. The synthetic center is all .5, with radius2/255 and four fixed
concrete probes. ACT must retain exact SparseHZ for the router and all four full
experts (dense fallback disabled), and all six static pairs must match their
explicit variable-weight expressions. External plain CROWN uses the existing
pinned environment, CPU/float64, matrix convolution and pair{0,1}, nine fixed
class0 margins. It must return finite ordered bounds and pass lowered concrete
conformance, not positive bounds. Each backend has300 seconds; failures remain.
Probe tolerance1e-10 is a conversion check, not a SAFE acceptance policy.
This gate does not establish trained-model scalability or any output certificate.

First full-size attempt `conv_fullshape_20260915_r1` retained all five ACT sparse
components and passed all six concrete pair probes. External graph conversion
and concrete parity passed, but CROWN convolution created a float32 intermediate
against float64 weights: the runner had omitted setting the default dtype.
The failed attempt is retained. R2 only sets the declared float64 default after
loading the same float32-initialized model; no model, box, backend or tolerance
is changed. The registered non-dyadic singleton limitation remains unchanged.

## Training supervision, fixed before launch

`conv_training_supervisor` requires a clean pushed source, the successful full-
shape gate and a hash-bound CUDA smoke from the current worker. It archives that
Git revision into the run's own source directory. Training/audit run from this
snapshot, never from later checkout edits. Dependencies are unchanged; the
launch records the existing act-py312 package inventory, GPU/driver, source tar,
recipe, split and seven CIFAR raw-file hashes. Dataset downloads are disabled.

Numeric execution is float32, no AMP or TF32, deterministic algorithms enabled,
cuDNN benchmark disabled. Unsupported deterministic operations fail rather than
silently falling back. Seed17 defines the model and the randperm45k/5k split;
each training epoch uses shuffle/worker generator17+epoch; validation uses
10000+epoch and final test20000. Workers seed Python/NumPy from their Torch seed.
Two workers and two CPU threads, GPU0, nice10. This execution detail supplements
the frozen recipe; it changes none of its hyperparameters or selection rules.

Before training and independent final evaluation, require8GiB free GPU and10GiB
free disk. Otherwise wait30s, at most24h. Every25 batches write an atomic
heartbeat. A dead child produces FAILED; a live child with heartbeat older than
30min produces STALLED_SUSPECTED, not a false failure. Read races receive three
retries. Owned training/audit limits are72h/1h. No other job is stopped.

Each completed epoch retains an immutable checkpoint with model, AdamW and cosine
states, RNG and loader-generator states; atomic epoch metadata binds its hash.
An earliest maximum **validation correct count** selects the checkpoint. All
100 epochs run even if validation stagnates. Test is evaluated only after that
choice. An independent process reconstructs the split, checks all100 epoch
denominators, LR values and checkpoint hashes/metadata, and reloads the selected
model to repeat full5k validation and10k test. Only then write
`CONV_LANDED_summary.json` with `LANDED_AUDITED`.

There is no automatic retry/resume, no test/verification-based model selection,
no dependency installation and no training-result push from the background
worker. Failed attempts and partial checkpoints remain. RNG/optimizer retention
enables a later explicitly audited recovery; it is not a claim that arbitrary
mid-epoch restarts are already supported. The outer supervisor uses a family-
wide lock. Its final artifact is available locally even if this chat is closed.

CUDA smoke is a separate discarded two-augmented-batch control using the exact
full architecture and training hyperparameters. It checks finite gradients,
nonzero router update, exact checkpoint inference replay and one identical
optimizer continuation update after restore. It never supplies production
weights. Unit controls additionally cover immutable checkpoints, complete split,
earliest ties, cosine endpoint, read races, resource waits and dead/stale workers.

## Prelaunch results (2026-09-15)

Compact review: `../results/conv_pretraining_review_20260915_r1.json`, generated
by `archive_conv_controls`. Both attempts retain the identical checkpoint and
represented-input hashes. R2 fullshape ACT passes in0.886s worker time with
771.54MiB peak RSS; CROWN passes in0.590s with1052.42MiB peak RSS and84 graph
nodes. Timings are diagnostic worker timings, not end-to-end competitiveness.
No exact/sound numerical certification claim follows from conformance probes.

CUDA smoke passes in2.63s, peak allocated84.42MiB. Two batches produce nonzero
router gradient and maximum parameter update0.0020014; versioned checkpoint
inference replay is bitwise equal and the restored optimizer's next identical
update matches. Those weights are discarded. The100-epoch production run must
start from the original seed17 factory, not this smoke checkpoint.
