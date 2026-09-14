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

Current stage: model construction and registration. Training supervisor,
external conformance and the three-arm evaluation are not yet completed.

Compatibility controls: four tests pass for RNG preservation, versioned exact
checkpoint replay, weighted concrete semantics, dyadic point lowering and a
nondegenerate sparse conv/pool box. Initial non-dyadic singleton testing exposed
inconsistent independent floating bounds at a ReLU; sparse propagation failed
closed. No tolerance or production rule was relaxed. This remains a numerical
limitation, not evidence that every full-sized CNN request will be consumable.
The dyadic control checks arithmetic/layout separately; it does not quantize the
registered training model. Full-shape compatibility is still a prelaunch gate.
